"""
ML Predictor Service.

Runs the LSTM + SARIMAX ensemble to predict crop suitability for a given
region, season, and feature vector.

Model selection rule (from architecture spec):
  - SARIMAX: always run (handles seasonal baselines + <3 years of data).
  - LSTM:    run when ml/artifacts/{region_code}/lstm_model.pt exists
             (requires ≥3 years of weather + yield history).
  - Ensemble: weighted average when both models are available.

Artifacts layout (one directory per region_code):
    ml/artifacts/{region_code}/
        scaler.pkl              # sklearn StandardScaler fitted on training data
        sarimax_results.pkl     # statsmodels SARIMAXResults serialised with pickle
        lstm_model.pt           # PyTorch state_dict for the LSTM network
        lstm_config.json        # LSTM hyperparameters (input_size, hidden_size, etc.)
        crop_index.json         # {crop_name: int_label, ...} mapping
        model_meta.json         # training date, region, data range, ensemble weights
"""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from agents.state import CropPrediction, GISFeatures, WeatherSnapshot
from ml.pipeline.features import FEATURE_COLUMNS, SHC_FEATURE_COLUMNS, SHC_FEATURE_DEFAULTS

logger = structlog.get_logger(__name__)

_ARTIFACTS_ROOT = Path("ml/artifacts")

_FEATURE_COLUMNS = FEATURE_COLUMNS

# SARIMAX seasonal order — (P, D, Q, s) where s=2 for biannual kharif/rabi cycle
_SARIMAX_ORDER  = (1, 1, 1)
_SARIMAX_SEASONAL_ORDER = (1, 1, 0, 2)


class ModelNotFoundError(Exception):
    """Raised when no trained artifacts exist for the requested region_code."""


# ── LSTM network definition (must match architecture used during training) ────

class _CropLSTM(nn.Module):
    """LSTM network for crop yield prediction.

    Architecture:
        LSTM(input_size, hidden_size, num_layers) → Linear(hidden_size, num_crops)
        Output is a raw logit vector — apply softmax to get crop confidence scores.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_crops: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_crops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logit tensor of shape (batch, num_crops).
        """
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ── Artifact loading ──────────────────────────────────────────────────────────

def _artifact_dir(region_code: str) -> Path:
    return _ARTIFACTS_ROOT / region_code


def _load_scaler(region_code: str) -> StandardScaler:
    path = _artifact_dir(region_code) / "scaler.pkl"
    if not path.exists():
        raise ModelNotFoundError(
            f"No scaler found at {path}. Run ml/train/train_sarimax.py --region {region_code} first."
        )
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local artifact


def _load_crop_index(region_code: str) -> dict[str, int]:
    path = _artifact_dir(region_code) / "crop_index.json"
    if not path.exists():
        raise ModelNotFoundError(f"No crop_index.json at {path}.")
    return json.loads(path.read_text())


def _load_model_meta(region_code: str) -> dict[str, Any]:
    path = _artifact_dir(region_code) / "model_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_drift_report(region_code: str) -> dict[str, Any]:
    """Load drift report if present.

    Args:
        region_code: Runtime region identifier.

    Returns:
        Drift report dictionary or empty dict when not available.
    """
    path = _artifact_dir(region_code) / "drift_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_sarimax(region_code: str) -> Any:
    path = _artifact_dir(region_code) / "sarimax_results.pkl"
    if not path.exists():
        raise ModelNotFoundError(
            f"No SARIMAX results at {path}. Run ml/train/train_sarimax.py --region {region_code}."
        )
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local artifact


def _load_lstm(region_code: str, num_features: int, num_crops: int) -> _CropLSTM | None:
    """Load the LSTM model if artifacts exist, else return None."""
    config_path = _artifact_dir(region_code) / "lstm_config.json"
    weights_path = _artifact_dir(region_code) / "lstm_model.pt"

    if not config_path.exists() or not weights_path.exists():
        return None

    config = json.loads(config_path.read_text())
    model = _CropLSTM(
        input_size=num_features,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        num_crops=num_crops,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Feature vector construction ───────────────────────────────────────────────

def _build_feature_vector(
    gis: GISFeatures,
    weather: WeatherSnapshot,
) -> tuple[np.ndarray, list[str]]:
    """Build an ordered feature vector from GIS + weather state dicts.

    Logic Flow:
        Reads core values in FEATURE_COLUMNS order.
        Appends SHC biological health features when present in GIS state;
        falls back to SHC_FEATURE_DEFAULTS otherwise to keep vector shape stable.
        Soil texture (categorical) is intentionally excluded — SARIMAX and LSTM
        both operate on numeric features only.

    Args:
        gis:     GISFeatures TypedDict from the graph state.
        weather: WeatherSnapshot TypedDict from the graph state.

    Returns:
        Tuple of (feature array of shape (1, n_features), list of feature column names used).
    """
    row = [
        gis["soil_nitrogen"],
        gis["soil_phosphorus"],
        gis["soil_potassium"],
        gis["soil_ph"],
        gis["elevation_m"],
        gis["slope_deg"],
        weather["rainfall_7d_mm"],
        weather["temp_avg_c"],
        weather["temp_min_c"],
        weather["temp_max_c"],
        weather["ndvi"],
    ]
    cols = list(FEATURE_COLUMNS)

    # Append SHC features if available; fall back to defaults
    for shc_col, default_key in [
        ("soil_organic_carbon_pct", "soil_organic_carbon_pct"),
        ("soil_ec_ds_m", "soil_ec_ds_m"),
        ("climate_anomaly_trend_mm", "climate_anomaly_trend_mm"),
        ("climate_temp_anomaly_c", "climate_temp_anomaly_c"),
    ]:
        row.append(float(gis.get(shc_col, SHC_FEATURE_DEFAULTS[default_key])))  # type: ignore[call-overload]
        cols.append(shc_col)

    return np.array(row, dtype=np.float32).reshape(1, -1), cols


def _detect_feature_anomalies(raw_feature_map: dict[str, float], drift_report: dict[str, Any]) -> tuple[bool, str]:
    """Detect whether live features are outside historical drift-safe bounds.

    Logic Flow:
        Reads per-feature anomaly bounds from drift_report.
        Flags anomalies when current value breaches both IQR and tail bounds.

    Args:
        raw_feature_map: Current GIS+weather features keyed by FEATURE_COLUMNS.
        drift_report: Drift report generated during training.

    Returns:
        Tuple of (is_anomalous, reason_text).
    """
    features_meta = drift_report.get("features", {}) if drift_report else {}
    breaches: list[str] = []

    for feat, value in raw_feature_map.items():
        feat_meta = features_meta.get(feat, {})
        bounds = feat_meta.get("anomaly_bounds", {})
        if not bounds:
            continue

        iqr_low = float(bounds.get("iqr_low", value))
        iqr_high = float(bounds.get("iqr_high", value))
        tail_low = float(bounds.get("tail_low", value))
        tail_high = float(bounds.get("tail_high", value))

        outside_iqr = value < iqr_low or value > iqr_high
        outside_tail = value < tail_low or value > tail_high
        if outside_iqr and outside_tail:
            breaches.append(
                f"{feat}={round(value, 3)} outside [{round(tail_low, 3)}, {round(tail_high, 3)}]"
            )

    if breaches:
        return True, "; ".join(breaches[:3])

    return False, "none"


def _compute_soil_health_index(gis: GISFeatures) -> float:
    """Compute a 0.0–1.0 composite soil health index from SHC indicators.

    Logic Flow:
        1. Organic carbon score: OC% mapped to 0-1 (0.5% = 0.5, 1.0%+ = 1.0).
        2. EC penalty: EC > 2 dS/m starts to penalise; EC > 4 = severe.
        3. Trend bonus: improving OC trend adds 0.1; declining deducts 0.15.
        4. Biological collapse risk: deducts 0.20 if flagged.
        Clamps result to [0.0, 1.0].

    Args:
        gis: GISFeatures TypedDict (may contain SHC fields or their defaults).

    Returns:
        Composite soil health index between 0.0 and 1.0.
    """
    oc = float(gis.get("soil_organic_carbon_pct", SHC_FEATURE_DEFAULTS["soil_organic_carbon_pct"]))  # type: ignore[call-overload]
    ec = float(gis.get("soil_ec_ds_m", SHC_FEATURE_DEFAULTS["soil_ec_ds_m"]))  # type: ignore[call-overload]
    oc_trend = str(gis.get("organic_carbon_trend", "unknown"))  # type: ignore[call-overload]
    collapse_risk = bool(gis.get("biological_collapse_risk", False))  # type: ignore[call-overload]

    # OC score: 0.5% → 0.5, 0.75% → 0.75, 1.0%+ → capped at 1.0
    oc_score = float(np.clip(oc, 0.0, 1.0))

    # EC penalty: saline soils (>2 dS/m) reduce health
    if ec > 4.0:
        ec_penalty = 0.30
    elif ec > 2.0:
        ec_penalty = 0.15
    else:
        ec_penalty = 0.0

    trend_adj = {"improving": 0.10, "stable": 0.0, "declining": -0.15}.get(oc_trend, 0.0)
    collapse_adj = -0.20 if collapse_risk else 0.0

    return float(np.clip(oc_score - ec_penalty + trend_adj + collapse_adj, 0.0, 1.0))


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_sarimax(
    sarimax_results: Any,
    feature_vec: np.ndarray,
    crop_index: dict[str, int],
) -> dict[str, float]:
    """Generate per-crop confidence scores from the fitted SARIMAX model.

    Logic Flow:
        SARIMAX was trained per-crop (one model per crop × region).
        sarimax_results is a dict[crop_name, SARIMAXResults].
        For each crop, call .forecast(steps=1) with exog=feature_vec.
        Normalise raw forecasts to [0, 1] via softmax.

    Args:
        sarimax_results: Dict mapping crop_name → fitted SARIMAXResults object.
        feature_vec:     Scaled feature vector of shape (1, n_features).
        crop_index:      Crop name → integer label mapping.

    Returns:
        Dict of crop_name → confidence score (0-1, sum ≈ 1).
    """
    raw: dict[str, float] = {}
    for crop_name, result in sarimax_results.items():
        try:
            forecast = result.forecast(steps=1, exog=feature_vec)
            raw[crop_name] = float(np.clip(forecast[0], 0.0, None))
        except Exception:  # noqa: BLE001
            raw[crop_name] = 0.0

    if not raw or sum(raw.values()) == 0.0:
        # Fallback: uniform distribution
        return {c: 1.0 / len(crop_index) for c in crop_index}

    total = sum(raw.values())
    return {c: v / total for c, v in raw.items()}


def _run_lstm(
    model: _CropLSTM,
    feature_vec: np.ndarray,
    crop_index: dict[str, int],
) -> dict[str, float]:
    """Generate per-crop confidence scores from the LSTM model.

    Logic Flow:
        Wraps feature_vec in a (1, 1, n_features) tensor (batch=1, seq=1).
        Passes through model → softmax → confidence dict.

    Args:
        model:       Loaded _CropLSTM in eval mode.
        feature_vec: Scaled feature vector of shape (1, n_features).
        crop_index:  Crop name → integer label mapping.

    Returns:
        Dict of crop_name → confidence score (0-1, sum ≈ 1).
    """
    x = torch.from_numpy(feature_vec).unsqueeze(0)  # (1, 1, n_features)
    with torch.no_grad():
        logits = model(x)                             # (1, num_crops)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    idx_to_crop = {v: k for k, v in crop_index.items()}
    return {idx_to_crop[i]: float(probs[i]) for i in range(len(probs)) if i in idx_to_crop}


def _ensemble(
    sarimax_scores: dict[str, float],
    lstm_scores: dict[str, float] | None,
    meta: dict[str, Any],
) -> tuple[dict[str, float], str]:
    """Blend SARIMAX and LSTM confidence scores via weighted average.

    Logic Flow:
        Reads ensemble weights from model_meta.json (defaults: 0.4 SARIMAX, 0.6 LSTM).
        If lstm_scores is None, returns SARIMAX scores unmodified.
        Normalises blended scores to sum to 1.

    Args:
        sarimax_scores: Per-crop confidence from SARIMAX.
        lstm_scores:    Per-crop confidence from LSTM, or None.
        meta:           model_meta.json content (may be empty dict).

    Returns:
        Tuple of (blended score dict, model_used label string).
    """
    if lstm_scores is None:
        return sarimax_scores, "sarimax"

    w_sarimax = float(meta.get("ensemble_weight_sarimax", 0.4))
    w_lstm     = float(meta.get("ensemble_weight_lstm",    0.6))

    crops = set(sarimax_scores) | set(lstm_scores)
    blended = {
        c: w_sarimax * sarimax_scores.get(c, 0.0) + w_lstm * lstm_scores.get(c, 0.0)
        for c in crops
    }
    total = sum(blended.values()) or 1.0
    return {c: v / total for c, v in blended.items()}, "ensemble"


# ── Public service ─────────────────────────────────────────────────────────────

class MLPredictorService:
    """Loads region-specific ML artifacts and runs the crop prediction ensemble."""

    async def predict(
        self,
        gis_features: GISFeatures,
        weather_snapshot: WeatherSnapshot,
        region_code: str,
        season: str,
        top_n: int = 20,
    ) -> list[CropPrediction]:
        """Predict crop suitability scores for a location/season.

        Logic Flow:
            1. Load scaler + crop_index + model_meta from artifacts dir.
            2. Build and scale the feature vector from GIS + weather state.
            3. Run SARIMAX inference (always).
            4. Run LSTM inference if lstm_model.pt artifact exists.
            5. Ensemble the scores; sort by confidence descending.
            6. Return top_n CropPrediction dicts with yield_estimate_kg_ha
               computed from the SARIMAX per-crop forecast value.

        Args:
            gis_features:    GISFeatures TypedDict from graph state.
            weather_snapshot: WeatherSnapshot TypedDict from graph state.
            region_code:     Runtime region identifier (e.g. 'IN').
            season:          Season string (e.g. 'kharif_2026').
            top_n:           Maximum number of predictions to return.

        Returns:
            Sorted list of CropPrediction TypedDicts (highest confidence first).

        Expected Exceptions:
            ModelNotFoundError: No artifacts in ml/artifacts/{region_code}/.
        """
        log = logger.bind(region_code=region_code, season=season)
        log.info("ml.predict.start")

        # All I/O is synchronous (file reads); wrap in executor to stay async
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            None,
            self._predict_sync,
            gis_features, weather_snapshot, region_code, season, top_n,
        )

        log.info("ml.predict.complete", num_predictions=len(predictions))
        return predictions

    def _predict_sync(
        self,
        gis_features: GISFeatures,
        weather_snapshot: WeatherSnapshot,
        region_code: str,
        season: str,
        top_n: int,
    ) -> list[CropPrediction]:
        """Synchronous prediction logic (runs in a thread executor)."""
        scaler      = _load_scaler(region_code)
        crop_index  = _load_crop_index(region_code)
        meta        = _load_model_meta(region_code)
        drift_report = _load_drift_report(region_code)
        sarimax     = _load_sarimax(region_code)
        lstm_model  = _load_lstm(region_code, len(_FEATURE_COLUMNS), len(crop_index))

        # Build and scale features
        raw_vec, feat_cols = _build_feature_vector(gis_features, weather_snapshot)
        raw_feature_map = {
            col: float(raw_vec[0, idx])
            for idx, col in enumerate(feat_cols)
        }
        scaled_vec = scaler.transform(raw_vec).astype(np.float32)

        # Inference
        sarimax_scores = _run_sarimax(sarimax, scaled_vec, crop_index)
        lstm_scores    = _run_lstm(lstm_model, scaled_vec, crop_index) if lstm_model else None
        blended, model_used = _ensemble(sarimax_scores, lstm_scores, meta)

        # Build yield estimates: use SARIMAX raw forecast values as proxy
        # (unit: kg/ha).  These are the per-crop historical yield averages
        # scaled by the sarimax confidence.
        avg_yield_by_crop: dict[str, float] = meta.get("avg_yield_kg_ha", {})
        uncertainty_profile = meta.get("uncertainty_profile", {})
        variability_index = float(uncertainty_profile.get("variability_index", 0.0))
        band_half_width = float(uncertainty_profile.get("suggested_relative_band_half_width", 0.20))
        band_half_width = float(np.clip(band_half_width, 0.05, 0.60))

        anomaly_flag, anomaly_reason = _detect_feature_anomalies(raw_feature_map, drift_report)
        soil_health_idx = _compute_soil_health_index(gis_features)
        fertilizer_suf = str(gis_features.get("n_sufficiency", "unknown"))  # type: ignore[call-overload]
        bio_collapse = bool(gis_features.get("biological_collapse_risk", False))  # type: ignore[call-overload]

        results: list[CropPrediction] = []
        for crop_name, confidence in sorted(blended.items(), key=lambda kv: -kv[1]):
            yield_est = avg_yield_by_crop.get(crop_name, 2000.0) * confidence * len(blended)
            yield_min = max(0.0, yield_est * (1.0 - band_half_width))
            yield_max = yield_est * (1.0 + band_half_width)
            calibrated_probability = float(np.clip(confidence * (1.0 - 0.4 * variability_index), 0.0, 1.0))
            results.append({
                "crop_name":               crop_name,
                "confidence":              round(confidence, 4),
                "probability":             round(calibrated_probability, 4),
                "yield_estimate_kg_ha":    round(yield_est, 1),
                "yield_min_kg_ha":         round(yield_min, 1),
                "yield_median_kg_ha":      round(yield_est, 1),
                "yield_max_kg_ha":         round(yield_max, 1),
                "uncertainty_band_pct":    round(band_half_width, 4),
                "anomaly_flag":            anomaly_flag,
                "anomaly_reason":          anomaly_reason,
                "model_used":              model_used,
                "fertilizer_sufficiency":  fertilizer_suf,
                "soil_health_index":       round(soil_health_idx, 3),
                "biological_collapse_risk": bio_collapse,
            })

        return results[:top_n]
