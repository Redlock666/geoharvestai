"""Data drift diagnostics for training feature stability and uncertainty profiling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from ml.pipeline.features import DRIFT_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


@dataclass
class DriftConfig:
    """Threshold configuration for drift warnings."""

    mean_delta_pct_warn: float = 20.0
    skew_delta_warn: float = 0.75
    slope_abs_warn: float = 0.05   # unit per year
    anomaly_iqr_multiplier: float = 1.5
    anomaly_tail_quantile: float = 0.05


def _calc_slope_per_year(times: pd.Series, values: pd.Series) -> float:
    """Compute linear trend slope per year.

    Args:
        times: Timestamp series.
        values: Numeric feature values.

    Returns:
        Estimated slope in feature-units per year.
    """
    if len(values) < 2:
        return 0.0

    x = (pd.to_datetime(times, utc=True).astype("int64") / 1e9) / (365.25 * 24 * 3600)
    y = values.astype(float).to_numpy()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _calc_skew(values: pd.Series) -> float:
    """Compute sample skewness with stable float-only arithmetic."""
    arr = values.astype(float).to_numpy()
    if arr.size < 3:
        return 0.0

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if abs(std) < 1e-12:
        return 0.0

    centered = arr - mean
    m3 = float(np.mean(centered ** 3))
    return m3 / (std ** 3)


def _calc_cv(values: pd.Series) -> float:
    """Compute coefficient of variation (std / mean)."""
    arr = values.astype(float).to_numpy()
    if arr.size < 2:
        return 0.0

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if abs(mean) < 1e-9:
        return 0.0
    return std / abs(mean)


def _safe_quantile(values: pd.Series, q: float, default: float = 0.0) -> float:
    """Return a quantile with empty-series guard."""
    arr = values.astype(float).to_numpy()
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def analyze_drift(exog_by_time: pd.DataFrame, config: DriftConfig | None = None) -> dict:
    """Build feature drift report with slope and skewness.

    Logic Flow:
        Splits the timeline into early-half and recent-half windows.
        For each canonical feature computes:
            - early_mean / recent_mean / pct delta
            - early_skew / recent_skew / skew delta
            - slope_per_year (global trend)
            - warn flag by configured thresholds

    Args:
        exog_by_time: Canonical feature dataset containing 'time' and features.
        config: Optional drift thresholds.

    Returns:
        Dict serialisable as JSON report.
    """
    cfg = config or DriftConfig()

    df = exog_by_time.copy().sort_values("time").reset_index(drop=True)
    mid = max(1, len(df) // 2)
    early = df.iloc[:mid]
    recent = df.iloc[mid:]

    features: dict[str, dict] = {}
    cv_list: list[float] = []
    slope_list: list[float] = []
    warn_count = 0

    for col in DRIFT_FEATURE_COLUMNS:
        if col not in df.columns:
            continue

        s_all = df[col].astype(float)
        s_early = early[col].astype(float)
        s_recent = recent[col].astype(float) if len(recent) > 0 else s_early

        early_mean = float(s_early.mean()) if len(s_early) else 0.0
        recent_mean = float(s_recent.mean()) if len(s_recent) else early_mean
        mean_delta_pct = 0.0 if abs(early_mean) < 1e-9 else ((recent_mean - early_mean) / abs(early_mean)) * 100.0

        early_skew = _calc_skew(s_early)
        recent_skew = _calc_skew(s_recent)
        skew_delta = recent_skew - early_skew
        cv = _calc_cv(s_all)

        slope = _calc_slope_per_year(df["time"], s_all)

        q_low = _safe_quantile(s_all, cfg.anomaly_tail_quantile)
        q_high = _safe_quantile(s_all, 1.0 - cfg.anomaly_tail_quantile)
        q1 = _safe_quantile(s_all, 0.25)
        q3 = _safe_quantile(s_all, 0.75)
        iqr = q3 - q1
        iqr_low = q1 - cfg.anomaly_iqr_multiplier * iqr
        iqr_high = q3 + cfg.anomaly_iqr_multiplier * iqr

        warn = (
            abs(mean_delta_pct) >= cfg.mean_delta_pct_warn
            or abs(skew_delta) >= cfg.skew_delta_warn
            or abs(slope) >= cfg.slope_abs_warn
        )
        if warn:
            warn_count += 1

        cv_list.append(cv)
        slope_list.append(abs(slope))

        recent_std = float(s_recent.std()) if len(s_recent) > 1 else 0.0
        early_std = float(s_early.std()) if len(s_early) > 1 else recent_std
        intra_variability_ratio = 1.0 if abs(early_std) < 1e-9 else recent_std / abs(early_std)

        features[col] = {
            "early_mean": round(early_mean, 6),
            "recent_mean": round(recent_mean, 6),
            "mean_delta_pct": round(mean_delta_pct, 4),
            "early_skewness": round(early_skew, 6),
            "recent_skewness": round(recent_skew, 6),
            "skewness_delta": round(skew_delta, 6),
            "coefficient_of_variation": round(cv, 6),
            "slope_per_year": round(slope, 6),
            "distribution": {
                "min": round(float(s_all.min()), 6),
                "p10": round(_safe_quantile(s_all, 0.10), 6),
                "median": round(float(s_all.median()), 6),
                "p90": round(_safe_quantile(s_all, 0.90), 6),
                "max": round(float(s_all.max()), 6),
            },
            "anomaly_bounds": {
                "tail_low": round(q_low, 6),
                "tail_high": round(q_high, 6),
                "iqr_low": round(iqr_low, 6),
                "iqr_high": round(iqr_high, 6),
            },
            "intra_variability": {
                "early_std": round(early_std, 6),
                "recent_std": round(recent_std, 6),
                "ratio_recent_vs_early": round(intra_variability_ratio, 6),
            },
            "warn": warn,
        }

    warn_features = [k for k, v in features.items() if v["warn"]]

    macro_variability = {
        "median_coefficient_of_variation": round(float(np.median(cv_list)) if cv_list else 0.0, 6),
        "max_coefficient_of_variation": round(float(np.max(cv_list)) if cv_list else 0.0, 6),
        "median_abs_slope_per_year": round(float(np.median(slope_list)) if slope_list else 0.0, 6),
    }

    inter_feature_variability = {
        "feature_count": len(features),
        "warn_feature_count": warn_count,
        "warn_feature_ratio": round((warn_count / max(len(features), 1)), 6),
    }

    intra_region_variability = {
        "median_recent_vs_early_std_ratio": round(
            float(np.median([
                v.get("intra_variability", {}).get("ratio_recent_vs_early", 1.0)
                for v in features.values()
            ])) if features else 1.0,
            6,
        )
    }

    # A single uncertainty index to propagate to inference artifacts.
    uncertainty_profile = {
        "variability_index": round(
            min(
                1.0,
                0.4 * macro_variability["median_coefficient_of_variation"]
                + 0.3 * macro_variability["median_abs_slope_per_year"]
                + 0.3 * inter_feature_variability["warn_feature_ratio"],
            ),
            6,
        ),
        "suggested_relative_band_half_width": round(
            min(
                0.6,
                0.12
                + 0.35 * macro_variability["median_coefficient_of_variation"]
                + 0.2 * inter_feature_variability["warn_feature_ratio"],
            ),
            6,
        ),
    }

    report = {
        "n_rows": len(df),
        "window_split": {
            "early_rows": len(early),
            "recent_rows": len(recent),
        },
        "thresholds": {
            "mean_delta_pct_warn": cfg.mean_delta_pct_warn,
            "skew_delta_warn": cfg.skew_delta_warn,
            "slope_abs_warn": cfg.slope_abs_warn,
            "anomaly_iqr_multiplier": cfg.anomaly_iqr_multiplier,
            "anomaly_tail_quantile": cfg.anomaly_tail_quantile,
        },
        "macro_variability": macro_variability,
        "inter_feature_variability": inter_feature_variability,
        "intra_region_variability": intra_region_variability,
        "uncertainty_profile": uncertainty_profile,
        "warn_features": warn_features,
        "features": features,
    }
    logger.info("drift.analysis.complete", warn_features=len(warn_features))
    return report


def save_drift_report(report: dict, out_dir: Path) -> None:
    """Save drift report JSON and flattened CSV.

    Args:
        report: Drift report output from analyze_drift().
        out_dir: Artifact directory for the region.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "drift_report.json").write_text(json.dumps(report, indent=2))

    flat_rows: list[dict] = []
    for feat, vals in report.get("features", {}).items():
        flat_rows.append({"feature": feat, **vals})

    if flat_rows:
        pd.DataFrame(flat_rows).to_csv(out_dir / "drift_report.csv", index=False)
