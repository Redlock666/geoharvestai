"""
Train SARIMAX per-crop models for a given region.

Reads crop yield history from TimescaleDB crop_yield_obs and weather
history from weather_obs.  Fits one SARIMAX(1,1,1)(1,1,0,2) model
per crop × region, then serialises:
    ml/artifacts/{region_code}/
        scaler.pkl           — StandardScaler fitted on weather+soil exog features
        sarimax_results.pkl  — dict[crop_name, SARIMAXResults]
        crop_index.json      — {crop_name: int_label}
        model_meta.json      — training metadata + avg yield per crop

Usage:
    python ml/train/train_sarimax.py --region IN
    python ml/train/train_sarimax.py --region IN --min-seasons 4 --top-crops 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import warnings
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import]

from ml.pipeline import FEATURE_COLUMNS, analyze_drift, build_training_bundle, save_drift_report

logger = structlog.get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)  # statsmodels convergence noise

_ARTIFACTS_ROOT = Path("ml/artifacts")

# SARIMAX order — (p,d,q)(P,D,Q,s) where s=2 (biannual kharif/rabi cycle)
_ORDER          = (1, 1, 1)
_SEASONAL_ORDER = (1, 1, 0, 2)

# Exogenous feature columns used in SARIMAX.
# Must match the canonical shared feature schema.
_EXOG_COLS = FEATURE_COLUMNS


# ── Data loading ──────────────────────────────────────────────────────────────

async def _load_yield_data(region_code: str) -> pd.DataFrame:
    """Load crop yield observations from TimescaleDB.

    Logic Flow:
        Connects to TimescaleDB using environment variables.
        Queries crop_yield_obs for all rows matching region_code.
        Returns a DataFrame with columns: time, state, district, crop_name,
        season, area_ha, production_t, yield_kg_ha.

    Args:
        region_code: Runtime region identifier (e.g. 'IN').

    Returns:
        DataFrame of yield observations.

    Expected Exceptions:
        asyncpg.PostgresConnectionError: TimescaleDB unreachable.
        ValueError: No yield data found for this region.
    """
    conn = await asyncpg.connect(
        host=os.environ.get("TIMESCALE_HOST", "localhost"),
        port=int(os.environ.get("TIMESCALE_PORT", "5433")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ.get("TIMESCALE_DB", "geoharvestai_ts"),
    )
    try:
        rows = await conn.fetch(
            """
            SELECT time, state, district, crop_name, season,
                   area_ha, production_t, yield_kg_ha
            FROM   crop_yield_obs
            WHERE  region_code = $1
            ORDER  BY time ASC
            """,
            region_code,
        )
    finally:
        await conn.close()

    if not rows:
        raise ValueError(
            f"No yield data for region_code={region_code}. "
            "Run scripts/ingest_apy.py first."
        )
    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["time"])
    logger.info("yield.data.loaded", rows=len(df), crops=df["crop_name"].nunique())
    return df


async def _load_weather_agg(region_code: str) -> pd.DataFrame:
    """Load aggregated national weather statistics by season from TimescaleDB.

    Logic Flow:
        Computes national monthly means from weather_obs.
        Joins to yield data via time (month) for exogenous SARIMAX features.
        Returns one row per (season_year, month).

    Args:
        region_code: Runtime region identifier (e.g. 'IN').

    Returns:
        DataFrame with columns: month, rainfall_7d_mm, temp_avg_c,
        temp_min_c, temp_max_c.
    """
    conn = await asyncpg.connect(
        host=os.environ.get("TIMESCALE_HOST", "localhost"),
        port=int(os.environ.get("TIMESCALE_PORT", "5433")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ.get("TIMESCALE_DB", "geoharvestai_ts"),
    )
    try:
        rows = await conn.fetch(
            """
            SELECT
                date_trunc('month', time)   AS month,
                SUM(rainfall_mm)            AS rainfall_7d_mm,
                AVG(temp_avg_c)             AS temp_avg_c,
                MIN(temp_min_c)             AS temp_min_c,
                MAX(temp_max_c)             AS temp_max_c
            FROM   weather_obs
            WHERE  region_code = $1
            GROUP  BY 1
            ORDER  BY 1 ASC
            """,
            region_code,
        )
    finally:
        await conn.close()

    if not rows:
        logger.warning("weather.data.empty", region_code=region_code)
        return pd.DataFrame(columns=["month"] + _EXOG_COLS)

    df = pd.DataFrame([dict(r) for r in rows])
    df["month"] = pd.to_datetime(df["month"])
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def _select_top_crops(df: pd.DataFrame, top_n: int, min_seasons: int) -> list[str]:
    """Select the most data-rich crops for training.

    Logic Flow:
        Counts the number of unique season observations per crop.
        Returns the top_n crops with at least min_seasons observations.

    Args:
        df:          Yield observations DataFrame.
        top_n:       Maximum number of crops to train.
        min_seasons: Minimum season count required to include a crop.

    Returns:
        Sorted list of crop name strings.
    """
    counts = (
        df.groupby("crop_name")["time"]
        .count()
        .rename("count")
        .reset_index()
    )
    eligible = counts[counts["count"] >= min_seasons]
    top = eligible.nlargest(top_n, "count")["crop_name"].tolist()
    logger.info("crops.selected", count=len(top), min_seasons=min_seasons)
    return sorted(top)


def _build_crop_series(
    df_yield: pd.DataFrame,
    df_exog_by_time: pd.DataFrame,
    crop_name: str,
) -> tuple[pd.Series, pd.DataFrame] | None:
    """Build yield time series and exogenous weather features for one crop.

    Logic Flow:
        Filters yield data to the given crop.
        Aggregates to national-level annual yield per season (mean across districts).
        Left-joins to weather_agg on the harvest month.
        Returns None if the crop has fewer than 3 data points after joining.

    Args:
        df_yield:   Full yield DataFrame.
        df_weather: National weather aggregated by month.
        crop_name:  Crop to prepare.

    Returns:
        Tuple of (yield_series, exog_df) or None if insufficient data.
    """
    crop_df = df_yield[df_yield["crop_name"] == crop_name].copy()
    crop_df = (
        crop_df.groupby("time")["yield_kg_ha"]
        .mean()
        .reset_index()
        .sort_values("time")
        .set_index("time")
    )
    crop_df.index = pd.DatetimeIndex(crop_df.index)

    if df_exog_by_time.empty:
        exog = pd.DataFrame(
            np.zeros((len(crop_df), len(_EXOG_COLS))),
            columns=_EXOG_COLS,
            index=crop_df.index,
        )
    else:
        df_exog_idx = df_exog_by_time.set_index("time")
        exog = crop_df.join(df_exog_idx[_EXOG_COLS], how="left").ffill().bfill().fillna(0.0)[_EXOG_COLS]

    if len(crop_df) < 3:
        return None

    return crop_df["yield_kg_ha"], exog


def _fit_sarimax(
    y: pd.Series,
    exog: pd.DataFrame,
    crop_name: str,
) -> object | None:
    """Fit a SARIMAX model for a single crop.

    Logic Flow:
        Uses order=(1,1,1) and seasonal_order=(1,1,0,2).
        Fits with enforce_stationarity=False to handle short series.
        Returns the fitted SARIMAXResults object, or None on failure.

    Args:
        y:         Yield time series (pandas Series, DatetimeIndex).
        exog:      Exogenous weather features (same index as y).
        crop_name: Crop name (used in error logging only).

    Returns:
        Fitted SARIMAXResults or None.
    """
    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=_ORDER,
            seasonal_order=_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        logger.info("sarimax.fit.ok", crop=crop_name, aic=round(result.aic, 2))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("sarimax.fit.failed", crop=crop_name, error=str(exc))
        return None


# ── Serialisation ─────────────────────────────────────────────────────────────

def _save_artifacts(
    region_code: str,
    sarimax_dict: dict[str, object],
    scaler: StandardScaler,
    crop_index: dict[str, int],
    avg_yield: dict[str, float],
    drift_report: dict,
    timeline_checkpoint: dict,
) -> None:
    """Persist all SARIMAX artifacts to ml/artifacts/{region_code}/.

    Logic Flow:
        Creates the artifact directory if it doesn't exist.
        Serialises: sarimax_results.pkl, scaler.pkl, crop_index.json,
        model_meta.json.

    Args:
        region_code:  Region identifier used as directory name.
        sarimax_dict: Dict mapping crop_name → fitted SARIMAXResults.
        scaler:       Fitted StandardScaler.
        crop_index:   Crop name → integer label.
        avg_yield:    Crop name → average yield (kg/ha).
    """
    art_dir = _ARTIFACTS_ROOT / region_code
    art_dir.mkdir(parents=True, exist_ok=True)

    with (art_dir / "sarimax_results.pkl").open("wb") as f:
        pickle.dump(sarimax_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with (art_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    (art_dir / "crop_index.json").write_text(json.dumps(crop_index, indent=2))
    (art_dir / "model_meta.json").write_text(json.dumps({
        "region_code":      region_code,
        "trained_crops":    list(sarimax_dict.keys()),
        "avg_yield_kg_ha":  avg_yield,
        "ensemble_weight_sarimax": 0.4,
        "ensemble_weight_lstm":    0.6,
        "uncertainty_profile": drift_report.get("uncertainty_profile", {}),
        "warn_features": drift_report.get("warn_features", []),
        "timeline_checkpoint": timeline_checkpoint,
        "trained_at":       pd.Timestamp.utcnow().isoformat(),
    }, indent=2))

    save_drift_report(drift_report, art_dir)

    logger.info(
        "artifacts.saved",
        dir=str(art_dir),
        crops=len(sarimax_dict),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(region_code: str, top_crops: int, min_seasons: int) -> None:
    """End-to-end SARIMAX training pipeline.

    Logic Flow:
        1. Load yield data from TimescaleDB.
        2. Load national weather aggregates.
        3. Select top crops by data richness.
        4. Fit one SARIMAX per crop.
        5. Fit a shared StandardScaler on the exog features.
        6. Save all artifacts.

    Args:
        region_code: Runtime region identifier (e.g. 'IN').
        top_crops:   Maximum number of crops to train.
        min_seasons: Minimum seasons of data required per crop.
    """
    logger.info("sarimax.train.start", region_code=region_code)

    bundle = await build_training_bundle(region_code)
    df_yield = bundle.yield_long
    df_exog_by_time = bundle.exog_by_time
    timeline_checkpoint = bundle.timeline_checkpoint

    drift_report = analyze_drift(df_exog_by_time)
    crops      = _select_top_crops(df_yield, top_crops, min_seasons)

    sarimax_dict: dict[str, object] = {}
    avg_yield: dict[str, float] = {}
    all_exog_frames: list[pd.DataFrame] = []

    for crop in crops:
        result = _build_crop_series(df_yield, df_exog_by_time, crop)
        if result is None:
            continue
        y, exog = result
        avg_yield[crop] = float(y.mean())
        all_exog_frames.append(exog)

        fitted = _fit_sarimax(y, exog, crop)
        if fitted is not None:
            sarimax_dict[crop] = fitted

    if not sarimax_dict:
        raise RuntimeError("No crops could be fitted. Check your input data.")

    # Fit scaler on concatenated exog data
    all_exog = pd.concat(all_exog_frames, ignore_index=True).fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(all_exog.values)

    crop_index = {c: i for i, c in enumerate(sorted(sarimax_dict.keys()))}
    _save_artifacts(
        region_code,
        sarimax_dict,
        scaler,
        crop_index,
        avg_yield,
        drift_report,
        timeline_checkpoint,
    )

    logger.info(
        "sarimax.train.complete",
        region_code=region_code,
        crops_fitted=len(sarimax_dict),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARIMAX crop models.")
    parser.add_argument("--region",      required=True,      help="Region code, e.g. IN")
    parser.add_argument("--top-crops",   type=int, default=40, help="Max crops to train")
    parser.add_argument("--min-seasons", type=int, default=4,  help="Min seasons per crop")
    args = parser.parse_args()

    asyncio.run(main(args.region, args.top_crops, args.min_seasons))
