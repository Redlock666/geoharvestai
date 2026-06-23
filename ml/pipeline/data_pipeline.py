"""Reusable training data pipeline for canonical feature modelling."""

from __future__ import annotations

import os
from dataclasses import dataclass

import asyncpg
import pandas as pd
import structlog

from ml.pipeline.features import DEFAULT_STATIC_FEATURES, FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

_MIN_YIELD_ROWS = int(os.environ.get("MIN_YIELD_ROWS", "200"))
_MIN_YIELD_CROPS = int(os.environ.get("MIN_YIELD_CROPS", "8"))
_MIN_YIELD_TIMESTEPS = int(os.environ.get("MIN_YIELD_TIMESTEPS", "24"))


@dataclass
class TrainingDataBundle:
    """Canonical training bundle for both SARIMAX and LSTM pipelines."""

    yield_long: pd.DataFrame
    exog_by_time: pd.DataFrame
    timeline_checkpoint: dict


def _assert_time_column(df: pd.DataFrame, name: str) -> None:
    """Validate existence and basic integrity of the time column.

    Args:
        df: DataFrame expected to contain a ``time`` column.
        name: Dataset label for diagnostics.

    Expected Exceptions:
        ValueError: If the time column is missing, null, duplicated, or unsorted.
    """
    if "time" not in df.columns:
        raise ValueError(f"{name} is missing required 'time' column.")

    if df["time"].isna().any():
        raise ValueError(f"{name} contains null timestamps; timeline checkpoint failed.")

    if df["time"].duplicated().any():
        dupes = int(df["time"].duplicated().sum())
        raise ValueError(f"{name} contains {dupes} duplicate timestamps; timeline checkpoint failed.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError(f"{name} timeline is not sorted ascending; timeline checkpoint failed.")


def _build_timeline_checkpoint(yield_long: pd.DataFrame, exog_by_time: pd.DataFrame) -> dict:
    """Create and enforce timeline consistency checkpoint.

    Logic Flow:
        Normalizes both datasets to unique sorted monthly timestamps.
        Ensures every yield timestamp is represented in exogenous features.
        Computes cadence diagnostics and raises on severe mismatch.

    Args:
        yield_long: Long-form yield observations.
        exog_by_time: Canonical exogenous features by timestamp.

    Returns:
        Dict with checkpoint diagnostics and pass/fail status.

    Expected Exceptions:
        ValueError: If yield timestamps are missing from exogenous timeline.
    """
    yield_time = (
        yield_long[["time"]]
        .drop_duplicates()
        .sort_values("time")
        .reset_index(drop=True)
    )
    exog_time = (
        exog_by_time[["time"]]
        .drop_duplicates()
        .sort_values("time")
        .reset_index(drop=True)
    )

    _assert_time_column(yield_time, "yield_timeline")
    _assert_time_column(exog_time, "exog_timeline")

    missing_from_exog = yield_time.loc[~yield_time["time"].isin(exog_time["time"]), "time"]
    if not missing_from_exog.empty:
        preview = [str(t) for t in missing_from_exog.head(5).tolist()]
        raise ValueError(
            "Timeline checkpoint failed: exogenous timeline missing yield timestamps. "
            f"missing_count={len(missing_from_exog)} sample={preview}"
        )

    y_deltas = yield_time["time"].diff().dropna().apply(lambda d: float(d.total_seconds() / 86400.0))
    x_deltas = exog_time["time"].diff().dropna().apply(lambda d: float(d.total_seconds() / 86400.0))

    checkpoint = {
        "status": "pass",
        "yield_timestamps": int(len(yield_time)),
        "exog_timestamps": int(len(exog_time)),
        "yield_start": str(yield_time["time"].min()),
        "yield_end": str(yield_time["time"].max()),
        "exog_start": str(exog_time["time"].min()),
        "exog_end": str(exog_time["time"].max()),
        "yield_median_gap_days": float(y_deltas.median()) if not y_deltas.empty else 0.0,
        "exog_median_gap_days": float(x_deltas.median()) if not x_deltas.empty else 0.0,
        "coverage_ratio": 1.0,
    }

    logger.info("pipeline.timeline.checkpoint", **checkpoint)
    return checkpoint


def _timescale_conn_kwargs() -> dict:
    return {
        "host": os.environ.get("TIMESCALE_HOST", "localhost"),
        "port": int(os.environ.get("TIMESCALE_PORT", "5433")),
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database": os.environ.get("TIMESCALE_DB", "geoharvestai_ts"),
    }


async def _load_yield(region_code: str) -> pd.DataFrame:
    """Load and aggregate crop yields by month/crop.

    Logic Flow:
        Reads crop_yield_obs for region_code.
        Aggregates district-level rows into national monthly crop means.

    Args:
        region_code: Runtime region identifier.

    Returns:
        DataFrame columns: time, crop_name, season, yield_kg_ha.

    Expected Exceptions:
        ValueError: No rows found for the given region_code.
    """
    conn = await asyncpg.connect(**_timescale_conn_kwargs())
    try:
        rows = await conn.fetch(
            """
            SELECT
                date_trunc('month', time) AS time,
                crop_name,
                season,
                AVG(yield_kg_ha) AS yield_kg_ha
            FROM crop_yield_obs
            WHERE region_code = $1
            GROUP BY 1, 2, 3
            ORDER BY 1, 2
            """,
            region_code,
        )
    finally:
        await conn.close()

    if not rows:
        raise ValueError(
            f"No crop_yield_obs rows for region_code={region_code}. "
            "Ingest APY CSV or run demo seeding first."
        )

    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Data sufficiency gate: do not train on sparse targets.
    n_rows = int(len(df))
    n_crops = int(df["crop_name"].nunique())
    n_timesteps = int(df["time"].nunique())
    if (
        n_rows < _MIN_YIELD_ROWS
        or n_crops < _MIN_YIELD_CROPS
        or n_timesteps < _MIN_YIELD_TIMESTEPS
    ):
        raise ValueError(
            "Insufficient APY coverage for reliable training. "
            f"rows={n_rows} (min={_MIN_YIELD_ROWS}), "
            f"crops={n_crops} (min={_MIN_YIELD_CROPS}), "
            f"timesteps={n_timesteps} (min={_MIN_YIELD_TIMESTEPS}). "
            "Ingest broader APY data before training to avoid guess-quality predictions."
        )

    logger.info(
        "pipeline.yield.coverage.ok",
        rows=n_rows,
        crops=n_crops,
        timesteps=n_timesteps,
    )
    return df


async def _load_weather_ndvi(region_code: str) -> pd.DataFrame:
    """Load monthly weather and NDVI features.

    Logic Flow:
        Aggregates weather_obs to monthly region means.
        Left-joins monthly ndvi_obs averages.
        Returns one row per month.

    Args:
        region_code: Runtime region identifier.

    Returns:
        DataFrame columns: time + dynamic weather/ndvi feature columns.
    """
    conn = await asyncpg.connect(**_timescale_conn_kwargs())
    try:
        weather_rows = await conn.fetch(
            """
            SELECT
                date_trunc('month', time) AS time,
                SUM(rainfall_mm) AS rainfall_7d_mm,
                AVG(temp_avg_c) AS temp_avg_c,
                MIN(temp_min_c) AS temp_min_c,
                MAX(temp_max_c) AS temp_max_c
            FROM weather_obs
            WHERE region_code = $1
            GROUP BY 1
            ORDER BY 1
            """,
            region_code,
        )

        ndvi_rows = await conn.fetch(
            """
            SELECT
                date_trunc('month', time) AS time,
                AVG(ndvi) AS ndvi
            FROM ndvi_obs
            WHERE region_code = $1
            GROUP BY 1
            ORDER BY 1
            """,
            region_code,
        )
    finally:
        await conn.close()

    weather = pd.DataFrame([dict(r) for r in weather_rows])
    ndvi = pd.DataFrame([dict(r) for r in ndvi_rows])

    if weather.empty:
        weather = pd.DataFrame(columns=["time", "rainfall_7d_mm", "temp_avg_c", "temp_min_c", "temp_max_c"])
    if ndvi.empty:
        ndvi = pd.DataFrame(columns=["time", "ndvi"])

    if not weather.empty:
        weather["time"] = pd.to_datetime(weather["time"], utc=True)
    if not ndvi.empty:
        ndvi["time"] = pd.to_datetime(ndvi["time"], utc=True)

    merged = weather.merge(ndvi, on="time", how="left")
    merged["ndvi"] = merged.get("ndvi", pd.Series(dtype=float)).fillna(0.45)
    return merged


def _attach_static_features(df_time: pd.DataFrame) -> pd.DataFrame:
    """Add static GIS proxy features so feature vector shape remains canonical.

    Logic Flow:
        Adds soil + terrain static defaults.
        Preserves dynamic weather columns.

    Args:
        df_time: DataFrame with at least a time column and weather features.

    Returns:
        DataFrame with all FEATURE_COLUMNS present.
    """
    out = df_time.copy()
    for col, val in DEFAULT_STATIC_FEATURES.items():
        out[col] = val

    # Ensure dynamic columns exist even when weather ingest isn't complete.
    for col, val in {
        "rainfall_7d_mm": 0.0,
        "temp_avg_c": 27.0,
        "temp_min_c": 20.0,
        "temp_max_c": 34.0,
        "ndvi": 0.45,
    }.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)

    return out[["time", *FEATURE_COLUMNS]]


async def build_training_bundle(region_code: str) -> TrainingDataBundle:
    """Build canonical training dataset for a region.

    Logic Flow:
        1. Load yield rows from crop_yield_obs.
        2. Load weather + NDVI monthly rows.
        3. Attach static GIS defaults to complete canonical feature vector.
        4. Align features to yield timestamps.

    Args:
        region_code: Runtime region identifier.

    Returns:
        TrainingDataBundle containing:
            - yield_long: long-form target table
            - exog_by_time: canonical exogenous features indexed by month

    Expected Exceptions:
        ValueError: If no yield rows exist for region_code.
    """
    logger.info("pipeline.build.start", region_code=region_code)

    yield_long = await _load_yield(region_code)
    weather_ndvi = await _load_weather_ndvi(region_code)
    exog = _attach_static_features(weather_ndvi)

    # Align to target time grid
    time_grid = pd.DataFrame({"time": sorted(yield_long["time"].unique())})
    exog_by_time = time_grid.merge(exog, on="time", how="left")

    # Fill missing aligned rows with defaults
    for col in FEATURE_COLUMNS:
        if col in DEFAULT_STATIC_FEATURES:
            exog_by_time[col] = exog_by_time[col].fillna(DEFAULT_STATIC_FEATURES[col])
        elif col == "ndvi":
            exog_by_time[col] = exog_by_time[col].fillna(0.45)
        elif col == "rainfall_7d_mm":
            exog_by_time[col] = exog_by_time[col].fillna(0.0)
        elif col in {"temp_avg_c", "temp_min_c", "temp_max_c"}:
            exog_by_time[col] = exog_by_time[col].fillna(27.0 if col == "temp_avg_c" else (20.0 if col == "temp_min_c" else 34.0))

    exog_by_time = exog_by_time.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    timeline_checkpoint = _build_timeline_checkpoint(yield_long, exog_by_time)

    logger.info(
        "pipeline.build.complete",
        region_code=region_code,
        yield_rows=len(yield_long),
        exog_rows=len(exog_by_time),
    )
    return TrainingDataBundle(
        yield_long=yield_long,
        exog_by_time=exog_by_time,
        timeline_checkpoint=timeline_checkpoint,
    )
