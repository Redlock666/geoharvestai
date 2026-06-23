"""Check whether ingested data is sufficient for reliable model training.

This script validates coverage in TimescaleDB for:
- crop_yield_obs (target data; mandatory)
- weather_obs and ndvi_obs (feature data; strongly recommended)

Usage:
    python3 scripts/check_training_readiness.py --region IN
"""

from __future__ import annotations

import argparse
import asyncio
import os

import asyncpg

from db.settings import get_timescale_dsn

_MIN_YIELD_ROWS = int(os.environ.get("MIN_YIELD_ROWS", "200"))
_MIN_YIELD_CROPS = int(os.environ.get("MIN_YIELD_CROPS", "8"))
_MIN_YIELD_TIMESTEPS = int(os.environ.get("MIN_YIELD_TIMESTEPS", "24"))


async def _fetch_yield_coverage(conn: asyncpg.Connection, region_code: str) -> dict:
    """Fetch APY target coverage stats for a region.

    Logic Flow:
        Aggregates row count, unique crops, and unique monthly timestamps.

    Args:
        conn: Active TimescaleDB asyncpg connection.
        region_code: Runtime region identifier.

    Returns:
        Dict with rows, crops, timesteps, min_time, max_time.

    Expected Exceptions:
        asyncpg.PostgresError: Query failure or table unavailable.
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT crop_name) AS crops,
            COUNT(DISTINCT date_trunc('month', time)) AS timesteps,
            MIN(time) AS min_time,
            MAX(time) AS max_time
        FROM crop_yield_obs
        WHERE region_code = $1
        """,
        region_code,
    )
    return dict(row) if row else {"rows": 0, "crops": 0, "timesteps": 0, "min_time": None, "max_time": None}


async def _fetch_feature_coverage(conn: asyncpg.Connection, region_code: str) -> dict:
    """Fetch weather and NDVI feature coverage stats.

    Logic Flow:
        Queries weather_obs and ndvi_obs for row counts and time windows.

    Args:
        conn: Active TimescaleDB asyncpg connection.
        region_code: Runtime region identifier.

    Returns:
        Dict containing weather_* and ndvi_* stats.

    Expected Exceptions:
        asyncpg.PostgresError: Query failure or table unavailable.
    """
    weather = await conn.fetchrow(
        """
        SELECT COUNT(*) AS rows, MIN(time) AS min_time, MAX(time) AS max_time
        FROM weather_obs
        WHERE region_code = $1
        """,
        region_code,
    )
    ndvi = await conn.fetchrow(
        """
        SELECT COUNT(*) AS rows, MIN(time) AS min_time, MAX(time) AS max_time
        FROM ndvi_obs
        WHERE region_code = $1
        """,
        region_code,
    )

    return {
        "weather_rows": int(weather["rows"]) if weather else 0,
        "weather_min_time": weather["min_time"] if weather else None,
        "weather_max_time": weather["max_time"] if weather else None,
        "ndvi_rows": int(ndvi["rows"]) if ndvi else 0,
        "ndvi_min_time": ndvi["min_time"] if ndvi else None,
        "ndvi_max_time": ndvi["max_time"] if ndvi else None,
    }


def _evaluate_readiness(yield_stats: dict, feature_stats: dict) -> tuple[bool, list[str]]:
    """Evaluate readiness rules for training.

    Logic Flow:
        Enforces hard APY sufficiency gates.
        Adds soft warnings for missing weather/NDVI features.

    Args:
        yield_stats: Coverage stats from crop_yield_obs.
        feature_stats: Coverage stats from weather_obs/ndvi_obs.

    Returns:
        Tuple: (is_ready, messages)

    Expected Exceptions:
        None.
    """
    messages: list[str] = []
    is_ready = True

    rows = int(yield_stats.get("rows", 0))
    crops = int(yield_stats.get("crops", 0))
    timesteps = int(yield_stats.get("timesteps", 0))

    if rows < _MIN_YIELD_ROWS:
        is_ready = False
        messages.append(f"APY rows too low: {rows} < {_MIN_YIELD_ROWS}")
    if crops < _MIN_YIELD_CROPS:
        is_ready = False
        messages.append(f"APY crop diversity too low: {crops} < {_MIN_YIELD_CROPS}")
    if timesteps < _MIN_YIELD_TIMESTEPS:
        is_ready = False
        messages.append(f"APY timeline too short: {timesteps} < {_MIN_YIELD_TIMESTEPS} monthly steps")

    weather_rows = int(feature_stats.get("weather_rows", 0))
    ndvi_rows = int(feature_stats.get("ndvi_rows", 0))
    if weather_rows == 0:
        messages.append("WARNING: weather_obs has 0 rows; model quality will degrade")
    if ndvi_rows == 0:
        messages.append("WARNING: ndvi_obs has 0 rows; model quality will degrade")

    return is_ready, messages


async def main(region_code: str) -> int:
    """Run data readiness checks and print a summary.

    Logic Flow:
        Connects to TimescaleDB.
        Fetches APY and feature coverage stats.
        Evaluates hard and soft readiness rules.
        Prints result and exits with shell-friendly code.

    Args:
        region_code: Runtime region identifier.

    Returns:
        0 when ready for training; 1 otherwise.

    Expected Exceptions:
        asyncpg.PostgresError: Connection or query failure.
    """
    conn = await asyncpg.connect(**get_timescale_dsn())
    try:
        yield_stats = await _fetch_yield_coverage(conn, region_code)
        feature_stats = await _fetch_feature_coverage(conn, region_code)
    finally:
        await conn.close()

    ready, messages = _evaluate_readiness(yield_stats, feature_stats)

    print("\n=== Training Readiness Report ===")
    print(f"Region: {region_code}")
    print(
        "APY target coverage: "
        f"rows={yield_stats['rows']}, crops={yield_stats['crops']}, timesteps={yield_stats['timesteps']}"
    )
    print(f"APY time range: {yield_stats['min_time']} -> {yield_stats['max_time']}")
    print(
        "Feature coverage: "
        f"weather_rows={feature_stats['weather_rows']}, ndvi_rows={feature_stats['ndvi_rows']}"
    )
    print(f"Weather time range: {feature_stats['weather_min_time']} -> {feature_stats['weather_max_time']}")
    print(f"NDVI time range: {feature_stats['ndvi_min_time']} -> {feature_stats['ndvi_max_time']}")

    if messages:
        print("\nNotes:")
        for msg in messages:
            print(f"- {msg}")

    if ready:
        print("\n[PASS] Ready for SARIMAX/LSTM training.")
        return 0

    print("\n[FAIL] Not ready for training. Ingest more APY target data first.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether ingested data is sufficient for reliable training")
    parser.add_argument("--region", required=True, help="Region code (e.g. IN)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.region)))
