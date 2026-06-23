"""
Daily data refresh worker.

Runs continuously as the Docker Compose 'worker' service.
Every 24 hours at 02:00 UTC, fetches:
  1. Last 7 days of weather from Open-Meteo (no API key required).
  2. Latest Sentinel-2 NDVI via Sentinel Hub (if last fetch > 5 days ago).

Writes into TimescaleDB weather_obs and ndvi_obs.  These feeds power the
real-time WeatherAgentService cache, ensuring staleness never exceeds 24 h
for weather and 5 days for NDVI (Sentinel-2 revisit cycle).

Open-Meteo resolution: ~11 km grid → assigned to H3 res-7 hexes by
snapping to nearest grid point (same strategy as ERA5 ingest).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import asyncpg
import h3
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.india import BBOX, H3_RESOLUTION
from db.settings import get_timescale_dsn

logger = structlog.get_logger(__name__)

_REFRESH_HOUR_UTC  = 2       # run at 02:00 UTC daily
_WEATHER_DAYS_BACK = 7       # how many days of weather history to refresh
_NDVI_STALE_DAYS   = 5       # only refresh NDVI if last fetch is older than this
_OPEN_METEO_URL    = "https://api.open-meteo.com/v1/forecast"
_CHUNK_SIZE        = 5_000


# ── Weather via Open-Meteo ────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
async def _fetch_openmeteo(
    session: httpx.AsyncClient,
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> dict:
    """Fetch daily weather from Open-Meteo for a coordinate.

    Logic Flow:
        Calls Open-Meteo /v1/forecast with daily aggregates.
        Returns raw JSON response dict.

    Args:
        session: Shared httpx.AsyncClient.
        lat:     Latitude (WGS84).
        lon:     Longitude (WGS84).
        start:   Start date string 'YYYY-MM-DD'.
        end:     End date string 'YYYY-MM-DD'.

    Returns:
        Open-Meteo API response JSON as dict.

    Expected Exceptions:
        httpx.HTTPStatusError: Rate limited or server error (retried).
    """
    resp = await session.get(
        _OPEN_METEO_URL,
        params={
            "latitude":                lat,
            "longitude":               lon,
            "daily": [
                "precipitation_sum",
                "temperature_2m_mean",
                "temperature_2m_min",
                "temperature_2m_max",
                "windspeed_10m_max",
                "relativehumidity_2m_mean",
            ],
            "timezone":                "UTC",
            "start_date":              start,
            "end_date":                end,
        },
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


async def _refresh_weather(
    conn: asyncpg.Connection,
    region_code: str,
    hex_ids: list[str],
) -> int:
    """Fetch last _WEATHER_DAYS_BACK days of weather and upsert to TimescaleDB.

    Logic Flow:
        Groups hex IDs by rounded (lat, lon) to minimise Open-Meteo calls
        (multiple hexes share the same ~11 km grid point).
        Fetches all grid points concurrently with an asyncio semaphore
        (max 10 concurrent, free tier limit).
        Upserts into weather_obs with ON CONFLICT DO NOTHING.

    Args:
        conn:        Active asyncpg connection to TimescaleDB.
        region_code: Runtime region identifier.
        hex_ids:     H3 resolution-7 hexes to refresh.

    Returns:
        Total rows upserted.

    Expected Exceptions:
        httpx.HTTPStatusError: Open-Meteo API error.
        asyncpg.PostgresError: TimescaleDB connection failure.
    """
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=_WEATHER_DAYS_BACK)
    log   = logger.bind(start=str(start), end=str(end), hexes=len(hex_ids))
    log.info("weather.refresh.start")

    # Deduplicate by 0.1° grid point (Open-Meteo ~11 km resolution)
    grid_to_hexes: dict[tuple[float, float], list[str]] = {}
    for hex_id in hex_ids:
        lat, lon = h3.h3_to_geo(hex_id)
        grid_lat = round(lat / 0.1) * 0.1
        grid_lon = round(lon / 0.1) * 0.1
        grid_to_hexes.setdefault((grid_lat, grid_lon), []).append(hex_id)

    semaphore = asyncio.Semaphore(10)
    rows: list[tuple] = []

    async def _fetch_one(grid_lat: float, grid_lon: float, hexes: list[str]) -> None:
        async with semaphore:
            async with httpx.AsyncClient(timeout=30.0) as session:
                data = await _fetch_openmeteo(
                    session, grid_lat, grid_lon, str(start), str(end)
                )
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        for i, dt_str in enumerate(dates):
            dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            for hex_id in hexes:
                rows.append((
                    dt,
                    hex_id,
                    region_code,
                    (daily.get("precipitation_sum") or [None])[i],
                    (daily.get("temperature_2m_mean") or [None])[i],
                    (daily.get("temperature_2m_min") or [None])[i],
                    (daily.get("temperature_2m_max") or [None])[i],
                    (daily.get("relativehumidity_2m_mean") or [None])[i],
                    (daily.get("windspeed_10m_max") or [None])[i],
                    "open_meteo",
                ))

    await asyncio.gather(*[
        _fetch_one(lat, lon, hexes)
        for (lat, lon), hexes in grid_to_hexes.items()
    ])

    insert_sql = """
        INSERT INTO weather_obs
            (time, hex_id, region_code, rainfall_mm, temp_avg_c,
             temp_min_c, temp_max_c, humidity_pct, wind_speed_ms, source)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        ON CONFLICT DO NOTHING
    """
    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(insert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])

    log.info("weather.refresh.complete", rows=inserted)
    return inserted


# ── NDVI via Sentinel Hub ─────────────────────────────────────────────────


async def _refresh_ndvi(
    conn: asyncpg.Connection,
    region_code: str,
    hex_ids: list[str],
) -> int:
    """Fetch latest Sentinel-2 NDVI and upsert to TimescaleDB ndvi_obs.

    Logic Flow:
        Checks the most recent ndvi_obs timestamp for the region.
        Skips if data is fresher than _NDVI_STALE_DAYS.
        Authenticates with Sentinel Hub via SENTINELHUB_CLIENT_ID + SECRET.
        Requests a composite NDVI image for India's bbox at 100 m resolution.
        Samples each H3 hex centroid from the returned raster.
        Upserts into ndvi_obs.

    Args:
        conn:        Active asyncpg connection to TimescaleDB.
        region_code: Runtime region identifier.
        hex_ids:     H3 resolution-7 hexes to refresh.

    Returns:
        Total rows upserted (0 if skipped due to freshness).

    Expected Exceptions:
        sentinelhub.exceptions.SHRuntimeWarning: Cloud coverage too high.
        asyncpg.PostgresError: TimescaleDB connection failure.
    """
    log = logger.bind(region=region_code)

    # Check staleness
    latest = await conn.fetchval(
        "SELECT MAX(time) FROM ndvi_obs WHERE region_code = $1",
        region_code,
    )
    if latest is not None:
        age_days = (datetime.now(timezone.utc) - latest).days
        if age_days < _NDVI_STALE_DAYS:
            log.info("ndvi.fresh.skip", age_days=age_days)
            return 0

    try:
        from sentinelhub import (  # type: ignore[import]
            BBox,
            BBoxSplitter,
            CRS,
            DataCollection,
            MimeType,
            SentinelHubRequest,
            SHConfig,
            bbox_to_dimensions,
        )
    except ImportError:
        log.warning("sentinelhub.not.installed.skipping.ndvi")
        return 0

    config = SHConfig()
    config.sh_client_id     = os.environ.get("SENTINELHUB_CLIENT_ID", "")
    config.sh_client_secret = os.environ.get("SENTINELHUB_CLIENT_SECRET", "")
    if not config.sh_client_id:
        log.warning("sentinelhub.credentials.missing.skipping.ndvi")
        return 0

    india_bbox  = BBox(
        (BBOX["min_lon"], BBOX["min_lat"], BBOX["max_lon"], BBOX["max_lat"]),
        crs=CRS.WGS84,
    )
    size = bbox_to_dimensions(india_bbox, resolution=1000)  # 1 km for full coverage

    evalscript = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["B04","B08"] }], output: { bands: 1 } };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  return [ndvi];
}
"""
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(
                (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d"),
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            ),
            mosaicking_order="leastCC",
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=india_bbox,
        size=size,
        config=config,
    )

    import numpy as np
    import rasterio  # type: ignore[import]
    from rasterio.transform import from_bounds

    ndvi_data = request.get_data()[0][:, :, 0].astype(np.float32)
    transform = from_bounds(
        BBOX["min_lon"], BBOX["min_lat"],
        BBOX["max_lon"], BBOX["max_lat"],
        ndvi_data.shape[1], ndvi_data.shape[0],
    )

    acq_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
    rows: list[tuple] = []
    for hex_id in hex_ids:
        lat, lon = h3.h3_to_geo(hex_id)
        from rasterio.transform import rowcol  # type: ignore[import]
        r_idx, c_idx = rowcol(transform, lon, lat)
        r_idx = int(max(0, min(r_idx, ndvi_data.shape[0] - 1)))
        c_idx = int(max(0, min(c_idx, ndvi_data.shape[1] - 1)))
        val = float(ndvi_data[r_idx, c_idx])
        if np.isnan(val) or val < -1 or val > 1:
            continue
        rows.append((acq_time, hex_id, region_code, val, None, "sentinel2"))

    insert_sql = """
        INSERT INTO ndvi_obs (time, hex_id, region_code, ndvi, cloud_cover_pct, source)
        VALUES ($1,$2,$3,$4,$5,$6)
        ON CONFLICT DO NOTHING
    """
    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(insert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])

    log.info("ndvi.refresh.complete", rows=inserted)
    return inserted


# ── Main loop ─────────────────────────────────────────────────────────────


async def _run_once(region_code: str, hex_ids: list[str]) -> None:
    """Execute one full refresh cycle (weather + NDVI).

    Logic Flow:
        Opens a single TimescaleDB connection.
        Refreshes weather, then NDVI.
        Closes connection cleanly on completion or error.

    Args:
        region_code: Runtime region identifier.
        hex_ids:     Pre-generated H3 hex IDs for the region.

    Expected Exceptions:
        asyncpg.PostgresError: TimescaleDB unreachable.
    """
    log = logger.bind(region=region_code)
    log.info("refresh.cycle.start")
    conn = await asyncpg.connect(**get_timescale_dsn())
    try:
        w_rows = await _refresh_weather(conn, region_code, hex_ids)
        n_rows = await _refresh_ndvi(conn, region_code, hex_ids)
        log.info("refresh.cycle.complete", weather_rows=w_rows, ndvi_rows=n_rows)
    except Exception as exc:
        log.exception("refresh.cycle.error", error=str(exc))
    finally:
        await conn.close()


async def main_loop() -> None:
    """Daily refresh loop — runs indefinitely as Docker worker service.

    Logic Flow:
        Generates H3 hexes for India once at startup.
        Sleeps until next 02:00 UTC.
        Runs _run_once() then sleeps 24 h.

    Expected Exceptions:
        asyncio.CancelledError: Graceful shutdown via Docker stop signal.
    """
    log = logger.bind(service="daily_refresh")
    log.info("worker.start")

    # Pre-generate hexes once (expensive, ~1s for India res-7)
    bbox_poly = {
        "type": "Polygon",
        "coordinates": [[
            [BBOX["min_lon"], BBOX["min_lat"]],
            [BBOX["max_lon"], BBOX["min_lat"]],
            [BBOX["max_lon"], BBOX["max_lat"]],
            [BBOX["min_lon"], BBOX["max_lat"]],
            [BBOX["min_lon"], BBOX["min_lat"]],
        ]],
    }
    hex_ids = list(h3.polyfill(bbox_poly, H3_RESOLUTION))
    region_code = os.environ.get("DEFAULT_REGION_CODE", "IN")
    log.info("hexes.ready", count=len(hex_ids), region=region_code)

    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(
            hour=_REFRESH_HOUR_UTC, minute=0, second=0, microsecond=0
        )
        if next_run <= now:
            next_run += timedelta(days=1)

        sleep_secs = (next_run - now).total_seconds()
        log.info("worker.sleeping", until=str(next_run), seconds=int(sleep_secs))
        await asyncio.sleep(sleep_secs)

        await _run_once(region_code, hex_ids)


if __name__ == "__main__":
    asyncio.run(main_loop())
