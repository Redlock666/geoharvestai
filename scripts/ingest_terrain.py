"""
Ingest SRTM GL1 30 m terrain data into PostGIS terrain_raw.

Downloads a single GeoTIFF for India's bounding box via the OpenTopography API,
computes per-pixel slope using numpy gradient, then samples elevation and slope
at H3 resolution-7 hex centroids and upserts into PostGIS.

Usage:
    python scripts/ingest_terrain.py --region IN
    python scripts/ingest_terrain.py --region IN --data-dir data/raw
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
import h3
import numpy as np
import rasterio
import structlog
from rasterio.transform import rowcol
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.india import BBOX, H3_RESOLUTION, OPENTOPOGRAPHY_API_URL
from db.settings import get_postgis_dsn

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 5_000


# ── Download ──────────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=60))
def _download_srtm(out_path: Path) -> None:
    """Download SRTM GL1 GeoTIFF for India from OpenTopography API.

    Logic Flow:
        Reads OPENTOPOGRAPHY_API_KEY from environment.
        Appends API key to the pre-built URL in configs.india.
        Streams response to disk.

    Args:
        out_path: Destination file path for the GeoTIFF.

    Expected Exceptions:
        KeyError: OPENTOPOGRAPHY_API_KEY not set.
        httpx.HTTPStatusError: API responded with non-2xx status.
    """
    import httpx  # local import — only needed in download path

    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")
    url = f"{OPENTOPOGRAPHY_API_URL}&API_Key={api_key}" if api_key else OPENTOPOGRAPHY_API_URL

    log = logger.bind(url=url[:80])
    log.info("srtm.download.start")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300.0) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

    log.info("srtm.download.complete", path=str(out_path), size_mb=out_path.stat().st_size // 1_048_576)


# ── Slope computation ─────────────────────────────────────────────────────


def _compute_slope(dem_path: Path) -> tuple[np.ndarray, np.ndarray, rasterio.DatasetReader]:
    """Compute per-pixel slope (degrees) from a DEM GeoTIFF.

    Logic Flow:
        Opens DEM, reads full elevation band.
        Approximates cell size in metres using geographic coordinates
        (x_res_m = x_deg × 111 320 × cos(centre_lat), y_res_m = y_deg × 110 540).
        Computes gradient with numpy, converts to degrees via arctan.

    Args:
        dem_path: Path to DEM GeoTIFF (WGS84, single band, metres).

    Returns:
        Tuple of (elevation_array, slope_array, open rasterio dataset).
        The caller is responsible for closing the dataset.

    Expected Exceptions:
        rasterio.errors.RasterioIOError: Corrupt or missing file.
    """
    src = rasterio.open(dem_path)
    elev = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan

    centre_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2.0
    res_y_deg  = abs(src.transform.e)
    res_x_deg  = abs(src.transform.a)
    dy_m = res_y_deg * 110_540.0
    dx_m = res_x_deg * 111_320.0 * np.cos(np.radians(centre_lat))

    grad_y, grad_x = np.gradient(np.nan_to_num(elev, nan=0.0), dy_m, dx_m)
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2))).astype(np.float32)
    # Restore NaN mask from elevation
    slope[np.isnan(elev)] = np.nan

    logger.info("slope.computed", shape=slope.shape)
    return elev, slope, src


# ── Sampling ──────────────────────────────────────────────────────────────


def _sample_terrain(dem_path: Path, hex_ids: list[str]) -> list[tuple]:
    """Sample elevation and slope at H3 resolution-7 hex centroids.

    Logic Flow:
        Computes slope from DEM.
        Vectorises (lat, lon) → (row, col) for all hex centroids at once.
        Skips hexes where elevation is NaN (ocean / no data).

    Args:
        dem_path: Path to SRTM GeoTIFF.
        hex_ids:  List of H3 resolution-7 hex IDs to sample.

    Returns:
        List of (hex_id, elevation_m, slope_deg) tuples.

    Expected Exceptions:
        rasterio.errors.RasterioIOError: Corrupt GeoTIFF.
    """
    log = logger.bind(hexes=len(hex_ids))
    log.info("terrain.sampling.start")

    elev, slope, src = _compute_slope(dem_path)

    latlons = np.array([h3.h3_to_geo(h) for h in hex_ids])
    lats = latlons[:, 0]
    lons = latlons[:, 1]

    rows_idx, cols_idx = rowcol(src.transform, lons, lats)
    rows_idx = np.clip(rows_idx, 0, elev.shape[0] - 1)
    cols_idx = np.clip(cols_idx, 0, elev.shape[1] - 1)
    src.close()

    elevs  = elev[rows_idx,  cols_idx]
    slopes = slope[rows_idx, cols_idx]

    result = []
    for i, hex_id in enumerate(hex_ids):
        e = elevs[i]
        s = slopes[i]
        if np.isnan(e):
            continue
        result.append((hex_id, float(e), float(s)))

    log.info("terrain.sampling.complete", rows=len(result))
    return result


# ── DB upsert ─────────────────────────────────────────────────────────────


async def _upsert_terrain(conn: asyncpg.Connection, rows: list[tuple]) -> int:
    """Bulk-upsert terrain rows and refresh terrain_by_hex materialized view.

    Logic Flow:
        Executes parameterized INSERT … ON CONFLICT in _CHUNK_SIZE batches.
        Refreshes terrain_by_hex after all batches complete.

    Args:
        conn: Active asyncpg connection to PostGIS.
        rows: List of (hex_id, elevation_m, slope_deg) tuples.

    Returns:
        Total rows upserted.

    Expected Exceptions:
        asyncpg.PostgresError: Constraint or connection failure.
    """
    upsert_sql = """
        INSERT INTO terrain_raw (hex_id, elevation_m, slope_deg, source)
        VALUES ($1, $2, $3, 'srtm_gl1_30m')
        ON CONFLICT (hex_id) DO UPDATE SET
            elevation_m = EXCLUDED.elevation_m,
            slope_deg   = EXCLUDED.slope_deg,
            ingested_at = now()
    """
    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(upsert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])
        logger.info("terrain.upsert.progress", inserted=inserted)

    await conn.execute("REFRESH MATERIALIZED VIEW terrain_by_hex")
    logger.info("view.refreshed", view="terrain_by_hex")
    return inserted


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, data_dir: Path) -> None:
    """Orchestrate the terrain ingest pipeline.

    Logic Flow:
        1. Check if DEM GeoTIFF already downloaded; download if not.
        2. Generate H3 res-7 hexes covering India bbox.
        3. Sample elevation + slope from DEM.
        4. Upsert into PostGIS terrain_raw and refresh view.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        data_dir:    Root directory for raw data files.

    Expected Exceptions:
        KeyError: OPENTOPOGRAPHY_API_KEY missing and DEM not downloaded.
        asyncpg.PostgresError: DB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_terrain")
    log.info("ingest.start")

    dem_path = data_dir / "terrain" / "srtm_india_30m.tif"
    if not dem_path.exists():
        log.info("dem.not.found.downloading")
        _download_srtm(dem_path)

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
    log.info("hexes.generated", count=len(hex_ids))

    rows = _sample_terrain(dem_path, hex_ids)

    conn = await asyncpg.connect(**get_postgis_dsn())
    try:
        inserted = await _upsert_terrain(conn, rows)
        log.info("ingest.complete", rows_inserted=inserted)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SRTM terrain data → PostGIS terrain_raw"
    )
    parser.add_argument("--region",   required=True, help="Region code (e.g. IN)")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    asyncio.run(run(args.region, Path(args.data_dir)))


if __name__ == "__main__":
    main()
