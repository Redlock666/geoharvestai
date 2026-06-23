"""
Ingest Köppen-Geiger climate zones into PostGIS climate_zones_raw.

Reads the Beck et al. 2018 global 1 km GeoTIFF (downloaded by
download_india_data.sh) and maps each H3 resolution-7 hex centroid to:
  - zone_code:     Köppen-Geiger class string (e.g. 'Aw', 'BSh', 'Cwa')
  - icar_zone_id:  ICAR agroclimatic zone 1-15 (approximated from lat/lon)
  - icar_zone_name: Human-readable ICAR zone name

Usage:
    python scripts/ingest_climate_zones.py --region IN
    python scripts/ingest_climate_zones.py --region IN --data-dir data/raw
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import asyncpg
import h3
import numpy as np
import rasterio
import structlog
from rasterio.transform import rowcol

from configs.india import BBOX, H3_RESOLUTION, ICAR_ZONES, KG_VALUE_TO_CODE, KOPPEN_GEIGER_URL
from db.settings import get_postgis_dsn

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 5_000


# ── ICAR zone approximation ───────────────────────────────────────────────
# Derived from ICAR's published 15-zone geographic description.
# Used when the official ICAR raster is unavailable.
# lat/lon → zone_id lookup (simplified rectangular approximation).
_ICAR_GEO_RULES: list[tuple[float, float, float, float, int]] = [
    # (min_lat, max_lat, min_lon, max_lon, zone_id)
    (32.0, 37.6, 68.0, 80.0,  1),  # Western Himalayan
    (26.0, 37.6, 80.0, 97.5,  2),  # Eastern Himalayan
    (22.0, 26.0, 85.0, 92.0,  3),  # Lower Gangetic Plains
    (24.0, 28.0, 80.0, 87.0,  4),  # Middle Gangetic Plains
    (26.0, 30.0, 76.0, 82.0,  5),  # Upper Gangetic Plains
    (28.0, 32.0, 73.0, 78.0,  6),  # Trans-Gangetic Plains
    (20.0, 25.0, 80.0, 88.0,  7),  # Eastern Plateau & Hills
    (20.0, 25.0, 75.0, 82.0,  8),  # Central Plateau & Hills
    (16.0, 22.0, 73.0, 80.0,  9),  # Western Plateau & Hills
    (12.0, 18.0, 75.0, 80.0, 10),  # Southern Plateau & Hills
    ( 8.0, 18.0, 78.0, 82.0, 11),  # East Coast Plains & Hills
    ( 8.0, 20.0, 72.0, 77.0, 12),  # West Coast Plains & Ghat
    (20.0, 25.0, 68.0, 75.0, 13),  # Gujarat Plains & Hills
    (24.0, 30.0, 68.0, 74.0, 14),  # Western Dry Region
    ( 6.5, 14.0, 92.0, 97.5, 15),  # The Islands (A&N, Lakshadweep)
]


def _icar_zone_for(lat: float, lon: float) -> tuple[int, str]:
    """Approximate the ICAR agroclimatic zone for a coordinate.

    Logic Flow:
        Iterates through _ICAR_GEO_RULES in priority order and returns
        the first matching zone. Falls back to zone 8 (Central Plateau)
        if no rule matches — the most spatially central zone for India.

    Args:
        lat: Latitude  (WGS84).
        lon: Longitude (WGS84).

    Returns:
        Tuple of (zone_id int, zone_name str).

    Expected Exceptions:
        None.
    """
    for min_lat, max_lat, min_lon, max_lon, zone_id in _ICAR_GEO_RULES:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return zone_id, ICAR_ZONES[zone_id]
    return 8, ICAR_ZONES[8]  # fallback: Central Plateau


# ── Sampling ──────────────────────────────────────────────────────────────


def _sample_climate_zones(kg_tiff_path: Path, hex_ids: list[str]) -> list[tuple]:
    """Sample KG climate zone and derive ICAR zone for each H3 hex centroid.

    Logic Flow:
        Opens Beck 2018 GeoTIFF, reads full band into memory.
        Converts all hex centroids to (row, col) in one vectorised call.
        Maps integer pixel value → KG code string via KG_VALUE_TO_CODE.
        Derives ICAR zone from lat/lon via _icar_zone_for().
        Skips hexes where pixel value == 0 (ocean/no data).

    Args:
        kg_tiff_path: Path to Beck 2018 GeoTIFF.
        hex_ids:      H3 resolution-7 hex IDs covering India bbox.

    Returns:
        List of (hex_id, zone_code, icar_zone_id, icar_zone_name) tuples.

    Expected Exceptions:
        FileNotFoundError: GeoTIFF not downloaded yet.
        rasterio.errors.RasterioIOError: Corrupt file.
    """
    log = logger.bind(hexes=len(hex_ids))
    log.info("climate.sampling.start")

    with rasterio.open(kg_tiff_path) as src:
        arr     = src.read(1)          # uint8 pixel values 0-31
        nodata  = src.nodata or 0
        t       = src.transform

    latlons  = np.array([h3.h3_to_geo(h) for h in hex_ids])
    lats     = latlons[:, 0]
    lons     = latlons[:, 1]

    rows_idx, cols_idx = rowcol(t, lons, lats)
    rows_idx = np.clip(rows_idx, 0, arr.shape[0] - 1)
    cols_idx = np.clip(cols_idx, 0, arr.shape[1] - 1)
    values   = arr[rows_idx, cols_idx]

    result = []
    for i, hex_id in enumerate(hex_ids):
        v = int(values[i])
        if v == 0 or v == nodata:
            continue
        zone_code              = KG_VALUE_TO_CODE.get(v, "unknown")
        icar_id, icar_name     = _icar_zone_for(float(lats[i]), float(lons[i]))
        result.append((hex_id, zone_code, icar_id, icar_name))

    log.info("climate.sampling.complete", rows=len(result))
    return result


# ── Download ──────────────────────────────────────────────────────────────


def _download_kg_raster(out_path: Path) -> None:
    """Download the Beck 2018 Köppen-Geiger GeoTIFF from figshare.

    Logic Flow:
        Streams the file to disk. ~70 MB download, no auth required.

    Args:
        out_path: Destination path for the GeoTIFF.

    Expected Exceptions:
        httpx.HTTPStatusError: figshare unavailable.
    """
    import httpx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("kg.download.start", url=KOPPEN_GEIGER_URL)
    with httpx.Client(timeout=300.0) as client:
        with client.stream("GET", KOPPEN_GEIGER_URL, follow_redirects=True) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_bytes(65536):
                    f.write(chunk)
    logger.info("kg.download.complete", path=str(out_path))


# ── DB upsert ─────────────────────────────────────────────────────────────


async def _upsert_climate(conn: asyncpg.Connection, rows: list[tuple]) -> int:
    """Bulk-upsert climate zone rows and refresh climate_zones_by_hex.

    Logic Flow:
        Parameterized INSERT … ON CONFLICT in _CHUNK_SIZE batches.
        Refreshes materialized view after completion.

    Args:
        conn: Active asyncpg connection to PostGIS.
        rows: List of (hex_id, zone_code, icar_zone_id, icar_zone_name) tuples.

    Returns:
        Total rows upserted.

    Expected Exceptions:
        asyncpg.PostgresError: Constraint or connection failure.
    """
    upsert_sql = """
        INSERT INTO climate_zones_raw
            (hex_id, zone_code, icar_zone_id, icar_zone_name, source)
        VALUES ($1, $2, $3, $4, 'beck2018_kg_1km')
        ON CONFLICT (hex_id) DO UPDATE SET
            zone_code      = EXCLUDED.zone_code,
            icar_zone_id   = EXCLUDED.icar_zone_id,
            icar_zone_name = EXCLUDED.icar_zone_name,
            ingested_at    = now()
    """
    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(upsert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])
        logger.info("climate.upsert.progress", inserted=inserted)

    await conn.execute("REFRESH MATERIALIZED VIEW climate_zones_by_hex")
    logger.info("view.refreshed", view="climate_zones_by_hex")
    return inserted


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, data_dir: Path) -> None:
    """Orchestrate the climate zone ingest pipeline.

    Logic Flow:
        1. Download KG raster if not present.
        2. Generate H3 res-7 hexes for India bbox.
        3. Sample KG zone and derive ICAR zone per hex.
        4. Upsert into PostGIS and refresh materialized view.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        data_dir:    Root directory for raw data files.

    Expected Exceptions:
        httpx.HTTPStatusError: figshare download failed.
        asyncpg.PostgresError: DB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_climate_zones")
    log.info("ingest.start")

    kg_path = data_dir / "koppen" / "koppen_geiger_1km.tif"
    if not kg_path.exists():
        log.info("kg.raster.not.found.downloading")
        _download_kg_raster(kg_path)

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

    rows = _sample_climate_zones(kg_path, hex_ids)

    conn = await asyncpg.connect(**get_postgis_dsn())
    try:
        inserted = await _upsert_climate(conn, rows)
        log.info("ingest.complete", rows_inserted=inserted)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Köppen-Geiger climate zones → PostGIS climate_zones_raw"
    )
    parser.add_argument("--region",   required=True, help="Region code (e.g. IN)")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    asyncio.run(run(args.region, Path(args.data_dir)))


if __name__ == "__main__":
    main()
