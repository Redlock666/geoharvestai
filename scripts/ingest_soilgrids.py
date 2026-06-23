"""
Ingest SoilGrids v2 data into PostGIS soil_raw table.

Two operating modes (auto-detected):
  1. GeoTIFF mode: if India-clipped GeoTIFFs exist in --data-dir, sample them
     with rasterio (fast, ~1 min for all of India).
  2. REST API mode: if GeoTIFFs are absent, query SoilGrids REST API at H3
     resolution-4 centroids (~1 800 points), then propagate values to all
     resolution-7 child hexes (~637 K hexes). Rate-limited to 3 req/s.

Usage:
    python scripts/ingest_soilgrids.py --region IN
    python scripts/ingest_soilgrids.py --region IN --data-dir data/raw
    python scripts/ingest_soilgrids.py --region IN --api-only   # force REST mode
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import h3
import httpx
import numpy as np
import structlog
from rasterio.transform import rowcol
from tenacity import retry, stop_after_attempt, wait_exponential

# rasterio import deferred to avoid import cost when unused
import rasterio  # noqa: E402

from configs.india import (
    BBOX,
    H3_RESOLUTION,
    H3_SOILGRIDS_BATCH_RESOLUTION,
    SOILGRIDS_REST_URL,
)
from db.settings import get_postgis_dsn

logger = structlog.get_logger(__name__)

_SOILGRIDS_PROPS = ["nitrogen", "phh2o", "clay", "sand", "silt"]
_API_CONCURRENCY = 3   # SoilGrids free tier: stay well under their rate limit
_API_DELAY_S     = 0.35


# ── USDA texture classification ───────────────────────────────────────────


def _classify_usda_texture(sand: float, silt: float, clay: float) -> str:
    """Classify soil texture using the USDA Soil Texture Triangle.

    Logic Flow:
        Evaluates clay percentage first (most diagnostic), then silt,
        then sand in decreasing order of specificity.
        Returns 'unknown' for NaN or all-zero inputs.

    Args:
        sand: Sand percentage (0–100).
        silt: Silt percentage (0–100).
        clay: Clay percentage (0–100).

    Returns:
        USDA texture class string (e.g. 'clay loam', 'sandy loam').

    Expected Exceptions:
        None — handles NaN and zero inputs gracefully.
    """
    if any(np.isnan(v) for v in [sand, silt, clay]):
        return "unknown"
    if clay >= 40:
        if sand > 45:
            return "sandy clay"
        if silt > 40:
            return "silty clay"
        return "clay"
    if clay >= 27:
        if sand > 45:
            return "sandy clay loam"
        if silt >= 28:
            return "clay loam"
        return "silty clay loam"
    if clay >= 20:
        if sand >= 45:
            return "sandy clay loam"
        if silt >= 50:
            return "silty clay loam"
        return "loam"
    if silt >= 80:
        return "silt" if clay < 12 else "silt loam"
    if silt >= 50:
        return "silt loam"
    if sand >= 85:
        return "sand"
    if sand >= 70:
        return "loamy sand"
    if sand >= 52:
        return "sandy loam"
    return "loam"


# ── GeoTIFF mode ─────────────────────────────────────────────────────────


def _sample_geotiffs(tiff_dir: Path, hex_ids: list[str]) -> list[dict]:
    """Sample all five SoilGrids GeoTIFFs at H3 res-7 hex centroids.

    Logic Flow:
        Opens each GeoTIFF once, reads the full raster band into memory,
        converts all hex centroids to (row, col) in one vectorised call,
        then indexes the array. Derives USDA texture from clay/sand/silt.
        Converts raw SoilGrids encoding to real units:
          nitrogen: cg/kg → g/kg  (÷ 100)
          phh2o:    pH×10 → pH    (÷ 10)
          clay/sand/silt: g/kg → % (÷ 10)

    Args:
        tiff_dir: Path containing *_0-5cm_india.tif files.
        hex_ids:  List of H3 resolution-7 hex IDs covering India.

    Returns:
        List of row dicts ready for asyncpg.executemany().

    Expected Exceptions:
        FileNotFoundError: A required GeoTIFF is missing.
        rasterio.errors.RasterioIOError: Corrupt file.
    """
    log = logger.bind(mode="geotiff", count=len(hex_ids))
    log.info("sampling.start")

    latlons = np.array([h3.h3_to_geo(h) for h in hex_ids])  # (lat, lon)
    lats = latlons[:, 0]
    lons = latlons[:, 1]

    bands: dict[str, np.ndarray] = {}
    for prop in _SOILGRIDS_PROPS:
        path = tiff_dir / f"{prop}_0-5cm_india.tif"
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            rows_idx, cols_idx = rowcol(src.transform, lons, lats)
            rows_idx = np.clip(rows_idx, 0, arr.shape[0] - 1)
            cols_idx = np.clip(cols_idx, 0, arr.shape[1] - 1)
            bands[prop] = arr[rows_idx, cols_idx]
        log.info("geotiff.loaded", prop=prop)

    rows = []
    for i, hex_id in enumerate(hex_ids):
        n   = bands["nitrogen"][i]
        ph  = bands["phh2o"][i]
        cl  = bands["clay"][i]
        sa  = bands["sand"][i]
        si  = bands["silt"][i]

        # Skip hexes with no data at all (ocean / outside India)
        if np.isnan(n) and np.isnan(ph):
            continue

        rows.append({
            "hex_id":           hex_id,
            "nitrogen_g_kg":    float(n / 100) if not np.isnan(n) else None,
            "ph":               float(ph / 10) if not np.isnan(ph) else None,
            "clay_pct":         float(cl / 10) if not np.isnan(cl) else None,
            "sand_pct":         float(sa / 10) if not np.isnan(sa) else None,
            "silt_pct":         float(si / 10) if not np.isnan(si) else None,
            "texture":          _classify_usda_texture(
                                    float(sa / 10) if not np.isnan(sa) else 33.0,
                                    float(si / 10) if not np.isnan(si) else 33.0,
                                    float(cl / 10) if not np.isnan(cl) else 33.0,
                                ),
        })

    log.info("sampling.complete", rows=len(rows))
    return rows


# ── REST API mode ─────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
async def _query_soilgrids_api(
    session: httpx.AsyncClient,
    lon: float,
    lat: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, float | None]:
    """Query SoilGrids v2 REST API for one coordinate.

    Logic Flow:
        Acquires semaphore to honour rate limit, sleeps _API_DELAY_S,
        then issues GET with all five properties in a single request.
        Parses the response into a flat {prop: value} dict.
        Converts raw units (cg/kg, pH×10, g/kg) to real units.

    Args:
        session:   Shared httpx.AsyncClient.
        lon:       Longitude (WGS84).
        lat:       Latitude  (WGS84).
        semaphore: Limits concurrent requests to _API_CONCURRENCY.

    Returns:
        Dict with keys: nitrogen_g_kg, ph, clay_pct, sand_pct, silt_pct.

    Expected Exceptions:
        httpx.HTTPStatusError: Non-2xx response after retries.
        httpx.TimeoutException: Network timeout.
    """
    async with semaphore:
        await asyncio.sleep(_API_DELAY_S)
        resp = await session.get(
            SOILGRIDS_REST_URL,
            params={
                "lon":      lon,
                "lat":      lat,
                "property": _SOILGRIDS_PROPS,
                "depth":    ["0-5cm"],
                "value":    ["mean"],
            },
        )
        resp.raise_for_status()

    data = resp.json()
    result: dict[str, float | None] = {}
    for layer in data.get("properties", {}).get("layers", []):
        name = layer["name"]
        for depth_entry in layer.get("depths", []):
            if depth_entry["label"] == "0-5cm":
                val = depth_entry["values"].get("mean")
                result[name] = float(val) if val is not None else None

    return {
        "nitrogen_g_kg": (result.get("nitrogen") or 0.0) / 100,
        "ph":            (result.get("phh2o")    or 70.0) / 10,
        "clay_pct":      (result.get("clay")     or 0.0)  / 10,
        "sand_pct":      (result.get("sand")     or 0.0)  / 10,
        "silt_pct":      (result.get("silt")     or 0.0)  / 10,
    }


async def _sample_via_api(bbox: dict[str, float]) -> list[dict]:
    """Fetch soil data for India via SoilGrids REST API.

    Logic Flow:
        Generates H3 resolution-4 hexes (~1 800) covering the bbox.
        Queries SoilGrids API for each res-4 centroid concurrently
        (bounded by semaphore).
        Propagates each res-4 value to all res-7 child hexes.
        Derives USDA texture from clay/sand/silt per hex.

    Args:
        bbox: Bounding box dict with min/max lat/lon keys.

    Returns:
        List of row dicts ready for asyncpg.executemany().

    Expected Exceptions:
        httpx.HTTPStatusError: API failure after retries.
    """
    log = logger.bind(mode="rest_api")

    bbox_poly = {
        "type": "Polygon",
        "coordinates": [[
            [bbox["min_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["min_lat"]],
            [bbox["max_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["max_lat"]],
            [bbox["min_lon"], bbox["min_lat"]],
        ]],
    }
    h4_hexes = list(h3.polyfill(bbox_poly, H3_SOILGRIDS_BATCH_RESOLUTION))
    log.info("api.hex4.count", count=len(h4_hexes))

    semaphore = asyncio.Semaphore(_API_CONCURRENCY)
    async with httpx.AsyncClient(timeout=30.0) as session:
        tasks = []
        for h4 in h4_hexes:
            lat, lon = h3.h3_to_geo(h4)
            tasks.append(_query_soilgrids_api(session, lon, lat, semaphore))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    rows = []
    for h4, result in zip(h4_hexes, results):
        if isinstance(result, BaseException):
            log.warning("api.hex.failed", hex=h4, error=str(result))
            continue
        # result is dict[str, float | None] after the BaseException guard above
        soil: dict[str, float | None] = result  # type: ignore[assignment]
        texture = _classify_usda_texture(
            soil["sand_pct"] or 0.0,
            soil["silt_pct"] or 0.0,
            soil["clay_pct"] or 0.0,
        )
        # Propagate to all resolution-7 children
        for h7 in h3.h3_to_children(h4, H3_RESOLUTION):
            rows.append({
                "hex_id":        h7,
                "nitrogen_g_kg": soil["nitrogen_g_kg"],
                "ph":            soil["ph"],
                "clay_pct":      soil["clay_pct"],
                "sand_pct":      soil["sand_pct"],
                "silt_pct":      soil["silt_pct"],
                "texture":       texture,
            })

    log.info("api.rows.prepared", count=len(rows))
    return rows


# ── DB upsert ─────────────────────────────────────────────────────────────


async def _upsert_soil(conn: asyncpg.Connection, rows: list[dict]) -> int:
    """Bulk-upsert soil rows into soil_raw and refresh materialized view.

    Logic Flow:
        Executes parameterized INSERT … ON CONFLICT DO UPDATE in chunks
        of 5 000 rows to avoid exceeding asyncpg parameter limits.
        Refreshes soil_by_hex materialized view after all inserts.

    Args:
        conn: Active asyncpg connection to PostGIS.
        rows: List of row dicts from _sample_geotiffs or _sample_via_api.

    Returns:
        Number of rows upserted.

    Expected Exceptions:
        asyncpg.PostgresError: Constraint violation or connection failure.
    """
    log = logger.bind(total_rows=len(rows))
    upsert_sql = """
        INSERT INTO soil_raw
            (hex_id, nitrogen_g_kg, ph, clay_pct, sand_pct, silt_pct, texture, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (hex_id) DO UPDATE SET
            nitrogen_g_kg = EXCLUDED.nitrogen_g_kg,
            ph            = EXCLUDED.ph,
            clay_pct      = EXCLUDED.clay_pct,
            sand_pct      = EXCLUDED.sand_pct,
            silt_pct      = EXCLUDED.silt_pct,
            texture       = EXCLUDED.texture,
            ingested_at   = now()
    """
    chunk_size = 5_000
    inserted = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        await conn.executemany(
            upsert_sql,
            [
                (
                    r["hex_id"],
                    r["nitrogen_g_kg"],
                    r["ph"],
                    r["clay_pct"],
                    r["sand_pct"],
                    r["silt_pct"],
                    r["texture"],
                    "soilgrids_v2",
                )
                for r in chunk
            ],
        )
        inserted += len(chunk)
        log.info("upsert.progress", inserted=inserted)

    await conn.execute("REFRESH MATERIALIZED VIEW soil_by_hex")
    log.info("view.refreshed", view="soil_by_hex")
    return inserted


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, data_dir: Path, api_only: bool) -> None:
    """Orchestrate the full SoilGrids ingest pipeline.

    Logic Flow:
        1. Determine operating mode (GeoTIFF vs REST API).
        2. Generate H3 hexes covering the India bbox.
        3. Sample soil properties for each hex.
        4. Upsert into PostGIS and refresh materialized view.
        5. Write result to ingest_log.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        data_dir:    Path to directory containing downloaded GeoTIFFs.
        api_only:    If True, skip GeoTIFF detection and use REST API.

    Expected Exceptions:
        FileNotFoundError: GeoTIFFs missing and api_only is False.
        asyncpg.PostgresError: DB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_soilgrids")
    log.info("ingest.start")

    tiff_dir = data_dir / "soilgrids"
    tiffs_present = not api_only and all(
        (tiff_dir / f"{p}_0-5cm_india.tif").exists() for p in _SOILGRIDS_PROPS
    )
    log.info("mode.selected", mode="geotiff" if tiffs_present else "rest_api")

    # Generate all H3 res-7 hexes for the bbox (GeoTIFF mode samples these directly)
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

    if tiffs_present:
        hex_ids = list(h3.polyfill(bbox_poly, H3_RESOLUTION))
        log.info("hexes.generated", resolution=H3_RESOLUTION, count=len(hex_ids))
        rows = _sample_geotiffs(tiff_dir, hex_ids)
    else:
        rows = await _sample_via_api(BBOX)

    conn = await asyncpg.connect(**get_postgis_dsn())
    try:
        inserted = await _upsert_soil(conn, rows)
        log.info("ingest.complete", rows_inserted=inserted)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SoilGrids v2 soil data → PostGIS soil_raw"
    )
    parser.add_argument(
        "--region", required=True,
        help="Region code supplied at runtime (e.g. IN)"
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root directory for downloaded raw data files (default: data/raw)"
    )
    parser.add_argument(
        "--api-only", action="store_true",
        help="Force REST API mode even if GeoTIFFs are present"
    )
    args = parser.parse_args()
    asyncio.run(run(args.region, Path(args.data_dir), args.api_only))


if __name__ == "__main__":
    main()
