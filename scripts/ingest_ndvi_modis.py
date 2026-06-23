"""
Ingest MODIS MOD13A2 NDVI history into TimescaleDB ndvi_obs.

Downloads MOD13A2 Version 061 (16-day NDVI composite, 1 km, 2010–present)
for India's bounding box via NASA earthaccess, parses the HDF files with
rioxarray, maps each H3 resolution-7 hex to its nearest MODIS pixel, and
inserts into ndvi_obs.

Usage:
    python scripts/ingest_ndvi_modis.py --region IN --years 2010-2025
    python scripts/ingest_ndvi_modis.py --region IN --years 2023-2023
"""

from __future__ import annotations

import argparse
import asyncio
import netrc
import os
from datetime import datetime
from pathlib import Path

import asyncpg
import h3
import numpy as np
import structlog

from configs.india import (
    BBOX,
    H3_RESOLUTION,
    MODIS_NDVI_SHORT_NAME,
    MODIS_NDVI_VERSION,
)
from db.settings import get_timescale_dsn

logger = structlog.get_logger(__name__)

_CHUNK_SIZE  = 10_000
_NDVI_SCALE  = 0.0001   # MOD13A2 NDVI scale factor (raw int16 × 0.0001 = NDVI)
_NDVI_FILL   = -3000    # MOD13A2 fill value for missing/masked pixels


# ── Download via earthaccess ──────────────────────────────────────────────


def _ensure_earthdata_auth() -> None:
    """Ensure Earthdata credentials are available before download.

    Logic Flow:
        Checks environment variables EARTHDATA_USERNAME/EARTHDATA_PASSWORD.
        If absent, attempts to read ~/.netrc entry for urs.earthdata.nasa.gov.
        If netrc entry exists, exports credentials to env for earthaccess.
        Raises a clear error if neither source is available.

    Expected Exceptions:
        RuntimeError: If credentials are unavailable in env and ~/.netrc.
    """
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return

    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None

    if auth:
        username, _, password = auth
        if username and password:
            os.environ["EARTHDATA_USERNAME"] = username
            os.environ["EARTHDATA_PASSWORD"] = password
            return

    raise RuntimeError(
        "Earthdata credentials missing. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD "
        "or add ~/.netrc entry for machine urs.earthdata.nasa.gov."
    )


def _download_modis_year(year: int, data_dir: Path) -> list[Path]:
    """Download MOD13A2 HDF files for India for a given year.

    Logic Flow:
        Authenticates with NASA EarthData using EARTHDATA_USERNAME and
        EARTHDATA_PASSWORD environment variables.
        Searches for MOD13A2.061 granules intersecting India's bbox.
        Downloads to data_dir / 'ndvi' / str(year).
        Returns paths to downloaded files.

    Args:
        year:     Calendar year to download.
        data_dir: Root raw data directory.

    Returns:
        List of local HDF4 file paths.

    Expected Exceptions:
        earthaccess.exceptions.LoginError: Invalid NASA credentials.
        Exception: Network error during download.
    """
    import earthaccess  # type: ignore[import]

    out_dir = data_dir / "ndvi" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logger.bind(year=year, out_dir=str(out_dir))
    log.info("modis.download.start")

    _ensure_earthdata_auth()

    earthaccess.login(
        strategy="environment",  # uses EARTHDATA_USERNAME + EARTHDATA_PASSWORD
    )

    results = earthaccess.search_data(
        short_name=MODIS_NDVI_SHORT_NAME,
        version=MODIS_NDVI_VERSION,
        bounding_box=(
            BBOX["min_lon"], BBOX["min_lat"],
            BBOX["max_lon"], BBOX["max_lat"],
        ),
        temporal=(f"{year}-01-01", f"{year}-12-31"),
    )
    log.info("modis.search.results", count=len(results))

    files = earthaccess.download(results, local_path=str(out_dir))
    log.info("modis.download.complete", count=len(files))
    return [Path(f) for f in files]


# ── HDF parsing ───────────────────────────────────────────────────────────


def _parse_modis_hdf(hdf_path: Path, region_code: str, hex_ids: list[str]) -> list[tuple]:
    """Parse one MOD13A2 HDF file into per-hex NDVI rows.

    Logic Flow:
        Opens HDF with rioxarray, reads the '1 km 16 days NDVI' subdataset.
        Extracts acquisition date from filename (e.g. MOD13A2.A2023001).
        Converts raw int16 to float NDVI (× 0.0001).
        Masks fill values (_NDVI_FILL).
        For each hex centroid snaps to nearest MODIS pixel (1 km grid).
        Skips pixels where NDVI is masked (cloud cover or missing).

    Args:
        hdf_path:    Path to MOD13A2 .hdf file.
        region_code: Runtime region identifier (e.g. 'IN').
        hex_ids:     H3 resolution-7 hex IDs to sample.

    Returns:
        List of (time, hex_id, region_code, ndvi, cloud_cover_pct, source) tuples.

    Expected Exceptions:
        rasterio.errors.RasterioIOError: Corrupt HDF file.
        KeyError: Expected subdataset not present in HDF.
    """
    try:
        import rioxarray  # type: ignore[import]
        import xarray as xr  # type: ignore[import]
    except ImportError as e:
        raise ImportError("rioxarray required: pip install rioxarray") from e

    log = logger.bind(file=hdf_path.name)

    # Parse acquisition date from filename: MOD13A2.AYYYYDDD.*.hdf
    fname = hdf_path.stem
    parts = fname.split(".")
    try:
        acq_date = datetime.strptime(parts[1], "A%Y%j")
    except (IndexError, ValueError):
        log.warning("modis.filename.parse.failed", fname=fname)
        return []

    # Open the NDVI subdataset
    try:
        da = rioxarray.open_rasterio(
            f'HDF4_EOS:EOS_GRID:"{hdf_path}":MODIS_Grid_16DAY_1km_500m_VI:1 km 16 days NDVI'
        ).squeeze()
    except Exception:
        # Fallback: open full HDF and select by variable name
        ds = xr.open_dataset(hdf_path, engine="rasterio")
        candidates = [v for v in ds.data_vars if "NDVI" in v.upper()]
        if not candidates:
            log.warning("modis.ndvi.variable.not.found", file=fname)
            return []
        da = ds[candidates[0]].squeeze()

    ndvi_raw = da.values.astype(np.float32)
    ndvi_raw[ndvi_raw <= _NDVI_FILL] = np.nan
    ndvi_scaled = ndvi_raw * _NDVI_SCALE
    ndvi_scaled = np.clip(ndvi_scaled, -1.0, 1.0)

    # MODIS sinusoidal → WGS84 handled by rioxarray
    latlons = np.array([h3.h3_to_geo(hx) for hx in hex_ids])

    rows: list[tuple] = []
    for i, hex_id in enumerate(hex_ids):
        lat = float(latlons[i, 0])
        lon = float(latlons[i, 1])

        try:
            val = float(da.sel(y=lat, x=lon, method="nearest").values)
            if val <= _NDVI_FILL:
                continue
            ndvi = float(np.clip(val * _NDVI_SCALE, -1.0, 1.0))
        except Exception:
            continue

        rows.append((
            acq_date,
            hex_id,
            region_code,
            ndvi,
            None,         # cloud_cover_pct not in MOD13A2 directly
            "modis_mod13a2",
        ))

    log.info("modis.parse.complete", rows=len(rows), date=acq_date.date())
    return rows


# ── DB insert ─────────────────────────────────────────────────────────────


async def _insert_ndvi(conn: asyncpg.Connection, rows: list[tuple]) -> int:
    """Bulk-insert NDVI rows into TimescaleDB ndvi_obs.

    Logic Flow:
        Parameterized INSERT … ON CONFLICT DO NOTHING for idempotency.
        Processes in _CHUNK_SIZE batches.

    Args:
        conn: Active asyncpg connection to TimescaleDB.
        rows: List of ndvi_obs row tuples.

    Returns:
        Total rows inserted.

    Expected Exceptions:
        asyncpg.PostgresError: Schema mismatch or connection failure.
    """
    insert_sql = """
        INSERT INTO ndvi_obs
            (time, hex_id, region_code, ndvi, cloud_cover_pct, source)
        VALUES ($1,$2,$3,$4,$5,$6)
        ON CONFLICT DO NOTHING
    """
    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(insert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])
        logger.info("ndvi.insert.progress", inserted=inserted)
    return inserted


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, years: range, data_dir: Path) -> None:
    """Orchestrate the MODIS NDVI ingest pipeline.

    Logic Flow:
        For each year in the range:
          1. Download MOD13A2 HDF files via earthaccess.
          2. Generate H3 res-7 hexes for India bbox.
          3. Parse each HDF → per-hex NDVI rows.
          4. Insert into TimescaleDB ndvi_obs.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        years:       Range of calendar years to ingest.
        data_dir:    Root directory for downloaded HDF files.

    Expected Exceptions:
        earthaccess.exceptions.LoginError: NASA credentials invalid.
        asyncpg.PostgresError: TimescaleDB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_ndvi_modis",
                      years=f"{years.start}-{years.stop - 1}")
    log.info("ingest.start")

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

    conn = await asyncpg.connect(**get_timescale_dsn())
    try:
        total_inserted = 0
        for year in years:
            hdf_files = _download_modis_year(year, data_dir)
            for hdf_path in hdf_files:
                rows = _parse_modis_hdf(hdf_path, region_code, hex_ids)
                if rows:
                    inserted = await _insert_ndvi(conn, rows)
                    total_inserted += inserted
            log.info("year.complete", year=year)

        log.info("ingest.complete", total_rows=total_inserted)
    finally:
        await conn.close()


def _parse_year_range(s: str) -> range:
    parts = s.split("-")
    return range(int(parts[0]), int(parts[-1]) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest MODIS MOD13A2 NDVI → TimescaleDB ndvi_obs"
    )
    parser.add_argument("--region",   required=True, help="Region code (e.g. IN)")
    parser.add_argument("--years",    required=True, help="Year or range, e.g. 2010-2025")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    asyncio.run(run(args.region, _parse_year_range(args.years), Path(args.data_dir)))


if __name__ == "__main__":
    main()
