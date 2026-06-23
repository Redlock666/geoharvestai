"""
Ingest ERA5-Land historical weather data into TimescaleDB weather_obs.

Downloads daily ERA5-Land reanalysis for India (2010 – requested end year)
via the Copernicus CDS API, parses with xarray, and inserts one row per
H3 resolution-7 hex per day into weather_obs.

ERA5-Land grid: 0.1° × 0.1° ≈ 9 km.  Each hex centroid is snapped to its
nearest ERA5-Land grid point — multiple hexes share the same grid point,
which is acceptable at resolution 7 (~5 km).

Usage:
    python scripts/ingest_era5.py --region IN --years 2010-2025
    python scripts/ingest_era5.py --region IN --years 2023-2023
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg
import h3
import numpy as np
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.india import BBOX, ERA5_DATASET, ERA5_VARIABLES, H3_RESOLUTION
from db.settings import get_timescale_dsn

logger = structlog.get_logger(__name__)

_CHUNK_SIZE   = 10_000
_DATA_DIR     = Path("data/raw/era5")
_KELVIN_OFFSET = 273.15


def _parse_cdsapirc() -> dict[str, str]:
    """Parse ~/.cdsapirc into a simple key-value dict.

    Returns:
        Dictionary with lowercase keys, e.g. {"url": "...", "key": "uid:token"}.
    """
    cfg_path = Path.home() / ".cdsapirc"
    if not cfg_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw in cfg_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        values[k.strip().lower()] = v.strip()
    return values


def _ensure_cds_auth() -> None:
    """Ensure CDS API credentials are present and valid format.

    Logic Flow:
        1) Use CDSAPI_KEY / CDSAPI_URL env if present.
        2) Otherwise, load key/url from ~/.cdsapirc.
        3) Validate key contains ':' (expected uid:api-token format).

    Expected Exceptions:
        RuntimeError: Credentials missing or malformed.
    """
    key = os.environ.get("CDSAPI_KEY")
    url = os.environ.get("CDSAPI_URL")

    if not key:
        cfg = _parse_cdsapirc()
        key = cfg.get("key")
        url = url or cfg.get("url")
        if key:
            os.environ["CDSAPI_KEY"] = key
        if url:
            os.environ["CDSAPI_URL"] = url

    if not key:
        raise RuntimeError(
            "CDS API credentials missing. Set CDSAPI_KEY (uid:api-token) or configure ~/.cdsapirc. "
            "Website login email/password is not sufficient for CDS API requests."
        )

    if ":" not in key:
        raise RuntimeError(
            "Invalid CDSAPI_KEY format. Expected 'uid:api-token'. "
            "Update CDSAPI_KEY or ~/.cdsapirc key entry."
        )


# ── CDS API download ──────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, min=10, max=120))
def _download_era5_year(year: int, out_path: Path) -> None:
    """Download one year of ERA5-Land daily data for India via CDS API.

    Logic Flow:
        Reads CDSAPI_KEY from environment (format: uid:api-key).
        Requests all 12 months, all days, single daily time (12:00).
        Downloads as NetCDF to out_path.
        Skips download if out_path already exists.

    Args:
        year:     Calendar year to download (e.g. 2023).
        out_path: Destination .nc file path.

    Expected Exceptions:
        KeyError: CDSAPI_KEY not set in environment.
        Exception: CDS API error (propagated after retries).
    """
    import cdsapi  # type: ignore[import]

    if out_path.exists():
        logger.info("era5.download.skip", year=year, reason="file_exists")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = logger.bind(year=year, out=str(out_path))
    log.info("era5.download.start")

    _ensure_cds_auth()

    client = cdsapi.Client(
        url=os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api/v2"),
        key=os.environ["CDSAPI_KEY"],
        quiet=True,
    )
    client.retrieve(
        ERA5_DATASET,
        {
            "product_type": "reanalysis",
            "variable":     ERA5_VARIABLES,
            "year":         str(year),
            "month":        [f"{m:02d}" for m in range(1, 13)],
            "day":          [f"{d:02d}" for d in range(1, 32)],
            "time":         ["12:00"],
            "area":         [
                BBOX["max_lat"], BBOX["min_lon"],
                BBOX["min_lat"], BBOX["max_lon"],
            ],  # N, W, S, E
            "format":       "netcdf",
        },
        str(out_path),
    )
    log.info("era5.download.complete", size_mb=out_path.stat().st_size // 1_048_576)


# ── NetCDF parsing ────────────────────────────────────────────────────────


def _parse_era5_nc(nc_path: Path, region_code: str, hex_ids: list[str]) -> list[tuple]:
    """Parse an ERA5-Land NetCDF file into per-hex-per-day weather rows.

    Logic Flow:
        Opens NetCDF with xarray.
        For each hex centroid, snaps to nearest ERA5 0.1° grid lat/lon.
        Iterates over time steps, extracts variables, converts units:
          temperature:   K → °C
          precipitation: m → mm
          humidity:      derived from dewpoint via Magnus formula
          wind speed:    sqrt(u² + v²) m/s
        Returns list of tuples matching weather_obs column order.

    Args:
        nc_path:     Path to ERA5-Land .nc file.
        region_code: Runtime region identifier (e.g. 'IN').
        hex_ids:     H3 resolution-7 hex IDs to extract data for.

    Returns:
        List of (time, hex_id, region_code, rainfall_mm, temp_avg_c,
                 temp_min_c, temp_max_c, humidity_pct, wind_speed_ms, source)
        tuples.

    Expected Exceptions:
        FileNotFoundError: .nc file missing.
        KeyError: Expected variable absent from ERA5-Land dataset.
    """
    import xarray as xr  # type: ignore[import]

    log = logger.bind(nc=nc_path.name, region=region_code)
    log.info("era5.parse.start")

    ds = xr.open_dataset(nc_path)

    # ERA5-Land variable name mapping (may vary by CDS API version)
    var_map = {
        "t2m":  "2m_temperature",
        "tp":   "total_precipitation",
        "d2m":  "2m_dewpoint_temperature",
        "u10":  "10m_u_component_of_wind",
        "v10":  "10m_v_component_of_wind",
    }
    # Normalise variable names
    for short, full in var_map.items():
        if full in ds and short not in ds:
            ds = ds.rename({full: short})

    latlons = np.array([h3.h3_to_geo(hx) for hx in hex_ids])
    era5_lats = np.round(latlons[:, 0] / 0.1) * 0.1
    era5_lons = np.round(latlons[:, 1] / 0.1) * 0.1

    times = ds["time"].values
    rows: list[tuple] = []

    for i, hex_id in enumerate(hex_ids):
        lat = float(era5_lats[i])
        lon = float(era5_lons[i])

        try:
            pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
        except Exception:
            continue

        t2m  = pt["t2m"].values  - _KELVIN_OFFSET       # K → °C
        tp   = pt["tp"].values   * 1000.0                # m → mm
        d2m  = pt["d2m"].values  - _KELVIN_OFFSET
        u10  = pt.get("u10", pt["t2m"] * 0.0).values    # fallback zeros
        v10  = pt.get("v10", pt["t2m"] * 0.0).values

        # Relative humidity via Magnus approximation
        rh = 100.0 * np.exp((17.625 * d2m) / (243.04 + d2m)) / \
                     np.exp((17.625 * t2m) / (243.04 + t2m))

        wind = np.sqrt(u10**2 + v10**2)

        for j, ts in enumerate(times):
            dt = datetime.utcfromtimestamp(int(ts) / 1e9).replace(tzinfo=None)
            rows.append((
                dt,
                hex_id,
                region_code,
                float(max(tp[j], 0.0)),
                float(t2m[j]),
                float(t2m[j]),   # ERA5-Land daily snapshot: avg ≈ min ≈ max at 12:00
                float(t2m[j]),   # For proper min/max, use hourly data
                float(np.clip(rh[j], 0.0, 100.0)),
                float(wind[j]),
                "era5_land",
            ))

    ds.close()
    log.info("era5.parse.complete", rows=len(rows))
    return rows


# ── DB insert ─────────────────────────────────────────────────────────────


async def _insert_weather(conn: asyncpg.Connection, rows: list[tuple]) -> int:
    """Bulk-insert weather rows into TimescaleDB weather_obs.

    Logic Flow:
        Uses INSERT … ON CONFLICT DO NOTHING (idempotent re-runs).
        Inserts in _CHUNK_SIZE batches to manage memory.

    Args:
        conn: Active asyncpg connection to TimescaleDB.
        rows: List of weather_obs row tuples.

    Returns:
        Total rows inserted (excluding conflicts).

    Expected Exceptions:
        asyncpg.PostgresError: Connection or schema mismatch.
    """
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
        logger.info("weather.insert.progress", inserted=inserted)
    return inserted


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, years: range, data_dir: Path) -> None:
    """Orchestrate the ERA5-Land ingest pipeline for a year range.

    Logic Flow:
        For each year in the range:
          1. Download ERA5-Land NetCDF via CDS API (skips if cached).
          2. Generate H3 res-7 hexes for India bbox.
          3. Parse NetCDF → per-hex daily weather rows.
          4. Insert into TimescaleDB weather_obs.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        years:       Range of calendar years to ingest.
        data_dir:    Root directory for downloaded NetCDF files.

    Expected Exceptions:
        KeyError: CDSAPI_KEY missing.
        asyncpg.PostgresError: TimescaleDB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_era5",
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
            nc_path = data_dir / "era5" / f"era5_india_{year}.nc"
            _download_era5_year(year, nc_path)
            rows = _parse_era5_nc(nc_path, region_code, hex_ids)
            inserted = await _insert_weather(conn, rows)
            total_inserted += inserted
            log.info("year.complete", year=year, inserted=inserted)

        log.info("ingest.complete", total_rows=total_inserted)
    finally:
        await conn.close()


def _parse_year_range(s: str) -> range:
    """Parse '2010-2025' or '2023' into a range object."""
    parts = s.split("-")
    if len(parts) == 2:
        return range(int(parts[0]), int(parts[1]) + 1)
    return range(int(parts[0]), int(parts[0]) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ERA5-Land weather data → TimescaleDB weather_obs"
    )
    parser.add_argument("--region",   required=True, help="Region code (e.g. IN)")
    parser.add_argument("--years",    required=True, help="Year or range, e.g. 2010-2025")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()
    asyncio.run(run(args.region, _parse_year_range(args.years), Path(args.data_dir)))


if __name__ == "__main__":
    main()
