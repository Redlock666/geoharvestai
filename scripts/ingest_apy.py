"""
Ingest APY (Area, Production, Yield) crop data into TimescaleDB crop_yield_obs.

This is the PRIMARY ML TRAINING TARGET for the LSTM + SARIMAX ensemble.

The APY Portal (aps.dac.gov.in/APY) provides district-level, season-level,
crop-level data covering all Indian states from 1966 to present.  The CSV
must be downloaded manually (browser interaction required) — see
download_india_data.sh for instructions.

Expected CSV columns (case-insensitive, flexible order):
    APS-native:
        State Name | District Name | Crop | Year | Season | Area (in Ha) | Production (in Tonnes)
    Fallback-compatible (data.gov/Kaggle/state DES variants):
        state/district/crop/year/season with either:
          (area + production) OR (yield/yield_kg_ha)

Year format: '2021-22'  (Indian financial year: April YYYY to March YYYY+1)
Season values: Kharif | Rabi | Zaid | Whole Year

Harvest date derivation (stored in 'time' column):
    Kharif YYYY-YY  →  YYYY-10-01   (Oct harvest, year = first part)
    Rabi   YYYY-YY  →  YYYY+1-04-01 (Apr harvest, year = second part)
    Zaid   YYYY-YY  →  YYYY+1-07-01 (Jul harvest)
    Whole Year       →  YYYY+1-03-01 (Mar, e.g. sugarcane)

Usage:
    python scripts/ingest_apy.py --region IN --file data/raw/apy/apy_india_all.csv
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

import asyncpg
import pandas as pd
import structlog

from configs.india import REGION_CODE, SEASON_HARVEST_MONTH, SEASON_YEAR_OFFSET
from db.settings import get_timescale_dsn

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 10_000

# Canonical season name → normalised key
_SEASON_ALIASES: dict[str, str] = {
    "kharif":     "kharif",
    "rabi":       "rabi",
    "zaid":       "zaid",
    "whole year": "whole_year",
    "whole_year": "whole_year",
    "annual":     "whole_year",
}


# ── CSV parsing ───────────────────────────────────────────────────────────


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise APY CSV column names to a canonical set.

    Logic Flow:
        Strips whitespace, lowercases headers, then maps common APY
        Portal column name variants to canonical names.
        Raises ValueError if any required column is absent after mapping.

    Args:
        df: Raw DataFrame from pd.read_csv().

    Returns:
        DataFrame with canonical columns where available:
        state, district, crop, year, season, area_ha, production_t,
        yield_kg_ha.

    Expected Exceptions:
        ValueError: Core columns (crop/year/season and target inputs) are missing.
    """
    df.columns = df.columns.str.strip().str.lower()

    col_map: dict[str, str] = {
        "state name":           "state",
        "state_name":           "state",
        "state":                "state",
        "district name":        "district",
        "district_name":        "district",
        "district":             "district",
        "districts":            "district",
        "crop":                 "crop",
        "crop name":            "crop",
        "crop_name":            "crop",
        "commodity":            "crop",
        "year":                 "year",
        "crop year":            "year",
        "crop_year":            "year",
        "season":               "season",
        "area (in ha)":         "area_ha",
        "area(in ha)":          "area_ha",
        "area_in_ha":           "area_ha",
        "area":                 "area_ha",
        "area_ha":              "area_ha",
        "area (ha)":            "area_ha",
        "production (in tonnes)": "production_t",
        "production(in tonnes)":  "production_t",
        "production_in_tonnes":   "production_t",
        "production":           "production_t",
        "production_t":         "production_t",
        "yield":                "yield_kg_ha",
        "yield (kg/ha)":        "yield_kg_ha",
        "yield kg/ha":          "yield_kg_ha",
        "yield_kg_ha":          "yield_kg_ha",
    }
    df = df.rename(columns=col_map)

    required_core = {"crop", "year", "season"}
    missing_core = required_core - set(df.columns)
    if missing_core:
        raise ValueError(f"APY CSV missing core columns after normalisation: {missing_core}")

    has_area_prod = {"area_ha", "production_t"}.issubset(set(df.columns))
    has_yield = "yield_kg_ha" in df.columns
    if not has_area_prod and not has_yield:
        raise ValueError(
            "APY CSV must include either (area_ha + production_t) or yield_kg_ha after normalisation."
        )

    # Optional geography columns fallback for alternate sources.
    if "state" not in df.columns:
        df["state"] = "Unknown"
    if "district" not in df.columns:
        df["district"] = "Unknown"

    # Ensure all canonical columns exist for downstream logic.
    for col in ["area_ha", "production_t", "yield_kg_ha"]:
        if col not in df.columns:
            df[col] = pd.NA

    required = {"state", "district", "crop", "year", "season", "area_ha", "production_t", "yield_kg_ha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"APY CSV missing columns after normalisation: {missing}")

    return df[list(required)]


def _parse_harvest_date(year_str: str, season_norm: str) -> date | None:
    """Derive harvest date from APY financial year string and season.

    Logic Flow:
        Splits '2021-22' into base_year=2021 and offset_year=2022.
        Looks up season in SEASON_YEAR_OFFSET and SEASON_HARVEST_MONTH.
        Returns None for unparseable year strings.

    Args:
        year_str:    APY year string, e.g. '2021-22' or '2021'.
        season_norm: Normalised season key ('kharif', 'rabi', etc.).

    Returns:
        date object representing the approximate harvest date, or None.

    Expected Exceptions:
        None — returns None on any parse error.
    """
    try:
        parts = str(year_str).strip().split("-")
        base_year = int(parts[0])
    except (ValueError, IndexError):
        return None

    year_offset = SEASON_YEAR_OFFSET.get(season_norm, 0)
    harvest_year = base_year + year_offset
    harvest_month = SEASON_HARVEST_MONTH.get(season_norm, 10)

    try:
        return date(harvest_year, harvest_month, 1)
    except ValueError:
        return None


def _load_and_clean(csv_path: Path, region_code: str) -> pd.DataFrame:
    """Load and clean the APY CSV into a normalised DataFrame.

    Logic Flow:
        Reads CSV with flexible encoding detection (UTF-8 → Latin-1 fallback).
        Normalises column names.
        Drops rows with 0 or NaN area (no cultivation recorded).
        Normalises state, district, crop to title-case.
        Normalises season to lowercase key.
        Parses harvest date, drops rows with unparseable dates.
    Computes yield_kg_ha = production_t × 1000 / area_ha when possible,
    otherwise uses provided yield_kg_ha from fallback datasets.
        Clips implausible yield values (>100 000 kg/ha → NaN).

    Args:
        csv_path:    Path to APY CSV file.
        region_code: Runtime region identifier assigned to all rows.

    Returns:
        Cleaned DataFrame ready for DB insert.

    Expected Exceptions:
        FileNotFoundError: CSV file absent.
        ValueError: Required columns missing.
    """
    log = logger.bind(file=str(csv_path), region=region_code)
    log.info("apy.load.start")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    log.info("apy.raw.rows", count=len(df))

    df = _normalise_columns(df)

    # Coerce numeric columns
    df["area_ha"]      = pd.to_numeric(df["area_ha"],      errors="coerce")
    df["production_t"] = pd.to_numeric(df["production_t"], errors="coerce")
    df["yield_kg_ha"]  = pd.to_numeric(df["yield_kg_ha"],  errors="coerce")

    # Keep rows that can support either computed or explicit yield
    has_area_prod = df["area_ha"].notna() & (df["area_ha"] > 0) & df["production_t"].notna()
    has_direct_yield = df["yield_kg_ha"].notna() & (df["yield_kg_ha"] > 0)
    df = df[has_area_prod | has_direct_yield].copy()

    # String normalisation
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()
    df["crop"]     = df["crop"].str.strip().str.title()
    df["year"]     = df["year"].str.strip()
    df["season"]   = (df["season"].str.strip().str.lower()
                      .map(lambda s: _SEASON_ALIASES.get(s, s)))

    # Harvest date
    df["harvest_date"] = [
        _parse_harvest_date(r["year"], r["season"])
        for _, r in df.iterrows()
    ]
    df = df[df["harvest_date"].notna()].copy()

    # Yield computation (prefer area+production when both available)
    calc_mask = df["area_ha"].notna() & (df["area_ha"] > 0) & df["production_t"].notna()
    df.loc[calc_mask, "yield_kg_ha"] = (df.loc[calc_mask, "production_t"] * 1000.0) / df.loc[calc_mask, "area_ha"]
    df.loc[df["yield_kg_ha"] > 100_000, "yield_kg_ha"] = None  # implausible values

    df["region_code"] = region_code

    log.info("apy.clean.complete", rows=len(df))
    return df


# ── DB insert ─────────────────────────────────────────────────────────────


async def _insert_yields(conn: asyncpg.Connection, df: pd.DataFrame) -> int:
    """Bulk-insert cleaned APY rows into TimescaleDB crop_yield_obs.

    Logic Flow:
        Converts DataFrame to list of tuples.
        Inserts in _CHUNK_SIZE batches using ON CONFLICT DO NOTHING
        for idempotent re-runs.

    Args:
        conn: Active asyncpg connection to TimescaleDB.
        df:   Cleaned APY DataFrame from _load_and_clean().

    Returns:
        Total rows inserted.

    Expected Exceptions:
        asyncpg.PostgresError: Schema mismatch or connection failure.
    """
    insert_sql = """
        INSERT INTO crop_yield_obs
            (time, region_code, state, district, crop_name, season,
             apy_year, area_ha, production_t, yield_kg_ha, source)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'apy_moa')
        ON CONFLICT DO NOTHING
    """
    rows = [
        (
            row["harvest_date"],
            row["region_code"],
            row["state"],
            row["district"],
            row["crop"],
            row["season"],
            row["year"],
            row["area_ha"]      if pd.notna(row["area_ha"])      else None,
            row["production_t"] if pd.notna(row["production_t"]) else None,
            row["yield_kg_ha"]  if pd.notna(row["yield_kg_ha"])  else None,
        )
        for _, row in df.iterrows()
    ]

    inserted = 0
    for i in range(0, len(rows), _CHUNK_SIZE):
        await conn.executemany(insert_sql, rows[i : i + _CHUNK_SIZE])
        inserted += len(rows[i : i + _CHUNK_SIZE])
        logger.info("yield.insert.progress", inserted=inserted)

    return inserted


# ── Summary stats ─────────────────────────────────────────────────────────


def _print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of ingested data to stdout."""
    print("\n─── APY Ingest Summary ────────────────────────────────")
    print(f"  Total rows:   {len(df):>10,}")
    print(f"  States:       {df['state'].nunique():>10,}")
    print(f"  Districts:    {df['district'].nunique():>10,}")
    print(f"  Crops:        {df['crop'].nunique():>10,}")
    print(f"  Seasons:      {sorted(df['season'].unique())}")
    print(f"  Year range:   {df['year'].min()} → {df['year'].max()}")
    print("  Top 10 crops by area:")
    top = df.groupby("crop")["area_ha"].sum().nlargest(10)
    for crop, area in top.items():
        print(f"    {crop:<25} {area:>12,.0f} ha")
    print("───────────────────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────


async def run(region_code: str, csv_path: Path) -> None:
    """Orchestrate the APY crop yield ingest pipeline.

    Logic Flow:
        1. Load and clean APY CSV.
        2. Print summary statistics.
        3. Connect to TimescaleDB.
        4. Insert into crop_yield_obs.

    Args:
        region_code: User-supplied region identifier (e.g. 'IN').
        csv_path:    Path to downloaded APY CSV file.

    Expected Exceptions:
        FileNotFoundError: CSV file not downloaded yet.
        asyncpg.PostgresError: TimescaleDB unreachable.
    """
    log = logger.bind(region=region_code, script="ingest_apy", file=str(csv_path))
    log.info("ingest.start")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"APY CSV not found: {csv_path}\n"
            "Download manually from https://aps.dac.gov.in/APY/Public_Report1.aspx\n"
            "See scripts/download_india_data.sh for detailed instructions."
        )

    df = _load_and_clean(csv_path, region_code)
    _print_summary(df)

    conn = await asyncpg.connect(**get_timescale_dsn())
    try:
        inserted = await _insert_yields(conn, df)
        log.info("ingest.complete", rows_inserted=inserted)
        print(f"✅ Inserted {inserted:,} rows into crop_yield_obs")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest APY crop yield CSV → TimescaleDB crop_yield_obs"
    )
    parser.add_argument("--region", required=True, help="Region code (e.g. IN)")
    parser.add_argument(
        "--file", required=True,
        help="Path to APY CSV (download from aps.dac.gov.in/APY)"
    )
    args = parser.parse_args()
    asyncio.run(run(args.region, Path(args.file)))


if __name__ == "__main__":
    main()
