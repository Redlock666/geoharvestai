"""
Ingest India Soil Health Card (SHC) data into soil_health_raw.

Source: India Soil Health Card Portal (soilhealth.dac.gov.in)
        or equivalent state DES / ICAR district-level CSV exports.

Expected CSV columns (column names are normalised during load — see _COLUMN_ALIASES):
    district, state, latitude, longitude      — location (at least lat/lon required)
    ph                                         — soil pH (optional; SoilGrids used as fallback)
    ec                                         — electrical conductivity (dS/m)
    organic_carbon                             — OC % (WALKLEY-BLACK method)
    nitrogen                                   — available N (kg/ha, KMnO4)
    phosphorus                                 — available P (kg/ha, Olsen/Bray)
    potassium                                  — available K (kg/ha, NH4OAc)
    sulphur, zinc, iron, copper, manganese,
    boron                                      — micronutrients (mg/kg)
    year, survey_year                          — SHC survey year

Logic Flow:
    1. Load and normalise CSV column names.
    2. Convert lat/lon to H3 hex cells (resolution 7).
    3. Aggregate multiple cards per hex_id to median values.
    4. Classify sufficiency per ICAR critical limits.
    5. Compute NPK trend and organic carbon trend from multi-year data.
    6. Set biological_collapse_risk flag.
    7. Upsert into soil_health_raw (ON CONFLICT DO UPDATE).
    8. Refresh soil_health_by_hex materialized view.

Usage:
    python scripts/ingest_soil_health_cards.py --region IN --file data/raw/shc/shc_india.csv
    python scripts/ingest_soil_health_cards.py --region IN --file data/raw/shc/shc_india.csv --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg
import h3
import numpy as np
import pandas as pd
import structlog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.settings import get_postgis_dsn

logger = structlog.get_logger(__name__)

_H3_RESOLUTION = 7

# ── Column name normalisation ──────────────────────────────────────────────────
_COLUMN_ALIASES: dict[str, str] = {
    # location
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "long": "longitude",
    "longitude": "longitude",
    # pH
    "ph": "ph",
    "soil_ph": "ph",
    "ph_water": "ph",
    # EC
    "ec": "ec",
    "electrical_conductivity": "ec",
    "ec_ds_m": "ec",
    "ec (ds/m)": "ec",
    # Organic carbon
    "oc": "organic_carbon",
    "organic_carbon": "organic_carbon",
    "oc_%": "organic_carbon",
    "organic carbon (%)": "organic_carbon",
    # Macronutrients
    "avail_n": "nitrogen",
    "available_n": "nitrogen",
    "nitrogen": "nitrogen",
    "n": "nitrogen",
    "avail_p": "phosphorus",
    "available_p": "phosphorus",
    "phosphorus": "phosphorus",
    "p": "phosphorus",
    "avail_k": "potassium",
    "available_k": "potassium",
    "potassium": "potassium",
    "k": "potassium",
    # Micronutrients
    "s": "sulphur",
    "sulphur": "sulphur",
    "sulfur": "sulphur",
    "zn": "zinc",
    "zinc": "zinc",
    "fe": "iron",
    "iron": "iron",
    "cu": "copper",
    "copper": "copper",
    "mn": "manganese",
    "manganese": "manganese",
    "b": "boron",
    "boron": "boron",
    # Survey year
    "year": "survey_year",
    "survey_year": "survey_year",
    "sampleyear": "survey_year",
}

# ── ICAR critical limits for sufficiency classification ──────────────────────
# Source: ICAR Handbook of Agriculture (2016 edition)
# These are widely accepted national benchmarks.
_ICAR_LIMITS = {
    "nitrogen":  {"low": 280.0, "high": 560.0},   # kg/ha available N
    "phosphorus": {"low": 10.0, "high": 25.0},     # kg/ha available P
    "potassium": {"low": 108.0, "high": 280.0},    # kg/ha available K
    # Organic carbon ICAR thresholds: low <0.5%, medium 0.5-0.75%, high >0.75%
    "organic_carbon_low": 0.5,
    "organic_carbon_medium": 0.75,
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to canonical names via _COLUMN_ALIASES."""
    renamed = {col: _COLUMN_ALIASES.get(col.strip().lower().replace(" ", "_"), col.strip().lower())
               for col in df.columns}
    return df.rename(columns=renamed)


def _classify_sufficiency(value: float | None, nutrient: str) -> str:
    """Classify nutrient sufficiency per ICAR critical limits.

    Args:
        value: Measured nutrient value (units depend on nutrient).
        nutrient: One of 'nitrogen', 'phosphorus', 'potassium'.

    Returns:
        'deficient' | 'sufficient' | 'excessive' | 'unknown'
    """
    if value is None or np.isnan(value):
        return "unknown"
    limits = _ICAR_LIMITS.get(nutrient)
    if limits is None:
        return "unknown"
    if value < limits["low"]:
        return "deficient"
    if value > limits["high"] * 1.5:
        return "excessive"
    return "sufficient"


def _classify_oc(oc_pct: float | None) -> str:
    """Classify organic carbon level per ICAR thresholds."""
    if oc_pct is None or np.isnan(oc_pct):
        return "unknown"
    if oc_pct < _ICAR_LIMITS["organic_carbon_low"]:
        return "low"
    if oc_pct < _ICAR_LIMITS["organic_carbon_medium"]:
        return "medium"
    return "high"


def _compute_trend(series: pd.Series) -> str:
    """Compute a simple directional trend from a numeric time series.

    Logic Flow:
        Fits a linear regression over the series values.
        Returns 'improving', 'stable', or 'declining' based on slope.

    Args:
        series: Numeric pandas Series (typically year-ordered values).

    Returns:
        'improving' | 'stable' | 'declining' | 'unknown'
    """
    vals = series.dropna()
    if len(vals) < 2:
        return "unknown"
    x = np.arange(len(vals), dtype=float)
    slope = float(np.polyfit(x, np.array(vals, dtype=float), 1)[0])
    relative = abs(slope) / (abs(vals.mean()) + 1e-9)
    if relative < 0.01:
        return "stable"
    return "improving" if slope > 0 else "declining"


def _biological_collapse_risk(
    n_suf: str, p_suf: str, k_suf: str, oc_suf: str, npk_trend: str
) -> bool:
    """Flag potential biological collapse: NPK sufficient or excessive, OC declining.

    Logic Flow:
        When soil shows adequate/excessive NPK but organic carbon is low or declining,
        it suggests the microbial community that mediates nutrient availability is
        compromised — a hallmark of over-fertilization induced biological degradation.

    Returns:
        True if biological collapse risk is present.
    """
    npk_ok = all(s in ("sufficient", "excessive") for s in (n_suf, p_suf, k_suf))
    oc_degraded = oc_suf == "low"
    oc_declining = npk_trend == "declining"
    return npk_ok and (oc_degraded or oc_declining)


def _load_and_aggregate(csv_path: Path, region_code: str) -> pd.DataFrame:
    """Load SHC CSV, map to H3 hex cells, and aggregate to hex-level medians.

    Logic Flow:
        1. Read CSV, normalise columns, drop rows with no lat/lon.
        2. Compute H3 hex_id for each row.
        3. Group by hex_id + survey_year to get year-level medians.
        4. Compute per-feature trend across years.
        5. Aggregate to one row per hex_id using latest-year values + trends.

    Args:
        csv_path: Path to SHC CSV file.
        region_code: Runtime region identifier.

    Returns:
        DataFrame with one row per hex_id ready for DB upsert.
    """
    log = logger.bind(file=str(csv_path), region_code=region_code)
    log.info("shc.load.start")

    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalise_columns(df)
    log.info("shc.load.columns_normalised", columns=list(df.columns))

    # Require lat/lon
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(
            "SHC CSV must contain latitude and longitude columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Compute H3 hex_id
    df["hex_id"] = df.apply(
        lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], _H3_RESOLUTION), axis=1
    )

    # Coerce numeric columns
    numeric_cols = [
        "organic_carbon", "ec", "nitrogen", "phosphorus", "potassium",
        "sulphur", "zinc", "iron", "copper", "manganese", "boron", "survey_year",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Default survey_year if missing
    if "survey_year" not in df.columns:
        df["survey_year"] = pd.Timestamp.now().year
        log.warning("shc.load.no_survey_year", fallback=df["survey_year"].iloc[0])

    # Group by hex_id + survey_year → median per year
    agg_cols = {
        c: "median" for c in numeric_cols if c in df.columns and c != "survey_year"
    }
    agg_cols["survey_year"] = "first"
    yearly = df.groupby(["hex_id", "survey_year"]).agg(agg_cols).reset_index()
    yearly.columns = [
        c if c not in ("hex_id", "survey_year", "level_0", "level_1") else c
        for c in yearly.columns
    ]

    # Compute trends across years per hex_id
    def _hex_trends(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values("survey_year")
        oc_col = "organic_carbon" if "organic_carbon" in grp.columns else None
        npk_cols = [c for c in ("nitrogen", "phosphorus", "potassium") if c in grp.columns]
        npk_mean = grp[npk_cols].mean(axis=1) if npk_cols else pd.Series(dtype=float)

        return pd.Series({
            "organic_carbon_trend": _compute_trend(grp[oc_col]) if oc_col else "unknown",
            "npk_trend_direction": _compute_trend(npk_mean) if len(npk_mean) else "unknown",
            "survey_year_latest": int(grp["survey_year"].max()),
            "cards_aggregated": len(grp),
        })

    trends = yearly.groupby("hex_id").apply(_hex_trends).reset_index()

    # Latest-year values per hex_id
    latest = (
        yearly.sort_values("survey_year")
        .groupby("hex_id")
        .last()
        .reset_index()
    )

    merged = latest.merge(trends, on="hex_id", how="left")
    merged["region_code"] = region_code

    # Classify sufficiency
    def _get(row: pd.Series, col: str) -> float | None:
        val = row.get(col)
        return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

    merged["n_sufficiency"]  = merged.apply(lambda r: _classify_sufficiency(_get(r, "nitrogen"), "nitrogen"), axis=1)
    merged["p_sufficiency"]  = merged.apply(lambda r: _classify_sufficiency(_get(r, "phosphorus"), "phosphorus"), axis=1)
    merged["k_sufficiency"]  = merged.apply(lambda r: _classify_sufficiency(_get(r, "potassium"), "potassium"), axis=1)
    merged["oc_sufficiency"] = merged.apply(lambda r: _classify_oc(_get(r, "organic_carbon")), axis=1)

    merged["biological_collapse_risk"] = merged.apply(
        lambda r: _biological_collapse_risk(
            r["n_sufficiency"], r["p_sufficiency"], r["k_sufficiency"],
            r["oc_sufficiency"], r.get("npk_trend_direction", "unknown"),
        ),
        axis=1,
    )

    log.info("shc.load.aggregated", hexes=len(merged), cards=int(merged["cards_aggregated"].sum()))
    return merged


async def _upsert(conn: asyncpg.Connection, df: pd.DataFrame, dry_run: bool) -> tuple[int, int]:
    """Upsert aggregated SHC rows into soil_health_raw.

    Logic Flow:
        Iterates over DataFrame rows, upserts with ON CONFLICT (hex_id) DO UPDATE.
        Updates only if survey_year_latest in the incoming row is >= stored year.

    Returns:
        (inserted, updated) counts.
    """
    inserted = updated = 0

    def _float(val: object) -> float | None:
        try:
            f = float(val)  # type: ignore[arg-type]
            return None if (f != f) else f  # NaN check
        except (TypeError, ValueError):
            return None

    for _, row in df.iterrows():
        params = (
            row["hex_id"],
            row["region_code"],
            _float(row.get("organic_carbon")),
            _float(row.get("ec")),
            _float(row.get("nitrogen")),
            _float(row.get("phosphorus")),
            _float(row.get("potassium")),
            _float(row.get("sulphur")),
            _float(row.get("zinc")),
            _float(row.get("iron")),
            _float(row.get("copper")),
            _float(row.get("manganese")),
            _float(row.get("boron")),
            str(row.get("npk_trend_direction", "unknown")),
            str(row.get("organic_carbon_trend", "unknown")),
            str(row.get("n_sufficiency", "unknown")),
            str(row.get("p_sufficiency", "unknown")),
            str(row.get("k_sufficiency", "unknown")),
            str(row.get("oc_sufficiency", "unknown")),
            bool(row.get("biological_collapse_risk", False)),
            int(row.get("survey_year_latest", 0)) or None,
            int(row.get("cards_aggregated", 1)),
        )

        if dry_run:
            inserted += 1
            continue

        result = await conn.execute(
            """
            INSERT INTO soil_health_raw (
                hex_id, region_code,
                organic_carbon_pct, electrical_conductivity_ds_m,
                available_n_kg_ha, available_p_kg_ha, available_k_kg_ha,
                sulphur_mg_kg, zinc_mg_kg, iron_mg_kg, copper_mg_kg,
                manganese_mg_kg, boron_mg_kg,
                npk_trend_direction, organic_carbon_trend,
                n_sufficiency, p_sufficiency, k_sufficiency, oc_sufficiency,
                biological_collapse_risk,
                survey_year_latest, cards_aggregated
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
            ON CONFLICT (hex_id) DO UPDATE SET
                organic_carbon_pct           = EXCLUDED.organic_carbon_pct,
                electrical_conductivity_ds_m = EXCLUDED.electrical_conductivity_ds_m,
                available_n_kg_ha            = EXCLUDED.available_n_kg_ha,
                available_p_kg_ha            = EXCLUDED.available_p_kg_ha,
                available_k_kg_ha            = EXCLUDED.available_k_kg_ha,
                sulphur_mg_kg                = EXCLUDED.sulphur_mg_kg,
                zinc_mg_kg                   = EXCLUDED.zinc_mg_kg,
                iron_mg_kg                   = EXCLUDED.iron_mg_kg,
                copper_mg_kg                 = EXCLUDED.copper_mg_kg,
                manganese_mg_kg              = EXCLUDED.manganese_mg_kg,
                boron_mg_kg                  = EXCLUDED.boron_mg_kg,
                npk_trend_direction          = EXCLUDED.npk_trend_direction,
                organic_carbon_trend         = EXCLUDED.organic_carbon_trend,
                n_sufficiency                = EXCLUDED.n_sufficiency,
                p_sufficiency                = EXCLUDED.p_sufficiency,
                k_sufficiency                = EXCLUDED.k_sufficiency,
                oc_sufficiency               = EXCLUDED.oc_sufficiency,
                biological_collapse_risk     = EXCLUDED.biological_collapse_risk,
                survey_year_latest           = EXCLUDED.survey_year_latest,
                cards_aggregated             = EXCLUDED.cards_aggregated,
                ingested_at                  = now()
            WHERE soil_health_raw.survey_year_latest IS NULL
               OR EXCLUDED.survey_year_latest >= soil_health_raw.survey_year_latest
            """,
            *params,
        )
        if result == "INSERT 0 1":
            inserted += 1
        else:
            updated += 1

    return inserted, updated


async def _main(region_code: str, csv_path: Path, dry_run: bool) -> None:
    """Main ingestion entry point."""
    log = logger.bind(region_code=region_code, file=str(csv_path), dry_run=dry_run)

    if not csv_path.exists():
        log.error("shc.ingest.file_not_found")
        sys.exit(1)

    df = _load_and_aggregate(csv_path, region_code)

    if dry_run:
        log.info("shc.ingest.dry_run", rows=len(df))
        print(df[["hex_id", "organic_carbon", "n_sufficiency", "biological_collapse_risk"]].head(20).to_string())
        return

    dsn = get_postgis_dsn()
    conn = await asyncpg.connect(**dsn)
    try:
        inserted, updated = await _upsert(conn, df, dry_run=False)
        log.info("shc.ingest.upsert_done", inserted=inserted, updated=updated)

        # Refresh materialized view
        await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY soil_health_by_hex;")
        log.info("shc.ingest.view_refreshed")

        # Log to audit table
        await conn.execute(
            """
            INSERT INTO ingest_log (script, region_code, rows_inserted, rows_skipped, status, finished_at)
            VALUES ($1, $2, $3, $4, 'success', now())
            """,
            "ingest_soil_health_cards.py", region_code, inserted, updated,
        )
    finally:
        await conn.close()

    print(f"✓ SHC ingestion complete — {inserted} inserted, {updated} updated, {len(df)} hex cells")
    print(f"  Biological collapse risk flagged: {int(df['biological_collapse_risk'].sum())} hex cells")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest India Soil Health Card data")
    p.add_argument("--region", required=True, help="Region code, e.g. IN")
    p.add_argument("--file", required=True, help="Path to SHC CSV file")
    p.add_argument("--dry-run", action="store_true", help="Parse and aggregate only, no DB writes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(
        region_code=args.region,
        csv_path=Path(args.file),
        dry_run=args.dry_run,
    ))
