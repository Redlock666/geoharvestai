"""
Ecosystem Drift Batch Compute Script.

Queries TimescaleDB + PostGIS for time series data per H3 hex cell, runs the
ecosystem drift analysis pipeline, and stores results in `ecosystem_drift_raw`.

This script is intended to run:
  - Once after initial data ingestion (to seed the drift reports)
  - Weekly via the daily_refresh.py worker (to keep reports current)
  - On demand after new SHC or ERA5 data is ingested

Data sources queried:
  PostGIS   — soil_health_raw     (OC%, EC, NPK over survey years)
  PostGIS   — climate_trend_raw   (rainfall/temp anomaly vs 30yr baseline)
  TimescaleDB — ndvi_obs          (NDVI per hex per time)
  TimescaleDB — crop_yield_obs    (yield per hex per time per crop)

Minimum data requirements per hex:
  - At least 1 indicator with ≥ 2 seasonal observations
  - Hexes with no time-series data are skipped

Usage:
    python scripts/compute_ecosystem_drift.py --region IN
    python scripts/compute_ecosystem_drift.py --region IN --dry-run
    python scripts/compute_ecosystem_drift.py --region IN --hex 8765b4a4fffffff  # single hex
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import numpy as np
import structlog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.settings import get_postgis_dsn, get_timescale_dsn
from ml.pipeline.ecosystem_drift import (
    EcosystemBundle,
    IndicatorTimeSeries,
    analyze_ecosystem_drift,
)

logger = structlog.get_logger(__name__)


# ── Time series loaders ───────────────────────────────────────────────────────

async def _load_shc_timeseries(
    postgis: asyncpg.Connection, region_code: str, hex_id: str | None
) -> dict[str, dict[str, list]]:
    """Fetch OC% and EC multi-year time series from soil_health_raw.

    Logic Flow:
        Groups SHC records by hex_id and survey_year.
        Returns dict: hex_id → {oc: [values...], ec: [values...], years: [years...]}

    Returns:
        Dict keyed by hex_id.
    """
    where = "region_code = $1"
    params: list = [region_code]
    if hex_id:
        where += " AND hex_id = $2"
        params.append(hex_id)

    rows = await postgis.fetch(
        f"""
        SELECT hex_id, survey_year_latest AS year,
               organic_carbon_pct, electrical_conductivity_ds_m,
               available_n_kg_ha, available_p_kg_ha, available_k_kg_ha
        FROM soil_health_raw
        WHERE {where}
        ORDER BY hex_id, survey_year_latest
        """,
        *params,
    )

    result: dict[str, dict[str, list]] = {}
    for row in rows:
        hx = row["hex_id"]
        if hx not in result:
            result[hx] = {"oc": [], "ec": [], "npk": [], "years": []}
        if row["year"]:
            result[hx]["years"].append(row["year"])
        if row["organic_carbon_pct"] is not None:
            result[hx]["oc"].append(float(row["organic_carbon_pct"]))
        if row["electrical_conductivity_ds_m"] is not None:
            result[hx]["ec"].append(float(row["electrical_conductivity_ds_m"]))
        # NPK mean as proxy
        npk_vals = [v for v in [row["available_n_kg_ha"], row["available_p_kg_ha"], row["available_k_kg_ha"]] if v is not None]
        if npk_vals:
            result[hx]["npk"].append(float(np.mean(npk_vals)))

    return result


async def _load_climate_trend(
    postgis: asyncpg.Connection, region_code: str, hex_id: str | None
) -> dict[str, dict]:
    """Fetch ERA5 climate anomaly values from climate_trend_raw."""
    where = "region_code = $1"
    params: list = [region_code]
    if hex_id:
        where += " AND hex_id = $2"
        params.append(hex_id)

    rows = await postgis.fetch(
        f"""
        SELECT hex_id, rainfall_anomaly_mm, temp_anomaly_c
        FROM climate_trend_raw WHERE {where}
        """,
        *params,
    )

    # Climate trend is a single value per hex (not multi-year time series yet)
    # Treat as a single-point series; CUSUM gracefully handles short series
    return {
        row["hex_id"]: {
            "rainfall": [float(row["rainfall_anomaly_mm"])] if row["rainfall_anomaly_mm"] is not None else [],
            "temp": [float(row["temp_anomaly_c"])] if row["temp_anomaly_c"] is not None else [],
        }
        for row in rows
    }


async def _load_ndvi_timeseries(
    timescale: asyncpg.Connection, region_code: str, hex_id: str | None
) -> dict[str, list[float]]:
    """Fetch seasonal mean NDVI per hex from TimescaleDB ndvi_obs."""
    where = "region_code = $1"
    params: list = [region_code]
    if hex_id:
        where += " AND hex_id = $2"
        params.append(hex_id)

    try:
        rows = await timescale.fetch(
            f"""
            SELECT hex_id, AVG(ndvi) AS mean_ndvi
            FROM ndvi_obs
            WHERE {where}
            GROUP BY hex_id, DATE_TRUNC('year', time)
            ORDER BY hex_id, DATE_TRUNC('year', time)
            """,
            *params,
        )
        result: dict[str, list[float]] = {}
        for row in rows:
            hx = row["hex_id"]
            if hx not in result:
                result[hx] = []
            if row["mean_ndvi"] is not None:
                result[hx].append(float(row["mean_ndvi"]))
        return result
    except Exception as e:
        logger.warning("ndvi.load.failed", error=str(e))
        return {}


async def _load_yield_timeseries(
    timescale: asyncpg.Connection, region_code: str, hex_id: str | None
) -> dict[str, list[float]]:
    """Fetch mean annual yield across crops per hex from TimescaleDB."""
    where = "region_code = $1"
    params: list = [region_code]
    if hex_id:
        where += " AND hex_id = $2"
        params.append(hex_id)

    try:
        rows = await timescale.fetch(
            f"""
            SELECT hex_id, AVG(yield_kg_ha) AS mean_yield
            FROM crop_yield_obs
            WHERE {where}
            GROUP BY hex_id, DATE_TRUNC('year', time)
            ORDER BY hex_id, DATE_TRUNC('year', time)
            """,
            *params,
        )
        result: dict[str, list[float]] = {}
        for row in rows:
            hx = row["hex_id"]
            if hx not in result:
                result[hx] = []
            if row["mean_yield"] is not None:
                result[hx].append(float(row["mean_yield"]))
        return result
    except Exception as e:
        logger.warning("yield.load.failed", error=str(e))
        return {}


# ── Bundle assembly ───────────────────────────────────────────────────────────

def _make_series(name: str, values: list[float], baseline_override: float | None = None) -> IndicatorTimeSeries | None:
    """Create an IndicatorTimeSeries, computing baseline stats from the series itself."""
    if not values:
        return None
    arr = np.array(values, dtype=float)
    baseline_mean = baseline_override if baseline_override is not None else float(np.mean(arr))
    baseline_std = max(float(np.std(arr)), 1e-6)
    return IndicatorTimeSeries(
        name=name,
        values=arr,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
    )


def build_bundle(
    hex_id: str,
    region_code: str,
    shc: dict,
    climate: dict,
    ndvi_vals: list[float],
    yield_vals: list[float],
) -> EcosystemBundle:
    """Assemble all available time series into an EcosystemBundle."""
    return EcosystemBundle(
        hex_id=hex_id,
        region_code=region_code,
        oc=_make_series("oc", shc.get("oc", []), baseline_override=0.65),
        ec=_make_series("ec", shc.get("ec", []), baseline_override=0.8),
        npk_mean=_make_series("npk_mean", shc.get("npk", []), baseline_override=250.0),
        rainfall_anomaly=_make_series("rainfall_anomaly", climate.get("rainfall", []), baseline_override=0.0),
        temp_anomaly=_make_series("temp_anomaly", climate.get("temp", []), baseline_override=0.0),
        ndvi=_make_series("ndvi", ndvi_vals, baseline_override=0.45),
        yield_mean=_make_series("yield_mean", yield_vals, baseline_override=2200.0),
    )


# ── DB upsert ─────────────────────────────────────────────────────────────────

async def _upsert_report(
    postgis: asyncpg.Connection,
    region_code: str,
    report,
    dry_run: bool,
) -> None:
    """Upsert an EcosystemDriftReport into ecosystem_drift_raw."""
    from models.ecosystem import EcosystemDriftReport  # local import to avoid circular

    viable_now = [c.crop_name for c in report.crop_viability if c.viable_now]
    viable_proj = [c.crop_name for c in report.crop_viability if c.viable_projected]
    at_risk = [c.crop_name for c in report.crop_viability if c.viable_now and not c.viable_projected]
    phase_in = [c.crop_name for c in report.crop_viability if not c.viable_now and c.viable_projected]

    interventions_json = json.dumps([iv.model_dump() for iv in report.repair_interventions])

    if dry_run:
        logger.info(
            "ecosystem.upsert.dry_run",
            hex_id=report.hex_id,
            health=report.ecosystem_health_score,
            velocity=report.health_velocity,
            stressor=report.primary_stressor,
        )
        return

    await postgis.execute(
        """
        INSERT INTO ecosystem_drift_raw (
            hex_id, region_code,
            ecosystem_health_score, health_velocity,
            cusum_oc_signal, cusum_ec_signal,
            cusum_rainfall_signal, cusum_temp_signal,
            cusum_ndvi_signal, cusum_yield_signal,
            primary_stressor,
            projected_health_score, seasons_to_critical,
            repair_interventions,
            viable_crops_current, viable_crops_projected,
            crops_at_risk, crops_to_phase_in, soil_restorative_crops,
            drift_narrative, repair_summary, projection_narrative,
            data_quality, indicators_with_data, seasons_of_data
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14::jsonb, $15::jsonb, $16::jsonb,
            $17::jsonb, $18::jsonb, $19::jsonb, $20, $21, $22, $23, $24, $25
        )
        ON CONFLICT (hex_id) DO UPDATE SET
            ecosystem_health_score  = EXCLUDED.ecosystem_health_score,
            health_velocity         = EXCLUDED.health_velocity,
            cusum_oc_signal         = EXCLUDED.cusum_oc_signal,
            cusum_ec_signal         = EXCLUDED.cusum_ec_signal,
            cusum_rainfall_signal   = EXCLUDED.cusum_rainfall_signal,
            cusum_temp_signal       = EXCLUDED.cusum_temp_signal,
            cusum_ndvi_signal       = EXCLUDED.cusum_ndvi_signal,
            cusum_yield_signal      = EXCLUDED.cusum_yield_signal,
            primary_stressor        = EXCLUDED.primary_stressor,
            projected_health_score  = EXCLUDED.projected_health_score,
            seasons_to_critical     = EXCLUDED.seasons_to_critical,
            repair_interventions    = EXCLUDED.repair_interventions,
            viable_crops_current    = EXCLUDED.viable_crops_current,
            viable_crops_projected  = EXCLUDED.viable_crops_projected,
            crops_at_risk           = EXCLUDED.crops_at_risk,
            crops_to_phase_in       = EXCLUDED.crops_to_phase_in,
            soil_restorative_crops  = EXCLUDED.soil_restorative_crops,
            drift_narrative         = EXCLUDED.drift_narrative,
            repair_summary          = EXCLUDED.repair_summary,
            projection_narrative    = EXCLUDED.projection_narrative,
            data_quality            = EXCLUDED.data_quality,
            indicators_with_data    = EXCLUDED.indicators_with_data,
            seasons_of_data         = EXCLUDED.seasons_of_data,
            computed_at             = now()
        """,
        report.hex_id, region_code,
        report.ecosystem_health_score,
        report.health_velocity,
        # CUSUM signals
        next((r.signal for r in report.cusum_results if r.indicator == "oc"), "insufficient_data"),
        next((r.signal for r in report.cusum_results if r.indicator == "ec"), "insufficient_data"),
        next((r.signal for r in report.cusum_results if r.indicator == "rainfall_anomaly"), "insufficient_data"),
        next((r.signal for r in report.cusum_results if r.indicator == "temp_anomaly"), "insufficient_data"),
        next((r.signal for r in report.cusum_results if r.indicator == "ndvi"), "insufficient_data"),
        next((r.signal for r in report.cusum_results if r.indicator == "yield_mean"), "insufficient_data"),
        report.primary_stressor,
        report.projected_health_score,
        report.seasons_to_critical,
        interventions_json,
        json.dumps(viable_now), json.dumps(viable_proj),
        json.dumps(at_risk), json.dumps(phase_in),
        json.dumps(report.soil_restorative_crops),
        report.drift_narrative, report.repair_summary, report.projection_narrative,
        report.data_quality, report.indicators_with_data, report.seasons_of_data,
    )


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def _main(region_code: str, hex_id: str | None, dry_run: bool) -> None:
    """Fetch time series, run analysis for all hexes, upsert results."""
    log = logger.bind(region_code=region_code, hex_id=hex_id or "all", dry_run=dry_run)
    log.info("ecosystem_drift.compute.start")

    postgis_conn = await asyncpg.connect(**get_postgis_dsn())
    timescale_conn = await asyncpg.connect(**get_timescale_dsn())

    try:
        # Load all time series
        shc_data = await _load_shc_timeseries(postgis_conn, region_code, hex_id)
        climate_data = await _load_climate_trend(postgis_conn, region_code, hex_id)
        ndvi_data = await _load_ndvi_timeseries(timescale_conn, region_code, hex_id)
        yield_data = await _load_yield_timeseries(timescale_conn, region_code, hex_id)

        # Collect all hex IDs across all sources
        all_hexes = (
            set(shc_data.keys())
            | set(climate_data.keys())
            | set(ndvi_data.keys())
            | set(yield_data.keys())
        )
        if hex_id:
            all_hexes = {hex_id} & all_hexes

        log.info("ecosystem_drift.hexes_found", count=len(all_hexes))

        processed = skipped = 0
        for hx in sorted(all_hexes):
            bundle = build_bundle(
                hex_id=hx,
                region_code=region_code,
                shc=shc_data.get(hx, {}),
                climate=climate_data.get(hx, {}),
                ndvi_vals=ndvi_data.get(hx, []),
                yield_vals=yield_data.get(hx, []),
            )

            # Skip hexes with no usable time series
            has_data = any([
                bundle.oc, bundle.ec, bundle.rainfall_anomaly,
                bundle.temp_anomaly, bundle.ndvi, bundle.yield_mean,
            ])
            if not has_data:
                skipped += 1
                continue

            report = analyze_ecosystem_drift(bundle)
            await _upsert_report(postgis_conn, region_code, report, dry_run)
            processed += 1

        if not dry_run:
            await postgis_conn.execute(
                "REFRESH MATERIALIZED VIEW CONCURRENTLY ecosystem_drift_by_hex;"
            )
            log.info("ecosystem_drift.view_refreshed")

        log.info(
            "ecosystem_drift.compute.complete",
            processed=processed, skipped=skipped,
        )
        print(f"✓ Ecosystem drift computed — {processed} hex cells processed, {skipped} skipped (no data)")

    finally:
        await postgis_conn.close()
        await timescale_conn.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ecosystem drift analysis per H3 hex cell")
    p.add_argument("--region", required=True, help="Region code, e.g. IN")
    p.add_argument("--hex", default=None, help="Single hex_id to compute (default: all hexes)")
    p.add_argument("--dry-run", action="store_true", help="Run analysis without writing to DB")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(
        region_code=args.region,
        hex_id=args.hex,
        dry_run=args.dry_run,
    ))
