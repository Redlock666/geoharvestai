"""
Ecosystem Analyzer Service.

Loads pre-computed ecosystem drift reports from the PostGIS
`ecosystem_drift_by_hex` materialized view. Reports are computed offline
by `scripts/compute_ecosystem_drift.py` and refreshed weekly by the
daily refresh worker.

This service is called by GISResolverService during Stage 1 of the
LangGraph pipeline so the LLM reasoning layer in Stage 4 receives full
ecosystem context alongside the per-field recommendation.
"""

from __future__ import annotations

import json
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from sqlalchemy.ext.asyncio import AsyncSession

from models.ecosystem import (
    CropViabilityAssessment,
    EcosystemDriftReport,
    RepairIntervention,
)

logger = structlog.get_logger(__name__)


class EcosystemAnalyzerService:
    """Loads pre-computed ecosystem drift reports from PostGIS."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def fetch_report(self, hex_id: str) -> EcosystemDriftReport | None:
        """Fetch the pre-computed ecosystem drift report for an H3 hex cell.

        Logic Flow:
            Queries the ecosystem_drift_by_hex materialized view.
            Deserialises JSONB fields (repair_interventions, crop viability arrays).
            Returns None gracefully when no report exists yet — the recommendation
            pipeline continues with standard GIS features only, and the LLM
            reasoning layer notes the absence.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            EcosystemDriftReport or None if not yet computed for this hex.

        Expected Exceptions:
            sqlalchemy.exc.OperationalError: PostGIS connection failure.
        """
        log = logger.bind(hex_id=hex_id)

        result = await self._db.execute(
            "SELECT ecosystem_health_score, health_velocity, primary_stressor, "
            "       projected_health_score, seasons_to_critical, "
            "       repair_interventions, viable_crops_current, viable_crops_projected, "
            "       crops_at_risk, crops_to_phase_in, soil_restorative_crops, "
            "       drift_narrative, repair_summary, projection_narrative, data_quality "
            "FROM ecosystem_drift_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()

        if row is None:
            log.info(
                "ecosystem.report.missing",
                note="No drift report computed yet for this hex — run compute_ecosystem_drift.py",
            )
            return None

        # Deserialise JSONB fields
        def _parse_json(field: object) -> list:
            if field is None:
                return []
            if isinstance(field, str):
                return json.loads(field)
            return field if isinstance(field, list) else []

        repair_raw = _parse_json(row.get("repair_interventions"))
        repair_interventions = [RepairIntervention(**r) for r in repair_raw if isinstance(r, dict)]

        viable_current = _parse_json(row.get("viable_crops_current"))
        viable_projected = _parse_json(row.get("viable_crops_projected"))
        at_risk = _parse_json(row.get("crops_at_risk"))
        phase_in = _parse_json(row.get("crops_to_phase_in"))
        restorative = _parse_json(row.get("soil_restorative_crops"))

        # Reconstruct minimal crop viability list from stored arrays
        crop_viability: list[CropViabilityAssessment] = []
        all_mentioned = set(viable_current + viable_projected + at_risk + phase_in)
        for crop_name in all_mentioned:
            viable_now = crop_name in viable_current
            viable_proj = crop_name in viable_projected
            if viable_now and not viable_proj:
                transition = "phase_out"
            elif not viable_now and viable_proj:
                transition = "phase_in_after_intervention"
            else:
                transition = "recommended" if crop_name in restorative else "neutral"
            crop_viability.append(CropViabilityAssessment(
                crop_name=crop_name,
                viable_now=viable_now,
                viable_projected=viable_proj,
                soil_restorative=crop_name in restorative,
                transition_priority=transition,
            ))

        report = EcosystemDriftReport(
            hex_id=hex_id,
            region_code="",  # not stored in view — populated by caller if needed
            ecosystem_health_score=float(row.get("ecosystem_health_score") or 0.5),
            health_velocity=str(row.get("health_velocity") or "stable"),
            primary_stressor=row.get("primary_stressor"),
            projected_health_score=row.get("projected_health_score"),
            seasons_to_critical=row.get("seasons_to_critical"),
            repair_interventions=repair_interventions,
            crop_viability=crop_viability,
            soil_restorative_crops=restorative,
            drift_narrative=str(row.get("drift_narrative") or ""),
            repair_summary=str(row.get("repair_summary") or ""),
            projection_narrative=str(row.get("projection_narrative") or ""),
            data_quality=str(row.get("data_quality") or "insufficient"),
        )

        log.info(
            "ecosystem.report.loaded",
            health_score=report.ecosystem_health_score,
            velocity=report.health_velocity,
            primary_stressor=report.primary_stressor,
            data_quality=report.data_quality,
        )
        return report
