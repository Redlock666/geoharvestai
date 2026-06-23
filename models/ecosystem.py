"""Pydantic V2 schemas for ecosystem drift analysis output."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CUSUMResult(BaseModel):
    """CUSUM control chart result for a single ecosystem indicator."""

    model_config = ConfigDict(populate_by_name=True)

    indicator: str
    signal: str = "stable"            # 'degrading' | 'stable' | 'improving' | 'insufficient_data'
    cusum_pos: float = 0.0            # upper CUSUM statistic (detects upward shift)
    cusum_neg: float = 0.0            # lower CUSUM statistic (detects downward shift)
    trend_slope: float = 0.0          # linear regression slope (units per season)
    n_points: int = 0                 # number of observations used
    baseline_mean: float = 0.0
    baseline_std: float = 0.0


class IndicatorProjection(BaseModel):
    """Linear extrapolation of one ecosystem indicator N seasons forward."""

    model_config = ConfigDict(populate_by_name=True)

    indicator: str
    current_value: Optional[float] = None
    projected_values: List[float] = Field(default_factory=list)   # 6 seasons
    seasons_to_critical: Optional[int] = None  # None = no breach within horizon
    critical_threshold: Optional[float] = None
    direction: str = "stable"          # 'improving' | 'stable' | 'degrading'


class RepairIntervention(BaseModel):
    """A single agronomic intervention to slow or reverse ecosystem drift."""

    model_config = ConfigDict(populate_by_name=True)

    priority: int = Field(..., ge=1, le=4)
    # 1=critical (act this season) | 2=high | 3=medium | 4=advisory

    category: str
    # 'soil_biology' | 'water_management' | 'salinity' | 'climate_adaptation' | 'nutrient'

    intervention: str                  # What to do
    mechanism: str                     # Why it works (one sentence)
    expected_seasons_to_effect: int    # Seasons until measurable response
    evidence_basis: str                # e.g. 'ICAR Handbook 2016', 'ICRISAT Trial 2019'


class CropViabilityAssessment(BaseModel):
    """Viability of a specific crop given current and projected ecosystem state."""

    model_config = ConfigDict(populate_by_name=True)

    crop_name: str
    viable_now: bool
    viable_projected: bool      # at 6-season projection
    confidence: str = "medium"  # 'high' | 'medium' | 'low'
    risk_factors: List[str] = Field(default_factory=list)
    # e.g. ['rainfall_deficit', 'rising_salinity', 'declining_oc']
    soil_restorative: bool = False
    # True = growing this crop actively reverses drift (legumes, cover crops)
    transition_priority: str = "neutral"
    # 'recommended' | 'neutral' | 'phase_out'


class EcosystemDriftReport(BaseModel):
    """
    Full ecosystem drift analysis for one H3 hex cell.

    Captures the slow, cumulative transformation of the agricultural
    ecosystem — individual micro-changes undetectable in isolation but
    collectively shifting the viable crop space and yield ceiling over
    seasons and years.
    """

    model_config = ConfigDict(populate_by_name=True)

    hex_id: str
    region_code: str

    # ── Overall health ────────────────────────────────────────────────────
    ecosystem_health_score: float = Field(0.5, ge=0.0, le=1.0)
    # 0.0 = severely degraded, 1.0 = excellent biological health
    health_velocity: str = "stable"
    # 'fast_decline' | 'moderate_decline' | 'stable' | 'slow_recovery' | 'recovering'
    primary_stressor: Optional[str] = None
    # Leading driver: 'organic_carbon_decline' | 'drying_trend' | 'salinity_rise'
    # | 'temperature_warming' | 'vegetation_degradation' | 'yield_collapse'

    # ── Per-indicator CUSUM results ───────────────────────────────────────
    cusum_results: List[CUSUMResult] = Field(default_factory=list)

    # ── 6-season forward projections ─────────────────────────────────────
    projections: List[IndicatorProjection] = Field(default_factory=list)
    projected_health_score: Optional[float] = None     # composite at season +6
    seasons_to_critical: Optional[int] = None          # seasons to primary threshold breach

    # ── Repair interventions (ordered by priority) ────────────────────────
    repair_interventions: List[RepairIntervention] = Field(default_factory=list)

    # ── Crop guidance ─────────────────────────────────────────────────────
    crop_viability: List[CropViabilityAssessment] = Field(default_factory=list)
    soil_restorative_crops: List[str] = Field(default_factory=list)
    # Crops to grow NOW that actively help reverse the drift

    # ── Narrative summaries (for LLM reasoning context) ───────────────────
    drift_narrative: str = ""       # what is happening and why (2-3 sentences)
    repair_summary: str = ""        # priority actions (2-3 sentences)
    projection_narrative: str = ""  # where ecosystem is heading (2-3 sentences)

    # ── Data quality ──────────────────────────────────────────────────────
    data_quality: str = "insufficient"
    # 'high' | 'medium' | 'low' | 'insufficient'
    indicators_with_data: int = 0
    seasons_of_data: Optional[int] = None
