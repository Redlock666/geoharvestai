"""
LangGraph State Schema — CropRecommendationState.

Central TypedDict that flows through every node in the agent graph.
No global variables; all state is passed explicitly via this schema.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class LocationContext(TypedDict):
    lat: float
    lon: float
    h3_hex: str
    region_code: str   # user-supplied at runtime — never hardcoded


class GISFeatures(TypedDict):
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    soil_ph: float
    soil_texture: str
    elevation_m: float
    slope_deg: float
    climate_zone: str
    # Soil Health Card biological layer (None when SHC not yet ingested)
    soil_organic_carbon_pct: float         # % OC — microbial biomass proxy
    soil_ec_ds_m: float                    # electrical conductivity (dS/m) — salinity
    npk_trend_direction: str               # 'improving' | 'stable' | 'declining' | 'unknown'
    organic_carbon_trend: str              # 'improving' | 'stable' | 'declining' | 'unknown'
    n_sufficiency: str                     # 'deficient' | 'sufficient' | 'excessive' | 'unknown'
    p_sufficiency: str
    k_sufficiency: str
    oc_sufficiency: str                    # 'low' | 'medium' | 'high' | 'unknown'
    biological_collapse_risk: bool         # True = high NPK + declining OC (over-fertilized)
    # ERA5 climate trend layer (None when trend not yet computed)
    climate_anomaly_trend_mm: float        # 5yr rolling vs 30yr baseline rainfall (mm)
    climate_temp_anomaly_c: float          # 5yr rolling vs 30yr baseline temperature (°C)
    climate_regime_shift: str             # 'stable' | 'drying' | 'wetting' | 'warming' | 'extreme_shift'


class WeatherSnapshot(TypedDict):
    rainfall_7d_mm: float
    temp_avg_c: float
    temp_min_c: float
    temp_max_c: float
    ndvi: float                # Sentinel-2 NDVI index (-1 to 1)
    ndvi_freshness_days: int   # Age of NDVI data in days (max 5)


class CropPrediction(TypedDict):
    crop_name: str
    confidence: float          # 0.0 – 1.0
    probability: float         # calibrated confidence after drift penalty
    yield_estimate_kg_ha: float
    yield_min_kg_ha: float
    yield_median_kg_ha: float
    yield_max_kg_ha: float
    uncertainty_band_pct: float
    anomaly_flag: bool
    anomaly_reason: str
    model_used: str            # "lstm" | "sarimax" | "ensemble"
    fertilizer_sufficiency: str  # 'deficient' | 'sufficient' | 'excessive' | 'unknown'
    soil_health_index: float     # 0.0–1.0 composite from OC%, EC, trend direction
    biological_collapse_risk: bool  # True when NPK adequate but OC declining


class CropRecommendationState(TypedDict):
    """Full agent state passed between all LangGraph nodes."""
    location: LocationContext
    season: str                                    # e.g. "kharif_2026", "rabi_2026"
    gis_features: GISFeatures
    weather_snapshot: WeatherSnapshot
    ml_predictions: list[CropPrediction]
    reasoning: str                                 # LLM-generated explanation
    ecosystem_health_score: Optional[float]        # 0-1 composite; None if not computed
    ecosystem_health_velocity: str                 # 'fast_decline' | 'stable' | 'recovering'
    ecosystem_primary_stressor: Optional[str]      # leading degradation driver
    ecosystem_seasons_to_critical: Optional[int]   # seasons to primary threshold breach
    ecosystem_drift_narrative: str                 # 2-3 sentence summary for LLM context
    ecosystem_repair_summary: str                  # priority interventions for LLM context
    ecosystem_projection_narrative: str            # trajectory summary for LLM context
    soil_restorative_crops: list[str]              # crops that actively reverse drift
    messages: Annotated[list, add_messages]  # LangGraph message history
