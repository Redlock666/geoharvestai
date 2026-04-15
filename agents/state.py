"""
LangGraph State Schema — CropRecommendationState.

Central TypedDict that flows through every node in the agent graph.
No global variables; all state is passed explicitly via this schema.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

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
    yield_estimate_kg_ha: float
    model_used: str            # "lstm" | "sarimax" | "ensemble"


class CropRecommendationState(TypedDict):
    """Full agent state passed between all LangGraph nodes."""
    location: LocationContext
    season: str                          # e.g. "kharif_2026", "rabi_2026"
    gis_features: GISFeatures
    weather_snapshot: WeatherSnapshot
    ml_predictions: list[CropPrediction]
    reasoning: str                       # LLM-generated explanation
    messages: Annotated[list, add_messages]  # LangGraph message history
