"""Pydantic V2 schemas for the crop recommendation API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RecommendRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(..., ge=-90, le=90, description="Latitude (WGS84)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (WGS84)")
    season: str = Field(..., description="Crop season, e.g. 'kharif_2026'")
    region_code: str = Field(..., description="User-supplied region identifier")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of crops to return")


class CropResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    crop_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    yield_estimate_kg_ha: float
    model_used: str


class RecommendResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    region_code: str
    season: str
    h3_hex: str
    recommendations: list[CropResult]
    reasoning: str
    ndvi_freshness_days: int
