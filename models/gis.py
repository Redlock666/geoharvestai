"""Pydantic V2 schema for the GIS feature vector output from GISResolverService."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GISFeatureVector(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    h3_hex: str
    lat: float
    lon: float
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    soil_ph: float
    soil_texture: str
    elevation_m: float
    slope_deg: float
    climate_zone: str
