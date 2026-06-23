"""Pydantic V2 schema for the GIS feature vector output from GISResolverService."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from models.ecosystem import EcosystemDriftReport


class SoilHealthProfile(BaseModel):
    """Biological and micronutrient indicators from India's Soil Health Card scheme."""

    model_config = ConfigDict(populate_by_name=True)

    organic_carbon_pct: Optional[float] = None          # % OC — microbial biomass proxy
    electrical_conductivity_ds_m: Optional[float] = None  # dS/m — salinity indicator
    available_n_kg_ha: Optional[float] = None
    available_p_kg_ha: Optional[float] = None
    available_k_kg_ha: Optional[float] = None
    sulphur_mg_kg: Optional[float] = None
    zinc_mg_kg: Optional[float] = None
    iron_mg_kg: Optional[float] = None
    npk_trend_direction: str = "unknown"                # 'improving'|'stable'|'declining'
    organic_carbon_trend: str = "unknown"
    n_sufficiency: str = "unknown"                      # 'deficient'|'sufficient'|'excessive'
    p_sufficiency: str = "unknown"
    k_sufficiency: str = "unknown"
    oc_sufficiency: str = "unknown"                     # 'low'|'medium'|'high'
    biological_collapse_risk: bool = False


class ClimateTrendProfile(BaseModel):
    """5-year rolling climate anomaly vs 30-year ERA5 baseline per hex cell."""

    model_config = ConfigDict(populate_by_name=True)

    rainfall_anomaly_mm: Optional[float] = None         # negative = drying trend
    rainfall_anomaly_pct: Optional[float] = None
    temp_anomaly_c: Optional[float] = None              # positive = warming trend
    climate_regime_shift: str = "stable"                # 'stable'|'drying'|'wetting'|'warming'|'extreme_shift'


class GISFeatureVector(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    h3_hex: str
    lat: float
    lon: float

    # SoilGrids v2 — chemical composition
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    soil_ph: float
    soil_texture: str

    # SRTM terrain
    elevation_m: float
    slope_deg: float

    # Köppen-Geiger climate zone
    climate_zone: str

    # Soil Health Card — biological health (None when SHC data not yet ingested)
    soil_health: Optional[SoilHealthProfile] = None

    # ERA5 climate trend (None when trend computation not yet run)
    climate_trend: Optional[ClimateTrendProfile] = None

    # Ecosystem drift report (None when batch compute not yet run)
    # Loaded separately to avoid circular imports; type annotation is forward ref
    ecosystem_drift: Optional["EcosystemDriftReport"] = None  # type: ignore[name-defined]
