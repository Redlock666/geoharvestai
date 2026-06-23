"""Canonical feature schema for training and inference alignment."""

from __future__ import annotations

# Core features — always available from SoilGrids + ERA5 + Sentinel-2
FEATURE_COLUMNS: list[str] = [
    "soil_nitrogen",
    "soil_phosphorus",
    "soil_potassium",
    "soil_ph",
    "elevation_m",
    "slope_deg",
    "rainfall_7d_mm",
    "temp_avg_c",
    "temp_min_c",
    "temp_max_c",
    "ndvi",
]

# Extended features — available when Soil Health Card data is ingested.
# These are appended to FEATURE_COLUMNS when present; absent features are
# filled with SHC_FEATURE_DEFAULTS to keep vector shape stable.
SHC_FEATURE_COLUMNS: list[str] = [
    "soil_organic_carbon_pct",      # Walkley-Black OC % — biological health proxy
    "soil_ec_ds_m",                 # Electrical conductivity (dS/m) — salinity
    "climate_anomaly_trend_mm",     # 5yr vs 30yr rainfall anomaly — climate drift
    "climate_temp_anomaly_c",       # 5yr vs 30yr temp anomaly — warming signal
]

# All features used when SHC data is available
EXTENDED_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS + SHC_FEATURE_COLUMNS

# Defaults used when region-level sources are not yet fully ingested.
# Keeps training and inference vector shapes stable for demo prototypes.
DEFAULT_STATIC_FEATURES: dict[str, float] = {
    "soil_nitrogen": 1.2,
    "soil_phosphorus": 18.0,
    "soil_potassium": 180.0,
    "soil_ph": 6.8,
    "elevation_m": 250.0,
    "slope_deg": 2.5,
}

# Defaults for SHC features when SHC data has not been ingested.
# 0.5% OC is the ICAR "low" boundary — conservative neutral starting point.
# 0.0 anomaly = no detectable climate drift from baseline.
SHC_FEATURE_DEFAULTS: dict[str, float] = {
    "soil_organic_carbon_pct": 0.5,
    "soil_ec_ds_m": 0.3,
    "climate_anomaly_trend_mm": 0.0,
    "climate_temp_anomaly_c": 0.0,
}

DRIFT_FEATURE_COLUMNS: list[str] = EXTENDED_FEATURE_COLUMNS.copy()
