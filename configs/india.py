"""
India-specific constants for data ingestion and spatial configuration.

These are static geographic and agronomic facts about India.
They are NEVER used as the runtime region_code — that is always
passed by the caller at runtime (e.g. --region IN).
"""

from __future__ import annotations

# ISO 3166-1 alpha-2 identifier used as region_code for India
REGION_CODE = "IN"

# WGS84 bounding box (0.5° buffer ensures full country coverage incl. islands)
BBOX: dict[str, float] = {
    "min_lon": 68.0,
    "max_lon": 97.5,
    "min_lat":  6.5,
    "max_lat": 37.6,
}

# H3 spatial resolution for all PostGIS lookups.
# Resolution 7 ≈ 5.16 km² avg hex area — matches SoilGrids 250 m tile resolution.
H3_RESOLUTION: int = 7

# Coarser resolution used for SoilGrids API batching (~1 800 hexes for India).
# Values are propagated to all resolution-7 children via h3.h3_to_children().
H3_SOILGRIDS_BATCH_RESOLUTION: int = 4

# APY season → approximate harvest month mapping.
# Used to convert (financial year string, season) → TIMESTAMPTZ harvest date.
#   kharif: sown Jun-Jul, harvested Oct-Nov  → year[0] Oct
#   rabi:   sown Oct-Nov, harvested Mar-Apr  → year[1] Apr
#   zaid:   sown Mar,     harvested Jul-Aug  → year[1] Jul
SEASON_HARVEST_MONTH: dict[str, int] = {
    "kharif":     10,
    "rabi":        4,
    "zaid":        7,
    "whole_year":  3,   # Perennials like sugarcane; use end-of-year harvest
}

# Whether the harvest falls in the second calendar year of the financial year.
# e.g. rabi "2021-22" → harvested April 2022 (year_offset=1)
SEASON_YEAR_OFFSET: dict[str, int] = {
    "kharif":     0,
    "rabi":        1,
    "zaid":        1,
    "whole_year":  1,
}

# ICAR 15 agroclimatic zones (pixel value → zone name).
# Source: ICAR (icar.gov.in) — used by ingest_climate_zones.py
ICAR_ZONES: dict[int, str] = {
    1:  "Western Himalayan Region",
    2:  "Eastern Himalayan Region",
    3:  "Lower Gangetic Plains Region",
    4:  "Middle Gangetic Plains Region",
    5:  "Upper Gangetic Plains Region",
    6:  "Trans-Gangetic Plains Region",
    7:  "Eastern Plateau and Hills Region",
    8:  "Central Plateau and Hills Region",
    9:  "Western Plateau and Hills Region",
    10: "Southern Plateau and Hills Region",
    11: "East Coast Plains and Hills Region",
    12: "West Coast Plains and Ghat Region",
    13: "Gujarat Plains and Hills Region",
    14: "Western Dry Region",
    15: "The Islands Region",
}

# Beck et al. 2018 Köppen-Geiger raster: pixel value → code string.
# Source: https://figshare.com/articles/dataset/Present_and_future_Koppen-Geiger.../6396959
KG_VALUE_TO_CODE: dict[int, str] = {
    1: "Af",  2: "Am",  3: "As",  4: "Aw",
    5: "BSh", 6: "BSk", 7: "BWh", 8: "BWk",
    9: "Cfa", 10: "Cfb", 11: "Cfc",
    12: "Csa", 13: "Csb", 14: "Csc",
    15: "Cwa", 16: "Cwb", 17: "Cwc",
    18: "Dfa", 19: "Dfb", 20: "Dfc", 21: "Dfd",
    22: "Dsa", 23: "Dsb", 24: "Dsc", 25: "Dsd",
    26: "Dwa", 27: "Dwb", 28: "Dwc", 29: "Dwd",
    30: "EF",  31: "ET",
}

# Dominant KG zones in India (for reference / validation)
# Aw (tropical savanna), BSh (hot semi-arid), BWh (hot desert),
# Cwa (humid subtropical), Cwb (oceanic highland), Am (tropical monsoon)
INDIA_KG_ZONES = {"Aw", "BSh", "BSk", "BWh", "BWk", "Cwa", "Cwb", "Am", "As"}

# ── External data source URLs ──────────────────────────────────────────────

# Beck et al. 2018 Köppen-Geiger 1 km GeoTIFF (global, ~70 MB)
KOPPEN_GEIGER_URL = "https://figshare.com/ndownloader/files/12407516"

# SoilGrids v2 REST API — single-point query
SOILGRIDS_REST_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids v2 WCS — India-clipped GeoTIFF (preferred for bulk ingest)
# Substitute {prop} with: nitrogen | phh2o | clay | sand | silt
SOILGRIDS_WCS_TEMPLATE = (
    "https://maps.isric.org/mapserv?map=/map/{prop}.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID={prop}_0-5cm_mean"
    "&FORMAT=image/tiff"
    "&SUBSET=X(68.0,97.5)&SUBSET=Y(6.5,37.6)"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    "&GEOTIFF:COMPRESSION=DEFLATE"
)

# OpenTopography SRTM GL1 30 m API — single call for full India bbox
OPENTOPOGRAPHY_API_URL = (
    "https://portal.opentopography.org/API/globaldem"
    "?demtype=SRTMGL1"
    f"&south={BBOX['min_lat']}&north={BBOX['max_lat']}"
    f"&west={BBOX['min_lon']}&east={BBOX['max_lon']}"
    "&outputFormat=GTiff"
)

# ERA5-Land via CDS API (requires CDSAPI_KEY in .env)
# Variables used: 2m_temperature, total_precipitation, 2m_dewpoint_temperature
ERA5_DATASET = "reanalysis-era5-land"
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# MODIS MOD13A2 — 16-day NDVI composites, 1 km, 2000–present
# Accessed via NASA earthaccess (requires EARTHDATA credentials in .env)
MODIS_NDVI_SHORT_NAME = "MOD13A2"
MODIS_NDVI_VERSION = "061"

# APY Portal — manual download URL (browser interaction required)
APY_PORTAL_URL = "https://aps.dac.gov.in/APY/Public_Report1.aspx"
