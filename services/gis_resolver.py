"""
GIS Spatial Resolver Service.

Resolves a (lat, lon) coordinate into a structured feature vector
by performing H3-indexed lookups against PostGIS soil and terrain layers.
"""

from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

import h3
from sqlalchemy.ext.asyncio import AsyncSession

from models.gis import GISFeatureVector, SoilHealthProfile, ClimateTrendProfile
from services.ecosystem_analyzer import EcosystemAnalyzerService

logger = structlog.get_logger(__name__)

_H3_RESOLUTION = 7  # ~5km hex cells — matches SoilGrids tile resolution


class GISResolverService:
    """Resolves geographic coordinates into ML-ready feature vectors via PostGIS."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def resolve(self, lat: float, lon: float) -> GISFeatureVector:
        """Resolve a coordinate pair into a GIS feature vector.

        Logic Flow:
            1. Compute H3 hex cell for fast spatial index lookup.
            2. Query PostGIS for soil composition (NPK, pH, texture) using hex_id.
            3. Query PostGIS for terrain features (elevation, slope) using hex_id.
            4. Query PostGIS for climate zone classification.
            5. Assemble and return a typed GISFeatureVector.

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).

        Returns:
            GISFeatureVector populated with soil, terrain, and climate attributes.

        Expected Exceptions:
            FeatureNotFoundError: No GIS data ingested for this hex cell yet.
            sqlalchemy.exc.OperationalError: PostGIS connection failure.
        """
        hex_id = h3.geo_to_h3(lat, lon, _H3_RESOLUTION)
        log = logger.bind(lat=lat, lon=lon, hex_id=hex_id)
        log.info("gis.resolve.start")

        soil = await self._fetch_soil(hex_id)
        terrain = await self._fetch_terrain(hex_id)
        climate_zone = await self._fetch_climate_zone(hex_id)
        soil_health = await self._fetch_soil_health(hex_id)
        climate_trend = await self._fetch_climate_trend(hex_id)
        ecosystem_drift = await EcosystemAnalyzerService(self._db).fetch_report(hex_id)

        log.info("gis.resolve.complete", climate_zone=climate_zone,
                 shc_available=soil_health is not None,
                 climate_trend_available=climate_trend is not None,
                 ecosystem_drift_available=ecosystem_drift is not None)
        return GISFeatureVector(
            h3_hex=hex_id,
            lat=lat,
            lon=lon,
            soil_nitrogen=soil["nitrogen"],
            soil_phosphorus=soil["phosphorus"],
            soil_potassium=soil["potassium"],
            soil_ph=soil["ph"],
            soil_texture=soil["texture"],
            elevation_m=terrain["elevation_m"],
            slope_deg=terrain["slope_deg"],
            climate_zone=climate_zone,
            soil_health=soil_health,
            climate_trend=climate_trend,
            ecosystem_drift=ecosystem_drift,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _fetch_soil(self, hex_id: str) -> dict:
        """Fetch soil composition for an H3 hex cell from PostGIS.

        Logic Flow:
            Executes a parameterized SELECT against the `soil_by_hex` materialized view.
            Retries up to 3 times on transient DB errors.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            Dict with keys: nitrogen, phosphorus, potassium, ph, texture.

        Expected Exceptions:
            FeatureNotFoundError: hex_id has no corresponding soil record.
        """
        result = await self._db.execute(
            "SELECT nitrogen, phosphorus, potassium, ph, texture "
            "FROM soil_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            raise FeatureNotFoundError(f"No soil data for hex_id={hex_id}. Run ingest first.")
        return dict(row)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _fetch_terrain(self, hex_id: str) -> dict:
        """Fetch terrain features (elevation, slope) for an H3 hex cell.

        Logic Flow:
            Queries the `terrain_by_hex` materialized view pre-computed from SRTM DEM.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            Dict with keys: elevation_m, slope_deg.

        Expected Exceptions:
            FeatureNotFoundError: hex_id has no terrain record.
        """
        result = await self._db.execute(
            "SELECT elevation_m, slope_deg FROM terrain_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            raise FeatureNotFoundError(f"No terrain data for hex_id={hex_id}.")
        return dict(row)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _fetch_climate_zone(self, hex_id: str) -> str:
        """Fetch the Köppen-Geiger climate zone for an H3 hex cell.

        Logic Flow:
            Queries `climate_zones_by_hex` materialized view.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            Köppen-Geiger zone code string (e.g. 'Aw', 'BSh', 'Cfa').

        Expected Exceptions:
            FeatureNotFoundError: hex_id has no climate classification.
        """
        result = await self._db.execute(
            "SELECT zone_code FROM climate_zones_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            raise FeatureNotFoundError(f"No climate zone for hex_id={hex_id}.")
        return str(row["zone_code"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _fetch_soil_health(self, hex_id: str) -> SoilHealthProfile | None:
        """Fetch Soil Health Card indicators for an H3 hex cell.

        Logic Flow:
            Queries soil_health_by_hex materialized view populated from India's
            Soil Health Card scheme (220M+ cards, soilhealth.dac.gov.in).
            Returns None gracefully when SHC data has not yet been ingested for
            this hex cell — the recommendation pipeline continues with SoilGrids
            chemical data only and notes the absence in the reasoning layer.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            SoilHealthProfile populated with biological and micronutrient fields,
            or None if no SHC record exists for this hex cell.

        Expected Exceptions:
            sqlalchemy.exc.OperationalError: PostGIS connection failure.
        """
        result = await self._db.execute(
            "SELECT organic_carbon_pct, electrical_conductivity_ds_m, "
            "       available_n_kg_ha, available_p_kg_ha, available_k_kg_ha, "
            "       sulphur_mg_kg, zinc_mg_kg, iron_mg_kg, "
            "       npk_trend_direction, organic_carbon_trend, "
            "       n_sufficiency, p_sufficiency, k_sufficiency, oc_sufficiency, "
            "       biological_collapse_risk "
            "FROM soil_health_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            logger.bind(hex_id=hex_id).info(
                "gis.resolve.shc_missing",
                note="SHC not yet ingested for this hex — continuing without biological health layer",
            )
            return None
        return SoilHealthProfile(**dict(row))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _fetch_climate_trend(self, hex_id: str) -> ClimateTrendProfile | None:
        """Fetch the ERA5-derived 5-year climate anomaly trend for an H3 hex cell.

        Logic Flow:
            Queries climate_trend_by_hex view, which stores 5-year rolling rainfall
            and temperature deviation from the 30-year ERA5 baseline (1991-2020).
            A negative rainfall_anomaly_mm indicates a drying trend consistent with
            climate change impact; a positive temp_anomaly_c indicates warming.
            Returns None when trend data has not yet been computed for this hex.

        Args:
            hex_id: H3 hex cell identifier at resolution 7.

        Returns:
            ClimateTrendProfile with anomaly values and regime classification,
            or None if trend data is unavailable.

        Expected Exceptions:
            sqlalchemy.exc.OperationalError: PostGIS connection failure.
        """
        result = await self._db.execute(
            "SELECT rainfall_anomaly_mm, rainfall_anomaly_pct, "
            "       temp_anomaly_c, climate_regime_shift "
            "FROM climate_trend_by_hex WHERE hex_id = :hex_id",
            {"hex_id": hex_id},
        )
        row = result.mappings().one_or_none()
        if row is None:
            return None
        return ClimateTrendProfile(**dict(row))


class FeatureNotFoundError(Exception):
    """Raised when no GIS data exists for the requested H3 hex cell."""
