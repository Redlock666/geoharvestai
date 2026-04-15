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

from models.gis import GISFeatureVector

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

        log.info("gis.resolve.complete", climate_zone=climate_zone)
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


class FeatureNotFoundError(Exception):
    """Raised when no GIS data exists for the requested H3 hex cell."""
