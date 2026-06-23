"""
Weather Agent Service.

Fetches the latest weather snapshot for a given H3 hex cell from TimescaleDB.
Weather history is pre-populated by ingest_era5.py; the daily_refresh.py
worker keeps the last 7 days current via Open-Meteo.

NDVI is served from ndvi_obs (MODIS history + Sentinel-2 live).
A cache miss on NDVI falls back to the most recent available observation,
reporting its age in ndvi_freshness_days.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import asyncpg
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import WeatherSnapshot

logger = structlog.get_logger(__name__)


class WeatherUnavailableError(Exception):
    """Raised when no weather data exists for this hex + region in TimescaleDB."""


class WeatherAgentService:
    """Serves weather snapshots from the TimescaleDB cache.

    The service never calls ERA5 or Open-Meteo directly at request time.
    All upstream fetches are handled offline by:
        - ingest_era5.py (historical backfill)
        - daily_refresh.py (running as Docker worker, refreshes every 24 h)
    """

    def __init__(self, db: asyncpg.Connection | None = None) -> None:
        """Initialise with an optional pre-connected asyncpg connection.

        Logic Flow:
            If db is None, a new connection is lazily created on first fetch()
            call using environment variables via get_timescale_dsn().

        Args:
            db: Optional pre-created asyncpg connection (injected in tests).
        """
        self._db = db
        self._owns_connection = db is None

    async def _get_conn(self) -> asyncpg.Connection:
        """Return (or create) the asyncpg connection to TimescaleDB.

        Logic Flow:
            Reads connection parameters from environment variables.
            Creates a new asyncpg connection if one was not injected.

        Expected Exceptions:
            asyncpg.PostgresConnectionError: TimescaleDB unreachable.
        """
        if self._db is not None:
            return self._db
        conn = await asyncpg.connect(
            host=os.environ.get("TIMESCALE_HOST", "timescaledb"),
            port=int(os.environ.get("TIMESCALE_PORT", "5432")),
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            database=os.environ.get("TIMESCALE_DB", "geoharvestai_ts"),
        )
        self._db = conn
        return conn

    async def close(self) -> None:
        """Close the asyncpg connection if this service owns it."""
        if self._owns_connection and self._db is not None:
            await self._db.close()
            self._db = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch(self, hex_id: str, region_code: str) -> WeatherSnapshot:
        """Return the most recent 7-day weather snapshot for a hex cell.

        Logic Flow:
            1. Query weather_7d continuous aggregate for the latest complete
               7-day bucket matching this hex_id.
            2. Query ndvi_obs for the most recent NDVI observation within 30 days.
            3. Compute ndvi_freshness_days from the observation timestamp.
            4. Assemble and return a typed WeatherSnapshot.

        Args:
            hex_id:      H3 resolution-7 hex identifier.
            region_code: Runtime region identifier (e.g. 'IN').

        Returns:
            WeatherSnapshot with weather and NDVI fields populated.

        Expected Exceptions:
            WeatherUnavailableError: No weather rows exist for this hex.
        """
        log = logger.bind(hex_id=hex_id, region_code=region_code)
        log.info("weather.fetch.start")

        conn = await self._get_conn()

        weather_row = await self._fetch_weather(conn, hex_id)
        ndvi_row = await self._fetch_ndvi(conn, hex_id)

        now = datetime.now(tz=timezone.utc)
        ndvi_freshness = 999
        ndvi_value = 0.0
        if ndvi_row is not None:
            ndvi_value = float(ndvi_row["ndvi"])
            obs_time: datetime = ndvi_row["time"]
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
            ndvi_freshness = (now - obs_time).days

        snapshot: WeatherSnapshot = {
            "rainfall_7d_mm": float(weather_row["rainfall_7d_mm"] or 0.0),
            "temp_avg_c":     float(weather_row["temp_avg_c"] or 25.0),
            "temp_min_c":     float(weather_row["temp_min_c"] or 20.0),
            "temp_max_c":     float(weather_row["temp_max_c"] or 35.0),
            "ndvi":           ndvi_value,
            "ndvi_freshness_days": ndvi_freshness,
        }
        log.info("weather.fetch.complete", ndvi_freshness_days=ndvi_freshness)
        return snapshot

    async def _fetch_weather(
        self, conn: asyncpg.Connection, hex_id: str
    ) -> asyncpg.Record:
        """Fetch the latest 7-day weather bucket from the continuous aggregate.

        Logic Flow:
            Queries weather_7d ordered by week DESC, LIMIT 1.
            Falls back to weather_obs raw table if aggregate is empty (e.g.
            immediately after ingest before TimescaleDB has refreshed policies).

        Args:
            conn:   Active asyncpg connection.
            hex_id: H3 resolution-7 hex identifier.

        Returns:
            asyncpg.Record with rainfall_7d_mm, temp_avg_c, temp_min_c,
            temp_max_c fields.

        Expected Exceptions:
            WeatherUnavailableError: No records in either table for this hex.
        """
        # Try the continuous aggregate first (fast path)
        row = await conn.fetchrow(
            """
            SELECT rainfall_7d_mm, temp_avg_c, temp_min_c, temp_max_c
            FROM   weather_7d
            WHERE  hex_id = $1
            ORDER  BY week DESC
            LIMIT  1
            """,
            hex_id,
        )
        if row is not None:
            return row

        # Fallback: raw weather_obs — aggregate sum/avg for last 7 days
        row = await conn.fetchrow(
            """
            SELECT
                SUM(rainfall_mm)  AS rainfall_7d_mm,
                AVG(temp_avg_c)   AS temp_avg_c,
                MIN(temp_min_c)   AS temp_min_c,
                MAX(temp_max_c)   AS temp_max_c
            FROM   weather_obs
            WHERE  hex_id = $1
              AND  time >= NOW() - INTERVAL '7 days'
            """,
            hex_id,
        )
        if row is not None and row["temp_avg_c"] is not None:
            return row

        raise WeatherUnavailableError(
            f"No weather data for hex_id={hex_id}. "
            "Run ingest_era5.py or wait for daily_refresh.py to populate the cache."
        )

    async def _fetch_ndvi(
        self, conn: asyncpg.Connection, hex_id: str
    ) -> asyncpg.Record | None:
        """Fetch the most recent NDVI observation within the last 30 days.

        Logic Flow:
            Queries ndvi_obs for the latest row within 30 days.
            Returns None (instead of raising) — NDVI absence is non-fatal;
            the caller will report ndvi_freshness_days=999.

        Args:
            conn:   Active asyncpg connection.
            hex_id: H3 resolution-7 hex identifier.

        Returns:
            asyncpg.Record with ndvi, time fields, or None if not found.
        """
        return await conn.fetchrow(
            """
            SELECT ndvi, time
            FROM   ndvi_obs
            WHERE  hex_id = $1
              AND  time >= NOW() - INTERVAL '30 days'
            ORDER  BY time DESC
            LIMIT  1
            """,
            hex_id,
        )
