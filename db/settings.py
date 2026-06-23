"""Database connection settings sourced from environment variables."""

from __future__ import annotations

import os


def get_db_url() -> str:
    """Build the async PostGIS connection URL from environment variables.

    Logic Flow:
        Reads POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB.
        Returns an asyncpg-compatible URL for SQLAlchemy.

    Expected Exceptions:
        KeyError: Required env var is missing.
    """
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "db")
    port = os.environ.get("POSTGRES_PORT", "5432")
    dbname = os.environ.get("POSTGRES_DB", "geoharvestai")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"


def get_timescale_url() -> str:
    """Build the async TimescaleDB connection URL from environment variables.

    Logic Flow:
        Reads POSTGRES_USER, POSTGRES_PASSWORD, TIMESCALE_HOST, TIMESCALE_PORT,
        TIMESCALE_DB. Falls back to sensible defaults for local Docker Compose.
        Returns an asyncpg-compatible URL for SQLAlchemy.

    Expected Exceptions:
        KeyError: POSTGRES_USER or POSTGRES_PASSWORD missing.
    """
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("TIMESCALE_HOST", "timescaledb")
    port = os.environ.get("TIMESCALE_PORT", "5432")
    dbname = os.environ.get("TIMESCALE_DB", "geoharvestai_ts")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"


def get_timescale_dsn() -> dict:
    """Return asyncpg-native DSN kwargs for direct asyncpg.connect() calls.

    Logic Flow:
        Used by batch ingest scripts that connect directly with asyncpg
        (not via SQLAlchemy).

    Returns:
        Dict with keys: host, port, user, password, database.

    Expected Exceptions:
        KeyError: POSTGRES_USER or POSTGRES_PASSWORD missing.
    """
    return {
        "host":     os.environ.get("TIMESCALE_HOST", "localhost"),
        "port":     int(os.environ.get("TIMESCALE_PORT", "5433")),
        "user":     os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database": os.environ.get("TIMESCALE_DB", "geoharvestai_ts"),
    }


def get_postgis_dsn() -> dict:
    """Return asyncpg-native DSN kwargs for direct asyncpg.connect() calls.

    Logic Flow:
        Used by batch ingest scripts that connect directly with asyncpg
        (not via SQLAlchemy).

    Returns:
        Dict with keys: host, port, user, password, database.

    Expected Exceptions:
        KeyError: POSTGRES_USER or POSTGRES_PASSWORD missing.
    """
    return {
        "host":     os.environ.get("POSTGRES_HOST", "localhost"),
        "port":     int(os.environ.get("POSTGRES_PORT", "5432")),
        "user":     os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database": os.environ.get("POSTGRES_DB", "geoharvestai"),
    }
