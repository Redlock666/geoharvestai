"""Database connection settings sourced from environment variables."""

from __future__ import annotations

import os


def get_db_url() -> str:
    """Build the async PostGIS connection URL from environment variables.

    Logic Flow:
        Reads POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB.
        Returns an asyncpg-compatible URL.

    Expected Exceptions:
        KeyError: Required env var is missing.
    """
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "db")
    port = os.environ.get("POSTGRES_PORT", "5432")
    dbname = os.environ.get("POSTGRES_DB", "geoharvestai")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
