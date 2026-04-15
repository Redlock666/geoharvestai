"""Async SQLAlchemy session factory for PostGIS."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.settings import get_db_url

engine = create_async_engine(get_db_url(), pool_pre_ping=True, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async DB session.

    Logic Flow:
        Opens a session from the pool, yields it to the route handler,
        and ensures rollback + close on any exception.

    Expected Exceptions:
        sqlalchemy.exc.OperationalError: DB unreachable.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
