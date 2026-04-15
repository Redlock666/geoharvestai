"""FastAPI application entry point for GeoHarvestAI."""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import recommend

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="GeoHarvestAI",
    description="Agentic GIS-powered crop recommendation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("geoharvestai.startup")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("geoharvestai.shutdown")
