"""FastAPI route: POST /recommend — crop recommendation endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agents.crop_graph import crop_recommendation_graph
from db.session import get_db
from models.schemas import RecommendRequest, RecommendResponse, CropResult
from services.gis_resolver import GISResolverService
from services.weather_agent import WeatherAgentService
from services.ml_predictor import MLPredictorService
from services.llm_reasoner import LLMReasonerService

router = APIRouter(prefix="/recommend", tags=["recommendations"])
logger = structlog.get_logger(__name__)


@router.post("", response_model=RecommendResponse)
async def recommend_crops(
    request: RecommendRequest,
    db: AsyncSession = Depends(get_db),
) -> RecommendResponse:
    """Recommend crops for a given location and season.

    Logic Flow:
        1. Instantiate services with injected DB session.
        2. Build initial LangGraph state from request payload.
        3. Invoke the compiled crop_recommendation_graph asynchronously.
        4. Map final graph state to RecommendResponse.

    Expected Exceptions:
        FeatureNotFoundError (422): No GIS data for this location.
        ModelNotFoundError (422): No ML model trained for this region yet.
    """
    log = logger.bind(lat=request.lat, lon=request.lon, region=request.region_code)
    log.info("recommend.request.received")

    services = {
        "gis": GISResolverService(db=db),
        "weather": WeatherAgentService(db=db),
        "ml": MLPredictorService(),
        "llm": LLMReasonerService(),
    }

    initial_state = {
        "location": {
            "lat": request.lat,
            "lon": request.lon,
            "h3_hex": "",           # populated by resolve_gis_node
            "region_code": request.region_code,
        },
        "season": request.season,
        "gis_features": {},
        "weather_snapshot": {},
        "ml_predictions": [],
        "reasoning": "",
        "messages": [],
        "_services": services,
    }

    final_state = await crop_recommendation_graph.ainvoke(initial_state)
    log.info("recommend.request.complete")

    return RecommendResponse(
        region_code=request.region_code,
        season=request.season,
        h3_hex=final_state["location"]["h3_hex"],
        recommendations=[
            CropResult(**p) for p in final_state["ml_predictions"][: request.top_n]
        ],
        reasoning=final_state["reasoning"],
        ndvi_freshness_days=final_state["weather_snapshot"].get("ndvi_freshness_days", -1),
    )
