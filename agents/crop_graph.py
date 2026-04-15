"""
Crop Recommendation Agent Graph.

Defines the LangGraph StateGraph that orchestrates the full
GIS → Weather → ML → LLM reasoning pipeline.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from agents.state import CropRecommendationState
from services.gis_resolver import GISResolverService
from services.weather_agent import WeatherAgentService
from services.ml_predictor import MLPredictorService
from services.llm_reasoner import LLMReasonerService

logger = structlog.get_logger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────

async def resolve_gis_node(state: CropRecommendationState) -> dict:
    """Node 1: Resolve lat/lon → GIS feature vector via PostGIS + H3.

    Logic Flow:
        Calls GISResolverService.resolve() with coordinates from state.location.
        Updates state with gis_features.

    Expected Exceptions:
        FeatureNotFoundError: No soil/terrain data ingested for this location yet.
    """
    log = logger.bind(node="resolve_gis", hex=state["location"].get("h3_hex"))
    log.info("node.start")
    # Service is injected via graph compilation (see build_graph)
    gis_service: GISResolverService = state["_services"]["gis"]  # type: ignore[index]
    features = await gis_service.resolve(
        lat=state["location"]["lat"],
        lon=state["location"]["lon"],
    )
    log.info("node.complete")
    return {"gis_features": features.model_dump()}


async def fetch_weather_node(state: CropRecommendationState) -> dict:
    """Node 2: Fetch real-time weather snapshot from ERA5 + Sentinel NDVI.

    Logic Flow:
        Calls WeatherAgentService.fetch() with hex_id and region_code.
        Weather data is served from TimescaleDB cache (refreshed daily by worker).
        NDVI served from cache unless ndvi_freshness_days > 5.

    Expected Exceptions:
        WeatherUnavailableError: Cache miss and upstream API unreachable.
    """
    log = logger.bind(node="fetch_weather")
    log.info("node.start")
    weather_service: WeatherAgentService = state["_services"]["weather"]  # type: ignore[index]
    snapshot = await weather_service.fetch(
        hex_id=state["location"]["h3_hex"],
        region_code=state["location"]["region_code"],
    )
    log.info("node.complete", ndvi_age_days=snapshot["ndvi_freshness_days"])
    return {"weather_snapshot": snapshot}


async def predict_crops_node(state: CropRecommendationState) -> dict:
    """Node 3: Run LSTM + SARIMAX ensemble to predict crop suitability.

    Logic Flow:
        Merges gis_features + weather_snapshot into a feature vector.
        Loads region-specific model artifacts from ml/artifacts/{region_code}/.
        Returns top-N crop predictions with confidence scores.

    Expected Exceptions:
        ModelNotFoundError: No trained model for this region_code yet.
    """
    log = logger.bind(node="predict_crops", region=state["location"]["region_code"])
    log.info("node.start")
    ml_service: MLPredictorService = state["_services"]["ml"]  # type: ignore[index]
    predictions = await ml_service.predict(
        gis_features=state["gis_features"],
        weather_snapshot=state["weather_snapshot"],
        region_code=state["location"]["region_code"],
        season=state["season"],
    )
    log.info("node.complete", num_predictions=len(predictions))
    return {"ml_predictions": predictions}


async def generate_reasoning_node(state: CropRecommendationState) -> dict:
    """Node 4: Generate agronomic reasoning via LLM (GPT-4o via LangChain LCEL).

    Logic Flow:
        Passes full state context to LLMReasonerService.explain().
        Returns a structured explanation: why each crop is recommended,
        risk factors, and market timing advice.

    Expected Exceptions:
        openai.APIError: LLM call failure — returns empty reasoning string.
    """
    log = logger.bind(node="generate_reasoning")
    log.info("node.start")
    llm_service: LLMReasonerService = state["_services"]["llm"]  # type: ignore[index]
    reasoning = await llm_service.explain(
        predictions=state["ml_predictions"],
        gis_features=state["gis_features"],
        weather_snapshot=state["weather_snapshot"],
        season=state["season"],
        region_code=state["location"]["region_code"],
    )
    log.info("node.complete")
    return {"reasoning": reasoning}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble and compile the CropRecommendationState graph.

    Logic Flow:
        Registers four sequential nodes: GIS → Weather → ML → LLM.
        No conditional branching in v1; all nodes execute in sequence.

    Returns:
        Compiled LangGraph StateGraph ready for async invocation.

    Expected Exceptions:
        None — compilation is deterministic.
    """
    graph = StateGraph(CropRecommendationState)

    graph.add_node("resolve_gis", resolve_gis_node)
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("predict_crops", predict_crops_node)
    graph.add_node("generate_reasoning", generate_reasoning_node)

    graph.set_entry_point("resolve_gis")
    graph.add_edge("resolve_gis", "fetch_weather")
    graph.add_edge("fetch_weather", "predict_crops")
    graph.add_edge("predict_crops", "generate_reasoning")
    graph.add_edge("generate_reasoning", END)

    return graph.compile()


# Singleton compiled graph — imported by the FastAPI route
crop_recommendation_graph = build_graph()
