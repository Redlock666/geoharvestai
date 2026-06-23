"""
LLM Reasoner Service.

Generates a structured agronomic explanation for the top crop recommendations
using a LangChain LCEL chain backed by o3 (OpenAI reasoning model).

The explanation covers:
  - Why each top-3 crop is recommended (soil fit, weather fit, season fit)
  - Key risk factors (low rainfall, acidic soil, high elevation, etc.)
  - Market timing advice based on season

The chain is a simple prompt → o3 → StrOutputParser pipeline.
No RAG or retrieval in v1; a future Purple8 integration can inject
ICAR/IMD advisory context into the system prompt.

Model note:
    Uses 'o3' by default (OpenAI's flagship reasoning model).
    Override via the OPENAI_REASONING_MODEL env var, e.g.:
        OPENAI_REASONING_MODEL=gpt-5.4   (or gpt-5.5, o4, etc.)
    o-series models do NOT accept temperature — reasoning_effort is used instead.
    GPT-5.x chat models accept temperature; set OPENAI_MODEL_FAMILY=chat to
    re-enable it.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.state import CropPrediction, GISFeatures, WeatherSnapshot

logger = structlog.get_logger(__name__)

# Default reasoning model — override with OPENAI_REASONING_MODEL env var.
# o-series (o3, o4, …) are reasoning models: no temperature, use reasoning_effort.
# GPT-5.x chat models (gpt-5.4, gpt-5.5) accept temperature; set
# OPENAI_MODEL_FAMILY=chat to switch behaviour.
_DEFAULT_MODEL   = "o3"
_MODEL_FAMILY    = os.environ.get("OPENAI_MODEL_FAMILY", "reasoning")   # "reasoning" | "chat"

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert agronomist with deep knowledge of Indian crop science, \
soils, and regional farming practices. You explain crop recommendations \
clearly and practically for farmers and agri-advisors.

Your explanation must:
1. Briefly justify why the top 3 recommended crops suit the given soil and \
   weather conditions.
2. Highlight the single most important risk factor for this location and season.
3. Give one concrete market timing tip (when to sow, when to expect peak price).
4. Be written in plain English, 150–250 words, no bullet points, no headers.
"""

_HUMAN_PROMPT = """\
Location data:
- Season: {season}
- Region: {region_code}
- Soil: pH={soil_ph}, N={soil_nitrogen} g/kg, P={soil_phosphorus} mg/kg, \
K={soil_potassium} mg/kg, texture={soil_texture}
- Terrain: elevation={elevation_m} m, slope={slope_deg}°
- Climate zone: {climate_zone}
- Weather (last 7 days): rainfall={rainfall_7d_mm} mm, \
temp={temp_avg_c}°C avg ({temp_min_c}–{temp_max_c}°C)
- NDVI: {ndvi:.3f} (data age: {ndvi_freshness_days} days)

Top crop recommendations (model: {model_used}):
{crops_summary}

Explain why these crops are recommended for this location and season, \
key risks, and market timing advice.
"""


# ── LCEL chain ────────────────────────────────────────────────────────────────

def _build_chain() -> Runnable:
    """Build the LangChain LCEL chain: prompt → reasoning model → StrOutputParser.

    Logic Flow:
        Reads OPENAI_API_KEY from environment.
        Selects model from OPENAI_REASONING_MODEL env var (default: o3).
        o-series reasoning models: reasoning_effort='high', no temperature.
        GPT-5.x chat models: temperature=0.3 (set OPENAI_MODEL_FAMILY=chat).
        Returns a Runnable chain (supports .ainvoke()).

    Returns:
        LCEL chain object.

    Expected Exceptions:
        KeyError: OPENAI_API_KEY not set in environment.
    """
    api_key_str = os.environ.get("OPENAI_API_KEY", "")
    model_name  = os.environ.get("OPENAI_REASONING_MODEL", _DEFAULT_MODEL)
    is_reasoning = _MODEL_FAMILY == "reasoning"

    llm_kwargs: dict[str, Any] = {
        "model":   model_name,
        "api_key": SecretStr(api_key_str),
    }
    if is_reasoning:
        # o-series: reasoning_effort controls depth; temperature is not accepted
        llm_kwargs["model_kwargs"] = {"reasoning_effort": "high"}
    else:
        # GPT-5.x chat models: temperature accepted
        llm_kwargs["temperature"] = 0.3

    llm = ChatOpenAI(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# ── Public service ─────────────────────────────────────────────────────────────

class LLMReasonerService:
    """Generates agronomic reasoning for crop recommendations via o3 / GPT-5.x."""

    def __init__(self) -> None:
        self._chain: Runnable = _build_chain()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
    async def explain(
        self,
        predictions: list[CropPrediction],
        gis_features: GISFeatures,
        weather_snapshot: WeatherSnapshot,
        season: str,
        region_code: str,
    ) -> str:
        """Generate a natural-language explanation for the top crop recommendations.

        Logic Flow:
            1. Format top-3 predictions into a readable crops_summary string.
            2. Build the prompt variables dict from GIS + weather + season.
            3. Call the LCEL chain asynchronously via .ainvoke().
            4. Return the stripped string response.

        Args:
            predictions:      List of CropPrediction dicts (already sorted by confidence).
            gis_features:     GISFeatures TypedDict from graph state.
            weather_snapshot: WeatherSnapshot TypedDict from graph state.
            season:           Season string (e.g. 'kharif_2026').
            region_code:      Runtime region identifier (e.g. 'IN').

        Returns:
            Agronomic reasoning string (150–250 words).

        Expected Exceptions:
            openai.APIError: LLM call failed after retries.
            Returns empty string on final failure (non-fatal — recommendation
            scores are still returned to the caller).
        """
        log = logger.bind(region_code=region_code, season=season)
        log.info("llm.explain.start", num_predictions=len(predictions))

        top3 = predictions[:3]
        crops_summary = "\n".join(
            f"  {i + 1}. {p['crop_name']} "
            f"(confidence={p['confidence']:.0%}, "
            f"yield_est={p['yield_estimate_kg_ha']:.0f} kg/ha)"
            for i, p in enumerate(top3)
        )
        model_used = top3[0]["model_used"] if top3 else "unknown"

        try:
            result: str = await self._chain.ainvoke({
                "season":              season,
                "region_code":         region_code,
                "soil_ph":             gis_features["soil_ph"],
                "soil_nitrogen":       gis_features["soil_nitrogen"],
                "soil_phosphorus":     gis_features["soil_phosphorus"],
                "soil_potassium":      gis_features["soil_potassium"],
                "soil_texture":        gis_features["soil_texture"],
                "elevation_m":         gis_features["elevation_m"],
                "slope_deg":           gis_features["slope_deg"],
                "climate_zone":        gis_features["climate_zone"],
                "rainfall_7d_mm":      weather_snapshot["rainfall_7d_mm"],
                "temp_avg_c":          weather_snapshot["temp_avg_c"],
                "temp_min_c":          weather_snapshot["temp_min_c"],
                "temp_max_c":          weather_snapshot["temp_max_c"],
                "ndvi":                weather_snapshot["ndvi"],
                "ndvi_freshness_days": weather_snapshot["ndvi_freshness_days"],
                "crops_summary":       crops_summary,
                "model_used":          model_used,
            })
            log.info("llm.explain.complete", chars=len(result))
            return result.strip()

        except Exception as exc:  # noqa: BLE001
            log.error("llm.explain.failed", error=str(exc))
            return ""
