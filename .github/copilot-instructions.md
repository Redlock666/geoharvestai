# GitHub Copilot Instructions — GIS Crop Recommendation System

## Project Overview
An agentic, GIS-powered crop recommendation API. Takes `(lat, lon, season)` as input — **region is always user-supplied at runtime, never hardcoded** — and returns ML-driven crop recommendations with LLM-generated reasoning. Built with **FastAPI + LangGraph + PostGIS + TimescaleDB**. Deployed via Docker; no graph DB (PostGIS handles all spatial relationships).

---

## Architecture: Separation of Concerns (Strict)

```
api/routes/          → FastAPI route handlers only (no business logic)
services/            → Business logic layer (one service per domain)
  gis_resolver.py    → Spatial joins: lat/lon → soil, terrain, climate features
  weather_agent.py   → LangGraph tool node: ERA5 + Sentinel NDVI fetcher
  ml_predictor.py    → Time-series forecasting pipeline (LSTM + SARIMAX ensemble)
  llm_reasoner.py    → LangChain LCEL chain: explains recommendations
agents/              → LangGraph graph definitions and TypedDict state schemas
models/              → Pydantic V2 schemas (request/response, DB models)
db/                  → PostGIS + TimescaleDB session management (never in services)
```

**Rule:** Services receive typed inputs and return typed outputs. They never import from `api/` and never hold DB sessions directly — use injected repositories.

---

## Key Patterns

### ML Models — LSTM + SARIMAX Ensemble
Use a **two-model ensemble** for predictions:
- **SARIMAX** (`statsmodels`) — handles seasonal crop cycles and interpretable trend decomposition. Use for regions with <3 years of historical data.
- **LSTM** (`PyTorch`) — captures non-linear weather-yield dependencies over multi-season sequences. Use when ≥3 years of timestamped weather + yield data exists per region.

```python
# ml/predictor.py pattern — never hardcode region
async def predict(features: CropFeatureVector, region_code: str) -> list[CropPrediction]:
    """
    Logic Flow: Load region-specific scaler → run SARIMAX for seasonal baseline
                → run LSTM for residual correction → weighted ensemble output.
    Expected Exceptions: ModelNotFoundError if region_code has no trained model yet.
    """
```

Model artifacts stored at `ml/artifacts/{region_code}/` — one directory per user-supplied region.

### Spatial Lookups — Always Use H3 Indexing
Pre-compute H3 hex cells (resolution 7) for all GIS layers. Never do raw `ST_Contains` on every request.
```python
import h3
hex_id = h3.geo_to_h3(lat, lon, resolution=7)  # Fast O(1) lookup key
```

### LangGraph State — TypedDict Only
```python
from typing import TypedDict, Annotated
class CropRecommendationState(TypedDict):
    location: dict          # {lat, lon, h3_hex}
    gis_features: dict      # soil NPK, pH, elevation, climate_zone
    weather_snapshot: dict  # ERA5 rainfall_7d, temp_avg, NDVI
    ml_predictions: list    # [{crop, confidence, yield_estimate}]
    reasoning: str          # LLM-generated explanation
```

### Async Everywhere
All service methods, DB calls, and LLM invocations must be `async def`. No blocking I/O.

### Real-Time Data Freshness Strategy
- **Soil/Terrain:** Cached indefinitely in PostGIS — never re-fetch per request
- **Weather (ERA5):** TimescaleDB, refreshed by a daily cron job — not per-request
- **NDVI (Sentinel-2):** 5-day revisit cycle; serve cached unless `force_refresh=True`

---

## Critical Developer Workflows

```bash
# Start full local stack (PostGIS + TimescaleDB + API)
docker-compose up --build

# Start only the DB services (for local API dev with hot-reload)
docker-compose up -d db timescaledb

# Run spatial data ingestion — region is ALWAYS a CLI arg, never hardcoded
python scripts/ingest_soil_grids.py --region <region_code>

# Run the API locally (outside Docker, for fast iteration)
uvicorn api.main:app --reload --port 8000

# Train models for a specific region
python ml/train.py --region <region_code> --model lstm
python ml/train.py --region <region_code> --model sarimax

# Run tests (spatial fixtures use pytest-postgresql)
pytest tests/ -v --asyncio-mode=auto
```

### Docker Compose Services
| Service | Purpose |
|---|---|
| `api` | FastAPI app (port 8000) |
| `db` | PostGIS for spatial/soil data |
| `timescaledb` | Time-series weather + NDVI data |
| `worker` | Daily cron: ERA5 + Sentinel data refresh |

---

## External Integrations & API Keys

| Service | Env Var | Notes |
|---|---|---|
| Sentinel Hub (NDVI) | `SENTINELHUB_CLIENT_ID`, `SENTINELHUB_CLIENT_SECRET` | 5-day data lag |
| Open-Meteo (weather) | No key needed | Free tier, rate-limit aware |
| ERA5 / CDS API | `CDSAPI_KEY` | Batch daily, not per-request |
| OpenAI (LLM) | `OPENAI_API_KEY` | GPT-4o for reasoning layer |

---

## Conventions Specific to This Project

- **Google-style docstrings** with `Logic Flow:` and `Expected Exceptions:` sections on every function
- **Structured JSON logging** — use `structlog` not `print` or stdlib `logging`
- **Parameterized SQL only** — never f-string queries into PostGIS/TimescaleDB
- **Pydantic V2** with `model_config = ConfigDict(populate_by_name=True)` for all schemas
- **PDCA error handling:** every service method wraps I/O in `try/except` with retry logic (use `tenacity`)
- **Region is always runtime input** — never hardcode a country, region, or locale anywhere in source code; always pass as a parameter
- **No Neo4j / graph DB** — PostGIS handles all spatial relationships; don't add graph dependencies
