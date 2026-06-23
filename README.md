# GeoHarvestAI

> Agentic GIS-powered crop intelligence — per-field recommendations and grid-level farming plans for any coordinate in India.

**LSTM + SARIMAX ensemble · PostGIS spatial resolution · Uncertainty quantification · Drift-aware inference · Grid Farming Planner · LLM agronomic reasoning**

---

## What's in the box

Two complementary capabilities built on a single data pipeline:

| Capability | API | What it solves |
|---|---|---|
| **Per-field recommendation** | `POST /recommend` | Tells a farmer exactly which crop to grow at a given lat/lon and season |
| **Grid Farming Planner** | `services/grid_planner.py` | Allocates which crops to grow across a cluster of farms to optimise system-level margin, water use, diversity, and deficit coverage |
| **Ecosystem Drift Analysis** | `services/ecosystem_analyzer.py` | Detects slow cumulative ecosystem degradation, projects trajectory 6 seasons forward, recommends repair interventions and which crops to grow vs phase out |

---

## Architecture

```
POST /recommend?lat=&lon=&season=&region_code=IN
         │
         ▼
 LangGraph (4 nodes, sequential)
  1. resolve_gis      → PostGIS H3 hex lookup → soil NPK/pH/texture + terrain + climate zone
  2. fetch_weather    → TimescaleDB: 7-day rainfall/temp + NDVI (5-day Sentinel-2 cycle)
  3. predict_crops    → LSTM + SARIMAX ensemble
                         - drift detection (IQR + tail-bound anomaly check)
                         - timeline checkpoint (null/gap/unsorted guard before every predict)
                         - uncertainty bands (yield_min / yield_median / yield_max kg/ha)
                         - calibrated probability & anomaly flag in every result
  4. generate_reason  → OpenAI o3 via LangChain LCEL → plain-English agronomic explanation

POST /grid-plan  (Grid Farming Planner — system-level)
         │
         ▼
 GridFarmingPlannerService
  - Multi-objective weighted scoring per crop × grid
    score = 0.30·margin − 0.20·risk − 0.15·water + 0.15·market_access
            + 0.10·diversity + 0.10·deficit_priority
  - Constrained allocator: concentration cap, minimum diversity, optional water budget
  - Output: per-grid crop allocations + portfolio-level summary (expected margin, total water)
```

---

## Codebase map

```
api/routes/          → FastAPI route handlers only
services/
  gis_resolver.py    → Spatial joins: lat/lon → soil, terrain, climate features
  weather_agent.py   → LangGraph tool node: ERA5 + Sentinel NDVI fetch
  ml_predictor.py    → Inference: drift detection, uncertainty bands, anomaly flags
  llm_reasoner.py    → LangChain LCEL chain: agronomic explanation
  grid_planner.py    → Grid Farming Planner: multi-objective scorer + allocator
  ecosystem_analyzer.py → Ecosystem Drift: loads pre-computed CUSUM drift reports
agents/
  crop_graph.py      → LangGraph graph definition
  state.py           → TypedDict state (ecosystem drift context fields added)
models/
  schemas.py         → Pydantic V2 API schemas (request/response)
  grid_planner.py    → Grid planner schemas (PlannerWeights, constraints, allocations)
  gis.py             → GIS feature models
ml/
  pipeline/
    data_pipeline.py → Canonical TrainingDataBundle builder (timeline checkpoint + sufficiency gate)
    drift.py         → Variability metrics, anomaly bounds, uncertainty profiles
    ecosystem_drift.py → CUSUM detector, projection engine, repair recommender, crop viability mapper
    features.py      → Feature engineering
  train/
    train_sarimax.py → SARIMAX training (wired to canonical bundle + drift artifact persistence)
    train_lstm.py    → LSTM training (requires ≥3 years of data)
  artifacts/         → Per-region model artifacts: {region_code}/
db/                  → PostGIS + TimescaleDB session management
scripts/             → Ingestion, download, preflight, and automation scripts
```

---

## Component status

| Component | File | Status |
|---|---|---|
| FastAPI route | `api/routes/recommend.py` | ✅ |
| LangGraph agent | `agents/crop_graph.py` | ✅ |
| GIS resolver | `services/gis_resolver.py` | ✅ |
| Weather agent | `services/weather_agent.py` | ✅ |
| ML predictor (with drift + uncertainty) | `services/ml_predictor.py` | ✅ |
| LLM reasoner | `services/llm_reasoner.py` | ✅ |
| Ecosystem Drift Engine | `ml/pipeline/ecosystem_drift.py` | ✅ |
| Ecosystem Analyzer Service | `services/ecosystem_analyzer.py` | ✅ |
| Grid Farming Planner | `services/grid_planner.py` | ✅ |
| Grid planner schemas | `models/grid_planner.py` | ✅ |
| Canonical training pipeline | `ml/pipeline/data_pipeline.py` | ✅ |
| Drift metrics | `ml/pipeline/drift.py` | ✅ |
| SARIMAX training | `ml/train/train_sarimax.py` | ✅ |
| LSTM training | `ml/train/train_lstm.py` | ✅ |
| Auth preflight | `scripts/check_data_auth.py` | ✅ |
| Training readiness checker | `scripts/check_training_readiness.py` | ✅ |
| Download validation + retry | `scripts/download_india_data.sh` | ✅ |

---

## Prediction output schema

Every `CropResult` now carries reliability metadata alongside the recommendation:

```json
{
  "crop_name": "Wheat",
  "confidence": 0.87,
  "probability": 0.73,
  "yield_estimate_kg_ha": 3420,
  "yield_min_kg_ha": 2940,
  "yield_median_kg_ha": 3380,
  "yield_max_kg_ha": 3820,
  "uncertainty_band_pct": 25.7,
  "anomaly_flag": false,
  "anomaly_reason": null,
  "model_used": "ensemble"
}
```

| Field | Description |
|---|---|
| `probability` | Calibrated probability, dampened by regional variability index from drift report |
| `yield_min/median/max_kg_ha` | Uncertainty band from historical variability profile |
| `uncertainty_band_pct` | Band width as % of median — surface this to users as a reliability signal |
| `anomaly_flag` | `true` if any input feature falls outside IQR-derived bounds |
| `anomaly_reason` | Human-readable description of which feature triggered the anomaly |

---

## ML pipeline safeguards

### Timeline checkpoint
Every call to `build_training_bundle()` runs a hard validation before any model sees data:
- No null timestamps in yield or exogenous series
- No duplicate timestamps
- Series is monotonically sorted
- Yield timestamps are fully covered by exogenous data
- Gaps diagnosed and reported

Raises `ValueError` immediately if any check fails — prevents silent degradation.

### Training sufficiency gate
`_load_yield()` enforces minimum data quality before training proceeds:

| Gate | Default | Env override |
|---|---|---|
| Minimum yield rows | 200 | `MIN_YIELD_ROWS` |
| Minimum distinct crops | 8 | `MIN_YIELD_CROPS` |
| Minimum monthly timesteps | 24 | `MIN_YIELD_TIMESTEPS` |

### Drift detection at inference
At every prediction call, `_detect_feature_anomalies()` loads the drift report for the region and checks each incoming feature against its IQR + tail bounds. Results surface in `anomaly_flag` and `anomaly_reason`.

---

## Grid Farming Planner

System-level crop planning for a group of farms or administrative grid cells.

```python
# Quick demo
python scripts/run_grid_planner_demo.py
# Output: G1 | Pulses | area=60.00 ha | score=0.4000
#         G1 | Millet | area=40.00 ha | score=0.2513
#         Portfolio: expected margin ₹8.49M, water 348,000 m³
```

**Objective function:**
```
score = w_margin·margin_norm
      − w_risk·risk_norm
      − w_water·water_per_ha_norm
      + w_access·market_access_norm
      + w_diversity·diversity_resilience_norm
      + w_deficit·deficit_priority_norm
```

**Constraints:**
- `max_crop_share` — no single crop gets more than 60% of grid area
- `min_diverse_crops` — at least 2 crops must be selected per grid
- `min_selected_share` — each selected crop must get at least 15% of grid area
- `max_water_m3_per_grid` — optional hard water budget cap

See `docs/GRID_FARMING_V1.md` for the full integration contract.

---

## Data ingestion (India)

### Preflight — check credentials before downloading
```bash
python scripts/check_data_auth.py
```
Validates Earthdata (MODIS NDVI) and CDS API (ERA5) credentials from env or dotfiles. Exit code 0 = all clear.

### Step 1 — Download static GIS files (one-time)
```bash
cp .env.example .env          # fill in API keys
bash scripts/download_india_data.sh data/raw
```
Downloads: SoilGrids v2 (nitrogen, phosphorus, potassium, pH), Köppen-Geiger 1 km raster, SRTM 30 m terrain.
The script validates each file (size threshold + XML/HTML payload detection) and retries up to 3 times before failing.
Manual steps printed at the end: APY yield CSV, ERA5 credentials, NASA EarthData.

### Step 2 — Start databases
```bash
docker-compose up -d db timescaledb
```

### Step 3 — Run ingestion in order
```bash
python scripts/ingest_soilgrids.py     --region IN
python scripts/ingest_terrain.py       --region IN
python scripts/ingest_climate_zones.py --region IN
python scripts/ingest_era5.py          --region IN --years 2010-2025
python scripts/ingest_ndvi_modis.py    --region IN --years 2010-2025
python scripts/ingest_apy.py           --region IN --file data/raw/apy/apy_india_all.csv
# Soil Health Card data — unlocks biological health layer + fertilizer sufficiency
python scripts/ingest_soil_health_cards.py --region IN --file data/raw/shc/shc_india.csv
# Compute ecosystem drift reports (run after SHC + ERA5 + NDVI are ingested)
python scripts/compute_ecosystem_drift.py --region IN
```

APY ingestion supports two column formats:
- **Form A** — `area_ha` + `production_tonnes` (yield derived automatically)
- **Form B** — `yield_kg_ha` directly

State and district columns are optional (default to `"Unknown"`).

### Step 4 — Check training readiness (automated gate)
```bash
make check-train-ready REGION=IN
# or — wait until ready, then train automatically:
make train-when-ready REGION=IN WAIT_MAX_MINUTES=180 WAIT_INTERVAL_SEC=300
```

`check-train-ready` queries TimescaleDB for APY/weather/NDVI coverage and prints a readiness report. Exit code 0 = ready, 1 = not ready.

`train-when-ready` polls on a configurable interval and launches training as soon as all gates pass.

### Step 5 — Train models
```bash
make train REGION=IN MODEL=sarimax
make train REGION=IN MODEL=lstm     # requires ≥3 years of data
```

Model artifacts stored at `ml/artifacts/{region_code}/`.

---

## Makefile targets

| Target | Description |
|---|---|
| `make up` | Start full stack (PostGIS + TimescaleDB + API) |
| `make up-db` | Start databases only (for hot-reload API dev) |
| `make ingest-soil REGION=IN` | Ingest SoilGrids data |
| `make ingest-climate REGION=IN` | Ingest Köppen-Geiger climate zones |
| `make ingest-era5 REGION=IN` | Ingest ERA5 weather history |
| `make ingest-ndvi REGION=IN` | Ingest MODIS NDVI history |
| `make ingest-apy REGION=IN APY_FILE=...` | Ingest crop yield data |
| `make ingest-shc REGION=IN SHC_FILE=...` | Ingest Soil Health Card data (biological health layer) |
| `make compute-ecosystem-drift REGION=IN` | Compute CUSUM drift reports for all hex cells |
| `make check-train-ready REGION=IN` | Run data sufficiency check |
| `make train-if-ready REGION=IN` | Train immediately if gates pass, else exit 1 |
| `make train-when-ready REGION=IN` | Poll until ready, then train automatically |
| `make test` | Run pytest suite |

---

## Running the full stack
```bash
docker-compose up --build
# API at http://localhost:8000
```

## Example request
```bash
curl -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"lat": 20.5, "lon": 78.9, "season": "kharif_2026", "region_code": "IN", "top_n": 5}'
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM reasoning layer |
| `OPENAI_REASONING_MODEL` | Model override — `o3`, `gpt-5.4`, etc. *(default: `o3`)* |
| `OPENAI_MODEL_FAMILY` | `reasoning` (default) or `chat` (enables temperature for GPT-5.x) |
| `CDSAPI_KEY` | ERA5 weather — format `uid:api-token` from cds.climate.copernicus.eu |
| `OPENTOPOGRAPHY_API_KEY` | SRTM terrain from opentopography.org |
| `EARTHDATA_USERNAME` | MODIS NDVI — NASA EarthData account |
| `EARTHDATA_PASSWORD` | MODIS NDVI — NASA EarthData account |
| `SENTINELHUB_CLIENT_ID` | Live Sentinel-2 NDVI (5-day cycle) |
| `SENTINELHUB_CLIENT_SECRET` | Live Sentinel-2 NDVI |
| `MIN_YIELD_ROWS` | Training sufficiency gate override (default: 200) |
| `MIN_YIELD_CROPS` | Training sufficiency gate override (default: 8) |
| `MIN_YIELD_TIMESTEPS` | Training sufficiency gate override (default: 24) |

---

## Soil biological health layer

When Soil Health Card data is ingested, every prediction gains three additional fields:

| Field | Description |
|---|---|
| `fertilizer_sufficiency` | `deficient` / `sufficient` / `excessive` per ICAR critical limits (N, P, K) |
| `soil_health_index` | 0.0–1.0 composite from organic carbon %, electrical conductivity, and OC trend |
| `biological_collapse_risk` | `true` when NPK is adequate but organic carbon is low or declining — the hallmark of over-fertilization induced microbial degradation |

The ML feature vector is automatically extended with:
- `soil_organic_carbon_pct` — Walkley-Black OC % (microbial biomass proxy)
- `soil_ec_ds_m` — electrical conductivity (salinity indicator)
- `climate_anomaly_trend_mm` — 5-year vs 30-year rainfall deviation (drying/wetting regime shift)
- `climate_temp_anomaly_c` — 5-year vs 30-year temperature deviation (warming signal)

When SHC data is not yet ingested, these features default to conservative neutral values and the system continues normally with SoilGrids chemical data only.

---

## Docs

| Document | Contents |
|---|---|
| `docs/GRID_FARMING_V1.md` | Grid Farming Planner integration contract, objective formula, constraint reference |
| `docs/DATA_SOURCE_FALLBACK_PLAYBOOK.md` | APY data source hierarchy, column mapping cheat sheet, fallback execution path |

---

## Ecosystem Drift Analysis

The `ml/pipeline/ecosystem_drift.py` module detects slow, cumulative micro-changes that transform agricultural ecosystems over seasons and years — the kind that point-in-time measurements miss entirely.

### Algorithm
| Stage | Method | Purpose |
|---|---|---|
| **CUSUM detection** | Page (1954) control charts, k=0.5σ, h=4σ | Detects sustained shifts below noise threshold |
| **Composite health score** | Weighted average of 5 normalised indicators | Single 0–1 signal of ecosystem state |
| **6-season projection** | Linear extrapolation + ICAR threshold ETA | "No-intervention trajectory" — where is this heading? |
| **Repair recommender** | Priority-ranked rule engine (8 rule sets) | Evidence-based interventions ordered by urgency |
| **Crop viability mapper** | 20-crop ICAR tolerance profile lookup | Which crops are viable now, in 6 seasons, and which are soil-restorative |

### Five monitored indicators
| Indicator | Degrading direction | Critical threshold |
|---|---|---|
| Organic carbon % | Declining | < 0.50% (ICAR low threshold) |
| Electrical conductivity | Rising | > 4.0 dS/m |
| 5yr rainfall anomaly vs 30yr baseline | More negative (drying) | < −200 mm |
| 5yr temperature anomaly vs 30yr baseline | More positive (warming) | > 1.0 °C |
| Seasonal NDVI | Declining | < 0.25 |

### Running ecosystem drift
```bash
# After ingesting SHC + ERA5 + NDVI:
make compute-ecosystem-drift REGION=IN

# Single hex for debugging:
docker compose run --rm train \
  python scripts/compute_ecosystem_drift.py --region IN --hex 8765b4a4fffffff --dry-run
```

The `ecosystem_drift_by_hex` materialized view is refreshed automatically. The `EcosystemAnalyzerService` loads reports during Stage 1 (GIS resolution) and makes them available to Stage 4 (LLM reasoning).

