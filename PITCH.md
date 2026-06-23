# GeoHarvestAI — Precision Crop Intelligence for India

> *Tell any field in India exactly what to grow. Tell any district exactly how to grow it together.*

GeoHarvestAI fuses satellite soil maps, 15 years of weather history, real-time crop health data, and AI reasoning to deliver three things no existing system provides:

1. **Per-field crop recommendations** — for any coordinate in India, in under three seconds.
2. **Grid Farming Plans** — system-level crop allocation across groups of farms, optimising for margin, water use, market balance, and national deficit coverage simultaneously.
3. **Ecosystem Drift Analysis** — detects the slow, cumulative transformation of the agricultural ecosystem across soil biology, climate patterns, and vegetation health; projects where it is heading; and recommends what to grow now vs what to transition to.

Built with **FastAPI + LangGraph + PostGIS + TimescaleDB**. Deployed via Docker.

---

## The Problem

Indian farmers lose an estimated **₹1.5 lakh crore annually** to wrong crop decisions. Existing advisory systems are generic, paper-based, or require expensive agronomist visits. None combine real-time satellite data, local soil science, and AI reasoning in a single query. And none address the deeper problem: even good individual decisions can produce bad collective outcomes — gluts in some crops, deficits in others, avoidable water stress, fragmented market access.

| Gap | Impact |
|---|---|
| Generic state-level advisories | Ignore district-level soil variation of up to 400% in nitrogen content across 50 km |
| Weather-blind decisions | Sowing choices made without current rainfall deficits or satellite crop health signal |
| No yield uncertainty | Buyers and lenders see a single number — no sense of risk or confidence interval |
| No collective planning | Individual optimal decisions aggregate into market gluts, water over-extraction, and persistent deficits in pulses, oilseeds, and protein crops |
| Ecosystem blindness | Nobody is tracking the slow degradation — declining organic carbon, drying trends, salinity accumulation — until yield collapses and it's too expensive to reverse |
| Expertise not scalable | 1 agronomist per 1,000 farmers — expert knowledge can't reach 140M farm households |

---

## What GeoHarvestAI Does

### Capability 1 — Per-Field Recommendation

A single API call — with just **latitude, longitude, and season** — triggers a four-stage AI pipeline:

**Stage 1 — Resolve the field's physical fingerprint**
Coordinates are mapped to an H3 hex cell (~5 km²). PostGIS returns soil nitrogen, phosphorus, potassium, pH, texture class, elevation, slope, and climate zone — all pre-indexed for sub-millisecond lookup.

The system also loads the field's **Soil Health Card (SHC) profile** — organic carbon percentage, electrical conductivity, micronutrient levels (sulphur, zinc, iron), and a derived NPK trend from historical cards. This is the layer that SoilGrids cannot provide: SoilGrids tells you how much nitrogen *is* in the soil; SHC data tells you whether the microbial community can actually make it available to the crop.

Finally, a **climate anomaly trend** is loaded: the 5-year rolling rainfall and temperature deviation from the 30-year ERA5 baseline. A field that averaged 800mm rainfall a decade ago and now averages 620mm is in a fundamentally different risk category than one that has always been at 620mm.

*Data sources: SoilGrids v2 (ISRIC) · SRTM 30m DEM (NASA) · Beck 2018 Köppen-Geiger · ICAR 15-Zone Classification · India Soil Health Card Portal (MoA&FW, 220M+ cards) · ERA5-Land 30-year baseline*

**Stage 2 — Fetch real-time weather & crop health**
7-day rainfall, temperature, and humidity from a TimescaleDB cache refreshed nightly via ERA5. NDVI — the satellite measure of crop canopy health — updated every 5 days via Sentinel-2.

*Data sources: ERA5-Land (Copernicus) · Open-Meteo · Sentinel-2 NDVI · MODIS MOD13A2 (15-year history)*

**Stage 3 — Run the ML ensemble with reliability checks**
A **SARIMAX** model captures seasonal crop cycles. An **LSTM** network learns non-linear weather-yield relationships from 15 years of district-level data. Before any prediction is made, the pipeline runs three automated safeguards:

- **Timeline checkpoint** — validates that yield and weather data are aligned in time, with no gaps, nulls, or unsorted entries. Raises a hard error if misaligned, preventing silent degradation.
- **Drift detection** — checks every incoming feature against IQR-derived bounds from the training distribution. Flags anomalies so downstream consumers know when predictions are operating outside their trained range.
- **Uncertainty quantification** — every crop result carries a full uncertainty band (min/median/max yield kg/ha), a calibrated probability dampened by regional variability, and a human-readable anomaly flag.

The ML feature vector now includes the soil biological health layer:
- `soil_organic_carbon_pct` — organic carbon % as a proxy for microbial biomass capacity
- `soil_ec_ds_m` — electrical conductivity; detects salinity-induced nutrient lock
- `climate_anomaly_trend_mm` — 5-year vs 30-year rainfall deviation; captures drying/wetting regime shifts
- `climate_temp_anomaly_c` — temperature drift from baseline; captures warming impact on crop suitability

Every prediction result now includes a **fertilizer sufficiency flag** and a **soil health index** (0–1 composite). When the system detects that NPK is adequate or excessive but organic carbon is low or declining, it raises a **biological collapse risk** flag — the hallmark of over-fertilization induced microbial degradation, one of the leading drivers of declining yield per unit of input across Indian farm land.

*Training data: APY Portal 1966–2026 (MoA&FW) · ICRISAT VDSA village-level data*

**Stage 4 — Generate agronomic reasoning**
OpenAI's o3 reasoning model explains *why* each top crop was recommended — covering soil fit, weather risk, market timing, and sowing advice — in plain English a field agent or farmer can act on immediately.

---

### Capability 2 — Grid Farming Planner *(new)*

A fundamentally different planning mode. Instead of advising individual farms independently, the Grid Farming Planner treats a cluster of farms as a **portfolio** and allocates crops across the group to optimise collective outcomes.

**Why this matters:**
- Individual-optimal decisions aggregate into district-level gluts or deficits
- Water-intensive crops (paddy) and water-efficient crops (millets, pulses) need to be balanced at the system level, not the farm level
- Domestic and export deficits in pulses, oilseeds, and coarse grains require directed area allocation — something market signals alone can't deliver
- FPO-level and government-level planners need a portfolio tool, not just a point recommendation

**How it works:**
The planner scores each candidate crop for each grid cell using a multi-objective weighted function:

```
score = 0.30 × net_margin
      − 0.20 × risk_penalty
      − 0.15 × water_intensity
      + 0.15 × market_access
      + 0.10 × diversity_resilience
      + 0.10 × deficit_priority
```

A constrained allocator then assigns area, respecting:
- No single crop exceeds 60% of grid area (concentration cap)
- At least 2 crops must be selected per grid (diversity floor)
- Each selected crop gets at least 15% of area (minimum viable share)
- Optional hard water budget per grid
- Selected crops must be **complementary, not contradictory** — crops with conflicting soil, water, or seasonal requirements are excluded from co-allocation in the same grid (e.g. paddy and chickpea are not co-allocated in a water-scarce kharif grid)

**Output:**
```json
{
  "grid_id": "G1",
  "allocations": [
    {"crop_name": "Pulses", "area_ha": 60.0, "score": 0.40, "expected_margin_inr": 5100000},
    {"crop_name": "Millet",  "area_ha": 40.0, "score": 0.25, "expected_margin_inr": 3390000}
  ],
  "summary": {
    "total_area_ha": 100.0,
    "total_expected_margin_inr": 8490000,
    "total_water_m3": 348000,
    "crop_count": 2
  }
}
```

---

### Capability 3 — Ecosystem Drift Analysis *(new)*

The hardest agricultural problem to solve is not "what should I grow today" — it is "what is happening to the land underneath me, where is it going, and what do I do about it."

Individual micro-changes — a 0.02% annual drop in organic carbon, a 15mm/year decline in seasonal rainfall vs the 30-year mean, a slow rise in soil salinity — are each invisible in isolation. Their cumulative effect over 10–15 seasons is a completely transformed ecosystem: different viable crops, different yield ceilings, different water requirements, different fertilizer responses.

GeoHarvestAI detects this drift using **CUSUM (Cumulative Sum) control charts** — a statistical technique from industrial process control that is purpose-built to detect sustained shifts below the noise threshold of ordinary monitoring. CUSUM accumulates evidence across seasons rather than evaluating each season independently.

**Five ecosystem dimensions monitored:**
| Indicator | Degradation signal | Source |
|---|---|---|
| Organic carbon % | Declining — biological capacity reducing | Soil Health Card (multi-year) |
| Electrical conductivity | Rising — salinity accumulating | Soil Health Card (multi-year) |
| Rainfall anomaly | More negative — ecosystem drying vs 30yr mean | ERA5 5yr rolling vs 1991–2020 |
| Temperature anomaly | More positive — sowing window compressing | ERA5 5yr rolling vs 1991–2020 |
| NDVI trend | Declining — canopy health degrading | MODIS 16-year history |

**What the system produces:**
- **Ecosystem health score** (0–1 composite) — declining score = degrading ecosystem
- **Health velocity** — `fast_decline` / `moderate_decline` / `stable` / `recovering`
- **Primary stressor** — which indicator is driving the most change
- **6-season projection** — where each indicator is heading at the current trend rate, with ICAR critical threshold ETA
- **Priority repair interventions** — ranked, evidence-based agronomic actions (e.g. "reduce inorganic fertilizer 30%, introduce chickpea intercropping — mechanism: restores Rhizobium nitrogen fixation suppressed by excess N")
- **Crop viability assessment** — which crops are viable *now*, which will still be viable in 6 seasons, which to phase out, which soil-restorative crops should be introduced to actively reverse the drift

**Example output for a degrading hex cell:**
```json
{
  "ecosystem_health_score": 0.38,
  "health_velocity": "moderate_decline",
  "primary_stressor": "organic_carbon_decline",
  "seasons_to_critical": 4,
  "drift_narrative": "Concurrent degradation detected across organic carbon and rainfall anomaly. The ecosystem health composite is 0.38 and moderate_decline. Co-occurring stressors typically amplify each other — soil biology collapse accelerates when combined with rainfall deficit.",
  "repair_summary": "Priority: address organic_carbon_decline. Soil-restorative pulse crops and organic matter addition are highest-ROI. Reduce inorganic fertilizer to stop amplifying the stressor.",
  "soil_restorative_crops": ["Chickpea", "Pigeon Pea", "Cowpea"],
  "crops_at_risk": ["Rice", "Sugarcane"],
  "crops_to_phase_in": ["Pearl Millet", "Finger Millet", "Sorghum"]
}
```

This output feeds directly into Stage 4 (LLM reasoning), which synthesises it into the final agronomic explanation the field agent sees.

---

## What the API Returns (per-field)

Every recommendation now includes reliability metadata:

```json
{
  "region_code": "IN",
  "season": "kharif_2026",
  "h3_hex": "8765b4a4fffffff",
  "recommendations": [
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
  ],
  "reasoning": "Wheat is strongly suited to this location's Cwa climate and loam texture. The 42mm rainfall over 7 days provides adequate soil moisture for germination...",
  "ndvi_freshness_days": 3
}
```

| Field | Description |
|---|---|
| `probability` | Calibrated probability, dampened by regional variability from drift report |
| `yield_min/median/max_kg_ha` | Uncertainty band from 15-year historical variability profile |
| `uncertainty_band_pct` | Band width as % of median — surfaces model confidence to downstream users |
| `anomaly_flag` | `true` if any input feature falls outside its IQR-derived training bounds |
| `anomaly_reason` | Which feature triggered the anomaly and by how much |
| `fertilizer_sufficiency` | `deficient` / `sufficient` / `excessive` — per ICAR critical limits |
| `soil_health_index` | 0.0–1.0 composite from organic carbon %, EC, and OC trend direction |
| `biological_collapse_risk` | `true` when NPK is adequate but organic carbon is low/declining (over-fertilization signal) |

---

## Data Sources

| Dataset | What it powers | Source | Cost |
|---|---|---|---|
| SoilGrids v2 | Soil NPK, pH, texture — 250 m | ISRIC | Free |
| SRTM GL1 DEM | Elevation + slope — 30 m | NASA | Free |
| Beck Köppen-Geiger | Climate zone — 1 km | figshare / Nature | Free |
| ERA5-Land | 15-year daily weather history | Copernicus / ECMWF | Free |
| Open-Meteo | Real-time weather (daily) | Open-Meteo.com | Free |
| Sentinel-2 NDVI | Crop health — 10 m, 5-day | ESA / Sentinel Hub | API key |
| MODIS MOD13A2 | 16-year NDVI history | NASA EarthData | Free |
| APY Portal | District × season × crop yield 1966–2026 | Govt of India | Free |
| ICRISAT VDSA | Village-level yield for LSTM fine-tuning | ICRISAT | Free (registration) |

**The vast majority of data powering this system is freely available from public sources.** No proprietary data lock-in.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **API** | FastAPI · Python 3.11 · Pydantic V2 · async throughout |
| **Agent pipeline** | LangGraph (4-node sequential graph) · LangChain LCEL |
| **Grid Farming Planner** | Multi-objective weighted scorer · constrained area allocator |
| **Ecosystem Drift Engine** | CUSUM control charts (Page 1954) · 6-season linear projection · ICAR threshold ETA |
| **Repair Recommender** | Priority-ranked rule engine · evidence-based agronomic interventions |
| **Crop Viability Mapper** | 20-crop ICAR tolerance profile lookup · soil-restorative crop identification |
| **Spatial database** | PostGIS 16 · H3 hex indexing (resolution 7) · Materialized views |
| **Time-series database** | TimescaleDB · Continuous aggregates · asyncpg |
| **Machine learning** | LSTM (PyTorch) · SARIMAX (statsmodels) · Weighted ensemble |
| **ML reliability** | Drift detection · Timeline checkpoint · Uncertainty quantification · Anomaly flags |
| **AI reasoning** | OpenAI o3 (default) · Model-agnostic — swap any LLM |
| **Training automation** | Readiness gate (APY/weather/NDVI coverage check) · Poll-until-ready loop |
| **Infrastructure** | Docker Compose · Daily refresh worker · JSON structured logging (structlog) |

Everything is **open-source or self-hostable**. The AI reasoning layer is model-agnostic.

---

## Who Uses This — and How

### Government & development banks
Plan MSP procurement, PM-KISAN targeting, and NABARD credit deployment with evidence-based yield forecasts. **Grid Farming Plans** provide the tool to direct area allocation toward national deficit crops — pulses, oilseeds, coarse grains — at the district or block level. On-premise deployment available with full data lineage.

### Agri-finance & crop insurance
Underwrite crop loans and insurance with location-specific yield *ranges* — not just point estimates. The uncertainty band and anomaly flag translate directly into actuarial risk scoring. A 5% reduction in spurious claims on a ₹500Cr book is ₹25Cr in savings.

### Input distribution companies
Plan fertiliser, seed, and pesticide inventory by district and season — driven by AI predictions of what will actually be planted, cross-validated against the Grid Farming Plan for the region.

### Agri supply chain & procurement
Know what volume to expect from which districts before the season starts. The Grid Farming Plan provides an authoritative pre-season forecast by crop and grid — beyond what satellite-only systems can deliver.

### Extension services, FPOs & NGOs
Equip field agents with AI-backed recommendations deliverable in local languages — no agronomist required. Grid plans give FPO-level coordinators a shared crop portfolio to work toward, replacing ad-hoc field-by-field decisions.

### Satellite & GIS platforms
Augment existing earth-observation products with ML-powered crop recommendations as a value-added layer. REST API integration in hours. H3-native output for spatial joins. White-label ready.

---

## Key Numbers

| Metric | Value |
|---|---|
| H3 hex cells covering India | 637,000+ |
| Years of historical weather | 15+ (ERA5-Land) |
| Crops modelled per region | 40+ |
| End-to-end recommendation latency | < 3 seconds |
| Yield history coverage | 1966 – 2026 (APY Portal) |
| Spatial resolution | ~5 km² per recommendation cell |
| NDVI refresh cycle | 5 days (Sentinel-2) |
| Uncertainty fields per prediction | 5 (min/median/max yield, band%, anomaly) |
| Ecosystem indicators monitored | 5 (OC, EC, rainfall, temperature, NDVI) |
| Crops profiled for viability | 20 major Indian crops with ICAR tolerance thresholds |
| Ecosystem projection horizon | 6 seasons (~3 years) |

---

## Integration

```bash
# Per-field recommendation
curl -X POST https://your-deployment/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "lat": 20.5,
    "lon": 78.9,
    "season": "kharif_2026",
    "region_code": "IN",
    "top_n": 5
  }'
```

- **Self-hosted:** Docker Compose, runs on any cloud or on-premise
- **API-first:** Standard REST + JSON — integrates with any platform
- **Extensible:** Add new regions by running the data ingestion pipeline for that geography
- **Secure:** No data leaves your infrastructure unless you choose to use a cloud LLM

---

## Deployment Options

| Option | Description |
|---|---|
| **Hosted SaaS** | API access via key, managed infrastructure, per-call or monthly pricing |
| **On-premise licence** | Full deployment in your own infrastructure, air-gap capable |
| **Source code acquisition** | Full IP transfer — integrate directly into your product stack |
| **White-label** | Branded deployment under your product name |
| **Government pilot** | 90-day district-level pilot — soil ingestion, model training, live recommendations + grid plans for one district, one season |

---

## Contact

**To request a live demo, a technical deep-dive, or a commercial proposal:**

📧 hello@geoharvestai.com

*The live demo is available at your request — we can walk through a real recommendation and grid farming plan for any district in India in a 30-minute call.*

---

*GeoHarvestAI · Precision crop intelligence for Indian agriculture · India · 2026*

|---|---|
| Generic state-level advisories | Ignore district-level soil variation of up to 400% in nitrogen content across 50 km |
| Weather-blind decisions | Sowing choices made without current rainfall deficits or satellite crop health signal |
| No yield forecasting | Buyers and input companies can't plan procurement or credit without district estimates |
| Expertise not scalable | 1 agronomist per 1,000 farmers — expert knowledge can't reach 140M farm households |

---

## What GeoHarvestAI Does

A single API call — with just **latitude, longitude, and season** — triggers a four-stage AI pipeline:

### Stage 1 — Resolve the field's physical fingerprint
Coordinates are mapped to an H3 hex cell (~5 km²). PostGIS returns soil nitrogen, phosphorus, potassium, pH, texture class, elevation, slope, and climate zone — all pre-indexed for sub-millisecond lookup.

**Data sources:** SoilGrids v2 (ISRIC) · SRTM 30m DEM (NASA) · Beck 2018 Köppen-Geiger · ICAR 15-Zone Classification

### Stage 2 — Fetch real-time weather & crop health
7-day rainfall, temperature, and humidity from a TimescaleDB cache refreshed nightly via ERA5. NDVI — the satellite measure of crop canopy health — updated every 5 days via Sentinel-2.

**Data sources:** ERA5-Land (Copernicus) · Open-Meteo · Sentinel-2 NDVI · MODIS MOD13A2 (15-year history)

### Stage 3 — Run the ML ensemble
A **SARIMAX** model captures seasonal crop cycles. An **LSTM** network learns non-linear weather-yield relationships from 15 years of district-level data. Both are ensembled into a confidence score and yield estimate (kg/ha) for each crop.

**Training data:** APY Portal 1966–2026 (MoA&FW) · ICRISAT VDSA village-level data

### Stage 4 — Generate agronomic reasoning
OpenAI's o3 reasoning model explains *why* each top crop was recommended — covering soil fit, weather risk, market timing, and sowing advice — in plain English a field agent or farmer can act on immediately.

---

## What the API Returns

Every response includes:

```json
{
  "region_code": "IN",
  "season": "kharif_2026",
  "h3_hex": "8765b4a4fffffff",
  "recommendations": [
    {
      "crop_name": "Wheat",
      "confidence": 0.87,
      "yield_estimate_kg_ha": 3420,
      "model_used": "ensemble"
    }
  ],
  "reasoning": "Wheat is strongly suited to this location's Cwa climate and loam texture. The 42mm rainfall over 7 days provides adequate soil moisture for germination...",
  "ndvi_freshness_days": 3
}
```

| Field | Description |
|---|---|
| `recommendations` | Up to 20 crops ranked by confidence, each with yield estimate and model attribution |
| `reasoning` | 150–250 word agronomic explanation in plain English |
| `h3_hex` | Spatial index for downstream mapping and aggregation |
| `ndvi_freshness_days` | Transparency on how current the satellite data is |

---

## Data Sources

| Dataset | What it powers | Source | Cost |
|---|---|---|---|
| SoilGrids v2 | Soil NPK, pH, texture — 250 m | ISRIC | Free |
| SRTM GL1 DEM | Elevation + slope — 30 m | NASA | Free |
| Beck Köppen-Geiger | Climate zone — 1 km | figshare / Nature | Free |
| ERA5-Land | 15-year daily weather history | Copernicus / ECMWF | Free |
| Open-Meteo | Real-time weather (daily) | Open-Meteo.com | Free |
| Sentinel-2 NDVI | Crop health — 10 m, 5-day | ESA / Sentinel Hub | API key |
| MODIS MOD13A2 | 16-year NDVI history | NASA EarthData | Free |
| APY Portal | District × season × crop yield 1966–2026 | Govt of India | Free |
| ICRISAT VDSA | Village-level yield for LSTM fine-tuning | ICRISAT | Free (registration) |

**The vast majority of the data powering this system is freely available from public sources.** No proprietary data lock-in.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **API** | FastAPI · Python 3.11 · Pydantic V2 · async throughout |
| **Agent pipeline** | LangGraph (4-node sequential graph) · LangChain LCEL |
| **Spatial database** | PostGIS 16 · H3 hex indexing · Materialized views |
| **Time-series database** | TimescaleDB · Continuous aggregates · asyncpg |
| **Machine learning** | LSTM (PyTorch) · SARIMAX (statsmodels) · Weighted ensemble |
| **AI reasoning** | OpenAI o3 (default) · Model-agnostic — swap any LLM |
| **Infrastructure** | Docker Compose · Daily refresh worker · JSON structured logging |

Everything is **open-source or self-hostable**. The AI reasoning layer is model-agnostic.

---

## Who Uses This — and How

### Agri-finance & crop insurance
Underwrite crop loans and insurance with location-specific yield estimates instead of district averages. Risk scoring per field, season-adjusted credit limits, automated early-warning for yield shortfalls.

### Input distribution companies
Plan fertiliser, seed, and pesticide inventory by district and season — driven by AI predictions of what will actually be planted. Soil-matched fertiliser and seed variety recommendations.

### Agri supply chain & procurement
Know what volume to expect from which districts before the season starts. Pre-season volume forecasts by crop, farmer advisory apps, FPO-level aggregation dashboards.

### Extension services & NGOs
Equip field agents with AI-backed recommendations deliverable in local languages — no agronomist required. Backend for field agent mobile apps, offline-capable district pre-computation.

### Satellite & GIS platforms
Augment existing earth-observation products with ML-powered crop recommendations as a value-added layer. REST API integration in hours, H3-native output for spatial joins, white-label ready.

### Government & development banks
Support MSP procurement planning, PM-KISAN targeting, and NABARD credit deployment with evidence-based yield forecasts. On-premise deployment available with full data lineage.

---

## Key Numbers

| Metric | Value |
|---|---|
| H3 hex cells covering India | 637,000+ |
| Years of historical weather | 15+ (ERA5-Land) |
| Crops modelled per region | 40+ |
| End-to-end recommendation latency | < 3 seconds |
| Yield history coverage | 1966 – 2026 (APY Portal) |
| Spatial resolution | ~5 km² per recommendation cell |
| NDVI refresh cycle | 5 days (Sentinel-2) |

---

## Integration

```bash
# Single POST request
curl -X POST https://your-deployment/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "lat": 20.5,
    "lon": 78.9,
    "season": "kharif_2026",
    "region_code": "IN",
    "top_n": 5
  }'
```

- **Self-hosted:** Docker Compose, runs on any cloud or on-premise
- **API-first:** Standard REST + JSON — integrates with any platform
- **Extensible:** Add new regions by running the data ingestion pipeline for that geography
- **Secure:** No data leaves your infrastructure unless you choose to use a cloud LLM

---

## Deployment Options

| Option | Description |
|---|---|
| **Hosted SaaS** | API access via key, managed infrastructure, per-call or monthly pricing |
| **On-premise licence** | Full deployment in your own infrastructure, air-gap capable |
| **Source code acquisition** | Full IP transfer — integrate directly into your product stack |
| **White-label** | Branded deployment under your product name |

---

## Contact

**To request a live demo, a technical deep-dive, or a commercial proposal:**

📧 hello@geoharvestai.com

*The live demo is available at your request — we can walk through a real recommendation for any district in India in a 30-minute call.*

---

*GeoHarvestAI · Precision crop intelligence for Indian agriculture · India · 2026*
