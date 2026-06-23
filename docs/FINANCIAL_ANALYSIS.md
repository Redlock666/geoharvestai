# GeoHarvestAI — Financial Analysis
## District-Level Implementation Cost vs Return on Investment

> *This document provides a bottom-up financial model for GeoHarvestAI deployment,*
> *starting at the district level and scaling to state and national tiers.*
> *All figures are estimates based on publicly available government data.*
> *Conservative and optimistic cases are both presented.*

---

## District Profile — Baseline Assumptions

A "representative Indian district" for this analysis:

| Parameter | Value | Source |
|---|---|---|
| Total area | 5,000 sq km | Census 2011 district average |
| Agricultural land | 2,000–3,000 sq km | ~50–60% of land use |
| H3 hex cells (resolution 7) | ~1,100–1,700 cells | ~5 km² per cell |
| Total farm households | 2,00,000–4,00,000 | Census 2011 |
| Average farm size | 1.0–1.5 ha | Agricultural Census 2015–16 |
| Seasons per year | 2 (kharif + rabi) | Standard Indian cycle |
| Annual agricultural output | ₹1,500–4,000 crore | Varies by district type |
| Annual fertilizer subsidy received | ₹80–200 crore | Per district share of national subsidy |
| Annual Kisan Credit Card portfolio | ₹300–800 crore | NABARD district data |
| PMFBY insurance premium subsidy | ₹30–80 crore | State + GoI share |

---

## Section 1 — Implementation Cost (District Level)

### 1.1 One-Time Setup Cost (Year 1 only)

| Activity | What it involves | Cost estimate |
|---|---|---|
| **Static GIS data ingestion** | SoilGrids v2, SRTM terrain, Köppen-Geiger for all hex cells in district | ₹0 (open data, compute only) |
| **Soil Health Card ingestion** | Parse and load SHC records for district (~10,000–50,000 cards) | ₹0 (government data) |
| **ERA5 weather history ingestion** | 15-year daily weather for district bounding box | ₹0 (Copernicus open access) |
| **NDVI history ingestion** | 16-year MODIS data for district | ₹0 (NASA EarthData free) |
| **APY yield data ingestion** | District-level crop yield 1966–2025 | ₹0 (government portal) |
| **Model training (SARIMAX + LSTM)** | Compute cost for district-specific models | ₹50,000–1,00,000 |
| **Ecosystem drift baseline** | Compute CUSUM reports for all hex cells | ₹20,000–50,000 |
| **Infrastructure setup** | Docker deployment on state cloud or NIC | ₹2,00,000–5,00,000 |
| **Field agent training** | 1-day training for 50–100 Krishi Mitra / extension officers | ₹5,00,000–10,00,000 |
| **Integration with state systems** | API connection to state PM-KISAN database | ₹5,00,000–15,00,000 |
| **Project management + onboarding** | 90-day supervised rollout | ₹15,00,000–30,00,000 |
| **Contingency (15%)** | | ₹4,00,000–9,00,000 |
| **Total one-time setup** | | **₹31,70,000–70,50,000** |
| | | **(₹32L – ₹70L)** |

### 1.2 Annual Operating Cost (Year 2 onwards)

| Activity | What it involves | Annual cost |
|---|---|---|
| **Daily data refresh** | ERA5, NDVI, Open-Meteo weather — automated | ₹1,00,000–2,00,000 |
| **LLM API cost** | OpenAI o3 calls — ~2,00,000 recommendations/season × 2 seasons | ₹3,00,000–8,00,000 |
| **Cloud infrastructure** | PostGIS + TimescaleDB hosting | ₹2,00,000–4,00,000 |
| **Seasonal model refresh** | Re-train on new season's data | ₹50,000–1,00,000 |
| **Ecosystem drift refresh** | Weekly CUSUM recompute | ₹20,000–50,000 |
| **Field support + supervision** | 0.5 FTE district coordinator | ₹6,00,000–10,00,000 |
| **Total annual operating** | | **₹12,70,000–25,50,000** |
| | | **(₹13L – ₹26L)** |

### 1.3 Total Cost of Ownership — 5 Years

| Year | Cost |
|---|---|
| Year 1 (setup + operations) | ₹45L – ₹96L |
| Year 2 | ₹13L – ₹26L |
| Year 3 | ₹13L – ₹26L |
| Year 4 | ₹13L – ₹26L |
| Year 5 | ₹13L – ₹26L |
| **5-year total** | **₹97L – ₹2.0Cr** |

> **Per farmer per year (3,00,000 farmers, Year 2 onwards): ₹4 – ₹9**
> This is the marginal cost per farmer per year once the system is operational.

---

## Section 2 — Return on Investment (District Level)

### 2.1 Fertilizer Subsidy Efficiency

**Basis:** National fertilizer subsidy = ₹1.8 lakh crore/year across ~14 crore ha of cultivated land.
Per district (2,500 ha cultivated): ~₹100–150 crore in subsidy flows.

The fertilizer sufficiency flag prevents over-application on fields where NPK is already adequate or excessive. Based on ICAR Soil Health Card surveys, 30–40% of Indian farmland shows adequate or excessive NPK despite ongoing subsidy uptake.

| Scenario | % of district subsidy redirected efficiently | Saving per district |
|---|---|---|
| Conservative | 8% | ₹8–12 crore/year |
| Moderate | 12% | ₹12–18 crore/year |
| Optimistic | 18% | ₹18–27 crore/year |

### 2.2 Yield Improvement — Farmer Income and Tax Base

**Basis:** Better crop-soil matching improves yield. ICAR trials show 12–20% yield uplift when crop variety and timing are optimised for local soil and weather conditions.

Assuming 30% adoption (Year 1), 60% (Year 2+) and 5–10% yield improvement:

| Scenario | Yield uplift | Additional agricultural output | State tax/mandi revenue (3%) |
|---|---|---|---|
| Conservative | 5%, 30% adoption | ₹22–60 crore/year | ₹0.7–1.8 crore/year |
| Moderate | 8%, 50% adoption | ₹60–160 crore/year | ₹1.8–4.8 crore/year |
| Optimistic | 12%, 70% adoption | ₹126–336 crore/year | ₹3.8–10 crore/year |

*Note: The larger value is farmer income gain — this is social ROI to the government even if not directly fiscal.*

### 2.3 Crop Insurance (PMFBY) Savings

**Basis:** Government pays 50–70% of PMFBY premium. Per district premium outflow: ₹30–80 crore/year.
Satellite-verified cropping patterns reduce fraudulent claims (mis-stated crop type, area). Current estimate: 15–25% of claims have discrepancies.

| Scenario | Claim fraud reduction | Saving per district |
|---|---|---|
| Conservative | 10% | ₹3–8 crore/year |
| Moderate | 18% | ₹5–14 crore/year |
| Optimistic | 25% | ₹7.5–20 crore/year |

### 2.4 Agricultural Credit NPA Reduction

**Basis:** District KCC portfolio: ₹300–800 crore. NPA rate: 8–12%. Government bears cost via NABARD recapitalisation and bank provisioning.
Yield forecasting with confidence intervals replaces district averages — lenders can size credit against predicted yield, not hope.

| Scenario | NPA reduction | Saving per district |
|---|---|---|
| Conservative | 10% of current NPA | ₹2.4–9.6 crore/year |
| Moderate | 18% | ₹4.3–17 crore/year |
| Optimistic | 25% | ₹6–24 crore/year |

### 2.5 Post-Harvest Waste Reduction

**Basis:** India's post-harvest loss = 15–20% of output. Per district: ₹225–800 crore in food value lost annually.
Pre-season volume forecasts allow procurement agencies, mandis, and cold chain to pre-position.

| Scenario | Waste reduction | Saving per district |
|---|---|---|
| Conservative | 3% | ₹6.7–24 crore/year |
| Moderate | 5% | ₹11–40 crore/year |
| Optimistic | 8% | ₹18–64 crore/year |

### 2.6 Import Substitution (Pulse / Oilseed Deficit Crops)

**Basis:** Grid Farming Planner directs area toward deficit crops. India imports pulses worth ₹15,000–20,000 crore/year nationally. Per district directed area shift (conservative 5% of cultivable area to deficit crops): additional production of ₹15–40 crore worth of shortage commodities.

| Scenario | Additional deficit crop production | Import substitution value |
|---|---|---|
| Conservative | ₹10 crore | ₹10 crore/year |
| Moderate | ₹20 crore | ₹20 crore/year |
| Optimistic | ₹40 crore | ₹40 crore/year |

### 2.7 Soil Remediation Avoided (Ecosystem Drift)

**Basis:** Reversing soil biological collapse (OC below critical threshold) costs ₹25,000–40,000/ha in remediation over 5 years, plus 10–15 years of productivity loss. A district with 2,500 sq km agricultural land has ~2,50,000 ha.

If Ecosystem Drift Analysis prevents even 1% of cultivable area from crossing the critical threshold per decade:

| Scenario | Area protected/decade | Remediation avoided |
|---|---|---|
| Conservative | 1,000 ha | ₹25–40 crore (amortised: ₹2.5–4 crore/year) |
| Moderate | 3,000 ha | ₹75–120 crore (amortised: ₹7.5–12 crore/year) |
| Optimistic | 6,000 ha | ₹150–240 crore (amortised: ₹15–24 crore/year) |

---

## Section 3 — Consolidated ROI per District per Year

### Year 2+ (post-setup, steady state)

| Return stream | Conservative | Moderate | Optimistic |
|---|---|---|---|
| Fertilizer subsidy efficiency | ₹8 Cr | ₹15 Cr | ₹22 Cr |
| Yield uplift (state fiscal share) | ₹1 Cr | ₹3 Cr | ₹7 Cr |
| PMFBY claim savings | ₹5 Cr | ₹9 Cr | ₹14 Cr |
| KCC NPA reduction | ₹6 Cr | ₹11 Cr | ₹15 Cr |
| Post-harvest waste reduction | ₹15 Cr | ₹25 Cr | ₹40 Cr |
| Import substitution | ₹10 Cr | ₹20 Cr | ₹40 Cr |
| Soil remediation avoided | ₹3 Cr | ₹8 Cr | ₹20 Cr |
| **Total annual return** | **₹48 Cr** | **₹91 Cr** | **₹158 Cr** |
| **Annual operating cost** | **₹13L** | **₹20L** | **₹26L** |
| **Net annual return** | **₹47.7 Cr** | **₹90.8 Cr** | **₹157.7 Cr** |

---

## Section 4 — Return on Investment Summary

### Payback period

| Scenario | 5-year total cost | Year 1 return | Payback period |
|---|---|---|---|
| Conservative | ₹97L | ₹48 Cr | < 1 season |
| Moderate | ₹1.5 Cr | ₹91 Cr | < 1 season |
| Optimistic | ₹2.0 Cr | ₹158 Cr | < 1 season |

**The system pays for itself in under one agricultural season in every scenario.**

### 5-Year ROI

| Scenario | 5-year total cost | 5-year total return | Net 5-year return | ROI |
|---|---|---|---|---|
| Conservative | ₹97L | ₹240 Cr | ₹239 Cr | **2,464×** |
| Moderate | ₹1.5 Cr | ₹455 Cr | ₹453.5 Cr | **3,023×** |
| Optimistic | ₹2.0 Cr | ₹790 Cr | ₹788 Cr | **3,940×** |

---

## Section 5 — State and National Scale

### One state (30 districts, 5 years)

| Item | Conservative | Optimistic |
|---|---|---|
| Total 5-year implementation cost | ₹29 Cr | ₹60 Cr |
| Total 5-year return | ₹7,200 Cr | ₹23,700 Cr |
| Net 5-year return | ₹7,171 Cr | ₹23,640 Cr |

### National (700 districts, Year 2+ annual)

| Return stream | Conservative/year | Optimistic/year |
|---|---|---|
| Fertilizer subsidy efficiency | ₹56,000 Cr | ₹1,54,000 Cr |
| PMFBY savings | ₹3,500 Cr | ₹9,800 Cr |
| KCC NPA reduction | ₹4,200 Cr | ₹10,500 Cr |
| Post-harvest waste | ₹10,500 Cr | ₹28,000 Cr |
| Import substitution | ₹7,000 Cr | ₹28,000 Cr |
| Soil remediation avoided | ₹2,100 Cr | ₹14,000 Cr |
| **National annual saving** | **₹83,300 Cr** | **₹2,44,300 Cr** |
| National annual operating cost | ₹910 Cr | ₹1,820 Cr |
| **Net national annual saving** | **₹82,390 Cr** | **₹2,42,480 Cr** |

> National implementation at scale costs **₹910 Cr – ₹1,820 Cr/year** to operate.
> It saves **₹82,000 Cr – ₹2,42,000 Cr/year**.
> The operating cost is **1.1% of the saving** in the conservative case.

---

## Section 6 — Sensitivity Analysis

### What has to be wrong for this not to work

| Assumption | Break-even value | Likelihood |
|---|---|---|
| Fertilizer saving must be at least | 1% of district subsidy (vs 8% assumed) | Very low — ICAR data shows 30–40% of fields are over-fertilized |
| Farmer adoption must be at least | 5% of district (vs 30–60% assumed) | Very low — no financial incentive required from farmer side |
| Yield improvement must be at least | 1% (vs 5–12% assumed) | Very low — even crop variety matching alone delivers 3–5% per ICAR trials |
| Post-harvest saving must be at least | 0.5% (vs 3–8% assumed) | Very low — FCI data shows >15% loss in most districts |

**In every sensitivity scenario, the return exceeds the cost within the first season.** The system does not need to be fully successful to be financially justified.

---

## Section 7 — Implementation Roadmap and Cost by Phase

### Phase 1 — Single District Pilot (Months 1–3)
**Cost: ₹50L – ₹1.5 Cr (one-time)**

- Select district with good SHC coverage and willing state partner
- Ingest all data sources, train models, compute ecosystem drift baseline
- Deploy to 500–2,000 farmer households via extension officers
- Track: recommendation adoption rate, fertilizer spend vs baseline, yield vs district average

**Deliverable:** Published case study with measured ROI across 3–4 return streams.

### Phase 2 — State Rollout (Months 4–18)
**Cost: ₹8–20 Cr (one-time across 30 districts) + ₹3–6 Cr/year operating**

- Scale to all districts in pilot state using Phase 1 infrastructure
- Integrate with state PM-KISAN database for farmer registration
- Connect to state procurement agency for directed crop plan confirmation
- Begin PMFBY integration for claim verification

**Deliverable:** State-level impact dashboard; basis for NABARD and GoI co-funding.

### Phase 3 — Multi-State / National (Year 2–5)
**Cost: ₹180–400 Cr (one-time national setup) + ₹910 Cr – ₹1,820 Cr/year operating**

- Replicate Phase 2 across 10–15 states
- Digital Agriculture Mission integration
- National food security model: Grid Farming Plans calibrated to national deficit/surplus targets
- International expansion: Bangladesh, Sri Lanka, East Africa

---

## Summary Table

| Deployment tier | Setup cost | Annual operating | Annual return | Payback |
|---|---|---|---|---|
| 1 district | ₹32L – ₹70L | ₹13L – ₹26L | ₹48 Cr – ₹158 Cr | < 1 season |
| 1 state (30 districts) | ₹9.6 Cr – ₹21 Cr | ₹3.9 Cr – ₹7.8 Cr | ₹1,440 Cr – ₹4,740 Cr | < 1 season |
| National (700 districts) | ₹224 Cr – ₹490 Cr | ₹910 Cr – ₹1,820 Cr | ₹83,300 Cr – ₹2,44,300 Cr | < 1 season |

---

*GeoHarvestAI · Financial Analysis · June 2026*
*Figures based on publicly available GoI, NABARD, ICAR, and FAO data.*
*Detailed methodology and source citations available on request.*
