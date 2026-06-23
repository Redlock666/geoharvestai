-- ============================================================
-- GeoHarvestAI — Soil Health Card (SHC) Schema
-- Stores biological and micronutrient indicators from India's
-- Soil Health Card scheme (220M+ cards issued by MoA&FW).
--
-- SHC cards capture what SoilGrids cannot: organic carbon,
-- electrical conductivity, sulphur, zinc, iron, and other
-- micronutrients that govern actual nutrient availability to crops
-- independent of raw NPK concentrations.
--
-- Populated by: scripts/ingest_soil_health_cards.py
-- ============================================================

-- ── Soil Health Card raw table ─────────────────────────────────────────────
-- One row per H3 hex cell (resolution 7 ≈ 5 km²).
-- Multiple SHC cards within a hex are aggregated to median values during
-- ingestion. Where multiple survey years exist, the most recent row is
-- used for static fields; npk_trend_direction is computed from all years.
CREATE TABLE IF NOT EXISTS soil_health_raw (
    hex_id                     TEXT     PRIMARY KEY,
    region_code                TEXT     NOT NULL,

    -- Biological health indicators
    organic_carbon_pct         REAL,    -- % OC — key proxy for microbial biomass
    electrical_conductivity_ds_m REAL,  -- dS/m — salinity indicator (>4 = toxic)

    -- Macronutrient availability (available, not total — agronomically meaningful)
    available_n_kg_ha          REAL,    -- kg/ha — Alkaline KMnO4 method
    available_p_kg_ha          REAL,    -- kg/ha — Olsen / Bray-II method
    available_k_kg_ha          REAL,    -- kg/ha — NH4OAc extraction

    -- Micronutrients (mg/kg — DTPA extractable)
    sulphur_mg_kg              REAL,
    zinc_mg_kg                 REAL,
    iron_mg_kg                 REAL,
    copper_mg_kg               REAL,
    manganese_mg_kg            REAL,
    boron_mg_kg                REAL,

    -- Derived trends (computed during ingestion from multi-year SHC data)
    -- Values: 'improving' | 'stable' | 'declining' | 'unknown'
    npk_trend_direction        TEXT     NOT NULL DEFAULT 'unknown',
    organic_carbon_trend       TEXT     NOT NULL DEFAULT 'unknown',

    -- Sufficiency classification (computed during ingestion)
    -- 'sufficient' | 'deficient' | 'excessive' per ICAR critical limits
    n_sufficiency              TEXT     NOT NULL DEFAULT 'unknown',
    p_sufficiency              TEXT     NOT NULL DEFAULT 'unknown',
    k_sufficiency              TEXT     NOT NULL DEFAULT 'unknown',
    oc_sufficiency             TEXT     NOT NULL DEFAULT 'unknown',  -- low/medium/high

    -- Biological collapse flag: NPK sufficient but yield declining = over-fertilization
    -- or microbial degradation
    biological_collapse_risk   BOOLEAN  NOT NULL DEFAULT FALSE,

    survey_year_latest         SMALLINT,          -- most recent SHC survey year
    cards_aggregated           INTEGER  DEFAULT 1, -- number of SHC cards in this hex
    source                     TEXT     NOT NULL DEFAULT 'india_shc_portal',
    ingested_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Index for region-level queries (used by training readiness checker)
CREATE INDEX IF NOT EXISTS soil_health_region_idx
    ON soil_health_raw (region_code);

-- ── Materialized view for GISResolverService ──────────────────────────────
-- GISResolverService._fetch_soil_health() queries this view.
-- Only expose columns needed for ML features + reasoning.
CREATE MATERIALIZED VIEW IF NOT EXISTS soil_health_by_hex AS
SELECT
    hex_id,
    organic_carbon_pct,
    electrical_conductivity_ds_m,
    available_n_kg_ha,
    available_p_kg_ha,
    available_k_kg_ha,
    sulphur_mg_kg,
    zinc_mg_kg,
    iron_mg_kg,
    npk_trend_direction,
    organic_carbon_trend,
    n_sufficiency,
    p_sufficiency,
    k_sufficiency,
    oc_sufficiency,
    biological_collapse_risk
FROM soil_health_raw;

CREATE UNIQUE INDEX IF NOT EXISTS soil_health_by_hex_pk ON soil_health_by_hex (hex_id);

-- ── Climate anomaly trend table ────────────────────────────────────────────
-- Stores 5-year rolling rainfall deviation vs 30-year baseline per hex cell.
-- Computed from ERA5-Land during nightly refresh worker.
-- Used to detect climate regime shifts (not just point-in-time weather).
--
-- Populated by: scripts/daily_refresh.py (ERA5 trend computation step)
CREATE TABLE IF NOT EXISTS climate_trend_raw (
    hex_id                      TEXT     PRIMARY KEY,
    region_code                 TEXT     NOT NULL,

    -- Rainfall trend
    baseline_rainfall_mm        REAL,    -- 30-year annual mean (ERA5 1991-2020)
    rolling_5yr_rainfall_mm     REAL,    -- 5-year rolling annual mean (current)
    rainfall_anomaly_mm         REAL,    -- rolling_5yr - baseline (negative = drying)
    rainfall_anomaly_pct        REAL,    -- anomaly as % of baseline

    -- Temperature trend
    baseline_temp_avg_c         REAL,    -- 30-year annual mean temperature
    rolling_5yr_temp_avg_c      REAL,    -- 5-year rolling mean
    temp_anomaly_c              REAL,    -- rolling_5yr - baseline (positive = warming)

    -- Derived regime classification
    -- 'stable' | 'drying' | 'wetting' | 'warming' | 'cooling' | 'extreme_shift'
    climate_regime_shift        TEXT     NOT NULL DEFAULT 'stable',

    computed_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS climate_trend_region_idx
    ON climate_trend_raw (region_code);

CREATE MATERIALIZED VIEW IF NOT EXISTS climate_trend_by_hex AS
SELECT
    hex_id,
    rainfall_anomaly_mm,
    rainfall_anomaly_pct,
    temp_anomaly_c,
    climate_regime_shift
FROM climate_trend_raw;

CREATE UNIQUE INDEX IF NOT EXISTS climate_trend_by_hex_pk ON climate_trend_by_hex (hex_id);
