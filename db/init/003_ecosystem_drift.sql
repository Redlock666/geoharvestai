-- ============================================================
-- GeoHarvestAI — Ecosystem Drift Schema
-- Stores pre-computed ecosystem drift analysis per H3 hex cell.
--
-- Ecosystem drift is the slow accumulation of micro-changes across
-- soil biology, climate patterns, and vegetation health that —
-- individually below any alert threshold — collectively transform
-- the agricultural ecosystem over seasons and years.
--
-- This table is written by:  scripts/compute_ecosystem_drift.py
-- Read by:                   services/ecosystem_analyzer.py
-- Refreshed by:              daily_refresh.py (weekly pass)
-- ============================================================

CREATE TABLE IF NOT EXISTS ecosystem_drift_raw (
    hex_id              TEXT     PRIMARY KEY,
    region_code         TEXT     NOT NULL,

    -- ── Composite ecosystem health ──────────────────────────────────────
    -- 0.0 = severely degraded, 1.0 = excellent biological health
    -- Weighted composite: OC(0.30) + rainfall(0.25) + NDVI(0.20) +
    --                     NPK_balance(0.15) + EC_inverse(0.10)
    ecosystem_health_score      REAL,
    health_score_prev_season    REAL,
    health_velocity             TEXT NOT NULL DEFAULT 'stable',
    -- 'fast_decline' (>0.1 drop/season) | 'moderate_decline' | 'stable'
    -- | 'slow_recovery' | 'recovering'

    -- ── CUSUM signals per indicator ──────────────────────────────────────
    -- CUSUM (Page 1954): detects sustained shift below moving-average noise.
    -- Signal: 'degrading' | 'stable' | 'improving' | 'insufficient_data'
    cusum_oc_signal             TEXT NOT NULL DEFAULT 'insufficient_data',
    cusum_ec_signal             TEXT NOT NULL DEFAULT 'insufficient_data',
    cusum_rainfall_signal       TEXT NOT NULL DEFAULT 'insufficient_data',
    cusum_temp_signal           TEXT NOT NULL DEFAULT 'insufficient_data',
    cusum_ndvi_signal           TEXT NOT NULL DEFAULT 'insufficient_data',
    cusum_yield_signal          TEXT NOT NULL DEFAULT 'insufficient_data',

    -- ── Primary stressor ─────────────────────────────────────────────────
    -- Which indicator is driving the most ecosystem change
    primary_stressor            TEXT,
    -- e.g. 'organic_carbon_decline' | 'drying_trend' | 'salinity_rise'
    --      | 'temperature_warming' | 'vegetation_degradation' | 'yield_collapse'

    -- ── 6-season forward projections ─────────────────────────────────────
    projected_oc_pct            REAL,           -- projected OC% in 6 seasons
    projected_rainfall_anomaly_mm REAL,         -- projected 5yr rainfall deviation
    projected_temp_anomaly_c    REAL,           -- projected temperature deviation
    projected_health_score      REAL,           -- projected composite score in 6 seasons
    seasons_to_critical         SMALLINT,       -- seasons until primary indicator breaches threshold
    -- NULL = no critical threshold breach projected within 12 seasons

    -- ── Repair interventions (ordered by priority) ────────────────────────
    -- JSON array of objects: [{priority, intervention, seasons_to_effect, evidence}]
    repair_interventions        JSONB,

    -- ── Crop guidance ────────────────────────────────────────────────────
    -- JSON arrays of crop names
    viable_crops_current        JSONB,          -- crops viable given current ecosystem state
    viable_crops_projected      JSONB,          -- crops still viable at 6-season projection
    crops_at_risk               JSONB,          -- currently viable but projected to fail
    crops_to_phase_in           JSONB,          -- drought/salt tolerant crops to transition to
    soil_restorative_crops      JSONB,          -- crops that actively reverse the drift (legumes etc.)

    -- ── Narrative summaries (for LLM reasoning context) ──────────────────
    drift_narrative             TEXT,           -- 2-3 sentence: what is happening and why
    repair_summary              TEXT,           -- 2-3 sentence: priority actions
    projection_narrative        TEXT,           -- 2-3 sentence: where the ecosystem is heading

    -- ── Data quality ─────────────────────────────────────────────────────
    -- 'high' (≥5 seasons all indicators) | 'medium' (≥3 seasons some)
    -- | 'low' (1-2 seasons) | 'insufficient' (no time series)
    data_quality                TEXT NOT NULL DEFAULT 'insufficient',
    indicators_with_data        SMALLINT NOT NULL DEFAULT 0,
    seasons_of_data             SMALLINT,       -- min seasons across all available indicators

    computed_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ecosystem_drift_region_idx
    ON ecosystem_drift_raw (region_code);

CREATE INDEX IF NOT EXISTS ecosystem_drift_health_idx
    ON ecosystem_drift_raw (ecosystem_health_score)
    WHERE ecosystem_health_score IS NOT NULL;

-- View for GISResolverService
CREATE MATERIALIZED VIEW IF NOT EXISTS ecosystem_drift_by_hex AS
SELECT
    hex_id,
    ecosystem_health_score,
    health_velocity,
    primary_stressor,
    projected_health_score,
    seasons_to_critical,
    repair_interventions,
    viable_crops_current,
    viable_crops_projected,
    crops_at_risk,
    crops_to_phase_in,
    soil_restorative_crops,
    drift_narrative,
    repair_summary,
    projection_narrative,
    data_quality
FROM ecosystem_drift_raw;

CREATE UNIQUE INDEX IF NOT EXISTS ecosystem_drift_by_hex_pk
    ON ecosystem_drift_by_hex (hex_id);
