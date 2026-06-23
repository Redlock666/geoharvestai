-- ============================================================
-- GeoHarvestAI — TimescaleDB Schema
-- Auto-executed by Docker Compose on first timescaledb container start.
-- All time-series tables are hypertables partitioned on 'time'.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Daily Weather (ERA5-Land + Open-Meteo) ────────────────────────────────
-- One row per hex per day.
-- Historical data: ERA5-Land via CDS API (ingest_era5.py)
-- Real-time refresh: Open-Meteo (daily_refresh.py)
CREATE TABLE IF NOT EXISTS weather_obs (
    time          TIMESTAMPTZ NOT NULL,
    hex_id        TEXT        NOT NULL,
    region_code   TEXT        NOT NULL,
    rainfall_mm   REAL,                  -- total precipitation (mm/day)
    temp_avg_c    REAL,                  -- 2 m temperature mean  (°C)
    temp_min_c    REAL,                  -- 2 m temperature min   (°C)
    temp_max_c    REAL,                  -- 2 m temperature max   (°C)
    humidity_pct  REAL,                  -- relative humidity     (%)
    wind_speed_ms REAL,                  -- wind speed at 10 m    (m/s)
    source        TEXT        NOT NULL DEFAULT 'era5_land'
);

SELECT create_hypertable('weather_obs', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_weather_hex_time    ON weather_obs (hex_id,      time DESC);
CREATE INDEX IF NOT EXISTS idx_weather_region_time ON weather_obs (region_code, time DESC);

-- Continuous aggregate: 7-day rolling sums/averages.
-- WeatherAgentService queries this instead of the raw table.
CREATE MATERIALIZED VIEW IF NOT EXISTS weather_7d
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('7 days', time) AS week,
    hex_id,
    region_code,
    SUM(rainfall_mm)            AS rainfall_7d_mm,
    AVG(temp_avg_c)             AS temp_avg_c,
    MIN(temp_min_c)             AS temp_min_c,
    MAX(temp_max_c)             AS temp_max_c,
    AVG(humidity_pct)           AS humidity_pct,
    AVG(wind_speed_ms)          AS wind_speed_ms
FROM weather_obs
GROUP BY 1, 2, 3
WITH NO DATA;

-- ── NDVI (MODIS MOD13A2 history + Sentinel-2 live) ───────────────────────
-- 16-day MODIS composites for ML training history (2010–present).
-- Replaced by Sentinel-2 5-day revisit for real-time use.
CREATE TABLE IF NOT EXISTS ndvi_obs (
    time            TIMESTAMPTZ NOT NULL,
    hex_id          TEXT        NOT NULL,
    region_code     TEXT        NOT NULL,
    ndvi            REAL        CHECK (ndvi BETWEEN -1 AND 1),
    cloud_cover_pct REAL,
    source          TEXT        NOT NULL DEFAULT 'modis_mod13a2'  -- or 'sentinel2'
);

SELECT create_hypertable('ndvi_obs', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ndvi_hex_time    ON ndvi_obs (hex_id,      time DESC);
CREATE INDEX IF NOT EXISTS idx_ndvi_region_time ON ndvi_obs (region_code, time DESC);

-- ── Crop Yield History — APY Portal (MoA&FW) ─────────────────────────────
-- PRIMARY ML TRAINING TARGET.
-- District × season × crop × year granularity.
-- Source: aps.dac.gov.in/APY — coverage 1966–present, all Indian states.
-- Populated by: scripts/ingest_apy.py
--
-- 'time' encodes the harvest date derived from season + year:
--   kharif YYYY-YY → YYYY-10-01  (Oct harvest)
--   rabi   YYYY-YY → YYYY+1-04-01 (Apr harvest)
--   zaid   YYYY-YY → YYYY+1-07-01 (Jul harvest)
CREATE TABLE IF NOT EXISTS crop_yield_obs (
    time         TIMESTAMPTZ NOT NULL,
    region_code  TEXT        NOT NULL,   -- 'IN' for India
    state        TEXT        NOT NULL,   -- state name (uppercase, as in APY CSV)
    district     TEXT        NOT NULL,   -- district name (uppercase)
    crop_name    TEXT        NOT NULL,   -- crop name (title-case normalised)
    season       TEXT        NOT NULL,   -- 'kharif' | 'rabi' | 'zaid' | 'whole_year'
    apy_year     TEXT        NOT NULL,   -- raw year string from APY, e.g. '2021-22'
    area_ha      REAL,                   -- area under cultivation (hectares)
    production_t REAL,                   -- total production (metric tonnes)
    yield_kg_ha  REAL,                   -- derived: production_t * 1000 / area_ha
    source       TEXT        NOT NULL DEFAULT 'apy_moa'
);

SELECT create_hypertable('crop_yield_obs', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_yield_region_crop   ON crop_yield_obs (region_code, crop_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_yield_district      ON crop_yield_obs (state, district, time DESC);
CREATE INDEX IF NOT EXISTS idx_yield_season_crop   ON crop_yield_obs (season, crop_name);

-- ── Ingestion audit log ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_log (
    id            BIGSERIAL   PRIMARY KEY,
    script        TEXT        NOT NULL,
    region_code   TEXT        NOT NULL,
    rows_inserted BIGINT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT        NOT NULL DEFAULT 'running',
    error_msg     TEXT
);
