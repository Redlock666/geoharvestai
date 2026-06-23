-- ============================================================
-- GeoHarvestAI — PostGIS Schema
-- Auto-executed by Docker Compose on first db container start.
-- Tables use H3 hex_id (resolution 7 ≈ 5 km²) as the primary key.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Ingestion audit log ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_log (
    id            BIGSERIAL    PRIMARY KEY,
    script        TEXT         NOT NULL,
    region_code   TEXT         NOT NULL,
    rows_inserted BIGINT,
    rows_skipped  BIGINT,
    started_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT         NOT NULL DEFAULT 'running',  -- running | success | error
    error_msg     TEXT
);

-- ── Soil ──────────────────────────────────────────────────────────────────
-- Raw soil composition per H3 hex cell.
-- Source: SoilGrids v2 (ISRIC) — 250 m global raster.
-- Populated by: scripts/ingest_soilgrids.py
--
-- Units stored here are already converted from SoilGrids encoding:
--   nitrogen_g_kg : g/kg  (raw cg/kg ÷ 100)
--   ph            : 0-14  (raw phh2o ÷ 10)
--   clay/sand/silt: %     (raw g/kg ÷ 10)
--   phosphorus, potassium: mg/kg — sourced from Soil Health Cards (SHC);
--                           NULL until ingest_shc.py is run.
CREATE TABLE IF NOT EXISTS soil_raw (
    hex_id           TEXT   PRIMARY KEY,
    nitrogen_g_kg    REAL,
    phosphorus_mg_kg REAL,    -- nullable until SHC data ingested
    potassium_mg_kg  REAL,    -- nullable until SHC data ingested
    ph               REAL,
    clay_pct         REAL,
    sand_pct         REAL,
    silt_pct         REAL,
    texture          TEXT,    -- USDA class derived from clay/sand/silt
    source           TEXT     NOT NULL DEFAULT 'soilgrids_v2',
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GISResolverService._fetch_soil() queries this view.
-- Column names match the query exactly: nitrogen, phosphorus, potassium, ph, texture.
CREATE MATERIALIZED VIEW IF NOT EXISTS soil_by_hex AS
SELECT
    hex_id,
    nitrogen_g_kg                AS nitrogen,
    COALESCE(phosphorus_mg_kg, 0) AS phosphorus,
    COALESCE(potassium_mg_kg, 0)  AS potassium,
    ph,
    texture
FROM soil_raw;

CREATE UNIQUE INDEX IF NOT EXISTS soil_by_hex_pk ON soil_by_hex (hex_id);

-- ── Terrain ───────────────────────────────────────────────────────────────
-- Elevation and slope per H3 hex cell.
-- Source: SRTM GL1 30 m via OpenTopography API.
-- Populated by: scripts/ingest_terrain.py
CREATE TABLE IF NOT EXISTS terrain_raw (
    hex_id      TEXT   PRIMARY KEY,
    elevation_m REAL,
    slope_deg   REAL,
    source      TEXT   NOT NULL DEFAULT 'srtm_gl1_30m',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GISResolverService._fetch_terrain() queries this view.
CREATE MATERIALIZED VIEW IF NOT EXISTS terrain_by_hex AS
SELECT hex_id, elevation_m, slope_deg FROM terrain_raw;

CREATE UNIQUE INDEX IF NOT EXISTS terrain_by_hex_pk ON terrain_by_hex (hex_id);

-- ── Climate Zones ─────────────────────────────────────────────────────────
-- Köppen-Geiger climate classification + ICAR agroclimatic zone per H3 hex.
-- Source: Beck et al. 2018 (1 km GeoTIFF, figshare) for KG codes.
--         ICAR 15-zone classification assigned by geographic lookup.
-- Populated by: scripts/ingest_climate_zones.py
CREATE TABLE IF NOT EXISTS climate_zones_raw (
    hex_id         TEXT     PRIMARY KEY,
    zone_code      TEXT,             -- Köppen-Geiger code  (e.g. 'Aw', 'BSh', 'Cwa')
    icar_zone_id   SMALLINT,         -- ICAR zone 1-15
    icar_zone_name TEXT,
    source         TEXT     NOT NULL DEFAULT 'beck2018_kg_1km',
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GISResolverService._fetch_climate_zone() queries this view.
-- The service expects a single column: zone_code.
CREATE MATERIALIZED VIEW IF NOT EXISTS climate_zones_by_hex AS
SELECT hex_id, zone_code, icar_zone_id, icar_zone_name FROM climate_zones_raw;

CREATE UNIQUE INDEX IF NOT EXISTS climate_zones_by_hex_pk ON climate_zones_by_hex (hex_id);
