# ============================================================
# GeoHarvestAI — Developer Makefile
# Run `make help` for a full command reference.
# All ingest/train commands run inside Docker (Python 3.11).
# ============================================================

REGION ?= IN
YEARS  ?= 2010-2025
WAIT_MAX_MINUTES ?= 180
WAIT_INTERVAL_SEC ?= 300

.PHONY: help env db-up db-down download \
        ingest-soil ingest-terrain ingest-climate \
        ingest-era5 ingest-ndvi ingest-apy ingest-shc ingest-all \
	check-train-ready \
        train-sarimax train-lstm train-all \
	train-if-ready \
	train-when-ready \
        demo stop logs build

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  GeoHarvestAI — command reference"
	@echo "  ────────────────────────────────────────────────────────────"
	@echo ""
	@echo "  SETUP"
	@echo "    make env           Copy .env.example → .env (edit it with your API keys)"
	@echo "    make build         Build Docker images"
	@echo ""
	@echo "  DATABASES"
	@echo "    make db-up         Start PostGIS + TimescaleDB (background)"
	@echo "    make db-down       Stop databases"
	@echo "    make logs          Tail all service logs"
	@echo ""
	@echo "  DATA DOWNLOAD  (run once, automatic where possible)"
	@echo "    make download      Download SoilGrids, Köppen-Geiger, SRTM (~600 MB total)"
	@echo "                       ⚠️  APY yield CSV must be downloaded manually — see instructions"
	@echo ""
	@echo "  DATA INGESTION  (run in order after download)"
	@echo "    make ingest-soil    Ingest SoilGrids v2 → PostGIS soil_raw"
	@echo "    make ingest-terrain Ingest SRTM DEM   → PostGIS terrain_raw"
	@echo "    make ingest-climate Ingest Köppen-Geiger → PostGIS climate_zones_raw"
	@echo "    make ingest-era5    Ingest ERA5-Land historical weather → TimescaleDB"
	@echo "    make ingest-ndvi    Ingest MODIS MOD13A2 NDVI history → TimescaleDB"
	@echo "    make ingest-apy     Ingest APY crop yield CSV → TimescaleDB (ML training target)"
	@echo "    make ingest-shc     Ingest Soil Health Card CSV → PostGIS (biological health layer)"

compute-ecosystem-drift:
	@echo "▶  Computing ecosystem drift reports per hex cell..."
	docker compose run --rm train \
	  python scripts/compute_ecosystem_drift.py --region $(REGION)
	@echo "    make ingest-all     Run all ingest steps in order (except ERA5/NDVI/APY)"
	@echo ""
	@echo "  MODEL TRAINING"
	@echo "    make check-train-ready  Validate APY/weather/NDVI coverage before training"
	@echo "    make train-sarimax  Fit SARIMAX per-crop models"
	@echo "    make train-lstm     Train LSTM (requires ≥3 years of data)"
	@echo "    make train-all      Run SARIMAX then LSTM"
	@echo "    make train-if-ready Run readiness check, then train-all if PASS"
	@echo "    make train-when-ready  Poll readiness until PASS, then train-all"
	@echo ""
	@echo "  RUN"
	@echo "    make demo          Start full stack — open http://localhost:8000"
	@echo "    make stop          Stop all services"
	@echo ""
	@echo "  Options (pass as environment overrides):"
	@echo "    REGION=IN   (default)   Region code"
	@echo "    YEARS=2010-2025         Year range for ERA5/NDVI ingest"
	@echo "    APY_FILE=data/raw/apy/apy_india_all.csv"
	@echo "    WAIT_MAX_MINUTES=180    Max wait before auto-train timeout"
	@echo "    WAIT_INTERVAL_SEC=300   Poll interval for readiness checks"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
env:
	@if [ -f .env ]; then echo "⚠️  .env already exists — skipping"; else cp .env.example .env && echo "✅ Created .env — fill in your API keys before proceeding"; fi

build:
	docker compose build

# ── Databases ─────────────────────────────────────────────────────────────
db-up:
	@echo "▶  Starting PostGIS + TimescaleDB..."
	docker compose up -d db timescaledb
	@echo "⏳ Waiting for databases to be healthy..."
	@docker compose exec db    pg_isready -U geo -d geoharvestai   -t 60 > /dev/null 2>&1 || (echo "❌ PostGIS not ready"; exit 1)
	@docker compose exec timescaledb pg_isready -U geo -d geoharvestai_ts -t 60 > /dev/null 2>&1 || (echo "❌ TimescaleDB not ready"; exit 1)
	@echo "✅ Databases are healthy"

db-down:
	docker compose stop db timescaledb

logs:
	docker compose logs -f

# ── Download ──────────────────────────────────────────────────────────────
download:
	@echo "▶  Downloading static GIS data..."
	@mkdir -p data/raw
	bash scripts/download_india_data.sh data/raw

# ── Ingest ────────────────────────────────────────────────────────────────
ingest-soil:
	@echo "▶  Ingesting SoilGrids → PostGIS..."
	docker compose run --rm ingest \
	  python scripts/ingest_soilgrids.py --region $(REGION) --data-dir /app/data/raw

ingest-terrain:
	@echo "▶  Ingesting SRTM terrain → PostGIS..."
	docker compose run --rm ingest \
	  python scripts/ingest_terrain.py --region $(REGION) --data-dir /app/data/raw

ingest-climate:
	@echo "▶  Ingesting Köppen-Geiger climate zones → PostGIS..."
	docker compose run --rm ingest \
	  python scripts/ingest_climate_zones.py --region $(REGION) --data-dir /app/data/raw

ingest-era5:
	@echo "▶  Ingesting ERA5-Land weather → TimescaleDB (this takes ~20 min)..."
	docker compose run --rm ingest \
	  python scripts/ingest_era5.py --region $(REGION) --years $(YEARS)

ingest-ndvi:
	@echo "▶  Ingesting MODIS NDVI → TimescaleDB (this takes ~30 min)..."
	docker compose run --rm ingest \
	  python scripts/ingest_ndvi_modis.py --region $(REGION) --years $(YEARS)

ingest-apy:
	$(eval APY_FILE ?= data/raw/apy/apy_india_all.csv)
	@if [ ! -f "$(APY_FILE)" ]; then \
	  echo "❌ APY file not found at $(APY_FILE)"; \
	  echo "   Download it manually from: https://aps.dac.gov.in/APY/Public_Report1.aspx"; \
	  echo "   Select All States | All Crops | All Seasons | Years 2001-2025"; \
	  echo "   Export CSV → save to $(APY_FILE)"; \
	  exit 1; \
	fi
	@echo "▶  Ingesting APY crop yields → TimescaleDB..."
	docker compose run --rm ingest \
	  python scripts/ingest_apy.py --region $(REGION) --file /app/$(APY_FILE)

ingest-shc:
	$(eval SHC_FILE ?= data/raw/shc/shc_india.csv)
	@if [ ! -f "$(SHC_FILE)" ]; then \
	  echo "❌ SHC file not found at $(SHC_FILE)"; \
	  echo "   Download from: https://soilhealth.dac.gov.in"; \
	  echo "   Or use state-level SHC export CSVs from your state agriculture department."; \
	  echo "   Minimum required columns: latitude, longitude, organic_carbon (or oc), ec"; \
	  exit 1; \
	fi
	@echo "▶  Ingesting Soil Health Card data → PostGIS (biological health layer)..."
	docker compose run --rm ingest \
	  python scripts/ingest_soil_health_cards.py --region $(REGION) --file /app/$(SHC_FILE)

# Static GIS only (no API keys needed beyond OpenTopography for terrain)
ingest-all: ingest-soil ingest-terrain ingest-climate
	@echo "✅ Static GIS ingestion complete."
	@echo "   Next: make ingest-era5  (needs CDSAPI_KEY)"
	@echo "         make ingest-ndvi  (needs EARTHDATA_USERNAME + PASSWORD)"
	@echo "         make ingest-apy   (needs manual CSV download)"

# ── Train ─────────────────────────────────────────────────────────────────
check-train-ready:
	@echo "▶  Checking training data readiness..."
	docker compose run --rm train \
	  python scripts/check_training_readiness.py --region $(REGION)

train-sarimax:
	@echo "▶  Training SARIMAX per-crop models..."
	docker compose run --rm train \
	  python ml/train/train_sarimax.py --region $(REGION)

train-lstm:
	@echo "▶  Training LSTM model (requires ≥3 years data)..."
	docker compose run --rm train \
	  python ml/train/train_lstm.py --region $(REGION)

train-all: train-sarimax train-lstm
	@echo "✅ Model training complete. Artifacts in ml/artifacts/$(REGION)/"

train-if-ready: check-train-ready train-all
	@echo "✅ Readiness passed and training completed."

train-when-ready:
	@echo "▶  Waiting for training readiness (max $(WAIT_MAX_MINUTES) min, interval $(WAIT_INTERVAL_SEC)s)..."
	@elapsed=0; \
	max_wait=`expr $(WAIT_MAX_MINUTES) \* 60`; \
	while true; do \
	  if docker compose run --rm train python scripts/check_training_readiness.py --region $(REGION); then \
	    echo "✅ Readiness passed. Starting training..."; \
	    $(MAKE) train-all REGION=$(REGION); \
	    break; \
	  fi; \
	  if [ $$elapsed -ge $$max_wait ]; then \
	    echo "❌ Timed out waiting for readiness after $(WAIT_MAX_MINUTES) minutes."; \
	    exit 1; \
	  fi; \
	  echo "⏳ Not ready yet. Retrying in $(WAIT_INTERVAL_SEC)s..."; \
	  sleep $(WAIT_INTERVAL_SEC); \
	  elapsed=`expr $$elapsed + $(WAIT_INTERVAL_SEC)`; \
	done

# ── Run demo ─────────────────────────────────────────────────────────────
demo:
	@echo "▶  Starting GeoHarvestAI full stack..."
	docker compose up --build
	@echo "🌾 Open http://localhost:8000"

stop:
	docker compose down
