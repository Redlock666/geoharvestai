# India Crop Data Fallback Playbook

This guide defines a practical source hierarchy and ingestion mapping when `aps.dac.gov.in` is slow/unreachable.

## 1) Source priority (targets for model training)

### Tier 1 (Preferred)
1. **APS / APY portal** (district-season-crop area/production)
   - https://aps.dac.gov.in/APY/Public_Report1.aspx
2. **DES data portal (official APY-style systems)**
   - https://data.desagri.gov.in/

### Tier 2 (Primary fallback)
3. **Open Government Data India (data.gov.in)**
   - https://www.data.gov.in/
   - Search terms: `crop production district season`, `area production yield`, `foodgrains state district`
4. **State DES / Agriculture department datasets**
   - state portals often publish district/year crop production tables in XLS/CSV/PDF

### Tier 3 (Gap-filling / coarse backup)
5. **FAOSTAT Crop Production (QCL)**
   - https://www.fao.org/faostat/en/#data/QCL
   - Use for aggregate trend checks, not district-level learning target.

---

## 2) Minimum schema required by ingestion

`script/ingest_apy.py` now supports both APS-native and fallback datasets.

### Core required fields
- `crop`
- `year`
- `season`

### Plus one of these target payload forms
- **Form A:** `area_ha` + `production_t`  
- **Form B:** `yield_kg_ha`

### Optional but recommended
- `state`
- `district`

If `state`/`district` are absent, ingest script fills `Unknown`.

---

## 3) Column mapping cheat sheet

The ingest script normalizes many common headers automatically.

### Geographic
- `State Name`, `state_name`, `state` -> `state`
- `District Name`, `district_name`, `district`, `districts` -> `district`

### Crop/time
- `Crop`, `Crop Name`, `crop_name`, `commodity` -> `crop`
- `Year`, `Crop Year`, `crop_year` -> `year`
- `Season` -> `season`

### Area/production
- `Area (in Ha)`, `Area(in Ha)`, `area_in_ha`, `Area`, `Area (ha)` -> `area_ha`
- `Production (in Tonnes)`, `Production(in Tonnes)`, `production_in_tonnes`, `Production` -> `production_t`

### Yield-direct
- `Yield`, `Yield (kg/ha)`, `Yield kg/ha`, `yield_kg_ha` -> `yield_kg_ha`

---

## 4) Quick execution path

1. Put your selected fallback CSV in:
   - `data/raw/apy/`
2. Ingest:
   - `python3 scripts/ingest_apy.py --region IN --file data/raw/apy/<your_file>.csv`
3. Train:
   - `python3 ml/train/train_sarimax.py --region IN`
   - `python3 ml/train/train_lstm.py --region IN`

The training pipeline includes a hard APY sufficiency gate and will fail early when target coverage is too sparse.

---

## 5) Readiness thresholds (to avoid guess-quality training)

The canonical pipeline blocks training if APY is too small:
- `MIN_YIELD_ROWS` (default `200`)
- `MIN_YIELD_CROPS` (default `8`)
- `MIN_YIELD_TIMESTEPS` (default `24`)

Override via env only when intentionally running a small demo.

---

## 6) Recommended weather/remote-sensing companion sources

- **ERA5-Land (CDS):** reliable historical weather baseline
- **MODIS MOD13A2:** stable long NDVI history
- **SoilGrids + SRTM:** static spatial covariates

Use these together with APY/fallback yield targets for reliable crop recommendation training.
