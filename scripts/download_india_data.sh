#!/usr/bin/env bash
# ==============================================================
# Download raw static data files required for India ingestion.
# Run this ONCE before running any Python ingest scripts.
#
# Usage:
#   bash scripts/download_india_data.sh [DATA_DIR]
#   DATA_DIR defaults to data/raw
#
# Prerequisites: curl, unzip
# ==============================================================

set -euo pipefail

_MAX_RETRIES=3
_MIN_BYTES_SOIL=100000
_MIN_BYTES_KOPPEN=100000

validate_raster_file() {
    local file_path="$1"
    local min_bytes="$2"

    if [[ ! -f "$file_path" ]]; then
        echo "❌ Missing file: $file_path"
        return 1
    fi

    local bytes
    bytes=$(wc -c < "$file_path" | tr -d ' ')
    if [[ "$bytes" -lt "$min_bytes" ]]; then
        echo "❌ File too small ($bytes bytes): $file_path"
        return 1
    fi

    local ftype
    ftype=$(file -b "$file_path" | tr '[:upper:]' '[:lower:]')
    if [[ "$ftype" == *"xml"* || "$ftype" == *"html"* || "$ftype" == *"empty"* || "$ftype" == *"ascii"* ]]; then
        echo "❌ Invalid raster type (${ftype}) for: $file_path"
        return 1
    fi

    # Also scan first bytes for OGC/API error payloads masquerading as .tif
    if head -n 3 "$file_path" | grep -qiE 'ExceptionReport|<html|error|ows:Exception'; then
        echo "❌ Detected error payload in: $file_path"
        return 1
    fi

    return 0
}

download_with_retries() {
    local out_file="$1"
    local min_bytes="$2"
    local url="$3"

    local attempt=1
    while [[ "$attempt" -le "$_MAX_RETRIES" ]]; do
        echo "  ↻ Download attempt ${attempt}/${_MAX_RETRIES}: $(basename "$out_file")"
        rm -f "$out_file"
        curl -fL --progress-bar -o "$out_file" "$url" || true

        if validate_raster_file "$out_file" "$min_bytes"; then
            echo "  ✅ Valid raster: $out_file"
            return 0
        fi

        echo "  ⚠️  Validation failed for $out_file"
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "❌ Failed to download valid raster after ${_MAX_RETRIES} attempts: $out_file"
    return 1
}

DATA_DIR="${1:-data/raw}"
SOIL_DIR="${DATA_DIR}/soilgrids"
KG_DIR="${DATA_DIR}/koppen"
TERRAIN_DIR="${DATA_DIR}/terrain"
APY_DIR="${DATA_DIR}/apy"

mkdir -p "${SOIL_DIR}" "${KG_DIR}" "${TERRAIN_DIR}" "${APY_DIR}"

echo "📁 Data directory: ${DATA_DIR}"
echo ""

# ── 1. Köppen-Geiger Beck et al. 2018 (global 1 km GeoTIFF, ~70 MB) ──────
# Source: https://www.nature.com/articles/sdata2018214
# Free, no login required.
KG_OUT="${KG_DIR}/koppen_geiger_1km.tif"
if [[ -f "${KG_OUT}" ]]; then
    if validate_raster_file "${KG_OUT}" "${_MIN_BYTES_KOPPEN}"; then
        echo "✅ Köppen-Geiger raster already exists — skipping"
    else
        echo "⚠️  Existing Köppen file invalid — re-downloading"
        download_with_retries "${KG_OUT}" "${_MIN_BYTES_KOPPEN}" "https://figshare.com/ndownloader/files/12407516"
        echo "✅ Köppen-Geiger saved to ${KG_OUT}"
    fi
else
    echo "⬇️  Downloading Köppen-Geiger Beck 2018 raster (~70 MB)..."
    download_with_retries "${KG_OUT}" "${_MIN_BYTES_KOPPEN}" "https://figshare.com/ndownloader/files/12407516"
    echo "✅ Köppen-Geiger saved to ${KG_OUT}"
fi

echo ""

# ── 2. SoilGrids v2 — India-clipped GeoTIFFs via ISRIC WCS ───────────────
# Properties: nitrogen, phh2o (pH), clay, sand, silt — all at 0-5 cm mean.
# ~10-30 MB each when clipped to India bbox.
# No login required.
SOILGRIDS_PROPS=("nitrogen" "phh2o" "clay" "sand" "silt")
ISRIC_BASE="https://maps.isric.org/mapserv"
INDIA_SUBSET="SUBSET=X(68.0,97.5)&SUBSET=Y(6.5,37.6)"
WCS_COMMON="SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&FORMAT=image/tiff&${INDIA_SUBSET}&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326&GEOTIFF:COMPRESSION=DEFLATE"

echo "⬇️  Downloading SoilGrids v2 GeoTIFFs for India (5 properties)..."
for PROP in "${SOILGRIDS_PROPS[@]}"; do
    OUT="${SOIL_DIR}/${PROP}_0-5cm_india.tif"
    if [[ -f "${OUT}" ]]; then
        if validate_raster_file "${OUT}" "${_MIN_BYTES_SOIL}"; then
            echo "  ✅ ${PROP} already exists — skipping"
            continue
        else
            echo "  ⚠️  ${PROP} existing file invalid — re-downloading"
        fi
    fi
    echo "  → ${PROP}..."
    download_with_retries \
        "${OUT}" \
        "${_MIN_BYTES_SOIL}" \
        "${ISRIC_BASE}?map=/map/${PROP}.map&${WCS_COMMON}&COVERAGEID=${PROP}_0-5cm_mean"
    echo "  ✅ Saved ${OUT}"
done

echo ""

# ── 3. SRTM GL1 Terrain — via OpenTopography API ─────────────────────────
# Requires a free API key: https://opentopography.org/developers
# Set OPENTOPOGRAPHY_API_KEY in your .env before running this block.
TERRAIN_OUT="${TERRAIN_DIR}/srtm_india_30m.tif"
if [[ -f "${TERRAIN_OUT}" ]]; then
    echo "✅ SRTM terrain GeoTIFF already exists — skipping"
elif [[ -z "${OPENTOPOGRAPHY_API_KEY:-}" ]]; then
    echo "⚠️  OPENTOPOGRAPHY_API_KEY not set — skipping terrain download"
    echo "    Register free at https://opentopography.org/developers"
    echo "    Then run: OPENTOPOGRAPHY_API_KEY=<key> bash scripts/download_india_data.sh"
else
    echo "⬇️  Downloading SRTM GL1 30 m DEM for India via OpenTopography (~500 MB)..."
    curl -L --progress-bar \
        -o "${TERRAIN_OUT}" \
        "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=6.5&north=37.6&west=68.0&east=97.5&outputFormat=GTiff&API_Key=${OPENTOPOGRAPHY_API_KEY}"
    echo "✅ Terrain DEM saved to ${TERRAIN_OUT}"
fi

echo ""

# ── 4. Manual steps (require browser or credentials) ─────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚙️  MANUAL STEPS REQUIRED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 APY Crop Yield Data (primary ML training target)"
echo "   1. Open: https://aps.dac.gov.in/APY/Public_Report1.aspx"
echo "   2. Select: All States | All Crops | All Seasons | Years: 2001-2025"
echo "   3. Click Generate → Export to Excel/CSV"
echo "   4. Save to: ${APY_DIR}/apy_india_all.csv"
echo "   Then run: python scripts/ingest_apy.py --region IN --file ${APY_DIR}/apy_india_all.csv"
echo ""
echo "🌡️  ERA5-Land Historical Weather (2010-2025)"
echo "   Requires: CDS API key at cds.climate.copernicus.eu (free registration)"
echo "   Then run: python scripts/ingest_era5.py --region IN --years 2010-2025"
echo ""
echo "🛰️  MODIS MOD13A2 NDVI History (2010-2025)"
echo "   Requires: NASA EarthData account at urs.earthdata.nasa.gov (free)"
echo "   Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env"
echo "   Then run: python scripts/ingest_ndvi_modis.py --region IN --years 2010-2025"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Automatic downloads complete. See manual steps above to finish."
echo ""
echo "📌 Full ingest order after downloads:"
echo "   1. docker-compose up -d db timescaledb"
echo "   2. python scripts/ingest_soilgrids.py --region IN"
echo "   3. python scripts/ingest_terrain.py   --region IN"
echo "   4. python scripts/ingest_climate_zones.py --region IN"
echo "   5. python scripts/ingest_era5.py      --region IN --years 2010-2025"
echo "   6. python scripts/ingest_ndvi_modis.py --region IN --years 2010-2025"
echo "   7. python scripts/ingest_apy.py       --region IN --file ${APY_DIR}/apy_india_all.csv"
