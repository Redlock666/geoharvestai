FROM python:3.11-slim

WORKDIR /app

# Install GDAL/GEOS system deps needed by geopandas + rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached until pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[dev]"

COPY . .

# Make sure local packages are importable without installation
ENV PYTHONPATH=/app

EXPOSE 8000
