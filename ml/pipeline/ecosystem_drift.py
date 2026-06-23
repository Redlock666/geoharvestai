"""
Ecosystem Drift Detection and Projection Pipeline.

Detects cumulative micro-changes in soil biology, climate patterns, and
vegetation health that — individually imperceptible — collectively transform
the agricultural ecosystem over seasons and years.

Algorithm overview:
    1. CUSUMDetector      Page (1954) one-sided cumulative sum control charts.
                          Sensitive to sustained shifts below moving-average noise.
                          k = 0.5σ allowable slack, h = 4σ decision threshold.

    2. DriftComposite     Weights five indicator streams into a 0–1 ecosystem
                          health score. Declining score = degrading ecosystem.
                          Weights: OC(0.30) + rainfall(0.25) + NDVI(0.20)
                                   + NPK_balance(0.15) + EC_inverse(0.10)

    3. DriftProjector     Linear extrapolation of each indicator's trend slope
                          6 seasons forward. Computes ICAR critical threshold
                          ETA in seasons.

    4. RepairRecommender  Priority-ranked rule engine. Detects combination
                          patterns (e.g. "NPK sufficient + OC declining" →
                          over-fertilization induced biological collapse) and
                          maps to specific agronomic interventions.

    5. CropViabilityMapper Maps current and projected ecosystem state to
                          viable / at-risk crop sets using ICAR water,
                          salinity, OC, and thermal tolerance profiles for
                          20 major Indian crops. Identifies soil-restorative
                          crops that actively reverse the drift.

Usage (batch compute script):
    bundle = EcosystemBundle(hex_id=..., oc_series=..., ...)
    report = analyze_ecosystem_drift(bundle)

All functions are pure (no I/O). The compute script handles DB reads/writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from models.ecosystem import (
    CUSUMResult,
    CropViabilityAssessment,
    EcosystemDriftReport,
    IndicatorProjection,
    RepairIntervention,
)

# ── ICAR / scientific critical thresholds ────────────────────────────────────
# Source: ICAR Handbook of Agriculture (2016), ICRISAT soil health guidelines
_OC_CRITICAL_LOW: float = 0.50        # % — ICAR low OC threshold
_OC_CRITICAL_VERY_LOW: float = 0.30   # % — biological function severely impaired
_EC_CONCERN: float = 2.0              # dS/m — moderate salinity begins to suppress yield
_EC_CRITICAL: float = 4.0             # dS/m — severe salinity; most crops fail
_RAINFALL_DEFICIT_CONCERN: float = -100.0   # mm — 5yr anomaly vs 30yr baseline
_RAINFALL_DEFICIT_CRITICAL: float = -200.0  # mm
_TEMP_ANOMALY_CONCERN: float = 0.5    # °C above 30yr mean
_TEMP_ANOMALY_CRITICAL: float = 1.0   # °C — sowing window compression begins
_NDVI_DECLINE_CONCERN: float = -0.05  # absolute NDVI drop per season
_NDVI_CRITICAL: float = 0.25          # absolute NDVI — below this, canopy severely stressed

# CUSUM control chart parameters (Page 1954, standard industrial SPC)
_K_FACTOR: float = 0.5   # allowable slack = 0.5 * baseline_std
_H_FACTOR: float = 4.0   # decision threshold = 4 * baseline_std (standard 4-sigma ARL)

# Minimum observations to run CUSUM vs fall back to slope direction
_CUSUM_MIN_POINTS: int = 4
_TREND_MIN_POINTS: int = 2

# Health score composite weights (must sum to 1.0)
_W_OC: float = 0.30
_W_RAINFALL: float = 0.25
_W_NDVI: float = 0.20
_W_NPK: float = 0.15
_W_EC: float = 0.10

# Projection horizon in seasons
_PROJECTION_SEASONS: int = 6


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class IndicatorTimeSeries:
    """One indicator's chronological history for a hex cell."""
    name: str
    values: np.ndarray          # ordered oldest → newest
    baseline_mean: float        # reference period mean (training distribution or long-run)
    baseline_std: float         # reference period std


@dataclass
class EcosystemBundle:
    """All available time series for a single H3 hex cell.

    Any field may be None when that data source has not yet been ingested
    for this hex. The analyzer degrades gracefully.
    """
    hex_id: str
    region_code: str
    oc: Optional[IndicatorTimeSeries] = None           # organic carbon % per survey year
    ec: Optional[IndicatorTimeSeries] = None           # electrical conductivity dS/m
    rainfall_anomaly: Optional[IndicatorTimeSeries] = None  # 5yr vs 30yr baseline (mm)
    temp_anomaly: Optional[IndicatorTimeSeries] = None      # 5yr vs 30yr baseline (°C)
    ndvi: Optional[IndicatorTimeSeries] = None         # seasonal mean NDVI
    npk_mean: Optional[IndicatorTimeSeries] = None     # mean available NPK (kg/ha) as proxy
    yield_mean: Optional[IndicatorTimeSeries] = None   # mean crop yield kg/ha across crops


# ── CUSUM detector ────────────────────────────────────────────────────────────

def run_cusum(series: IndicatorTimeSeries, degrading_direction: str = "down") -> CUSUMResult:
    """Run a one-sided CUSUM control chart on a single indicator time series.

    Logic Flow:
        Computes upper and lower CUSUM statistics across the series.
        'degrading_direction' specifies whether a downward (OC, NDVI, rainfall)
        or upward (EC, temp anomaly) sustained shift indicates ecosystem degradation.
        Signals 'degrading' when the relevant CUSUM exceeds the h threshold.
        Falls back to sign of slope when fewer than CUSUM_MIN_POINTS observations.

    Args:
        series: IndicatorTimeSeries with baseline_mean and baseline_std.
        degrading_direction: 'down' when decline = bad; 'up' when rise = bad.

    Returns:
        CUSUMResult with signal classification and statistic values.
    """
    values = np.array(series.values, dtype=float)
    n = len(values)

    if n < _TREND_MIN_POINTS:
        return CUSUMResult(
            indicator=series.name, signal="insufficient_data",
            n_points=n, baseline_mean=series.baseline_mean, baseline_std=series.baseline_std,
        )

    # Compute trend slope regardless of series length
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])

    if n < _CUSUM_MIN_POINTS:
        # Simple slope-based classification when insufficient points for CUSUM
        if abs(slope) < 0.05 * max(abs(series.baseline_mean), 1e-9):
            signal = "stable"
        elif (degrading_direction == "down" and slope < 0) or (degrading_direction == "up" and slope > 0):
            signal = "degrading"
        else:
            signal = "improving"

        return CUSUMResult(
            indicator=series.name, signal=signal,
            trend_slope=round(slope, 6), n_points=n,
            baseline_mean=series.baseline_mean, baseline_std=series.baseline_std,
        )

    # Full CUSUM
    std = max(series.baseline_std, 1e-9)
    k = _K_FACTOR * std
    h = _H_FACTOR * std

    mu = series.baseline_mean
    s_pos = 0.0  # upper CUSUM — detects upward shift
    s_neg = 0.0  # lower CUSUM — detects downward shift

    for x_i in values:
        s_pos = max(0.0, s_pos + (x_i - mu) - k)
        s_neg = max(0.0, s_neg - (x_i - mu) - k)

    alarm_up = s_pos > h
    alarm_down = s_neg > h

    if degrading_direction == "down":
        degrading = alarm_down
        improving = alarm_up
    else:  # "up" — rise is bad
        degrading = alarm_up
        improving = alarm_down

    if degrading:
        signal = "degrading"
    elif improving:
        signal = "improving"
    else:
        signal = "stable"

    return CUSUMResult(
        indicator=series.name,
        signal=signal,
        cusum_pos=round(s_pos, 4),
        cusum_neg=round(s_neg, 4),
        trend_slope=round(slope, 6),
        n_points=n,
        baseline_mean=series.baseline_mean,
        baseline_std=series.baseline_std,
    )


# ── Projection engine ─────────────────────────────────────────────────────────

def project_linear(
    series: IndicatorTimeSeries,
    n_seasons: int = _PROJECTION_SEASONS,
    critical_threshold: Optional[float] = None,
    threshold_direction: str = "below",
) -> IndicatorProjection:
    """Extrapolate indicator trend N seasons forward using linear regression.

    Logic Flow:
        Fits linear regression on the available time series.
        Projects forward by extending the regression line.
        Computes seasons-to-critical-threshold if threshold provided.
        Assumes the existing trend continues unchanged — represents the
        "no intervention" trajectory.

    Args:
        series: IndicatorTimeSeries.
        n_seasons: Number of seasons to project forward.
        critical_threshold: ICAR/scientific threshold to track.
        threshold_direction: 'below' when falling below threshold is critical;
                             'above' when rising above threshold is critical.

    Returns:
        IndicatorProjection with projected values and ETA.
    """
    values = np.array(series.values, dtype=float)
    n = len(values)
    current_value = float(values[-1]) if n > 0 else None

    if n < _TREND_MIN_POINTS:
        return IndicatorProjection(
            indicator=series.name,
            current_value=current_value,
            projected_values=[],
            direction="stable",
        )

    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, values, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    projected = [float(slope * (n + i) + intercept) for i in range(1, n_seasons + 1)]

    # Direction classification based on slope magnitude relative to baseline
    baseline = max(abs(series.baseline_mean), 1e-9)
    relative_slope = abs(slope) / baseline
    if relative_slope < 0.01:
        direction = "stable"
    elif (slope < 0 and threshold_direction == "below") or (slope > 0 and threshold_direction == "above"):
        direction = "degrading"
    else:
        direction = "improving"

    # Seasons to threshold
    seasons_to_critical: Optional[int] = None
    if critical_threshold is not None and current_value is not None and abs(slope) > 1e-9:
        if threshold_direction == "below" and current_value > critical_threshold:
            seasons_needed = (current_value - critical_threshold) / (-slope)
            if seasons_needed > 0:
                seasons_to_critical = max(1, int(np.ceil(seasons_needed)))
        elif threshold_direction == "above" and current_value < critical_threshold:
            seasons_needed = (critical_threshold - current_value) / slope
            if seasons_needed > 0:
                seasons_to_critical = max(1, int(np.ceil(seasons_needed)))

    return IndicatorProjection(
        indicator=series.name,
        current_value=current_value,
        projected_values=[round(v, 4) for v in projected],
        seasons_to_critical=seasons_to_critical,
        critical_threshold=critical_threshold,
        direction=direction,
    )


# ── Ecosystem health composite ────────────────────────────────────────────────

def _normalise(value: float, low: float, high: float) -> float:
    """Normalise value to [0, 1] where high = 1.0 (good)."""
    return float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))


def compute_health_score(bundle: EcosystemBundle) -> Optional[float]:
    """Compute 0–1 composite ecosystem health score.

    Logic Flow:
        Normalises each available indicator to [0, 1] where 1.0 = excellent.
        Weights: OC(0.30) + rainfall(0.25) + NDVI(0.20) + NPK(0.15) + EC_inv(0.10).
        Returns None when fewer than 2 indicators are available (insufficient).
        Missing indicators fall back to neutral 0.5 to avoid penalising gaps.

    Args:
        bundle: EcosystemBundle with current observation values.

    Returns:
        Float 0.0–1.0 or None if insufficient data.
    """
    scores: dict[str, float] = {}

    if bundle.oc and len(bundle.oc.values) > 0:
        # OC%: 0.0% = 0, 1.0%+ = 1.0 (ICAR high = 0.75%+)
        scores["oc"] = _normalise(float(bundle.oc.values[-1]), 0.0, 1.0)

    if bundle.rainfall_anomaly and len(bundle.rainfall_anomaly.values) > 0:
        # Rainfall anomaly: -300mm = 0, 0mm = 0.5, +200mm = 1.0
        scores["rainfall"] = _normalise(float(bundle.rainfall_anomaly.values[-1]), -300.0, 200.0)

    if bundle.ndvi and len(bundle.ndvi.values) > 0:
        # NDVI: 0.0 = 0, 0.8 = 1.0
        scores["ndvi"] = _normalise(float(bundle.ndvi.values[-1]), 0.0, 0.8)

    if bundle.npk_mean and len(bundle.npk_mean.values) > 0:
        # NPK mean kg/ha: 0 = 0, 500 = 1.0 (simplified — available NPK proxy)
        scores["npk"] = _normalise(float(bundle.npk_mean.values[-1]), 0.0, 500.0)

    if bundle.ec and len(bundle.ec.values) > 0:
        # EC: 0 = 1.0 (best), 4+ = 0.0 (worst) — inverted
        scores["ec"] = _normalise(-float(bundle.ec.values[-1]), -4.0, 0.0)

    available = len(scores)
    if available < 2:
        return None

    weights = {"oc": _W_OC, "rainfall": _W_RAINFALL, "ndvi": _W_NDVI, "npk": _W_NPK, "ec": _W_EC}
    total_w = sum(weights[k] for k in scores)

    composite = sum(scores[k] * weights[k] for k in scores) / total_w
    return round(float(np.clip(composite, 0.0, 1.0)), 3)


def classify_velocity(health_history: list[float]) -> str:
    """Classify ecosystem health velocity from a sequence of health scores.

    Args:
        health_history: Oldest-to-newest health scores (at least 2 needed).

    Returns:
        'fast_decline' | 'moderate_decline' | 'stable' | 'slow_recovery' | 'recovering'
    """
    if len(health_history) < 2:
        return "stable"

    x = np.arange(len(health_history), dtype=float)
    slope = float(np.polyfit(x, health_history, 1)[0])

    if slope < -0.10:
        return "fast_decline"
    if slope < -0.03:
        return "moderate_decline"
    if slope < 0.01:
        return "stable"
    if slope < 0.05:
        return "slow_recovery"
    return "recovering"


# ── Primary stressor identification ──────────────────────────────────────────

def identify_primary_stressor(cusum_results: List[CUSUMResult]) -> Optional[str]:
    """Identify the ecosystem indicator driving the most degradation.

    Logic Flow:
        Finds the degrading indicator with the highest CUSUM statistic magnitude.
        Falls back to the steepest negative slope when CUSUM is below threshold.

    Returns:
        Stressor label string or None if no degradation detected.
    """
    stressor_map = {
        "oc": "organic_carbon_decline",
        "ec": "salinity_rise",
        "rainfall_anomaly": "drying_trend",
        "temp_anomaly": "temperature_warming",
        "ndvi": "vegetation_degradation",
        "yield_mean": "yield_collapse",
    }

    degrading = [r for r in cusum_results if r.signal == "degrading"]
    if not degrading:
        return None

    # Primary stressor = degrading indicator with largest CUSUM exceedance
    strongest = max(
        degrading,
        key=lambda r: max(r.cusum_pos, r.cusum_neg) + abs(r.trend_slope) * 10,
    )
    return stressor_map.get(strongest.indicator, strongest.indicator)


# ── Repair recommendation engine ──────────────────────────────────────────────

_CROP_PROFILES: Dict[str, Dict] = {
    # Format: water_req_mm, oc_min_pct, ec_max_ds_m, temp_max_c,
    #         drought_tolerance (low/medium/high), salt_tolerance (low/medium/high),
    #         soil_restorative (True = fixes N / builds OC)
    "Rice":          {"water_req": 1200, "oc_min": 0.50, "ec_max": 2.0, "temp_max": 38, "drought": "low",    "salt": "low",    "restorative": False},
    "Wheat":         {"water_req": 450,  "oc_min": 0.40, "ec_max": 4.0, "temp_max": 32, "drought": "medium", "salt": "medium", "restorative": False},
    "Pearl Millet":  {"water_req": 300,  "oc_min": 0.25, "ec_max": 4.5, "temp_max": 45, "drought": "high",   "salt": "high",   "restorative": False},
    "Sorghum":       {"water_req": 380,  "oc_min": 0.30, "ec_max": 4.0, "temp_max": 43, "drought": "high",   "salt": "medium", "restorative": False},
    "Finger Millet": {"water_req": 350,  "oc_min": 0.25, "ec_max": 5.0, "temp_max": 40, "drought": "high",   "salt": "high",   "restorative": False},
    "Chickpea":      {"water_req": 350,  "oc_min": 0.30, "ec_max": 3.0, "temp_max": 40, "drought": "high",   "salt": "medium", "restorative": True},
    "Pigeon Pea":    {"water_req": 400,  "oc_min": 0.30, "ec_max": 3.0, "temp_max": 40, "drought": "high",   "salt": "low",    "restorative": True},
    "Lentil":        {"water_req": 300,  "oc_min": 0.30, "ec_max": 2.5, "temp_max": 30, "drought": "medium", "salt": "low",    "restorative": True},
    "Groundnut":     {"water_req": 500,  "oc_min": 0.50, "ec_max": 2.0, "temp_max": 38, "drought": "medium", "salt": "low",    "restorative": True},
    "Mustard":       {"water_req": 350,  "oc_min": 0.40, "ec_max": 5.0, "temp_max": 35, "drought": "medium", "salt": "high",   "restorative": False},
    "Cotton":        {"water_req": 700,  "oc_min": 0.50, "ec_max": 3.0, "temp_max": 42, "drought": "medium", "salt": "medium", "restorative": False},
    "Sugarcane":     {"water_req": 1500, "oc_min": 0.60, "ec_max": 1.7, "temp_max": 38, "drought": "low",    "salt": "low",    "restorative": False},
    "Maize":         {"water_req": 500,  "oc_min": 0.40, "ec_max": 2.5, "temp_max": 40, "drought": "medium", "salt": "low",    "restorative": False},
    "Barley":        {"water_req": 380,  "oc_min": 0.35, "ec_max": 6.0, "temp_max": 32, "drought": "medium", "salt": "high",   "restorative": False},
    "Cowpea":        {"water_req": 300,  "oc_min": 0.25, "ec_max": 3.5, "temp_max": 42, "drought": "high",   "salt": "medium", "restorative": True},
    "Black Gram":    {"water_req": 350,  "oc_min": 0.35, "ec_max": 2.5, "temp_max": 40, "drought": "medium", "salt": "low",    "restorative": True},
    "Green Gram":    {"water_req": 350,  "oc_min": 0.35, "ec_max": 3.0, "temp_max": 40, "drought": "medium", "salt": "low",    "restorative": True},
    "Sesame":        {"water_req": 300,  "oc_min": 0.30, "ec_max": 3.5, "temp_max": 42, "drought": "high",   "salt": "medium", "restorative": False},
    "Soybean":       {"water_req": 450,  "oc_min": 0.40, "ec_max": 3.0, "temp_max": 38, "drought": "medium", "salt": "low",    "restorative": True},
    "Sunflower":     {"water_req": 500,  "oc_min": 0.40, "ec_max": 4.0, "temp_max": 40, "drought": "medium", "salt": "medium", "restorative": False},
}


def recommend_repairs(
    cusum_results: List[CUSUMResult],
    projections: List[IndicatorProjection],
) -> List[RepairIntervention]:
    """Priority-ranked repair intervention engine.

    Logic Flow:
        Evaluates cusum signals and projection trajectories against a
        rule set covering the major ecosystem degradation patterns found
        in Indian agriculture: over-fertilization biological collapse,
        drying-trend drought stress, salinity accumulation, warming-driven
        sowing window compression, and vegetation health decline.

        Rules are checked in priority order; each matched rule appends
        interventions. Interventions are deduplicated by category.

    Returns:
        List of RepairIntervention ordered by priority (1 = most critical).
    """
    signal_map = {r.indicator: r.signal for r in cusum_results}
    slope_map = {r.indicator: r.trend_slope for r in cusum_results}
    projection_dir = {p.indicator: p.direction for p in projections}

    interventions: List[RepairIntervention] = []

    oc_degrading = signal_map.get("oc") == "degrading"
    ec_degrading = signal_map.get("ec") == "degrading"
    rain_degrading = signal_map.get("rainfall_anomaly") == "degrading"
    temp_degrading = signal_map.get("temp_anomaly") == "degrading"
    ndvi_degrading = signal_map.get("ndvi") == "degrading"
    npk_improving = signal_map.get("npk_mean") == "improving"  # NPK rising while OC falls

    # ── Rule 1: Biological collapse — OC declining while NPK is sufficient/rising
    if oc_degrading and npk_improving:
        interventions.extend([
            RepairIntervention(
                priority=1, category="soil_biology",
                intervention="Reduce inorganic fertilizer application by 20–30%",
                mechanism="Excess inorganic N suppresses mycorrhizal fungi and nitrifier populations, accelerating OC mineralisation",
                expected_seasons_to_effect=2,
                evidence_basis="ICAR Soil Microbiome Report 2021; Powlson et al. 2014, Nature Clim. Change",
            ),
            RepairIntervention(
                priority=1, category="soil_biology",
                intervention="Introduce pulse/legume intercropping (chickpea, pigeon pea, cowpea)",
                mechanism="Nitrogen fixation by Rhizobium rebuilds soil N pool biologically; root exudates feed microbial biomass and restore OC",
                expected_seasons_to_effect=3,
                evidence_basis="ICRISAT Dryland Systems 2019; ICAR Cropping Systems Research",
            ),
            RepairIntervention(
                priority=2, category="soil_biology",
                intervention="Apply farmyard manure or vermicompost (2–4 tonnes/ha)",
                mechanism="Organic matter addition directly replenishes OC, feeds soil food web, and buffers pH and nutrient cycling",
                expected_seasons_to_effect=2,
                evidence_basis="ICAR Handbook of Agriculture 2016, Chapter 4",
            ),
            RepairIntervention(
                priority=3, category="soil_biology",
                intervention="Adopt zero-till or minimum-till for at least one season",
                mechanism="Tillage disrupts mycorrhizal hyphal networks; reduced disturbance preserves microbial community continuity",
                expected_seasons_to_effect=2,
                evidence_basis="CIMMYT Zero-Till Wheat Programme; Gal et al. 2007 Soil & Tillage Research",
            ),
        ])

    # ── Rule 2: OC declining without NPK excess (genuine biological depletion)
    elif oc_degrading and not npk_improving:
        interventions.extend([
            RepairIntervention(
                priority=1, category="soil_biology",
                intervention="Grow soil-restorative pulse crops as primary or cover crop",
                mechanism="Legumes fix atmospheric N2 via Rhizobium symbiosis; root turnover adds 0.3–0.6% OC per season",
                expected_seasons_to_effect=3,
                evidence_basis="FAO/ITPS Status of World's Soil Resources 2015",
            ),
            RepairIntervention(
                priority=2, category="soil_biology",
                intervention="Incorporate crop residue instead of burning",
                mechanism="Residue burning destroys 70–80% of organic matter; incorporation adds 0.2–0.4% OC and stimulates decomposer biomass",
                expected_seasons_to_effect=2,
                evidence_basis="ICAR NAAS Policy Brief 2020",
            ),
        ])

    # ── Rule 3: Salinity accumulation
    if ec_degrading:
        ec_current = projections[0].current_value if projections else None
        severe = ec_current is not None and ec_current > _EC_CRITICAL
        interventions.extend([
            RepairIntervention(
                priority=1 if severe else 2, category="salinity",
                intervention="Apply gypsum (calcium sulphate) at 2–3 tonnes/ha before irrigation",
                mechanism="Ca2+ displaces Na+ from cation exchange sites; Na+ leaches to subsoil on irrigation, reducing surface EC",
                expected_seasons_to_effect=2,
                evidence_basis="ICAR Central Soil Salinity Research Institute (CSSRI) Technical Bulletin 2018",
            ),
            RepairIntervention(
                priority=2, category="salinity",
                intervention="Improve field drainage; avoid waterlogging (primary driver of secondary salinity)",
                mechanism="Waterlogging raises capillary fringe and evaporates dissolved salts at surface; drainage breaks the cycle",
                expected_seasons_to_effect=3,
                evidence_basis="CSSRI Karnal, Drainage Manual 2015",
            ),
            RepairIntervention(
                priority=3, category="salinity",
                intervention="Transition to salt-tolerant varieties (Barley KM65, Finger Millet GPU-67, Mustard CS-52)",
                mechanism="Salt-tolerant varieties maintain 70–80% yield at EC 4–6 dS/m; buffers farm income while soil treatment proceeds",
                expected_seasons_to_effect=1,
                evidence_basis="NBPGR Salt Tolerance Crop Database; CSSRI Varietal Releases 2022",
            ),
        ])

    # ── Rule 4: Rainfall deficit worsening
    if rain_degrading:
        rain_slope = slope_map.get("rainfall_anomaly", 0.0)
        accelerating = rain_slope < -20.0  # mm/season — fast drying
        interventions.extend([
            RepairIntervention(
                priority=1 if accelerating else 2, category="water_management",
                intervention="Transition to drought-tolerant improved varieties (ICRISAT ICMV-221 millet, ICRISAT ICCC-37 chickpea)",
                mechanism="Improved drought-tolerant varieties maintain 60–75% yield at 30% rainfall deficit; no infrastructure required",
                expected_seasons_to_effect=1,
                evidence_basis="ICRISAT Crop Improvement 2023; AICRP on Pearl Millet",
            ),
            RepairIntervention(
                priority=2, category="water_management",
                intervention="Install mulching on 30–50% of field area",
                mechanism="Mulch reduces soil moisture evaporation by 40–60%, extending effective water availability for an additional 10–15 days",
                expected_seasons_to_effect=1,
                evidence_basis="ICAR CRIDA Dryland Agronomy Bulletin 2019",
            ),
            RepairIntervention(
                priority=3, category="water_management",
                intervention="Construct farm pond or install in-situ rainwater harvesting (broad-bed furrow)",
                mechanism="Harvests 40–80mm of runoff per rainfall event; provides supplemental irrigation at critical growth stages",
                expected_seasons_to_effect=2,
                evidence_basis="ICAR CRIDA Watershed Technology; NABARD Farm Pond Scheme",
            ),
        ])

    # ── Rule 5: Temperature warming trend
    if temp_degrading:
        interventions.extend([
            RepairIntervention(
                priority=2, category="climate_adaptation",
                intervention="Advance sowing dates by 7–14 days to utilise cooler temperatures at anthesis",
                mechanism="1°C warming shifts optimal sowing window forward; early sowing avoids heat stress during grain filling",
                expected_seasons_to_effect=1,
                evidence_basis="ICAR-IARI Climate-Smart Agronomy Advisory 2022",
            ),
            RepairIntervention(
                priority=3, category="climate_adaptation",
                intervention="Introduce boundary tree planting (agroforestry) to reduce microclimate temperature by 1–2°C",
                mechanism="Tree canopy reduces solar radiation interception; reduces soil surface temperature and evapotranspiration demand",
                expected_seasons_to_effect=6,
                evidence_basis="ICRAF South Asia Agroforestry Manual; World Agroforestry 2020",
            ),
            RepairIntervention(
                priority=3, category="climate_adaptation",
                intervention="Transition to heat-tolerant crop varieties (NW-1014 wheat, Rajendra Bhagwati rice)",
                mechanism="Heat-tolerant varieties maintain pollen viability up to 3°C above conventional threshold",
                expected_seasons_to_effect=1,
                evidence_basis="ICAR-IARI Heat Tolerance Wheat Breeding Programme; DRR Technical Bulletin 2021",
            ),
        ])

    # ── Rule 6: NDVI decline without other clear driver
    if ndvi_degrading and not (oc_degrading or rain_degrading or ec_degrading):
        interventions.extend([
            RepairIntervention(
                priority=2, category="nutrient",
                intervention="Test for micronutrient deficiency (zinc, iron, boron) — apply 25 kg ZnSO4/ha if zinc-deficient",
                mechanism="Zinc deficiency causes 'khaira' in rice and 'white bud' in maize; NDVI recovers within one season of correction",
                expected_seasons_to_effect=1,
                evidence_basis="ICAR Soil Micronutrient Status Report 2020; IRRI Zinc in Rice Production",
            ),
        ])

    # Deduplicate by category + intervention text (keep highest priority)
    seen: set[str] = set()
    deduped: List[RepairIntervention] = []
    for iv in sorted(interventions, key=lambda x: x.priority):
        key = f"{iv.category}::{iv.intervention[:40]}"
        if key not in seen:
            seen.add(key)
            deduped.append(iv)

    return deduped[:8]  # cap at 8 interventions for readability


# ── Crop viability mapper ─────────────────────────────────────────────────────

def _current_ecosystem_state(bundle: EcosystemBundle) -> dict:
    """Extract current values from bundle for crop viability checks."""
    def _last(ts: Optional[IndicatorTimeSeries]) -> Optional[float]:
        if ts is None or len(ts.values) == 0:
            return None
        return float(ts.values[-1])

    return {
        "oc": _last(bundle.oc),
        "ec": _last(bundle.ec),
        "rainfall_anomaly": _last(bundle.rainfall_anomaly),
        "temp_anomaly": _last(bundle.temp_anomaly),
    }


def _project_ecosystem_state(
    bundle: EcosystemBundle,
    projections: List[IndicatorProjection],
) -> dict:
    """Extract projected (season+6) values for crop viability checks."""
    proj_map = {p.indicator: p.projected_values for p in projections if p.projected_values}
    current = _current_ecosystem_state(bundle)

    def _projected(key: str, ts: Optional[IndicatorTimeSeries]) -> Optional[float]:
        vals = proj_map.get(key)
        if vals:
            return float(vals[-1])
        return current.get(key)

    return {
        "oc": _projected("oc", bundle.oc),
        "ec": _projected("ec", bundle.ec),
        "rainfall_anomaly": _projected("rainfall_anomaly", bundle.rainfall_anomaly),
        "temp_anomaly": _projected("temp_anomaly", bundle.temp_anomaly),
    }


def _crop_viable_in_state(crop: str, profile: dict, state: dict) -> tuple[bool, list[str]]:
    """Check whether a crop is viable given an ecosystem state dict.

    Returns (viable, list_of_risk_factors).
    """
    risks: list[str] = []

    oc = state.get("oc")
    if oc is not None and oc < profile["oc_min"]:
        risks.append(f"oc_below_minimum_{profile['oc_min']}_pct")

    ec = state.get("ec")
    if ec is not None and ec > profile["ec_max"]:
        risks.append(f"salinity_exceeds_tolerance_{profile['ec_max']}_ds_m")

    rain_anomaly = state.get("rainfall_anomaly")
    if rain_anomaly is not None:
        # Approximate: if rainfall is 200mm below baseline, low-water crops survive
        effective_deficit = -rain_anomaly  # positive = deficit
        if effective_deficit > 150 and profile["drought"] == "low":
            risks.append("rainfall_deficit_exceeds_drought_tolerance")
        elif effective_deficit > 250 and profile["drought"] == "medium":
            risks.append("severe_rainfall_deficit")

    temp_anomaly = state.get("temp_anomaly")
    if temp_anomaly is not None and temp_anomaly > 1.0:
        # If regional temp is >1°C above 30yr mean, crops with low temp_max are at risk
        if profile["temp_max"] < 35:
            risks.append("temperature_warming_exceeds_thermal_optimum")

    return len(risks) == 0, risks


def map_crop_viability(
    bundle: EcosystemBundle,
    projections: List[IndicatorProjection],
) -> tuple[List[CropViabilityAssessment], List[str]]:
    """Assess crop viability under current and projected ecosystem state.

    Logic Flow:
        Evaluates each crop in _CROP_PROFILES against:
          (a) current ecosystem state (from latest bundle values)
          (b) projected ecosystem state (from 6-season linear projections)
        Classifies transition_priority: phase-out crops moving from viable → not viable.
        Identifies soil-restorative crops viable in current state.

    Returns:
        Tuple of (list of CropViabilityAssessment, list of soil_restorative crop names).
    """
    current_state = _current_ecosystem_state(bundle)
    projected_state = _project_ecosystem_state(bundle, projections)

    assessments: List[CropViabilityAssessment] = []
    restorative_crops: List[str] = []

    for crop_name, profile in _CROP_PROFILES.items():
        viable_now, risks_now = _crop_viable_in_state(crop_name, profile, current_state)
        viable_proj, risks_proj = _crop_viable_in_state(crop_name, profile, projected_state)

        if viable_now and profile["restorative"]:
            restorative_crops.append(crop_name)

        if viable_now and not viable_proj:
            transition = "phase_out"
        elif not viable_now and viable_proj:
            transition = "phase_in_after_intervention"
        elif viable_now and viable_proj:
            transition = "recommended" if profile["restorative"] else "neutral"
        else:
            transition = "not_viable"

        # Only include crops relevant to the assessment
        combined_risks = list(set(risks_now + risks_proj))
        assessments.append(CropViabilityAssessment(
            crop_name=crop_name,
            viable_now=viable_now,
            viable_projected=viable_proj,
            confidence="high" if len(current_state) >= 3 else "medium",
            risk_factors=combined_risks[:4],
            soil_restorative=bool(profile["restorative"]),
            transition_priority=transition,
        ))

    # Sort: restorative first, then viable+stable, then at-risk, then not viable
    priority_order = {"recommended": 0, "neutral": 1, "phase_out": 2,
                      "phase_in_after_intervention": 3, "not_viable": 4}
    assessments.sort(key=lambda a: priority_order.get(a.transition_priority, 5))

    return assessments, sorted(set(restorative_crops))


# ── Narrative generation ──────────────────────────────────────────────────────

def build_narratives(
    bundle: EcosystemBundle,
    cusum_results: List[CUSUMResult],
    projections: List[IndicatorProjection],
    primary_stressor: Optional[str],
    health_score: Optional[float],
    health_velocity: str,
    seasons_to_critical: Optional[int],
) -> tuple[str, str, str]:
    """Produce three short narrative strings for the LLM reasoning layer.

    Returns:
        (drift_narrative, repair_summary, projection_narrative)
    """
    degrading_indicators = [r.indicator for r in cusum_results if r.signal == "degrading"]
    n_degrading = len(degrading_indicators)

    # Drift narrative
    if n_degrading == 0:
        drift = "Ecosystem indicators are stable — no sustained degradation detected across the measured dimensions."
    elif n_degrading == 1:
        drift = (
            f"A sustained degradation signal has been detected in {degrading_indicators[0].replace('_', ' ')}. "
            "Other indicators are currently stable, but the CUSUM pattern indicates this shift has persisted "
            "beyond random variation — it is a structural change, not noise."
        )
    else:
        listed = ", ".join(d.replace("_", " ") for d in degrading_indicators[:-1])
        last = degrading_indicators[-1].replace("_", " ")
        drift = (
            f"Concurrent degradation signals detected across {listed} and {last}. "
            f"The ecosystem health composite is {f'{health_score:.2f}' if health_score else 'unavailable'} "
            f"and {health_velocity.replace('_', ' ')}. "
            "Co-occurring stressors typically amplify each other — soil biology collapse accelerates "
            "when combined with rainfall deficit."
        )

    # Projection narrative
    at_risk_proj = [p for p in projections if p.direction == "degrading" and p.seasons_to_critical]
    if at_risk_proj:
        earliest = min(at_risk_proj, key=lambda p: p.seasons_to_critical or 999)
        proj = (
            f"At current trajectory, {earliest.indicator.replace('_', ' ')} is projected to breach its critical "
            f"threshold in approximately {earliest.seasons_to_critical} seasons. "
            "Without intervention, viable crop options narrow and yield per unit of input continues to decline."
        )
    elif seasons_to_critical:
        proj = (
            f"The primary stressor is projected to reach a critical threshold in {seasons_to_critical} seasons "
            "under the current trend. Intervention this season or next will be significantly more cost-effective "
            "than remediation after the threshold is crossed."
        )
    else:
        proj = (
            "No critical threshold breach is projected within the 6-season horizon at the current trend rate. "
            "This does not preclude gradual yield ceiling compression — early preventive action is still warranted."
        )

    # Repair summary
    if primary_stressor:
        repair = (
            f"Priority action: address {primary_stressor.replace('_', ' ')}. "
            "Soil-restorative pulse crops and organic matter addition are the highest-ROI "
            "interventions when biological health is the primary driver. "
            "All interventions should be combined with reduced inorganic fertilizer to stop amplifying the stressor."
        )
    else:
        repair = "No priority interventions required at this time. Continue monitoring and maintain organic matter inputs."

    return drift, repair, proj


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_ecosystem_drift(bundle: EcosystemBundle) -> EcosystemDriftReport:
    """Run the full ecosystem drift analysis pipeline on a hex cell bundle.

    Logic Flow:
        1. Run CUSUM on each available indicator time series.
        2. Compute composite ecosystem health score.
        3. Project each indicator 6 seasons forward; compute threshold ETAs.
        4. Identify primary stressor.
        5. Generate repair interventions.
        6. Assess crop viability under current and projected state.
        7. Build narrative summaries.
        8. Assemble and return EcosystemDriftReport.

    Args:
        bundle: EcosystemBundle for one hex cell. Fields may be None when
                data has not yet been ingested for that indicator.

    Returns:
        EcosystemDriftReport. data_quality reflects how many indicators
        had sufficient observations.
    """
    # CUSUM — each indicator has its natural "bad" direction
    cusum_results: List[CUSUMResult] = []
    if bundle.oc:
        cusum_results.append(run_cusum(bundle.oc, degrading_direction="down"))
    if bundle.ec:
        cusum_results.append(run_cusum(bundle.ec, degrading_direction="up"))
    if bundle.rainfall_anomaly:
        cusum_results.append(run_cusum(bundle.rainfall_anomaly, degrading_direction="down"))
    if bundle.temp_anomaly:
        cusum_results.append(run_cusum(bundle.temp_anomaly, degrading_direction="up"))
    if bundle.ndvi:
        cusum_results.append(run_cusum(bundle.ndvi, degrading_direction="down"))
    if bundle.yield_mean:
        cusum_results.append(run_cusum(bundle.yield_mean, degrading_direction="down"))
    if bundle.npk_mean:
        cusum_results.append(run_cusum(bundle.npk_mean, degrading_direction="down"))

    # Projections
    projections: List[IndicatorProjection] = []
    if bundle.oc:
        projections.append(project_linear(bundle.oc, critical_threshold=_OC_CRITICAL_LOW,  threshold_direction="below"))
    if bundle.ec:
        projections.append(project_linear(bundle.ec, critical_threshold=_EC_CRITICAL,       threshold_direction="above"))
    if bundle.rainfall_anomaly:
        projections.append(project_linear(bundle.rainfall_anomaly, critical_threshold=_RAINFALL_DEFICIT_CRITICAL, threshold_direction="below"))
    if bundle.temp_anomaly:
        projections.append(project_linear(bundle.temp_anomaly, critical_threshold=_TEMP_ANOMALY_CRITICAL, threshold_direction="above"))
    if bundle.ndvi:
        projections.append(project_linear(bundle.ndvi, critical_threshold=_NDVI_CRITICAL,   threshold_direction="below"))
    if bundle.yield_mean:
        projections.append(project_linear(bundle.yield_mean))
    if bundle.npk_mean:
        projections.append(project_linear(bundle.npk_mean))

    # Health score
    health_score = compute_health_score(bundle)

    # Velocity (would need history — use single-season slope as proxy)
    health_velocity = "stable"
    if cusum_results:
        degrading_count = sum(1 for r in cusum_results if r.signal == "degrading")
        if degrading_count >= 3:
            health_velocity = "fast_decline"
        elif degrading_count >= 2:
            health_velocity = "moderate_decline"
        elif degrading_count == 1:
            health_velocity = "moderate_decline"

    # Primary stressor
    primary_stressor = identify_primary_stressor(cusum_results)

    # Seasons to critical (minimum across projected indicators)
    etas = [p.seasons_to_critical for p in projections if p.seasons_to_critical is not None]
    seasons_to_critical = min(etas) if etas else None

    # Projected health score
    projected_health_score: Optional[float] = None
    if health_score is not None:
        degrading_proj = [p for p in projections if p.direction == "degrading"]
        if degrading_proj:
            projected_health_score = max(0.0, health_score - 0.05 * len(degrading_proj))
        else:
            projected_health_score = health_score

    # Repairs
    repair_interventions = recommend_repairs(cusum_results, projections)

    # Crop viability
    crop_viability, restorative_crops = map_crop_viability(bundle, projections)

    # Narratives
    drift_narrative, repair_summary, projection_narrative = build_narratives(
        bundle, cusum_results, projections, primary_stressor,
        health_score, health_velocity, seasons_to_critical,
    )

    # Data quality
    available_series = sum(1 for s in [bundle.oc, bundle.ec, bundle.rainfall_anomaly,
                                        bundle.temp_anomaly, bundle.ndvi, bundle.yield_mean]
                           if s is not None and len(s.values) > 0)
    min_points = min((len(s.values) for s in [bundle.oc, bundle.ec, bundle.rainfall_anomaly,
                                               bundle.temp_anomaly, bundle.ndvi, bundle.yield_mean]
                      if s is not None and len(s.values) > 0), default=0)

    if available_series >= 4 and min_points >= 5:
        data_quality = "high"
    elif available_series >= 3 and min_points >= 3:
        data_quality = "medium"
    elif available_series >= 1:
        data_quality = "low"
    else:
        data_quality = "insufficient"

    return EcosystemDriftReport(
        hex_id=bundle.hex_id,
        region_code=bundle.region_code,
        ecosystem_health_score=health_score if health_score is not None else 0.5,
        health_velocity=health_velocity,
        primary_stressor=primary_stressor,
        cusum_results=cusum_results,
        projections=projections,
        projected_health_score=projected_health_score,
        seasons_to_critical=seasons_to_critical,
        repair_interventions=repair_interventions,
        crop_viability=crop_viability,
        soil_restorative_crops=restorative_crops,
        drift_narrative=drift_narrative,
        repair_summary=repair_summary,
        projection_narrative=projection_narrative,
        data_quality=data_quality,
        indicators_with_data=available_series,
        seasons_of_data=min_points if min_points > 0 else None,
    )
