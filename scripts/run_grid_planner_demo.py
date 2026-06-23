"""Run a minimal demo for GridFarmingPlannerService.

Usage:
    python3 scripts/run_grid_planner_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grid_planner import GridCropCandidate, GridPlannerRequest
from services.grid_planner import GridFarmingPlannerService


def main() -> None:
    request = GridPlannerRequest(
        region_code="IN",
        season="kharif_2026",
        candidates=[
            GridCropCandidate(
                grid_id="G1",
                h3_hex="8860145ad1fffff",
                crop_name="Rice",
                available_area_ha=100,
                predicted_yield_kg_ha=4200,
                farmgate_price_inr_per_kg=22,
                variable_cost_inr_per_ha=42000,
                risk_score=0.35,
                water_m3_per_ha=9500,
                storage_access_score=0.7,
                market_access_score=0.8,
                current_crop_share=0.6,
                deficit_priority_score=0.4,
            ),
            GridCropCandidate(
                grid_id="G1",
                h3_hex="8860145ad1fffff",
                crop_name="Millet",
                available_area_ha=100,
                predicted_yield_kg_ha=2800,
                farmgate_price_inr_per_kg=30,
                variable_cost_inr_per_ha=24000,
                risk_score=0.25,
                water_m3_per_ha=4200,
                storage_access_score=0.6,
                market_access_score=0.65,
                current_crop_share=0.1,
                deficit_priority_score=0.8,
            ),
            GridCropCandidate(
                grid_id="G1",
                h3_hex="8860145ad1fffff",
                crop_name="Pulses",
                available_area_ha=100,
                predicted_yield_kg_ha=1900,
                farmgate_price_inr_per_kg=65,
                variable_cost_inr_per_ha=22000,
                risk_score=0.30,
                water_m3_per_ha=3000,
                storage_access_score=0.55,
                market_access_score=0.6,
                current_crop_share=0.05,
                deficit_priority_score=0.85,
            ),
        ],
    )

    planner = GridFarmingPlannerService()
    response = planner.plan(request)

    print("=== Grid Planner Demo ===")
    for a in response.allocations:
        print(f"{a.grid_id} | {a.crop_name:<8} | area={a.allocated_area_ha:.2f} ha | score={a.weighted_score:.4f}")
    print("Summary:", response.summary.model_dump())


if __name__ == "__main__":
    main()
