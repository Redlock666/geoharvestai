# Grid Farming Planner v1 (Schema + Objective Function)

This document defines a concrete v1 design that plugs into the current pipeline.

## Goal

Move from single-farm recommendation to **grid-level portfolio planning** that balances:

- net margin,
- volatility/risk,
- water footprint,
- storage/market access,
- crop diversity resilience,
- domestic/export deficit priority.

---

## Input contract

Implemented in `models/grid_planner.py`:

- `PlannerWeights`
- `PlannerConstraints`
- `GridCropCandidate`
- `GridPlannerRequest`

Core fields per candidate:

- `grid_id`, `h3_hex`, `crop_name`
- `available_area_ha`
- `predicted_yield_kg_ha` (from ML pipeline)
- `farmgate_price_inr_per_kg`, `variable_cost_inr_per_ha`
- `risk_score`, `water_m3_per_ha`
- `storage_access_score`, `market_access_score`
- `current_crop_share`, `deficit_priority_score`

---

## Objective function (v1)

For each candidate (grid, crop), define:

$$
\text{score} =
w_m \cdot \hat m
- w_r \cdot \hat r
- w_w \cdot \hat w
+ w_a \cdot \hat a
+ w_d \cdot \widehat{(1-s)}
+ w_f \cdot \hat f
$$

Where:

- $\hat m$: normalized net margin per ha,
- $\hat r$: normalized risk,
- $\hat w$: normalized water use,
- $\hat a$: normalized access score (storage + market),
- $s$: current crop share in grid (so lower concentration gets bonus),
- $\hat f$: normalized deficit-priority signal.

Weights are configured in `PlannerWeights` and should sum to ~1.0.

Default v1 weights:

- net margin: 0.30
- risk penalty: 0.20
- water penalty: 0.15
- market access: 0.15
- diversity resilience: 0.10
- deficit priority: 0.10

---

## Hard constraints (v1)

Implemented in `PlannerConstraints`:

- `max_crop_share_per_grid` (default 0.60)
- `min_diverse_crops_per_grid` (default 2)
- `min_selected_crop_share` (default 0.15)
- `max_water_m3_per_grid` (optional)

Allocator strategy in `services/grid_planner.py`:

1. Preselect top crops to satisfy minimum diversity.
2. Assign base area floor for selected crops.
3. Allocate remaining area greedily by weighted score.
4. Respect crop concentration and optional water cap.

---

## Integration with current pipeline

Use existing stack artifacts to build `GridCropCandidate` rows:

1. **GIS + weather features** from current resolver/agent services.
2. **Predicted yield/confidence** from `MLPredictorService` outputs.
3. **Risk proxy** from drift/uncertainty profile in model metadata.
4. **Deficit priority score** from demand/export module (new feed).
5. **Price/cost/access** from market/logistics tables (new feed).

Then call:

- `GridFarmingPlannerService.plan(GridPlannerRequest)`

Returns:

- `GridPlannerResponse` with per-grid allocations and portfolio summary.

---

## v1 rollout notes

- Start district-level price/cost proxies if farm-level economics unavailable.
- Start with coarse demand-deficit signal and refine over time.
- Keep objective weights configurable by season and region.
- Track portfolio KPIs each season (margin uplift, deficit closure, water intensity).
