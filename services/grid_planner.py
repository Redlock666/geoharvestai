"""Grid farming planner service (v1 weighted objective + constrained allocation)."""

from __future__ import annotations

from collections import defaultdict

import structlog

from models.grid_planner import (
    GridCropAllocation,
    GridCropCandidate,
    GridPlannerRequest,
    GridPlannerResponse,
    GridPlannerSummary,
)

logger = structlog.get_logger(__name__)


def _minmax_norm(values: list[float]) -> list[float]:
    """Min-max normalize a vector into [0, 1] with flat-vector guard.

    Args:
        values: Numeric list.

    Returns:
        Normalized values in [0, 1].
    """
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if abs(mx - mn) < 1e-12:
        return [0.5 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


class GridFarmingPlannerService:
    """Compute v1 grid farming allocations using weighted multi-objective scoring.

    Logic Flow:
        1. Compute per-candidate component metrics (margin, risk, water, etc.).
        2. Normalize each component to [0, 1] across candidate set.
        3. Compute weighted score using configured weights.
        4. Allocate grid area with hard constraints:
            - max share per crop
            - minimum diversity target
            - optional water budget cap
        5. Return per-grid allocations and portfolio summary.

    Expected Exceptions:
        ValueError: Empty candidates list or invalid grid area/constraints.
    """

    def plan(self, request: GridPlannerRequest) -> GridPlannerResponse:
        """Generate constrained crop allocations for grouped farming grids.

        Args:
            request: Planner request with candidates, weights, and constraints.

        Returns:
            Planner response containing allocations and summary.

        Expected Exceptions:
            ValueError: If candidates are empty.
        """
        if not request.candidates:
            raise ValueError("Planner requires at least one candidate.")

        scored = self._score_candidates(request.candidates, request.weights.model_dump())
        allocations = self._allocate(scored, request.constraints.model_dump())

        summary = self._summarize(allocations)
        logger.info(
            "grid_planner.complete",
            region_code=request.region_code,
            season=request.season,
            allocations=len(allocations),
            objective_value=summary.objective_value,
        )

        return GridPlannerResponse(
            region_code=request.region_code,
            season=request.season,
            allocations=allocations,
            summary=summary,
        )

    def _score_candidates(self, candidates: list[GridCropCandidate], weights: dict) -> list[dict]:
        """Compute weighted objective score per candidate.

        Objective:
            score =
                w_margin * margin_norm
                - w_risk * risk_norm
                - w_water * water_norm
                + w_market * market_norm
                + w_diversity * diversity_bonus_norm
                + w_deficit * deficit_priority_norm

        Args:
            candidates: Candidate list.
            weights: Weight dictionary.

        Returns:
            List of dicts enriched with objective score and derived economics.
        """
        margin_per_ha = [
            (c.predicted_yield_kg_ha * c.farmgate_price_inr_per_kg) - c.variable_cost_inr_per_ha
            for c in candidates
        ]
        risk = [c.risk_score for c in candidates]
        water = [c.water_m3_per_ha for c in candidates]
        market = [(c.storage_access_score + c.market_access_score) / 2.0 for c in candidates]
        diversity_bonus = [1.0 - c.current_crop_share for c in candidates]
        deficit = [c.deficit_priority_score for c in candidates]

        m_norm = _minmax_norm(margin_per_ha)
        r_norm = _minmax_norm(risk)
        w_norm = _minmax_norm(water)
        mk_norm = _minmax_norm(market)
        d_norm = _minmax_norm(diversity_bonus)
        df_norm = _minmax_norm(deficit)

        out: list[dict] = []
        for i, c in enumerate(candidates):
            score = (
                weights["net_margin"] * m_norm[i]
                - weights["risk_penalty"] * r_norm[i]
                - weights["water_penalty"] * w_norm[i]
                + weights["market_access"] * mk_norm[i]
                + weights["diversity_resilience"] * d_norm[i]
                + weights["deficit_priority"] * df_norm[i]
            )
            out.append(
                {
                    "candidate": c,
                    "score": float(score),
                    "margin_per_ha": float(margin_per_ha[i]),
                }
            )
        return out

    def _allocate(self, scored: list[dict], constraints: dict) -> list[GridCropAllocation]:
        """Allocate area per grid under v1 hard constraints.

        Args:
            scored: Scored candidates.
            constraints: Constraint dictionary.

        Returns:
            List of grid crop allocations.
        """
        by_grid: dict[str, list[dict]] = defaultdict(list)
        for s in scored:
            by_grid[s["candidate"].grid_id].append(s)

        outputs: list[GridCropAllocation] = []

        for grid_id, options in by_grid.items():
            options = sorted(options, key=lambda x: x["score"], reverse=True)
            first = options[0]["candidate"]
            total_area = max(c["candidate"].available_area_ha for c in options)

            max_share = float(constraints["max_crop_share_per_grid"])
            min_diverse = int(constraints["min_diverse_crops_per_grid"])
            min_sel_share = float(constraints["min_selected_crop_share"])
            max_water = constraints.get("max_water_m3_per_grid")
            remaining_water = float(max_water) if max_water is not None else None

            remaining_area = total_area
            area_by_crop: dict[str, float] = {}

            # Step 1: enforce minimum diversity using top options.
            preselect = options[: min(min_diverse, len(options))]
            for item in preselect:
                c = item["candidate"]
                base_area = min(min_sel_share * total_area, max_share * total_area)
                if remaining_water is not None and c.water_m3_per_ha > 0:
                    base_area = min(base_area, remaining_water / c.water_m3_per_ha)
                base_area = max(0.0, min(base_area, remaining_area))
                if base_area <= 0.0:
                    continue
                area_by_crop[c.crop_name] = area_by_crop.get(c.crop_name, 0.0) + base_area
                remaining_area -= base_area
                if remaining_water is not None:
                    remaining_water -= base_area * c.water_m3_per_ha

            # Step 2: fill remaining area greedily by score under caps.
            for item in options:
                if remaining_area <= 1e-9:
                    break
                c = item["candidate"]
                current = area_by_crop.get(c.crop_name, 0.0)
                cap_area = max_share * total_area
                room = max(0.0, cap_area - current)
                if room <= 0.0:
                    continue

                assign = min(room, remaining_area)
                if remaining_water is not None and c.water_m3_per_ha > 0:
                    assign = min(assign, remaining_water / c.water_m3_per_ha)

                assign = max(0.0, assign)
                if assign <= 0.0:
                    continue

                area_by_crop[c.crop_name] = current + assign
                remaining_area -= assign
                if remaining_water is not None:
                    remaining_water -= assign * c.water_m3_per_ha

            # Build output rows in score order.
            for item in options:
                c = item["candidate"]
                area = area_by_crop.get(c.crop_name, 0.0)
                if area <= 1e-9:
                    continue

                expected_production_t = (area * c.predicted_yield_kg_ha) / 1000.0
                expected_revenue = area * c.predicted_yield_kg_ha * c.farmgate_price_inr_per_kg
                expected_margin = area * item["margin_per_ha"]
                expected_water = area * c.water_m3_per_ha

                outputs.append(
                    GridCropAllocation(
                        grid_id=grid_id,
                        h3_hex=c.h3_hex,
                        crop_name=c.crop_name,
                        allocated_area_ha=round(area, 4),
                        weighted_score=round(item["score"], 6),
                        expected_production_t=round(expected_production_t, 4),
                        expected_revenue_inr=round(expected_revenue, 2),
                        expected_margin_inr=round(expected_margin, 2),
                        expected_water_m3=round(expected_water, 2),
                    )
                )

            logger.info(
                "grid_planner.grid.complete",
                grid_id=grid_id,
                h3_hex=first.h3_hex,
                options=len(options),
                allocated_area_ha=round(sum(area_by_crop.values()), 4),
                unallocated_area_ha=round(max(0.0, remaining_area), 4),
            )

        return outputs

    def _summarize(self, allocations: list[GridCropAllocation]) -> GridPlannerSummary:
        """Summarize portfolio metrics for planner output.

        Args:
            allocations: Allocation rows.

        Returns:
            Portfolio summary.
        """
        total_area = sum(a.allocated_area_ha for a in allocations)
        total_prod = sum(a.expected_production_t for a in allocations)
        total_rev = sum(a.expected_revenue_inr for a in allocations)
        total_margin = sum(a.expected_margin_inr for a in allocations)
        total_water = sum(a.expected_water_m3 for a in allocations)
        objective = sum(a.allocated_area_ha * a.weighted_score for a in allocations)

        return GridPlannerSummary(
            total_allocated_area_ha=round(total_area, 4),
            total_expected_production_t=round(total_prod, 4),
            total_expected_revenue_inr=round(total_rev, 2),
            total_expected_margin_inr=round(total_margin, 2),
            total_expected_water_m3=round(total_water, 2),
            objective_value=round(objective, 6),
        )
