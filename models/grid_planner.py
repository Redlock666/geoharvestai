"""Pydantic schemas for v1 grid farming planner inputs and outputs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannerWeights(BaseModel):
    """Weights for the multi-objective grid planning score.

    Logic Flow:
        Represents the contribution of each normalized objective component.
        Components are expected to sum to 1.0 within a small tolerance.

    Expected Exceptions:
        ValueError: If weights do not sum to approximately 1.0.
    """

    model_config = ConfigDict(populate_by_name=True)

    net_margin: float = Field(default=0.30, ge=0.0, le=1.0)
    risk_penalty: float = Field(default=0.20, ge=0.0, le=1.0)
    water_penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    market_access: float = Field(default=0.15, ge=0.0, le=1.0)
    diversity_resilience: float = Field(default=0.10, ge=0.0, le=1.0)
    deficit_priority: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "PlannerWeights":
        """Validate that weights sum to ~1.0."""
        total = (
            self.net_margin
            + self.risk_penalty
            + self.water_penalty
            + self.market_access
            + self.diversity_resilience
            + self.deficit_priority
        )
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Planner weights must sum to ~1.0, got {total:.4f}")
        return self


class PlannerConstraints(BaseModel):
    """Hard constraints for v1 constrained area allocation."""

    model_config = ConfigDict(populate_by_name=True)

    max_crop_share_per_grid: float = Field(default=0.60, gt=0.0, le=1.0)
    min_diverse_crops_per_grid: int = Field(default=2, ge=1, le=10)
    min_selected_crop_share: float = Field(default=0.15, ge=0.0, le=1.0)
    max_water_m3_per_grid: Optional[float] = Field(default=None, gt=0.0)


class GridCropCandidate(BaseModel):
    """Candidate crop option for a planning grid."""

    model_config = ConfigDict(populate_by_name=True)

    grid_id: str
    h3_hex: str
    crop_name: str

    available_area_ha: float = Field(..., gt=0.0)
    predicted_yield_kg_ha: float = Field(..., ge=0.0)
    farmgate_price_inr_per_kg: float = Field(..., ge=0.0)
    variable_cost_inr_per_ha: float = Field(..., ge=0.0)

    risk_score: float = Field(..., ge=0.0, le=1.0, description="Higher = more volatile/risky")
    water_m3_per_ha: float = Field(..., ge=0.0)
    storage_access_score: float = Field(..., ge=0.0, le=1.0)
    market_access_score: float = Field(..., ge=0.0, le=1.0)

    current_crop_share: float = Field(default=0.0, ge=0.0, le=1.0)
    deficit_priority_score: float = Field(default=0.0, ge=0.0, le=1.0)


class GridPlannerRequest(BaseModel):
    """Request payload for v1 grid farming optimization."""

    model_config = ConfigDict(populate_by_name=True)

    region_code: str
    season: str
    weights: PlannerWeights = Field(default_factory=PlannerWeights)
    constraints: PlannerConstraints = Field(default_factory=PlannerConstraints)
    candidates: list[GridCropCandidate]


class GridCropAllocation(BaseModel):
    """Optimized allocation output for one crop in one grid."""

    model_config = ConfigDict(populate_by_name=True)

    grid_id: str
    h3_hex: str
    crop_name: str
    allocated_area_ha: float

    weighted_score: float
    expected_production_t: float
    expected_revenue_inr: float
    expected_margin_inr: float
    expected_water_m3: float


class GridPlannerSummary(BaseModel):
    """Portfolio-level summary across all grids."""

    model_config = ConfigDict(populate_by_name=True)

    total_allocated_area_ha: float
    total_expected_production_t: float
    total_expected_revenue_inr: float
    total_expected_margin_inr: float
    total_expected_water_m3: float
    objective_value: float


class GridPlannerResponse(BaseModel):
    """Response payload for v1 grid farming optimization."""

    model_config = ConfigDict(populate_by_name=True)

    region_code: str
    season: str
    allocations: list[GridCropAllocation]
    summary: GridPlannerSummary
