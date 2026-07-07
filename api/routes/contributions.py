"""
api/routes/contributions.py

GET /predict/{engine_id}/contributions?dataset_id=FD001

Returns per-sensor and per-module PC1 attribution for the given engine at
its latest test cycle. Consumed by the React EngineHealthMap component on
enginewatch.tech. Uses the identical pipeline path as the Streamlit
engine_diagram and narration_panel — attributions can never diverge between
the site and the dashboard.

No ML training. Inference only: scaler.transform + pca.components_ projection.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel



router = APIRouter(tags=["contributions"])


# ── Response schema ────────────────────────────────────────────────────────────


class SensorContribution(BaseModel):
    sensor_id: str  # "s11"
    symbol: str  # "Ps30"
    description: str  # "Static pressure at HPC outlet"
    signed_contribution: float
    abs_contribution: float


class ModuleHeat(BaseModel):
    module: str  # "hpc"
    display_name: str  # "HPC"
    direction: str  # "healthy" | "critical" | "inactive"
    signed_heat: float
    norm_magnitude: float  # [0,1] — dominant module = 1.0
    norm_signed: float  # [-1,1]
    active_sensors: list[SensorContribution]
    is_active: bool


class ContributionsResponse(BaseModel):
    engine_id: int
    dataset_id: str
    cycle: int  # latest cycle used
    dominant_module: str  # key of module with norm_magnitude == 1.0
    dominant_driver_text: str  # "HPC — Ps30, T30, phi driving degradation"
    modules: list[ModuleHeat]


# ── Endpoint ───────────────────────────────────────────────────────────────────

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}


@router.get(
    "/predict/{engine_id}/contributions",
    response_model=ContributionsResponse,
    summary="Per-module PC1 attribution for a single engine",
    description=(
        "Projects the engine's latest-cycle scaled sensor readings onto PC1 of "
        "the operative health axis and aggregates contributions by physical "
        "C-MAPSS module (Fan, LPC, HPC, HPT, LPT, Bypass, Core, Burner, EPR). "
        "Result drives the Engine Health Map colour coding on enginewatch.tech."
    ),
)
async def engine_contributions(
    engine_id: int,
    dataset_id: str = Query(default="FD001", description="One of FD001–FD004"),
) -> ContributionsResponse:
    if dataset_id not in VALID_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=f"dataset_id must be one of {sorted(VALID_DATASETS)}",
        )

    from api.main import _attribution_cache

    cache_key = f"{dataset_id}:{engine_id}"
    result = _attribution_cache.get(cache_key)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Engine {engine_id} not found in dataset {dataset_id} attribution cache",
        )

    return ContributionsResponse(**result)
