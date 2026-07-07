import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["analytics"])

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int

class TrendDecile(BaseModel):
    life_pct_bin: int
    mean_risk_score: float
    n_engines_contributing: int

class FleetAnalyticsResponse(BaseModel):
    dataset_id: str
    risk_histogram: list[HistogramBin]
    state_counts: dict[str, int]
    risk_trend: list[TrendDecile]

class FleetCompareRow(BaseModel):
    dataset_id: str
    fleet_size: int
    state_counts: dict[str, int]
    n_critical: int
    mean_rul: float
    median_rul: float

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}

@router.get(
    "/fleet/analytics",
    response_model=FleetAnalyticsResponse,
    summary="Fleet analytics and risk distribution",
)
async def fleet_analytics(
    dataset_id: str = Query()
) -> FleetAnalyticsResponse:
    if dataset_id not in VALID_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=f"dataset_id must be one of {sorted(VALID_DATASETS)}",
        )

    from api.main import _predict_cache, _fleet_summary_cache, _fleet_trend_cache

    if dataset_id not in _fleet_summary_cache or dataset_id not in _fleet_trend_cache:
        raise HTTPException(
            status_code=503,
            detail=f"Artifacts unavailable for {dataset_id}",
        )

    summary = _fleet_summary_cache[dataset_id]
    risk_trend_raw = _fleet_trend_cache[dataset_id]
    
    risk_trend = [
        TrendDecile(
            life_pct_bin=r["life_pct_bin"],
            mean_risk_score=r["mean_risk_score"],
            n_engines_contributing=r["n_engines_contributing"],
        )
        for r in risk_trend_raw
    ]

    # risk_histogram: risk_score bucketed into 10 fixed-width bins [0.0-1.0]
    risk_scores = [
        val["risk_score"]
        for key, val in _predict_cache.items()
        if key.startswith(f"{dataset_id}:")
    ]

    hist_counts, bin_edges = np.histogram(risk_scores, bins=10, range=(0.0, 1.0))
    risk_histogram = [
        HistogramBin(
            bin_start=float(bin_edges[i]),
            bin_end=float(bin_edges[i + 1]),
            count=int(hist_counts[i]),
        )
        for i in range(10)
    ]

    return FleetAnalyticsResponse(
        dataset_id=dataset_id,
        risk_histogram=risk_histogram,
        state_counts=summary["state_counts"],
        risk_trend=risk_trend,
    )

@router.get(
    "/fleet/compare",
    response_model=list[FleetCompareRow],
    summary="Compare fleet metrics across all datasets",
)
async def fleet_compare() -> list[FleetCompareRow]:
    from api.main import _fleet_summary_cache

    result = []
    for d_id in sorted(VALID_DATASETS):
        if d_id in _fleet_summary_cache:
            summary = _fleet_summary_cache[d_id]
            result.append(
                FleetCompareRow(
                    dataset_id=d_id,
                    fleet_size=summary["n_engines"],
                    state_counts=summary["state_counts"],
                    n_critical=summary["n_critical"],
                    mean_rul=summary["mean_rul"],
                    median_rul=summary["median_rul"],
                )
            )
    return result
