"""Pydantic Schemas for the EngineWatch Inference API

Defines the request/response contract. These mirror the columns the pipeline already produces, so no new computation is introduced here.
"""

from pydantic import BaseModel, Field


class EnginePrediction(BaseModel):
    """
    Purpose:     One engine's health + RUL prediction at its latest cycle.
    Output:     JSON returned by /predict. Mirrors model.predict.predict_engine() exactly - no recomputation in the API layer (read-only contract).
    """

    engine_id: int
    dataset_id: str
    health_index: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_state: str = Field(..., description="Healthy | Degrading | Critical")
    rul_cycles: float
    ci_lower: float | None = Field(None, description="Lower CI bound (cycles)")
    ci_upper: float | None = Field(None, description="Upper CI bound (cycles)")
    ci_std: float | None = Field(None, description="RF tree-variance std (cycles)")
    model_name: str = Field(..., description="Model that produced the point prediction")
    rmse: float = Field(..., description="That model's validation RMSE")


class FleetEngine(BaseModel):
    """One engine's risk summary in a fleet listing."""
    engine_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_state: str
    rul_cycles: float


class FleetSummary(BaseModel):
    """Fleet-level aggregate health snapshot."""
    dataset_id: str
    n_engines: int
    state_counts: dict = Field(..., description="{Healthy, Degrading, Critical: count}")
    n_critical: int
    mean_rul: float
    median_rul: float
    highest_risk_engine: int = Field(..., description="engine_id of max risk_score")
