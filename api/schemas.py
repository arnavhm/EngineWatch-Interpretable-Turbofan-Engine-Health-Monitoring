"""Pydantic Schemas for the EngineWatch Inference API

Defines the request/response contract. These mirror the columns the pipeline already produces, so no new computation is introduced here.
"""

from typing import Optional

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
    ci_lower: Optional[float] = Field(None, description="Lower CI bound (cycles)")
    ci_upper: Optional[float] = Field(None, description="Upper CI bound (cycles)")
    ci_std: Optional[float] = Field(None, description="RF tree-variance std (cycles)")
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


class FleetHandover(BaseModel):
    """
    Purpose:     Daily shift-handover report combining pipeline facts with an
                 optional LLM-authored narrative.
    Assumptions: narrative is null when Gemini is unavailable — never an error.
    """

    dataset_id: str
    facts: dict = Field(..., description="Structured fleet facts from the pipeline")
    narrative: Optional[str] = Field(
        None,
        description="Gemini-authored shift narrative; null when Gemini is unavailable",
    )
    narration_available: bool = Field(
        ..., description="True when the Gemini call succeeded"
    )


class ApiVersion(BaseModel):
    """
    Purpose:     Return the currently deployed git commit hash and short status.
    Input:       none.
    Output:      {"commit": str, "commit_short": str, "dirty": bool}
    Assumptions: process is running from within the git repo checkout.
    Failure:     git command fails or repo not found — raises explicitly (500).
    """

    commit: str
    commit_short: str
    dirty: bool
