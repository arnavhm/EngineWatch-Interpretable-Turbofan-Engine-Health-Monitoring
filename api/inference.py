"""Inference adapter for the API Layer.

Delegates to the pure prediction core (model/predict.py). The API never calls the pipeline directly - this guarantees API and dashboard return identical numbers. Read-only: no retraining, no mutation of outputs.
"""

from __future__ import annotations

from model.predict import predict_engine_by_id


def get_engine_prediction(engine_id: int, dataset_id: str = "FD001") -> dict:
    """
    Purpose: Resolve one engine's prediction for an API requst.
    Input: engine_id - CMAPPS unit number; dataset_id - CMAPPS subset (FD001–FD004)
    Output: prediction dict from the pure core (rul_cycles, risk_state, CI, etc.)
    Assumptions: engine_id exists in the dataset's test split
    Failure: ValueError if engine_id not found (caller maps to HTTP 404);
    FileNotFoundError if artifacts/raw data missing (maps to 503)
    """
    return predict_engine_by_id(engine_id, dataset_id)
