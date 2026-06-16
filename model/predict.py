"""Pure, Streamlit-free RUL prediction core.


Single Source of truth for engine predictions. Both the dashboard panel (app/components/rul_prediction.py) and the FastAPI service (api/) import predict_engine() so their numbers are provably identical. This module is read-only over the pipeline - it never retrains and never modifies outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.data_loader import load_pipeline_data_uncached
from app.utils.rul_artifacts import _load_rul_artifacts_uncached

FEATURE_COLUMNS = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


def _compute_rf_ci(
    rf_model: object, features: np.ndarray, point_pred: float
) -> tuple[float, float, float]:
    """
    Purpose:     RF tree-variance confidence interval, bound to the point prediction.
    Input:       rf_model with estimators_, features (1, n_features), GB point prediction
    Output:      (ci_lower, ci_upper, ci_std) in cycles; lower floored at 0.0
    Assumptions: rf_model is a fitted RandomForestRegressor
    Failure: AttributeError if rf_model has no estimators_
    """
    tree_preds = np.array([tree.predict(features)[0] for tree in rf_model.estimators_])
    ci_std = float(tree_preds.std())
    return max(point_pred - ci_std, 0.0), point_pred + ci_std, ci_std


def predict_engine(engine_df: pd.DataFrame, dataset_id: str = "FD001") -> dict:
    """
    Purpose:     Predict RUL + health state for one engine at its latest cycle.
    Input:       engine_df — single-engine DataFrame (all cycles, pipeline columns present);
                 dataset_id — one of FD001–FD004 (selects the right RUL artifacts)
    Output:      dict with engine_id, dataset_id, health_index, risk_score, risk_state,
                 rul_cycles, ci_lower, ci_upper — all native Python types (JSON-safe)
    Assumptions: FEATURE_COLUMNS and risk_state present on engine_df (pipeline already run)
    Failure:     KeyError if a feature column is missing; ValueError if engine_df is empty
    """
    if engine_df.empty:
        raise ValueError("engine_df is empty — no cycles to predict on")

    artifacts = _load_rul_artifacts_uncached(dataset_id=dataset_id)
    model = artifacts.best_model

    last_row = engine_df.iloc[[-1]]
    features = last_row[FEATURE_COLUMNS]

    predicted_rul = max(float(model.predict(features)[0]), 0.0)

    ci_lower = ci_upper = ci_std = None
    rf_model = artifacts.all_models.get("random_forest")
    if rf_model is not None and hasattr(rf_model, "estimators_"):
        ci_lower, ci_upper, ci_std = _compute_rf_ci(rf_model, features.values, predicted_rul)

    return {
        "engine_id": int(last_row["unit"].iloc[0]),
        "dataset_id": dataset_id,
        "health_index": float(last_row["health_index"].iloc[0]),
        "risk_score": float(last_row["risk_score"].iloc[0]),
        "risk_state": str(last_row["risk_state"].iloc[0]),
        "rul_cycles": predicted_rul,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_std": ci_std,
        "model_name": artifacts.best_model_name,
        "rmse": float(artifacts.evaluation_metrics[artifacts.best_model_name]["rmse"]),
    }


def predict_engine_by_id(engine_id: int, dataset_id: str = "FD001") -> dict:
    """
    Purpose:     Convenience path for the API — run the pipeline for a dataset,
                 slice one engine, predict.
    Input:       engine_id — CMAPSS unit number; dataset_id — FD001–FD004
    Output:      same dict as predict_engine()
    Assumptions: engine_id exists in the dataset's test split
    Failure:     ValueError if engine_id not found in the dataset
    """
    _, test_df = load_pipeline_data_uncached(dataset_id)
    engine_df = test_df[test_df["unit"] == engine_id]
    if engine_df.empty:
        raise ValueError(f"Engine {engine_id} not found in {dataset_id} test split")
    return predict_engine(engine_df, dataset_id)


def predict_fleet(dataset_id: str = "FD001") -> pd.DataFrame:
    """
    Purpose:     Score the entire fleet in ONE pipeline pass — last cycle per engine,
                 RUL predicted for all engines in a single model.predict() call.
                 Single source of truth for fleet endpoints; avoids re-running the
                 pipeline per engine.
    Input:       dataset_id — one of FD001–FD004
    Output:      DataFrame, one row per engine, columns:
                 engine_id, health_index, risk_score, risk_state, rul_cycles
                 (sorted by risk_score descending)
    Assumptions: pipeline produces FEATURE_COLUMNS + risk_state + unit on test split
    Failure:     FileNotFoundError if artifacts/raw data missing
    """
    _, test_df = load_pipeline_data_uncached(dataset_id)

    # last cycle per engine (CMAPSS inference convention)
    last = (
        test_df.sort_values("cycle")
        .groupby("unit")
        .last()
        .reset_index()
    )

    artifacts = _load_rul_artifacts_uncached(dataset_id=dataset_id)
    model = artifacts.best_model

    # single batched prediction across the whole fleet
    preds = model.predict(last[FEATURE_COLUMNS])
    preds = [max(float(p), 0.0) for p in preds]

    return pd.DataFrame({
        "engine_id": last["unit"].astype(int),
        "health_index": last["health_index"].astype(float),
        "risk_score": last["risk_score"].astype(float),
        "risk_state": last["risk_state"].astype(str),
        "rul_cycles": preds,
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)
