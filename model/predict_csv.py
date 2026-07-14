"""Transform-only CSV inference pipeline.

Scores arbitrary uploaded engine sensor logs WITHOUT a training split, by
applying PERSISTED transformers (scaler, PCA, KMeans, risk) loaded from disk.
Distinct from model/predict.py, which runs the full re-fit pipeline for known
CMAPSS engines. Read-only: no retraining, no mutation of artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.utils.rul_artifacts import _load_rul_artifacts_uncached
from data.load import load_config
from data.preprocess import preprocess_test
from features.health_index import apply_dual_health_index, assign_operative_features
from features.variability import compute_variability
from features.velocity import compute_velocity
from model.clustering import apply_clustering_per_fault_mode
# classify_engines lives in the fault-classification module — adjust this import
# to its real location (you referenced it as classify_engines).
from model.fault_classifier import \
    classify_engines  # TODO: confirm module path
from model.predict import FEATURE_COLUMNS, _compute_rf_ci
from model.risk import apply_risk_score_per_fault_mode

MIN_CYCLES = 20  # velocity/variability rolling window; last row needs >= this history


def _artifacts_dir(dataset_id: str, config: dict) -> Path:
    """Resolve the per-dataset model artifact directory (e.g. models/FD001)."""
    base = Path(config.get("rul", {}).get("save_path", "models"))
    return base / dataset_id


def predict_csv(raw_df: pd.DataFrame, dataset_id: str = "FD001") -> list[dict]:
    """
    Purpose:     Score uploaded raw engine sensor logs using PERSISTED transformers.
                 One prediction per engine (its latest cycle).
    Input:       raw_df — raw CMAPSS-shaped frame (unit, cycle, op settings, sensors);
                 dataset_id — selects which dataset's persisted transformers to apply.
    Output:      list of per-engine dicts (engine_id, rul_cycles, risk_state,
                 risk_score, health_index, ci_lower, ci_upper).
    Assumptions: raw_df columns match the dataset's expected schema; each engine has
                 >= MIN_CYCLES cycles (else it is skipped with a reason).
    Failure:     ValueError if raw_df empty or required artifacts missing;
                 KeyError if a fault_mode lacks persisted artifacts.
    """
    if raw_df.empty:
        raise ValueError("Uploaded CSV is empty")

    config = load_config()
    config["dataset_id"] = dataset_id
    art_dir = _artifacts_dir(dataset_id, config)

    # ── load persisted transformers (NOT re-fit) ──
    scaler = joblib.load(art_dir.parent / f"scaler_{dataset_id}.joblib")
    pca_by_axis = joblib.load(art_dir / "hi_pca_by_axis.joblib")
    hi_scaler_by_axis = joblib.load(art_dir / "hi_scaler_by_axis.joblib")
    fault_clf = joblib.load(art_dir / "fault_classifier.joblib")
    var_artifacts = joblib.load(art_dir / "variability_artifacts.joblib")
    cluster_by_fault = joblib.load(art_dir / "cluster_models_by_fault.joblib")
    risk_by_fault = joblib.load(art_dir / "risk_artifacts_by_fault.joblib")

    # ── transform-only chain (no fit anywhere) ──
    proc = preprocess_test(
        raw_df, config, scaler, persist_outputs=False
    )  # no disk side-effects
    hi = apply_dual_health_index(proc, pca_by_axis, hi_scaler_by_axis, config)
    vel = compute_velocity(hi, config)  # stateless slopes
    var, _ = compute_variability(
        vel, config, artifacts=var_artifacts
    )  # persisted bounds
    classified = classify_engines(var, fault_clf, config)  # persisted classifier
    classified = assign_operative_features(classified)
    clustered = apply_clustering_per_fault_mode(classified, cluster_by_fault)
    scored = apply_risk_score_per_fault_mode(clustered, cluster_by_fault, risk_by_fault)

    # ── per-engine prediction at latest cycle ──
    artifacts = _load_rul_artifacts_uncached(dataset_id=dataset_id)
    model = artifacts.best_model
    rf_model = artifacts.all_models.get("random_forest")
    model_name = artifacts.best_model_name
    rmse = float(artifacts.evaluation_metrics[model_name]["rmse"])

    results: list[dict] = []
    for engine_id, g in scored.groupby("unit"):
        if len(g) < MIN_CYCLES:
            results.append(
                {
                    "engine_id": int(engine_id),
                    "skipped": True,
                    "reason": f"only {len(g)} cycles; need >= {MIN_CYCLES} for velocity/variability",
                }
            )
            continue

        last = g.sort_values("cycle").iloc[[-1]]
        feats = last[FEATURE_COLUMNS]
        rul = max(float(model.predict(feats)[0]), 0.0)

        ci_lower = ci_upper = ci_std = None
        if rf_model is not None and hasattr(rf_model, "estimators_"):
            ci_lower, ci_upper, ci_std = _compute_rf_ci(rf_model, feats.values, rul)

        results.append(
            {
                "engine_id": int(engine_id),
                "dataset_id": dataset_id,
                "health_index": float(last["health_index"].iloc[0]),
                "risk_score": float(last["risk_score"].iloc[0]),
                "risk_state": str(last["risk_state"].iloc[0]),
                "rul_cycles": rul,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_std": ci_std,
                "model_name": model_name,
                "rmse": rmse,
                "skipped": False,
            }
        )

    return results
