"""Offline training entrypoint for RUL model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.data_loader import build_pipeline_data
from data.load import load_config, load_dataset
from model.rul import build_rul_model


def _attach_test_rul(test_df: pd.DataFrame, rul_offsets: pd.Series) -> pd.DataFrame:
    """Add cycle-level test RUL using CMAPSS offset convention."""
    test_with_rul = test_df.copy()
    max_cycle_by_unit = test_with_rul.groupby("unit")["cycle"].transform("max")
    final_rul_by_unit = test_with_rul["unit"].map(rul_offsets)

    if final_rul_by_unit.isna().any():
        missing_units = (
            test_with_rul.loc[final_rul_by_unit.isna(), "unit"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Missing ground-truth RUL offsets for test units: {missing_units}"
        )

    test_with_rul["RUL"] = (
        max_cycle_by_unit - test_with_rul["cycle"] + final_rul_by_unit
    )
    return test_with_rul


def main() -> None:
    """Train and persist RUL artifacts from the current code/config environment."""
    config = load_config()
    train_rs, test_rs = build_pipeline_data(persist_outputs=True)
    _, _, test_rul_offsets = load_dataset(config)
    test_with_rul = _attach_test_rul(test_rs, test_rul_offsets)

    predictions_df, artifacts = build_rul_model(train_rs, test_with_rul, config)

    save_path = Path(config["rul"]["save_path"]).resolve()
    print(f"Saved artifacts to: {save_path}")
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Prediction rows: {len(predictions_df)}")

    import joblib
    import os
    from model.predict import FEATURE_COLUMNS, _compute_rf_ci
    import numpy as np

    dataset_id = "FD001"
    
    # 1. Run the full inference pipeline for all 100 FD001 test engines using the already-trained artifacts
    last = test_rs.sort_values("cycle").groupby("unit").last().reset_index()
    preds = artifacts.best_model.predict(last[FEATURE_COLUMNS])
    preds = [max(float(p), 0.0) for p in preds]
    
    rf_model = artifacts.all_models.get("random_forest")
    env_ci = os.environ.get("ENABLE_CI")
    if env_ci is not None:
        enable_ci = env_ci.lower() in ("true", "1", "yes")
    else:
        enable_ci = config.get("api", {}).get("enable_ci", True)
    compute_ci = enable_ci and rf_model is not None and hasattr(rf_model, "estimators_")
    
    per_engine = {}
    engine_dicts_list = []
    
    for i, row in last.iterrows():
        engine_id = int(row["unit"])
        features = row[FEATURE_COLUMNS]
        predicted_rul = preds[i]
        
        ci_lower = ci_upper = ci_std = None
        if compute_ci:
            ci_lower, ci_upper, ci_std = _compute_rf_ci(rf_model, features.values.reshape(1, -1), predicted_rul)
            
        pred_dict = {
            "engine_id": engine_id,
            "dataset_id": dataset_id,
            "health_index": float(row["health_index"]),
            "risk_score": float(row["risk_score"]),
            "risk_state": str(row["risk_state"]),
            "rul_cycles": predicted_rul,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_std": ci_std,
            "model_name": artifacts.best_model_name,
            "rmse": float(artifacts.evaluation_metrics[artifacts.best_model_name]["rmse"])
        }
        per_engine[engine_id] = pred_dict
        engine_dicts_list.append(pred_dict)
        
    engine_dicts_list.sort(key=lambda x: x["risk_score"], reverse=True)
    
    counts = last["risk_state"].value_counts().to_dict()
    fleet_summary = {
        "dataset_id": dataset_id,
        "n_engines": len(per_engine),
        "state_counts": {
            "Healthy": int(counts.get("Healthy", 0)),
            "Degrading": int(counts.get("Degrading", 0)),
            "Critical": int(counts.get("Critical", 0))
        },
        "n_critical": int(counts.get("Critical", 0)),
        "mean_rul": round(float(np.mean(preds)), 2),
        "median_rul": round(float(np.median(preds)), 2),
        "highest_risk_engine": engine_dicts_list[0]["engine_id"]
    }
    
    top_risk = engine_dicts_list[:5]
    
    # 2. Build a fleet_cache dict
    fleet_cache = {
        "per_engine": per_engine,
        "fleet_summary": fleet_summary,
        "top_risk": top_risk
    }
    
    # 3. Save: joblib.dump(fleet_cache, "models/fleet_cache_FD001.pkl")
    root_dir = Path(__file__).resolve().parent.parent
    cache_path = root_dir / "models" / "fleet_cache_FD001.pkl"
    joblib.dump(fleet_cache, cache_path)
    
    # 4. Print: [artifacts] fleet_cache_FD001.pkl saved (100 engines)
    print(f"[artifacts] fleet_cache_FD001.pkl saved ({len(per_engine)} engines)")
    
    # 5. Build trajectory_cache for all 100 FD001 engines
    trajectory_cache = {}
    for engine_id, group in test_rs.groupby("unit"):
        engine_id = int(engine_id)
        group = group.sort_values("cycle")
        trajectory_cache[engine_id] = {
            "engine_id": engine_id,
            "dataset_id": dataset_id,
            "cycles": group["cycle"].astype(int).tolist(),
            "health_index": group["health_index"].astype(float).tolist(),
            "velocity": group["HI_velocity"].astype(float).tolist(),
            "variability": group["HI_variability"].astype(float).tolist()
        }
        
    # Save trajectory_cache
    traj_cache_path = root_dir / "models" / "trajectory_cache_FD001.pkl"
    joblib.dump(trajectory_cache, traj_cache_path)
    print(f"[artifacts] trajectory_cache_FD001.pkl saved ({len(trajectory_cache)} engines)")
    
    # 6. Build sensor_cache for all 100 FD001 engines
    # Map sensor_N columns to Saxena 2008 Table 2 physical names
    SENSOR_NAME_MAP = {
        "sensor_2": "T24", "sensor_3": "T30", "sensor_4": "T50",
        "sensor_7": "P30", "sensor_8": "Nf", "sensor_9": "Nc",
        "sensor_11": "Ps30", "sensor_12": "phi", "sensor_13": "NRf",
        "sensor_14": "NRc", "sensor_15": "BPR", "sensor_17": "htBleed",
        "sensor_20": "W31", "sensor_21": "W32"
    }
    sensor_cols = [c for c in test_rs.columns if c.startswith("sensor_")]
    sensor_cache = {}
    for eid, group in test_rs.groupby("unit"):
        eid = int(eid)
        group = group.sort_values("cycle")
        sensors = {}
        for col in sensor_cols:
            physical = SENSOR_NAME_MAP.get(col, col)
            sensors[physical] = group[col].astype(float).tolist()
        sensor_cache[eid] = {
            "engine_id": eid,
            "dataset_id": dataset_id,
            "cycles": group["cycle"].astype(int).tolist(),
            "sensors": sensors
        }
    sensor_cache_path = root_dir / "models" / "sensor_cache_FD001.pkl"
    joblib.dump(sensor_cache, sensor_cache_path)
    print(f"[artifacts] sensor_cache_FD001.pkl saved ({len(sensor_cache)} engines)")
    
    # 7. Build anomaly_cache for all 100 FD001 engines
    from evaluation.validation import detect_anomalous_engines
    anomaly_df = detect_anomalous_engines(test_rs)
    anomaly_cache = []
    for _, arow in anomaly_df.iterrows():
        eid = int(arow["unit"])
        engine_last = last[last["unit"] == eid].iloc[0]
        anomaly_cache.append({
            "engine_id": eid,
            "health_index": float(engine_last["health_index"]),
            "velocity": float(engine_last["HI_velocity"]),
            "is_anomaly": bool(arow["is_anomaly"]),
            "risk_state": str(engine_last["risk_state"])
        })
    anomaly_cache_path = root_dir / "models" / "anomaly_cache_FD001.pkl"
    joblib.dump(anomaly_cache, anomaly_cache_path)
    print(f"[artifacts] anomaly_cache_FD001.pkl saved ({len(anomaly_cache)} engines)")


if __name__ == "__main__":
    main()
