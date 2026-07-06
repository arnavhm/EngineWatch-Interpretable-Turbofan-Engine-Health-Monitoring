"""Offline training entrypoint for RUL model artifacts.

ARCHITECTURE NOTE:
    This script uses load_pipeline_data_uncached() — the same canonical pipeline
    that predict_engine_by_id() and the FastAPI /predict endpoint use at inference
    time. Using any other pipeline function will produce
    training features that diverge from inference features, causing the fleet_cache
    and live inference to disagree. Do not change this without updating the API too.

ARTIFACT PATHS:
    RUL model artifacts → models/{dataset_id}/  (where _load_rul_artifacts_uncached loads from)
    Fleet cache files   → models/               (where api/main.py loads from at startup)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.data_loader import load_pipeline_data_uncached
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


def main(dataset_id: str = "FD001") -> None:
    """Train and persist RUL artifacts using the canonical inference pipeline.

    Purpose:      Single entrypoint that guarantees training features == inference
                  features. Both use load_pipeline_data_uncached, so the fleet_cache
                  and the live /predict endpoint will always agree.
    Failure:      Raises if canonical gate (Engine 34) fails post-generation.
    """
    import joblib
    from data.regime import resolve_regime_config
    from model.predict import FEATURE_COLUMNS, _compute_rf_ci

    config = resolve_regime_config(load_config(), dataset_id)
    config["dataset_id"] = dataset_id
    config["dataset"]["name"] = dataset_id
    config["dataset"]["train_file"] = f"train_{dataset_id}.txt"
    config["dataset"]["test_file"] = f"test_{dataset_id}.txt"
    config["dataset"]["rul_file"] = f"RUL_{dataset_id}.txt"

    # ── 1. Run the canonical pipeline (same code path as the API) ─────────
    print(f"[train] Running canonical pipeline for {dataset_id}...")
    train_rs, test_rs, scaler = load_pipeline_data_uncached(dataset_id)

    _, test_raw, test_rul_offsets = load_dataset(config)
    test_with_rul = _attach_test_rul(test_rs, test_rul_offsets)

    # ── 2. Train RUL model ────────────────────────────────────────────────
    print("[train] Training RUL model...")
    predictions_df, artifacts = build_rul_model(train_rs, test_with_rul, config)

    best = artifacts.best_model_name
    rmse = float(artifacts.evaluation_metrics[best]["rmse"])
    print(f"[train] Best model: {best}  RMSE: {rmse:.3f}")

    # ── 3. Save RUL artifacts to models/{dataset_id}/ ────────────────────
    #    _load_rul_artifacts_uncached loads from this subdirectory path.
    root_dir = Path(__file__).resolve().parent.parent
    artifact_dir = root_dir / "models" / dataset_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts, artifact_dir / "rul_artifacts.joblib")
    for model_name, model_obj in artifacts.all_models.items():
        fname = f"rul_{model_name.replace(' ', '_')}.joblib"
        joblib.dump(model_obj, artifact_dir / fname)

    # Legacy flat path — only ever meant for FD001-era callers that predate
    # per-dataset artifact directories. Writing this for any other dataset_id
    # would silently overwrite whatever currently depends on it.
    if dataset_id == "FD001":
        legacy_dir = root_dir / "models"
        joblib.dump(artifacts, legacy_dir / "rul_artifacts.joblib")

    print(f"[train] RUL artifacts saved → {artifact_dir}")

    # ── 4. Build fleet cache from in-memory model + already-computed test_rs ──
    #    No second pipeline run. Uses test_rs from step 1 and artifacts.best_model
    #    directly. Values are identical to what /predict returns via live inference
    #    because both use the same features (load_pipeline_data_uncached) and the
    #    same model (artifacts.best_model).
    print("[train] Generating fleet cache from in-memory model...")
    import os

    enable_ci = os.environ.get("ENABLE_CI", "true").lower() in ("true", "1", "yes")
    rf_model = artifacts.all_models.get("random_forest")
    compute_ci = (
        enable_ci
        and rf_model is not None
        and hasattr(rf_model, "estimators_")
    )

    # ── 5. Build structured fleet_cache (matches api/main.py expectations) ─
    last = test_rs.sort_values("cycle").groupby("unit").last().reset_index()
    raw_preds = artifacts.best_model.predict(last[FEATURE_COLUMNS])

    per_engine: dict[int, dict] = {}
    engine_dicts_list: list[dict] = []

    for i, row in last.iterrows():
        engine_id = int(row["unit"])
        predicted_rul = max(float(raw_preds[i]), 0.0)
        ci_lower = ci_upper = ci_std = None

        if compute_ci:
            ci_lower, ci_upper, ci_std = _compute_rf_ci(
                rf_model,
                last.iloc[[i]][FEATURE_COLUMNS].values,
                predicted_rul,
            )

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
            "model_name": best,
            "rmse": rmse,
        }
        per_engine[engine_id] = pred_dict
        engine_dicts_list.append(pred_dict)

    engine_dicts_list.sort(key=lambda x: x["risk_score"], reverse=True)

    preds = [d["rul_cycles"] for d in engine_dicts_list]
    counts = last["risk_state"].value_counts().to_dict()
    fleet_summary = {
        "dataset_id": dataset_id,
        "n_engines": len(per_engine),
        "state_counts": {
            "Healthy": int(counts.get("Healthy", 0)),
            "Degrading": int(counts.get("Degrading", 0)),
            "Critical": int(counts.get("Critical", 0)),
        },
        "n_critical": int(counts.get("Critical", 0)),
        "mean_rul": round(float(np.mean(preds)), 2),
        "median_rul": round(float(np.median(preds)), 2),
        "highest_risk_engine": engine_dicts_list[0]["engine_id"],
    }

    fleet_cache = {
        "per_engine": per_engine,
        "fleet_summary": fleet_summary,
        "top_risk": engine_dicts_list[:5],
    }

    # ── 6. Save fleet_cache to models/ root ──────────────────────────────
    cache_path = root_dir / "models" / f"fleet_cache_{dataset_id}.pkl"
    joblib.dump(fleet_cache, cache_path)
    print(f"[artifacts] fleet_cache_{dataset_id}.pkl saved ({len(per_engine)} engines)")

    # ── 7. Trajectory cache ───────────────────────────────────────────────
    trajectory_cache: dict[int, dict] = {}
    for engine_id, group in test_rs.groupby("unit"):
        engine_id = int(engine_id)
        group = group.sort_values("cycle")
        trajectory_cache[engine_id] = {
            "engine_id": engine_id,
            "dataset_id": dataset_id,
            "cycles": group["cycle"].astype(int).tolist(),
            "health_index": group["health_index"].astype(float).tolist(),
            "velocity": group["HI_velocity"].astype(float).tolist(),
            "variability": group["HI_variability"].astype(float).tolist(),
        }
    traj_path = root_dir / "models" / f"trajectory_cache_{dataset_id}.pkl"
    joblib.dump(trajectory_cache, traj_path)
    print(
        f"[artifacts] trajectory_cache_{dataset_id}.pkl saved "
        f"({len(trajectory_cache)} engines)"
    )

    # ── 8. Sensor cache ───────────────────────────────────────────────────
    SENSOR_NAME_MAP = {
        "sensor_2": "T24", "sensor_3": "T30", "sensor_4": "T50",
        "sensor_7": "P30", "sensor_8": "Nf",  "sensor_9": "Nc",
        "sensor_11": "Ps30", "sensor_12": "phi", "sensor_13": "NRf",
        "sensor_14": "NRc", "sensor_15": "BPR", "sensor_17": "htBleed",
        "sensor_20": "W31", "sensor_21": "W32",
    }
    sensor_cols = [c for c in test_rs.columns if c.startswith("sensor_")]
    
    setting_cols = config["regimes"]["setting_cols"]
    test_rs_with_settings = test_rs.merge(
        test_raw[["unit", "cycle"] + setting_cols],
        on=["unit", "cycle"], how="left", validate="one_to_one",
    )
    fitted_sensor_cols = list(scaler._sensor_cols)
    physical_df = scaler.inverse_transform_df(test_rs_with_settings, fitted_sensor_cols)

    sensor_cache: dict[int, dict] = {}
    for eid, group in physical_df.groupby("unit"):
        eid = int(eid)
        group = group.sort_values("cycle")
        sensors = {
            SENSOR_NAME_MAP.get(col, col): group[col].astype(float).tolist()
            for col in sensor_cols
        }
        sensor_cache[eid] = {
            "engine_id": eid,
            "dataset_id": dataset_id,
            "cycles": group["cycle"].astype(int).tolist(),
            "sensors": sensors,
        }
    sensor_path = root_dir / "models" / f"sensor_cache_{dataset_id}.pkl"
    joblib.dump(sensor_cache, sensor_path)
    print(f"[artifacts] sensor_cache_{dataset_id}.pkl saved ({len(sensor_cache)} engines)")

    # ── 9. Anomaly cache ──────────────────────────────────────────────────
    from evaluation.validation import detect_anomalous_engines

    anomaly_df = detect_anomalous_engines(test_rs)
    anomaly_cache = []
    for _, arow in anomaly_df.iterrows():
        eid = int(arow["unit"])
        engine_last = last[last["unit"] == eid]
        if engine_last.empty:
            continue
        engine_last = engine_last.iloc[0]
        anomaly_cache.append(
            {
                "engine_id": eid,
                "health_index": float(engine_last["health_index"]),
                "velocity": float(engine_last["HI_velocity"]),
                "is_anomaly": bool(arow["is_anomaly"]),
                "risk_state": str(engine_last["risk_state"]),
            }
        )
    anomaly_path = root_dir / "models" / f"anomaly_cache_{dataset_id}.pkl"
    joblib.dump(anomaly_cache, anomaly_path)
    print(f"[artifacts] anomaly_cache_{dataset_id}.pkl saved ({len(anomaly_cache)} engines)")

    # ── 10. Canonical gate — fail loudly if Engine 34 is wrong ───────────
    if dataset_id == "FD001":
        e34 = per_engine.get(34)
        if e34 is None:
            raise RuntimeError("Engine 34 not found in fleet_cache — training aborted")

        CANONICAL_RISK  = 0.7402876566726511
        CANONICAL_RUL   = 3.698652753342952
        RISK_TOL        = 0.01
        RUL_TOL         = 0.5

        risk_ok = abs(e34["risk_score"] - CANONICAL_RISK) < RISK_TOL
        rul_ok  = abs(e34["rul_cycles"]  - CANONICAL_RUL)  < RUL_TOL

        print(f"\n[gate] Engine 34 / {dataset_id}")
        print(f"       risk_score : {e34['risk_score']:.6f}  (canonical {CANONICAL_RISK})")
        print(f"       rul_cycles : {e34['rul_cycles']:.6f}  (canonical {CANONICAL_RUL})")
        print(f"       rmse       : {e34['rmse']:.3f}")

        if not (risk_ok and rul_ok):
            raise RuntimeError(
                f"[gate] FAILED — training artifacts do not match canonical Engine 34 values.\n"
                f"       risk_score delta : {abs(e34['risk_score'] - CANONICAL_RISK):.6f} (tol {RISK_TOL})\n"
                f"       rul_cycles delta : {abs(e34['rul_cycles'] - CANONICAL_RUL):.6f} (tol {RUL_TOL})\n"
                f"       Artifacts NOT saved to production paths. Investigate before redeploying."
            )

        print("[gate] PASSED ✓  — fleet_cache matches canonical Engine 34 values")
    print("[train] Done.")


if __name__ == "__main__":
    dataset_id = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    main(dataset_id)
