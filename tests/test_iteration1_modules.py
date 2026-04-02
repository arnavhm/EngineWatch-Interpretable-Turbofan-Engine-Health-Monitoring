from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.utils import rul_artifacts as rul_artifacts_module
from data.load import load_config
from features.health_index import build_health_index
from features.variability import compute_variability
from features.velocity import compute_velocity
from model.clustering import build_clustering
from model.risk import build_risk_score
from model.rul import build_rul_model


def _synthetic_health_frames(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensors = config["selected_sensors"]

    def _make_df(unit_ids: list[int], n_cycles: int) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        for unit in unit_ids:
            for cycle in range(1, n_cycles + 1):
                row: dict[str, float | int] = {"unit": unit, "cycle": cycle}
                for idx, sensor in enumerate(sensors):
                    row[sensor] = 1.0 - (0.03 * cycle) + (0.002 * idx)
                rows.append(row)
        return pd.DataFrame(rows)

    train_df = _make_df([1, 2], 12)
    test_df = _make_df([3], 10)
    return train_df, test_df


def _clustering_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "cycle": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "health_index": [0.95, 0.92, 0.9, 0.55, 0.52, 0.5, 0.2, 0.18, 0.15],
            "HI_velocity": [
                -0.01,
                -0.012,
                -0.013,
                -0.03,
                -0.031,
                -0.032,
                -0.05,
                -0.051,
                -0.053,
            ],
            "HI_variability": [0.04, 0.05, 0.05, 0.18, 0.19, 0.2, 0.35, 0.36, 0.38],
        }
    )
    test = pd.DataFrame(
        {
            "unit": [4, 4, 4, 5, 5, 5],
            "cycle": [1, 2, 3, 1, 2, 3],
            "health_index": [0.91, 0.88, 0.86, 0.25, 0.22, 0.2],
            "HI_velocity": [-0.012, -0.013, -0.014, -0.048, -0.05, -0.052],
            "HI_variability": [0.05, 0.06, 0.06, 0.31, 0.33, 0.35],
        }
    )
    return train, test


def _rul_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)

    rows_train: list[dict] = []
    for unit in range(1, 7):
        max_cycle = 14
        for cycle in range(1, max_cycle + 1):
            health_index = 1.0 - (cycle / max_cycle) + rng.normal(0.0, 0.01)
            velocity = -0.06 + rng.normal(0.0, 0.004)
            variability = 0.08 + (cycle / max_cycle) * 0.28 + rng.normal(0.0, 0.01)
            risk_score = np.clip(1.0 - health_index + 0.1 * variability, 0.0, 1.0)
            rul = max_cycle - cycle
            rows_train.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "health_index": float(np.clip(health_index, 0.0, 1.0)),
                    "HI_velocity": float(velocity),
                    "HI_variability": float(np.clip(variability, 0.0, 1.0)),
                    "risk_score": float(risk_score),
                    "RUL": int(rul),
                }
            )

    rows_test: list[dict] = []
    for unit in range(7, 11):
        max_cycle = 10
        for cycle in range(1, max_cycle + 1):
            health_index = 1.0 - (cycle / max_cycle) + rng.normal(0.0, 0.01)
            velocity = -0.065 + rng.normal(0.0, 0.004)
            variability = 0.09 + (cycle / max_cycle) * 0.25 + rng.normal(0.0, 0.01)
            risk_score = np.clip(1.0 - health_index + 0.1 * variability, 0.0, 1.0)
            rul = max_cycle - cycle
            rows_test.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "health_index": float(np.clip(health_index, 0.0, 1.0)),
                    "HI_velocity": float(velocity),
                    "HI_variability": float(np.clip(variability, 0.0, 1.0)),
                    "risk_score": float(risk_score),
                    "RUL": int(rul),
                }
            )

    return pd.DataFrame(rows_train), pd.DataFrame(rows_test)


def test_health_index_fit_transform_and_transform_bounds() -> None:
    config = load_config("config/config.yaml")
    train_df, test_df = _synthetic_health_frames(config)

    train_hi, test_hi, artifacts = build_health_index(train_df, test_df, config)

    assert "health_index" in train_hi.columns
    assert "health_index" in test_hi.columns
    assert train_hi["health_index"].between(0.0, 1.0).all()
    assert test_hi["health_index"].between(0.0, 1.0).all()
    assert artifacts.explained_variance_ratio > 0.0


def test_velocity_and_variability_edge_windows() -> None:
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "health_index": [1.0, 0.9, 0.8, 0.7, 0.6],
        }
    )
    config = {
        "rolling": {
            "window_size": 3,
            "min_periods": 2,
        }
    }

    with_velocity = compute_velocity(df, config)
    with_variability, _ = compute_variability(with_velocity, config)

    assert "HI_velocity" in with_variability.columns
    assert "HI_variability" in with_variability.columns
    assert with_variability["HI_velocity"].notna().all()
    assert with_variability["HI_variability"].between(0.0, 1.0).all()


def test_clustering_label_mapping_and_risk_range() -> None:
    config = load_config("config/config.yaml")
    train_df, test_df = _clustering_frames()

    train_cl, test_cl, cl_art = build_clustering(train_df, test_df, config)
    train_rs, test_rs, _ = build_risk_score(train_cl, test_cl, cl_art)

    assert set(train_rs["risk_state"].astype(str)).issubset(
        {"Healthy", "Degrading", "Critical"}
    )
    assert train_rs["risk_score"].between(0.0, 1.0).all()
    assert test_rs["risk_score"].between(0.0, 1.0).all()

    risk_by_state = train_rs.groupby(train_rs["risk_state"].astype(str))[
        "risk_score"
    ].mean()
    assert risk_by_state["Critical"] > risk_by_state["Healthy"]


def test_rul_model_selection_is_deterministic(tmp_path) -> None:
    train_df, test_df = _rul_frames()
    config = load_config("config/config.yaml")
    config["rul"]["save_path"] = str(tmp_path / "models")
    config["rul"]["models"]["random_forest_n_estimators"] = 40
    config["rul"]["models"]["gradient_boosting_n_estimators"] = 40

    pred_df_1, artifacts_1 = build_rul_model(train_df, test_df, config)
    pred_df_2, artifacts_2 = build_rul_model(train_df, test_df, config)

    assert artifacts_1.best_model_name == artifacts_2.best_model_name
    assert pred_df_1["predicted_RUL"].tolist() == pred_df_2["predicted_RUL"].tolist()
    assert {"late", "early", "on_time"}.issubset(artifacts_1.prediction_balance.keys())


def test_artifact_loader_raises_when_no_artifact_available(
    monkeypatch, tmp_path
) -> None:
    config = load_config("config/config.yaml")
    config["rul"]["save_path"] = str(tmp_path / "missing_models")

    monkeypatch.setattr(rul_artifacts_module, "load_config", lambda: config)
    monkeypatch.setattr(
        rul_artifacts_module,
        "_candidate_artifact_paths",
        lambda _: [tmp_path / "missing_models" / "rul_artifacts.joblib"],
    )

    with pytest.raises(RuntimeError, match="Dashboard runtime does not retrain models"):
        rul_artifacts_module.load_or_rebuild_rul_artifacts()
