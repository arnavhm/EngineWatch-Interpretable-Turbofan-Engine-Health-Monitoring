
import pytest

from data.load import load_config, load_dataset
from data.preprocess import preprocess_test, preprocess_train
from features.health_index import build_health_index, assign_operative_features
from features.variability import build_variability
from features.velocity import build_velocity
from model.clustering import build_clustering_per_fault_mode
from model.fault_classifier import classify_engines, fit_fault_classifier
from model.risk import build_risk_score_per_fault_mode

REQUIRED_TRAIN_COLUMNS = [
    "unit",
    "cycle",
    "health_index",
    "HI_velocity",
    "HI_variability",
    "risk_state",
    "risk_score",
]

REQUIRED_TEST_COLUMNS = [
    "unit",
    "cycle",
    "health_index",
    "HI_velocity",
    "HI_variability",
    "risk_state",
    "risk_score",
]


@pytest.fixture(scope="module")
def fd001_pipeline_output():
    config = load_config()
    config["dataset_id"] = "FD001"
    config["dataset"]["name"] = "FD001"
    config["dataset"]["train_file"] = "train_FD001.txt"
    config["dataset"]["test_file"] = "test_FD001.txt"
    config["dataset"]["rul_file"] = "RUL_FD001.txt"

    train_raw, test_raw, _ = load_dataset(config)
    train_proc, scaler, _ = preprocess_train(train_raw, config, persist_outputs=False)
    test_proc = preprocess_test(test_raw, config, scaler, persist_outputs=False)
    train_hi, test_hi, _ = build_health_index(train_proc, test_proc, config)
    train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
    train_var, test_var, _ = build_variability(train_vel, test_vel, config)
    fault_artifacts = fit_fault_classifier(train_var, config)
    train_var = classify_engines(train_var, fault_artifacts, config)
    test_var = classify_engines(test_var, fault_artifacts, config)
    train_var = assign_operative_features(train_var)
    test_var = assign_operative_features(test_var)

    train_cl, test_cl, clusterers_by_mode = build_clustering_per_fault_mode(
        train_var, test_var, config
    )
    train_rs, test_rs, _ = build_risk_score_per_fault_mode(
        train_cl, test_cl, clusterers_by_mode
    )
    return train_rs, test_rs


def test_train_row_count(fd001_pipeline_output):
    train_rs, _ = fd001_pipeline_output
    assert len(train_rs) == 20631, f"Expected 20631 rows, got {len(train_rs)}"


def test_test_row_count(fd001_pipeline_output):
    _, test_rs = fd001_pipeline_output
    assert len(test_rs) == 13096, f"Expected 13096 rows, got {len(test_rs)}"


def test_required_train_columns_present(fd001_pipeline_output):
    train_rs, _ = fd001_pipeline_output
    missing = [c for c in REQUIRED_TRAIN_COLUMNS if c not in train_rs.columns]
    assert not missing, f"Missing columns in train output: {missing}"


def test_required_test_columns_present(fd001_pipeline_output):
    _, test_rs = fd001_pipeline_output
    missing = [c for c in REQUIRED_TEST_COLUMNS if c not in test_rs.columns]
    assert not missing, f"Missing columns in test output: {missing}"


def test_no_nan_in_train_features(fd001_pipeline_output):
    train_rs, _ = fd001_pipeline_output
    nan_cols = train_rs[REQUIRED_TRAIN_COLUMNS].isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    assert nan_cols.empty, f"NaN values found in train: {nan_cols.to_dict()}"


def test_no_nan_in_test_features(fd001_pipeline_output):
    _, test_rs = fd001_pipeline_output
    nan_cols = test_rs[REQUIRED_TEST_COLUMNS].isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    assert nan_cols.empty, f"NaN values found in test: {nan_cols.to_dict()}"


def test_health_index_range(fd001_pipeline_output):
    train_rs, test_rs = fd001_pipeline_output
    assert (
        train_rs["health_index"].between(0, 1).all()
    ), "health_index out of [0,1] on train"
    assert (
        test_rs["health_index"].between(0, 1).all()
    ), "health_index out of [0,1] on test"


def test_risk_score_range(fd001_pipeline_output):
    train_rs, test_rs = fd001_pipeline_output
    assert (
        train_rs["risk_score"].between(0, 1).all()
    ), "risk_score out of [0,1] on train"
    assert test_rs["risk_score"].between(0, 1).all(), "risk_score out of [0,1] on test"


def test_risk_state_valid_labels(fd001_pipeline_output):
    train_rs, test_rs = fd001_pipeline_output
    valid = {"Healthy", "Degrading", "Critical"}
    train_labels = set(train_rs["risk_state"].unique())
    test_labels = set(test_rs["risk_state"].unique())
    assert (
        train_labels <= valid
    ), f"Invalid risk_state labels in train: {train_labels - valid}"
    assert (
        test_labels <= valid
    ), f"Invalid risk_state labels in test: {test_labels - valid}"


def test_engine_count_train(fd001_pipeline_output):
    train_rs, _ = fd001_pipeline_output
    assert (
        train_rs["unit"].nunique() == 100
    ), f"Expected 100 train engines, got {train_rs['unit'].nunique()}"


def test_engine_count_test(fd001_pipeline_output):
    _, test_rs = fd001_pipeline_output
    assert (
        test_rs["unit"].nunique() == 100
    ), f"Expected 100 test engines, got {test_rs['unit'].nunique()}"


def test_scaler_not_fit_on_test(fd001_pipeline_output):
    # Verify test sensor columns have different mean than 0
    # (train is standardised to mean≈0; test is transformed but not refitted)
    _, test_rs = fd001_pipeline_output
    sensor_cols = [c for c in test_rs.columns if c.startswith("sensor_")]
    if sensor_cols:
        test_means = test_rs[sensor_cols].mean()
        # Test mean will not be exactly 0 — if it is, scaler was refit on test (bug)
        assert not (
            test_means.abs() < 1e-10
        ).all(), (
            "All test sensor means are 0.0 — scaler may have been refit on test data"
        )


def test_risk_score_state_agree(fd001_pipeline_output):
    """Risk score and state must agree in direction — guards the fan-inversion bug."""
    _, test_rs = fd001_pipeline_output
    last = test_rs.sort_values("cycle").groupby("unit").last()
    contradictions = last[(last.risk_score > 0.7) & (last.risk_state == "Healthy")]
    assert (
        len(contradictions) == 0
    ), f"{len(contradictions)} engines high-risk but Healthy"
