from pathlib import Path

import pytest

from data.load import load_config, load_dataset
from data.preprocess import (
    apply_scaler,
    compute_rul,
    preprocess_test,
    preprocess_train,
    select_sensors,
)


def _test_config(tmp_path: Path) -> dict:
    config = load_config("config/config.yaml")
    config["dataset"]["processed_path"] = str(tmp_path / "processed")
    config["save_scaler"] = True
    config["scaler_path"] = str(tmp_path / "models" / "scaler.joblib")
    return config


def test_compute_rul_has_zero_at_last_cycle() -> None:
    config = load_config("config/config.yaml")
    train_df, _, _ = load_dataset(config)

    with_rul = compute_rul(train_df)
    last_cycles = with_rul.groupby("unit", as_index=False)["cycle"].max()
    merged = with_rul.merge(last_cycles, on=["unit", "cycle"], how="inner")

    assert (merged["RUL"] == 0).all()


def test_select_sensors_missing_column_raises() -> None:
    config = load_config("config/config.yaml")
    train_df, _, _ = load_dataset(config)

    with pytest.raises(KeyError):
        select_sensors(train_df, ["sensor_999"])


def test_preprocess_train_and_test_pipeline_outputs(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    train_df, test_df, _ = load_dataset(config)

    train_processed, scaler, sensor_cols = preprocess_train(train_df, config)
    test_processed = preprocess_test(test_df, config, scaler)

    assert train_processed.shape[0] == 20631
    assert test_processed.shape[0] == 13096
    assert train_processed["RUL"].min() == 0
    assert "RUL" not in test_processed.columns

    means = train_processed[sensor_cols].mean()
    stds = train_processed[sensor_cols].std()

    assert (means.abs() < 0.01).all()
    assert (stds - 1).abs().max() < 0.01

    scaler_path = Path(config["scaler_path"])
    assert scaler_path.exists()


def test_apply_scaler_feature_mismatch_raises(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    train_df, _, _ = load_dataset(config)

    train_processed, scaler, sensor_cols = preprocess_train(train_df, config)

    with pytest.raises(ValueError):
        apply_scaler(train_processed, sensor_cols[:-1], scaler)
