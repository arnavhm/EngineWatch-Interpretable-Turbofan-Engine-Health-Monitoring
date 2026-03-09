"""
Module: data/preprocess.py

Purpose    : Transform raw CMAPSS data into model-ready features.
Assumptions:
  - Scaler ALWAYS fit on training data only — never on test data.
  - Sensor selection driven entirely by config — no hardcoded lists.
  - Training trajectories run to failure: last cycle RUL = 0.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) to a training dataframe.

    Purpose:
        Compute cycle-wise RUL using the rule: RUL = max_cycle(unit) - cycle.
    Input shape:
        DataFrame with columns `unit` and `cycle`, one row per cycle.
    Output shape:
        Same DataFrame shape + one additional `RUL` column.
    Assumptions:
        Trajectories run to failure, so last cycle RUL is 0.
    Failure conditions:
        Raises KeyError if `unit` or `cycle` is missing.
    """
    required_columns = {"unit", "cycle"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns for RUL computation: {sorted(missing_columns)}"
        )

    max_cycles = df.groupby("unit")["cycle"].max().reset_index(name="max_cycle")

    df_with_rul = df.merge(max_cycles, on="unit", how="left")
    df_with_rul["RUL"] = df_with_rul["max_cycle"] - df_with_rul["cycle"]
    df_with_rul = df_with_rul.drop(columns=["max_cycle"])

    return df_with_rul


def select_sensors(df: pd.DataFrame, selected_sensors: list[str]) -> pd.DataFrame:
    """
    Select identifiers and configured sensor columns.

    Purpose:
        Keep only `unit`, `cycle`, configured sensors, and optional `RUL`.
    Input shape:
        DataFrame with CMAPSS columns.
    Output shape:
        DataFrame with columns [`unit`, `cycle`] + selected_sensors + optional [`RUL`].
    Assumptions:
        Sensor names come from config and must be present in the dataframe.
    Failure conditions:
        Raises KeyError if required columns or any configured sensor is missing.
    """
    base_columns = ["unit", "cycle"]
    missing_base = [column for column in base_columns if column not in df.columns]
    if missing_base:
        raise KeyError(f"Missing required base columns: {missing_base}")

    missing_sensors = [
        sensor for sensor in selected_sensors if sensor not in df.columns
    ]
    if missing_sensors:
        raise KeyError(f"Configured sensors not found in dataframe: {missing_sensors}")

    output_columns = base_columns + selected_sensors
    if "RUL" in df.columns:
        output_columns.append("RUL")

    return df[output_columns].copy()


def fit_scaler(df: pd.DataFrame, sensor_cols: list[str]) -> StandardScaler:
    """
    Fit StandardScaler on training sensor columns only.

    Purpose:
        Learn normalization statistics from training sensor values.
    Input shape:
        Training DataFrame and sensor column list.
    Output shape:
        Fitted `StandardScaler`.
    Assumptions:
        Called on training data only.
    Failure conditions:
        Raises KeyError if any sensor column is missing.
    """
    missing_sensors = [sensor for sensor in sensor_cols if sensor not in df.columns]
    if missing_sensors:
        raise KeyError(f"Sensor columns missing for scaler fitting: {missing_sensors}")

    scaler = StandardScaler()
    scaler.fit(df[sensor_cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame, sensor_cols: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    """
    Apply a fitted scaler to sensor columns while preserving identifiers.

    Purpose:
        Standardize only sensor columns; keep `unit`, `cycle`, and `RUL` unchanged.
    Input shape:
        DataFrame, sensor columns list, and fitted scaler.
    Output shape:
        DataFrame with same shape and scaled sensor columns.
    Assumptions:
        Scaler was fit using the same sensor columns in the same order.
    Failure conditions:
        Raises KeyError for missing columns; ValueError for feature count mismatch.
    """
    missing_sensors = [sensor for sensor in sensor_cols if sensor not in df.columns]
    if missing_sensors:
        raise KeyError(
            f"Sensor columns missing for scaler transform: {missing_sensors}"
        )

    if scaler.n_features_in_ != len(sensor_cols):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, received {len(sensor_cols)}"
        )

    scaled_df = df.copy()
    scaled_df[sensor_cols] = scaler.transform(scaled_df[sensor_cols])
    return scaled_df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to parquet.

    Purpose:
        Persist processed datasets for downstream features/models.
    Input shape:
        Any processed DataFrame and output path string.
    Output shape:
        None.
    Assumptions:
        Parquet writer dependency is available.
    Failure conditions:
        Raises I/O or parquet engine errors from pandas.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def preprocess_train(
    df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, StandardScaler, list[str]]:
    """
    Preprocess training data end-to-end.

    Purpose:
        Compute RUL, keep configured sensors, fit scaler, scale sensors, and save parquet.
    Input shape:
        Raw training dataframe and loaded config.
    Output shape:
        Tuple of (processed_df, fitted_scaler, sensor_cols).
    Assumptions:
        Config contains `selected_sensors` and `dataset.processed_path`.
    Failure conditions:
        Raises KeyError for missing config/columns and downstream scaling or I/O errors.
    """
    sensor_cols: list[str] = config["selected_sensors"]

    train_with_rul = compute_rul(df)
    train_selected = select_sensors(train_with_rul, sensor_cols)

    scaler = fit_scaler(train_selected, sensor_cols)
    train_processed = apply_scaler(train_selected, sensor_cols, scaler)

    processed_path = Path(config["dataset"]["processed_path"])
    dataset_name = config["dataset"]["name"]
    output_file = processed_path / f"train_{dataset_name}_processed.parquet"
    save_processed(train_processed, str(output_file))

    return train_processed, scaler, sensor_cols


def preprocess_test(
    df: pd.DataFrame, config: dict, scaler: StandardScaler
) -> pd.DataFrame:
    """
    Preprocess test data using a pre-fitted training scaler.

    Purpose:
        Keep configured sensors, apply training scaler (no fit), and save parquet.
    Input shape:
        Raw test dataframe, loaded config, and fitted scaler.
    Output shape:
        Processed test dataframe.
    Assumptions:
        Scaler comes from `preprocess_train`.
    Failure conditions:
        Raises KeyError for missing config/columns and downstream scaling or I/O errors.
    """
    sensor_cols: list[str] = config["selected_sensors"]

    test_selected = select_sensors(df, sensor_cols)
    test_processed = apply_scaler(test_selected, sensor_cols, scaler)

    processed_path = Path(config["dataset"]["processed_path"])
    dataset_name = config["dataset"]["name"]
    output_file = processed_path / f"test_{dataset_name}_processed.parquet"
    save_processed(test_processed, str(output_file))

    return test_processed
