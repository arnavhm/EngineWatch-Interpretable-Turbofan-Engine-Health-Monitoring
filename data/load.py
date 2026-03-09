"""
Module : data/load.py

Purpose :Load raw CMAPPS text files into structured dataframes.
Why : CMAPPS Files have no headers - schema must be applied at load time.
Assumptions : Files are space separated, 26 columns, no missing values.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import yaml


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> dict:
    """
    Load central YAML configuration.

    Input : Path string or Path object to config.yaml
    Output: dict
    """
    path = Path(config_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_cmapss_file(file_path: str, columns: list) -> pd.DataFrame:
    """
    Load a single CMAPPS raw text file.

    Input Shape : Text file - N rows x 26 space-separated columns
    Output Shape : pd.DataFrame (N, 26)

    Raises:
        FileNotFoundError : If the specified file does not exist.
        Value Error : if column count != 26
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")

    if df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, got {df.shape[1]} ")

    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df


def load_rul_file(rul_path: str) -> pd.Series:
    """
    Load ground-truth RUL vectors for test set.

    Input Shape : Text file - M rows x 1 column
    Output Shape : pd.Series of length M, integer dtype, indexed from 1
    """
    path = Path(rul_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {rul_path}")

    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"], engine="python")
    rul.index = rul.index + 1

    return rul["rul"].astype(int)


def load_dataset(config: dict) -> tuple:
    """
    Orchestrate loading of train, test, and RUL files.

    Input  : config dict from load_config()
    Output : (train_df, test_df, rul)
             train_df shape : (N_train_cycles, 26)
             test_df shape  : (N_test_cycles, 26)
             rul shape      : (N_test_engines,)
    """
    raw_path = Path(config["dataset"]["raw_path"])
    columns = config["dataset"]["columns"]

    train_df = load_cmapss_file(raw_path / config["dataset"]["train_file"], columns)
    test_df = load_cmapss_file(raw_path / config["dataset"]["test_file"], columns)
    rul = load_rul_file(raw_path / config["dataset"]["rul_file"])

    print(f"[load] Train shape : {train_df.shape}")
    print(f"[load] Test shape  : {test_df.shape}")
    print(f"[load] RUL entries : {len(rul)}")

    return train_df, test_df, rul
