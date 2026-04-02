"""Utilities for loading and rebuilding RUL model artifacts safely."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.utils.data_loader import load_pipeline_data
from data.load import load_config, load_dataset
from model.rul import build_rul_model


def _project_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parent.parent.parent


def _candidate_artifact_paths(config: dict[str, Any]) -> list[Path]:
    """Return prioritized artifact file paths (config path first, then legacy path)."""
    root = _project_root()

    configured_dir = Path(config["rul"]["save_path"])
    if configured_dir.is_absolute():
        configured_path = configured_dir / "rul_artifacts.joblib"
    else:
        configured_path = root / configured_dir / "rul_artifacts.joblib"

    # Legacy artifact location kept as fallback for older runs.
    legacy_path = root / "notebooks" / "models" / "rul_artifacts.joblib"

    candidates = [configured_path.resolve()]
    if legacy_path.resolve() not in candidates:
        candidates.append(legacy_path.resolve())
    return candidates


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


def load_or_rebuild_rul_artifacts() -> Any:
    """
    Load RUL artifacts from disk, rebuilding if serialization is incompatible.

    Purpose:
        Handle scikit-learn/joblib incompatibilities across Python environments
        by retraining artifacts from the current code and dependency stack.
    """
    config = load_config()

    for path in _candidate_artifact_paths(config):
        if not path.exists():
            continue
        try:
            return joblib.load(path)
        except Exception:
            # Rebuild if an artifact cannot be deserialized in this environment.
            continue

    train_rs, test_rs = load_pipeline_data()
    _, _, test_rul_offsets = load_dataset(config)
    test_with_rul = _attach_test_rul(test_rs, test_rul_offsets)
    _, artifacts = build_rul_model(train_rs, test_with_rul, config)
    return artifacts
