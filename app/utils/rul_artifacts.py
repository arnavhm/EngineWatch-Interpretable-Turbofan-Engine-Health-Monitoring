"""Utilities for loading pre-trained RUL model artifacts safely."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from data.load import load_config


def _project_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parent.parent.parent


def _candidate_artifact_paths(config: dict[str, Any]) -> list[Path]:
    """Return artifact paths from config."""
    root = _project_root()

    configured_dir = Path(config["rul"]["save_path"])
    if configured_dir.is_absolute():
        configured_path = configured_dir / "rul_artifacts.joblib"
    else:
        configured_path = root / configured_dir / "rul_artifacts.joblib"

    return [configured_path.resolve()]


def _load_rul_artifacts_uncached(dataset_id: str = "FD001") -> Any:
    """
    Purpose:     Pure RUL artifact load — no Streamlit, no caching.
                 Importable from FastAPI, scripts, tests.
    Input:       dataset_id — one of FD001–FD004
    Output:      RULArtifacts dataclass for that dataset
    Assumptions: Artifacts exist at the config-resolved path for dataset_id
    Failure:     FileNotFoundError if the .joblib is missing
    """
    config = load_config()
    config["dataset_id"] = dataset_id
    # Try dataset-specific artifacts first (models/FD001/, models/FD002/, etc.)
    artifact_name = os.environ.get("RUL_ARTIFACT_NAME", "rul_artifacts.joblib")
    dataset_artifact_dir = Path(_project_root()) / "models" / dataset_id / artifact_name
    if dataset_artifact_dir.exists():
        return joblib.load(dataset_artifact_dir)

    for path in _candidate_artifact_paths(config):
        if path.exists():
            return joblib.load(path)

    candidate_paths = [str(dataset_artifact_dir)] + [str(path) for path in _candidate_artifact_paths(config)]
    raise FileNotFoundError(
        "Unable to load RUL artifacts for dashboard inference.\n"
        "Dashboard runtime does not retrain models.\n"
        "Run offline training first: python scripts/train_rul_artifacts.py\n"
        f"Searched paths: {candidate_paths}"
    )


@st.cache_resource
def load_or_rebuild_rul_artifacts(dataset_id: str = "FD001") -> Any:
    """Streamlit-cached wrapper. Dashboard entry point — behavior unchanged."""
    return _load_rul_artifacts_uncached(dataset_id)
