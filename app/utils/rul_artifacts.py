"""Utilities for loading pre-trained RUL model artifacts safely."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import joblib
from data.load import load_config


def _project_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parent.parent.parent


def _candidate_artifact_paths(config: dict[str, Any]) -> list[Path]:
    """Return artifact paths from config first, then known legacy locations."""
    root = _project_root()

    configured_dir = Path(config["rul"]["save_path"])
    if configured_dir.is_absolute():
        configured_path = configured_dir / "rul_artifacts.joblib"
    else:
        configured_path = root / configured_dir / "rul_artifacts.joblib"

    # Legacy notebooks location from earlier project iterations.
    legacy_notebooks_path = root / "notebooks" / "models" / "rul_artifacts.joblib"

    candidates = [configured_path.resolve(), legacy_notebooks_path.resolve()]

    # De-duplicate while preserving configured path priority.
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates


@st.cache_resource
def load_or_rebuild_rul_artifacts(dataset_id: str = "FD001") -> Any:
    """
    Load RUL artifacts from disk for the specified dataset.

    Purpose:
        Keep dashboard runtime read-only and fail fast when artifacts
        are missing or incompatible with the current environment.
        Cache key includes dataset_id so FD001 artifacts are not reused for FD002, etc.
    """
    config = load_config()
    config["dataset_id"] = dataset_id
    load_errors: list[str] = []

    # Try dataset-specific artifacts first (models/FD001/, models/FD002/, etc.)
    dataset_artifact_dir = (
        Path(_project_root()) / "models" / dataset_id / "rul_artifacts.joblib"
    )
    if dataset_artifact_dir.exists():
        try:
            return joblib.load(dataset_artifact_dir)
        except Exception as error:
            load_errors.append(f"{dataset_artifact_dir}: {error}")

    for path in _candidate_artifact_paths(config):
        if not path.exists():
            continue
        try:
            return joblib.load(path)
        except Exception as error:
            load_errors.append(f"{path}: {error}")
            continue

    candidate_paths = [str(path) for path in _candidate_artifact_paths(config)]
    error_details = (
        "\n".join(load_errors)
        if load_errors
        else "No candidate artifact file could be loaded."
    )
    raise RuntimeError(
        "Unable to load RUL artifacts for dashboard inference.\n"
        "Dashboard runtime does not retrain models.\n"
        "Run offline training first: python scripts/train_rul_artifacts.py\n"
        f"Searched paths: {candidate_paths}\n"
        f"Load errors: {error_details}"
    )
