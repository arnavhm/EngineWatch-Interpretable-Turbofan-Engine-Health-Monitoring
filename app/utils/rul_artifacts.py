"""Utilities for loading pre-trained RUL model artifacts safely."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from data.load import load_config


def _project_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parent.parent.parent


def _candidate_artifact_paths(config: dict[str, Any]) -> list[Path]:
    """Return artifact path from active project config."""
    root = _project_root()

    configured_dir = Path(config["rul"]["save_path"])
    if configured_dir.is_absolute():
        configured_path = configured_dir / "rul_artifacts.joblib"
    else:
        configured_path = root / configured_dir / "rul_artifacts.joblib"

    return [configured_path.resolve()]


def load_or_rebuild_rul_artifacts() -> Any:
    """
    Load RUL artifacts from disk.

    Purpose:
        Keep dashboard runtime read-only and fail fast when artifacts
        are missing or incompatible with the current environment.
    """
    config = load_config()
    load_errors: list[str] = []

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
