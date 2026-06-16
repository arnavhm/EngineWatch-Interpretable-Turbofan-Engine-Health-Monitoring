"""Offline multi-dataset RUL training entrypoint.

Purpose:
    Run the existing interpretable CMAPSS pipeline across multiple FD datasets
    (for example FD001-FD004) using one safe orchestration script.

Input shape:
    - Raw CMAPSS files under configured raw directory with names:
      train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt
    - Config dictionary loaded via data.load.load_config().

Output shape:
    - Per-dataset processed parquet files in configured processed directory.
    - Per-dataset scaler artifacts: models/scaler_FDxxx.joblib.
    - Per-dataset RUL model artifacts under models/FDxxx/.
    - Console summary table with best model and metrics for each dataset.

Assumptions:
    - Core modules are importable from project root.
    - Dataset files follow the canonical CMAPSS naming convention.

Failure conditions:
    - Missing required dataset files for a selected dataset.
    - Feature pipeline/model training errors raised by downstream modules.
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.load import load_config, load_dataset
from data.preprocess import preprocess_test, preprocess_train
from features.health_index import apply_dual_health_index, build_dual_health_index
from features.variability import build_variability
from features.velocity import build_velocity
from model.clustering import build_clustering_per_fault_mode
from model.fault_classifier import classify_engines, fit_fault_classifier
from model.risk import build_risk_score_per_fault_mode
from model.rul import build_rul_model


def _attach_test_rul(test_df: pd.DataFrame, rul_offsets: pd.Series) -> pd.DataFrame:
    """Add cycle-level test RUL using CMAPSS offset convention.

    Purpose:
        Convert engine-level RUL offsets into cycle-level targets for each row.
    Input shape:
        test_df: cycle-level test dataframe with `unit` and `cycle`.
        rul_offsets: pd.Series indexed by unit id with final-cycle RUL offsets.
    Output shape:
        DataFrame with an additional `RUL` column.
    Assumptions:
        rul_offsets index matches test engine ids (1-based CMAPSS convention).
    Failure conditions:
        Raises ValueError when offsets are missing for any test unit.
    """
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


def _discover_dataset_ids(raw_path: Path) -> list[str]:
    """Discover CMAPSS FD dataset ids available in a raw directory.

    Purpose:
        Build a safe default dataset list from files on disk rather than
        hardcoding dataset ids.
    Input shape:
        raw_path: directory containing train/test/RUL text files.
    Output shape:
        Sorted dataset id list like ["FD001", "FD002"].
    Assumptions:
        Files follow naming pattern: (train|test|RUL)_FDxxx.txt.
    Failure conditions:
        Returns an empty list if no complete datasets are found.
    """
    pattern = re.compile(r"^(train|test|RUL)_(FD\d{3})\.txt$")
    grouped: dict[str, set[str]] = {}

    for path in raw_path.glob("*.txt"):
        match = pattern.match(path.name)
        if not match:
            continue
        file_kind, dataset_id = match.groups()
        grouped.setdefault(dataset_id, set()).add(file_kind)

    complete_ids = [
        dataset_id
        for dataset_id, kinds in grouped.items()
        if kinds == {"train", "test", "RUL"}
    ]
    return sorted(complete_ids)


def _dataset_config(base_config: dict, dataset_id: str) -> dict:
    """Create a dataset-specific config clone that avoids artifact overwrites.

    Purpose:
        Keep config/config.yaml unchanged while running per-dataset training.
    Input shape:
        base_config: config loaded from YAML.
        dataset_id: dataset identifier like FD001.
    Output shape:
        Deep-copied config dict with dataset-specific file and save paths.
    Assumptions:
        base_config contains `dataset`, `scaler_path`, and `rul.save_path` keys.
    Failure conditions:
        KeyError from missing required config fields.
    """
    config = copy.deepcopy(base_config)

    config["dataset_id"] = dataset_id
    config["dataset"]["name"] = dataset_id
    config["dataset"]["train_file"] = f"train_{dataset_id}.txt"
    config["dataset"]["test_file"] = f"test_{dataset_id}.txt"
    config["dataset"]["rul_file"] = f"RUL_{dataset_id}.txt"

    config["scaler_path"] = str(Path("models") / f"scaler_{dataset_id}.joblib")

    save_root = Path(config["rul"]["save_path"])
    config["rul"]["save_path"] = str(save_root / dataset_id)

    regime_cfg = config.get("regimes", {})
    if regime_cfg.get("enabled", False):
        by_dataset = regime_cfg.get("by_dataset", {})
        if dataset_id not in by_dataset:
            raise KeyError(f"Missing config key: regimes.by_dataset.{dataset_id}")
        regime_cfg["n_regimes"] = int(by_dataset[dataset_id])

    return config


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multi-dataset training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train RUL artifacts across multiple CMAPSS datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset ids (for example: FD001 FD002 FD003 FD004).",
    )
    parser.add_argument(
        "--no-persist-processed",
        action="store_true",
        help="Disable writing processed parquet outputs while still saving model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute per-dataset pipeline and model training with isolated artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    base_config = load_config()

    raw_path = Path(base_config["dataset"]["raw_path"])
    discovered = _discover_dataset_ids(raw_path)

    dataset_ids = args.datasets if args.datasets else discovered
    if not dataset_ids:
        raise RuntimeError(
            "No complete datasets discovered. Ensure train/test/RUL files exist in the raw path."
        )

    persist_processed = not args.no_persist_processed
    run_summaries: list[dict[str, object]] = []

    for dataset_id in dataset_ids:
        config = _dataset_config(base_config, dataset_id)

        print(f"\n=== Running pipeline for {dataset_id} ===")
        train_raw, test_raw, test_rul_offsets = load_dataset(config)

        train_proc, scaler, _ = preprocess_train(
            train_raw,
            config,
            persist_outputs=persist_processed,
        )
        test_proc = preprocess_test(
            test_raw,
            config,
            scaler,
            persist_outputs=persist_processed,
        )

        train_hi, hi_pca_by_axis, hi_scaler_by_axis = build_dual_health_index(
            train_proc,
            config,
        )
        test_hi = apply_dual_health_index(
            test_proc,
            hi_pca_by_axis,
            hi_scaler_by_axis,
            config,
        )
        train_hi["health_index"] = train_hi["HI_hpc"]
        test_hi["health_index"] = test_hi["HI_hpc"]

        hi_artifact_dir = Path("models") / dataset_id
        hi_artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(hi_pca_by_axis, hi_artifact_dir / "hi_pca_by_axis.joblib")
        joblib.dump(
            hi_scaler_by_axis,
            hi_artifact_dir / "hi_scaler_by_axis.joblib",
        )
        train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
        train_var, test_var, var_artifacts = build_variability(train_vel, test_vel, config)
        joblib.dump(var_artifacts, hi_artifact_dir / "variability_artifacts.joblib")

        fault_artifacts = fit_fault_classifier(train_var, config)
        train_var = classify_engines(train_var, fault_artifacts, config)
        test_var = classify_engines(test_var, fault_artifacts, config)

        joblib.dump(fault_artifacts, hi_artifact_dir / "fault_classifier.joblib")

        train_cl, test_cl, clusterers_by_mode = build_clustering_per_fault_mode(
            train_var, test_var, config
        )
        train_rs, test_rs, risk_arts_by_mode = build_risk_score_per_fault_mode(
            train_cl, test_cl, clusterers_by_mode
        )

        joblib.dump(
            clusterers_by_mode,
            hi_artifact_dir / "cluster_models_by_fault.joblib",
        )
        joblib.dump(
            risk_arts_by_mode,
            hi_artifact_dir / "risk_artifacts_by_fault.joblib",
        )

        test_with_rul = _attach_test_rul(test_rs, test_rul_offsets)
        _, artifacts = build_rul_model(train_rs, test_with_rul, config)

        best_metrics = artifacts.evaluation_metrics[artifacts.best_model_name]
        run_summaries.append(
            {
                "dataset": dataset_id,
                "best_model": artifacts.best_model_name,
                "rmse": round(float(best_metrics["rmse"]), 3),
                "nasa_score": round(float(best_metrics["nasa_score"]), 3),
                "artifact_dir": config["rul"]["save_path"],
                "scaler_path": config["scaler_path"],
            }
        )

        print(
            "Completed "
            f"{dataset_id} | best={artifacts.best_model_name} "
            f"| rmse={best_metrics['rmse']:.3f} "
            f"| nasa={best_metrics['nasa_score']:.3f}"
        )

    summary_df = pd.DataFrame(run_summaries)
    print("\n=== Multi-dataset summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
