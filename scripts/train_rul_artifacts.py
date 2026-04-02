"""Offline training entrypoint for RUL model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.data_loader import build_pipeline_data
from data.load import load_config, load_dataset
from model.rul import build_rul_model


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


def main() -> None:
    """Train and persist RUL artifacts from the current code/config environment."""
    config = load_config()
    train_rs, test_rs = build_pipeline_data(persist_outputs=True)
    _, _, test_rul_offsets = load_dataset(config)
    test_with_rul = _attach_test_rul(test_rs, test_rul_offsets)

    predictions_df, artifacts = build_rul_model(train_rs, test_with_rul, config)

    save_path = Path(config["rul"]["save_path"]).resolve()
    print(f"Saved artifacts to: {save_path}")
    print(f"Best model: {artifacts.best_model_name}")
    print(f"Prediction rows: {len(predictions_df)}")


if __name__ == "__main__":
    main()
