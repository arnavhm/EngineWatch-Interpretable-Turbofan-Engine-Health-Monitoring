import pandas as pd
import pytest

from evaluation.validation import run_validation


def _base_validation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "RUL": [3, 2, 1, 3, 2, 1],
            "health_index": [1.0, 0.8, 0.6, 1.0, 0.85, 0.7],
            "HI_velocity": [-0.1, -0.1, -0.1, -0.08, -0.08, -0.08],
            "HI_variability": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "risk_state": [
                "Healthy",
                "Degrading",
                "Critical",
                "Healthy",
                "Degrading",
                "Critical",
            ],
            "risk_score": [0.1, 0.5, 0.9, 0.2, 0.55, 0.85],
        }
    )


def test_run_validation_happy_path() -> None:
    df = _base_validation_df()

    report = run_validation(df)

    assert report.n_engines == 2
    assert len(report.engine_results) == 2
    assert report.pct_valid_cluster == 100.0
    assert report.pct_monotonic_hi == 100.0
    assert report.risk_rul_pearson_r < 0


def test_run_validation_missing_required_column_raises() -> None:
    df = _base_validation_df().drop(columns=["risk_score"])

    with pytest.raises(KeyError):
        run_validation(df)


def test_run_validation_flags_regressive_cluster_sequence() -> None:
    df = _base_validation_df()
    df.loc[df["unit"] == 2, "risk_state"] = ["Healthy", "Critical", "Degrading"]

    report = run_validation(df)

    assert report.pct_valid_cluster == 50.0
    assert 2 in report.anomalous_engines


def test_run_validation_uses_config_thresholds_for_warnings() -> None:
    df = _base_validation_df()
    config = {
        "validation": {
            "monotonicity_abs_rho_threshold": 0.7,
            "weak_risk_rul_correlation_threshold": -0.999,
        }
    }

    with pytest.warns(UserWarning, match="Risk-RUL Pearson"):
        report = run_validation(df, config=config)

    assert report.weak_risk_rul_correlation_threshold == -0.999
    assert report.monotonicity_abs_rho_threshold == 0.7
