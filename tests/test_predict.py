import pytest
import pandas as pd
import numpy as np
from model.predict import predict_engine


def test_predict_engine_success():
    # Construct a sample DataFrame representing one cycle of an engine
    df = pd.DataFrame(
        {
            "health_index": [0.75],
            "HI_velocity": [-0.002],
            "HI_variability": [0.01],
            "risk_score": [0.1],
            "risk_state": ["Healthy"],
            "unit": [1],
        }
    )

    result = predict_engine(df, dataset_id="FD001")

    # Assert all expected keys exist in the output dictionary
    assert "rul_cycles" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "ci_std" in result
    assert "risk_state" in result
    assert "model_name" in result
    assert "rmse" in result

    # Check types and sanity bounds
    assert isinstance(result["rul_cycles"], float)
    assert result["rul_cycles"] >= 0.0
    assert result["risk_state"] == "Healthy"
    assert isinstance(result["model_name"], str)
    assert isinstance(result["rmse"], float)

    # Check CI bounds sanity if calculated
    if result["ci_std"] is not None:
        assert isinstance(result["ci_lower"], float)
        assert isinstance(result["ci_upper"], float)
        assert isinstance(result["ci_std"], float)
        assert result["ci_lower"] <= result["rul_cycles"] <= result["ci_upper"]


def test_predict_engine_missing_columns():
    # Missing 'risk_score' feature
    df = pd.DataFrame(
        {
            "health_index": [0.75],
            "HI_velocity": [-0.002],
            "HI_variability": [0.01],
            "unit": [1],
        }
    )

    with pytest.raises(KeyError) as exc_info:
        predict_engine(df, dataset_id="FD001")
    assert "risk_score" in str(exc_info.value)


def test_predict_engine_empty_dataframe():
    # Empty DataFrame
    df = pd.DataFrame(columns=["health_index", "HI_velocity", "HI_variability", "risk_score"])

    with pytest.raises(ValueError) as exc_info:
        predict_engine(df, dataset_id="FD001")
    assert "engine_df is empty" in str(exc_info.value)


def test_predict_engine_by_id_success():
    from model.predict import predict_engine_by_id
    result = predict_engine_by_id(34, "FD001")
    
    assert result["engine_id"] == 34
    assert result["dataset_id"] == "FD001"
    assert "rul_cycles" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "health_index" in result
    assert "risk_score" in result
    assert "risk_state" in result


def test_predict_engine_by_id_invalid_id():
    from model.predict import predict_engine_by_id
    with pytest.raises(ValueError) as exc_info:
        predict_engine_by_id(9999, "FD001")
    assert "not found in FD001 test split" in str(exc_info.value)
