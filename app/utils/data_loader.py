import pandas as pd
import streamlit as st

from data.load import load_config, load_dataset
from data.preprocess import preprocess_test, preprocess_train
from features.variability import build_variability
from features.velocity import build_velocity

_dataset_cache: dict[str, tuple] = {}


def get_cached_dataset(dataset_id: str, config: dict) -> tuple:
    """Return cached (train_df, test_df, rul_df). Load once, reuse forever."""
    if dataset_id not in _dataset_cache:
        _dataset_cache[dataset_id] = load_dataset(config)
    return _dataset_cache[dataset_id]



def load_pipeline_data_uncached(
    dataset_id: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, "RegimeScaler"]:
    """
    Purpose:     Pure pipeline run for a CMAPSS dataset — no Streamlit, no caching.
                 Importable from any context (FastAPI, scripts, tests).
    Input:       dataset_id — one of FD001–FD004
    Output:      (train_df, test_df, scaler) with all pipeline columns incl. risk_state, risk_score
    Assumptions: Raw data files present for dataset_id
    Failure:     FileNotFoundError if raw files missing; KeyError on malformed config
    """
    from data.regime import resolve_regime_config
    
    config = resolve_regime_config(load_config(), dataset_id)
    config["dataset_id"] = dataset_id
    config["dataset"]["name"] = dataset_id
    config["dataset"]["train_file"] = f"train_{dataset_id}.txt"
    config["dataset"]["test_file"] = f"test_{dataset_id}.txt"
    config["dataset"]["rul_file"] = f"RUL_{dataset_id}.txt"

    from features.health_index import (apply_dual_health_index,
                                       build_dual_health_index,
                                       assign_operative_features)
    from model.clustering import build_clustering_per_fault_mode
    from model.fault_classifier import classify_engines, fit_fault_classifier
    from model.risk import build_risk_score_per_fault_mode

    # Load with updated config
    train_raw, test_raw, _ = get_cached_dataset(dataset_id, config)
    train_proc, scaler, _ = preprocess_train(train_raw, config, persist_outputs=False)
    test_proc = preprocess_test(test_raw, config, scaler, persist_outputs=False)

    train_hi, hi_pca_by_axis, hi_scaler_by_axis = build_dual_health_index(
        train_proc, config
    )
    test_hi = apply_dual_health_index(
        test_proc, hi_pca_by_axis, hi_scaler_by_axis, config
    )

    train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
    train_var, test_var, _ = build_variability(train_vel, test_vel, config)

    fault_artifacts = fit_fault_classifier(train_var, config)
    train_var = classify_engines(train_var, fault_artifacts, config)
    test_var = classify_engines(test_var, fault_artifacts, config)

    train_var = assign_operative_features(train_var)
    test_var = assign_operative_features(test_var)

    train_cl, test_cl, clusterers_by_mode = build_clustering_per_fault_mode(
        train_var, test_var, config
    )
    train_rs, test_rs, risk_arts_by_mode = build_risk_score_per_fault_mode(
        train_cl, test_cl, clusterers_by_mode
    )

    return train_rs, test_rs, scaler


@st.cache_data
def load_pipeline_data(dataset_id: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, "Any"]:
    """Streamlit-cached wrapper. Dashboard entry point."""
    train_rs, test_rs, scaler = load_pipeline_data_uncached(dataset_id)
    return train_rs, test_rs, scaler
