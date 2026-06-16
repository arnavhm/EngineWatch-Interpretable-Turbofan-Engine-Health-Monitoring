import streamlit as st
import pandas as pd
from data.load import load_dataset, load_config
from data.preprocess import preprocess_train, preprocess_test
from features.health_index import build_health_index
from features.velocity import build_velocity
from features.variability import build_variability
from model.clustering import build_clustering
from model.risk import build_risk_score


def build_pipeline_data(
    persist_outputs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full feature and risk pipeline for train and test datasets."""
    config = load_config()
    train_raw, test_raw, _ = load_dataset(config)
    train_proc, scaler, _ = preprocess_train(
        train_raw,
        config,
        persist_outputs=persist_outputs,
    )
    test_proc = preprocess_test(
        test_raw,
        config,
        scaler,
        persist_outputs=persist_outputs,
    )
    train_hi, test_hi, _ = build_health_index(train_proc, test_proc, config)
    train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
    train_var, test_var, _ = build_variability(train_vel, test_vel, config)
    train_cl, test_cl, cl_art = build_clustering(train_var, test_var, config)
    train_rs, test_rs, _ = build_risk_score(train_cl, test_cl, cl_art)
    return train_rs, test_rs


def load_pipeline_data_uncached(
    dataset_id: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose:     Pure pipeline run for a CMAPSS dataset — no Streamlit, no caching.
                 Importable from any context (FastAPI, scripts, tests).
    Input:       dataset_id — one of FD001–FD004
    Output:      (train_df, test_df) with all pipeline columns incl. risk_state, risk_score
    Assumptions: Raw data files present for dataset_id
    Failure:     FileNotFoundError if raw files missing; KeyError on malformed config
    """
    config = load_config()
    config["dataset_id"] = dataset_id
    config["dataset"]["name"] = dataset_id
    config["dataset"]["train_file"] = f"train_{dataset_id}.txt"
    config["dataset"]["test_file"] = f"test_{dataset_id}.txt"
    config["dataset"]["rul_file"] = f"RUL_{dataset_id}.txt"

    from features.health_index import build_dual_health_index, apply_dual_health_index
    from model.fault_classifier import fit_fault_classifier, classify_engines
    from model.clustering import build_clustering_per_fault_mode
    from model.risk import build_risk_score_per_fault_mode

    # Load with updated config
    train_raw, test_raw, _ = load_dataset(config)
    train_proc, scaler, _ = preprocess_train(train_raw, config, persist_outputs=False)
    test_proc = preprocess_test(test_raw, config, scaler, persist_outputs=False)
    
    train_hi, hi_pca_by_axis, hi_scaler_by_axis = build_dual_health_index(train_proc, config)
    test_hi = apply_dual_health_index(test_proc, hi_pca_by_axis, hi_scaler_by_axis, config)
    train_hi["health_index"] = train_hi["HI_hpc"]
    test_hi["health_index"] = test_hi["HI_hpc"]
    
    train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
    train_var, test_var, _ = build_variability(train_vel, test_vel, config)
    
    fault_artifacts = fit_fault_classifier(train_var, config)
    train_var = classify_engines(train_var, fault_artifacts, config)
    test_var = classify_engines(test_var, fault_artifacts, config)
    
    train_cl, test_cl, clusterers_by_mode = build_clustering_per_fault_mode(train_var, test_var, config)
    train_rs, test_rs, risk_arts_by_mode = build_risk_score_per_fault_mode(train_cl, test_cl, clusterers_by_mode)
    
    return train_rs, test_rs


@st.cache_data
def load_pipeline_data(dataset_id: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Streamlit-cached wrapper. Dashboard entry point — behavior unchanged."""
    return load_pipeline_data_uncached(dataset_id)
