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


@st.cache_data
def load_pipeline_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pipeline outputs for the dashboard without writing artifacts."""
    return build_pipeline_data(persist_outputs=False)
