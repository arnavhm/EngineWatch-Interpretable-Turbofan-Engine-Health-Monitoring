import streamlit as st
from data.load import load_dataset, load_config
from data.preprocess import preprocess_train, preprocess_test
from features.health_index import build_health_index
from features.velocity import build_velocity
from features.variability import build_variability
from model.clustering import build_clustering
from model.risk import build_risk_score

@st.cache_data
def load_pipeline_data():
    config = load_config()
    train_raw, test_raw, _ = load_dataset(config)
    train_proc, scaler, sensor_cols = preprocess_train(train_raw, config)
    test_proc = preprocess_test(test_raw, config, scaler)
    train_hi, test_hi, hi_art = build_health_index(train_proc, test_proc, config)
    train_vel, test_vel, vel_art = build_velocity(train_hi, test_hi, config)
    train_var, test_var, var_art = build_variability(train_vel, test_vel, config)
    train_cl, test_cl, cl_art = build_clustering(train_var, test_var, config)
    train_rs, test_rs, risk_art = build_risk_score(train_cl, test_cl, cl_art)
    return train_rs, test_rs
