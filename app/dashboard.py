import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

# Page config must be the first Streamlit command
st.set_page_config(page_title="CMAPSS Engine Health Monitor", layout="wide")

from app.utils.data_loader import load_pipeline_data
from app.components.fleet_overview import render_fleet_overview
from app.components.engine_selector import render_engine_selector
from app.components.hi_plot import render_hi_plot
from app.components.dynamics_plots import render_dynamics_plots
from app.components.risk_gauge import render_risk_gauge
from app.components.cluster_timeline import render_cluster_timeline
from app.components.rul_prediction import render_rul_prediction

def main():
    st.title("CMAPSS Engine Health Monitor")
    
    # Run pipeline and cache data
    with st.spinner("Loading and processing engine data..."):
        train_rs, test_rs = load_pipeline_data()
        
    # Standardise on the test set for this dashboard as per typical setup, 
    # though the prompt mentions 100 training and 100 test engines. 
    # Let's combine them for the holistic fleet view if they are both available,
    # or just use test if that is the final scoring set.
    # The prompt says "100 training engines (IDs 1-100), 100 test engines", let's use the test set as it typically represents the live fleet.
    # Actually wait, let's use `test_rs` as the default source for "current fleet" in predictive maintenance.
    # If the user specifically wanted training set, we can adjust, but for now test_rs is usually the target.
    # To be safe and show 1-100 engines properly, let's use `test_rs`.
    df = test_rs.copy()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FLEET RISK OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    render_fleet_overview(df)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENGINE LEVEL DETAILS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    selected_engine_id = render_engine_selector(df)
    
    # Filter data for selected engine
    engine_df = df[df["unit"] == selected_engine_id].copy()
    
    st.markdown("---")
    # Health Index Trajectory
    render_hi_plot(engine_df, selected_engine_id)
    
    st.markdown("---")
    # Dynamics & Gauge
    col1, col2 = st.columns([2, 1])
    with col1:
        render_dynamics_plots(engine_df)
    with col2:
        render_risk_gauge(engine_df)
        render_rul_prediction(engine_df)
        
    st.markdown("---")
    # Cluster Progression Timeline
    render_cluster_timeline(engine_df, selected_engine_id)

if __name__ == "__main__":
    main()
