import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

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
from app.components.model_evaluation import render_model_evaluation


def main() -> None:
    # ── Global CSS injection ───────────────────────────────────────────
    st.markdown(
        """
    <style>
    /* Section headers */
    h2, h3 { 
        font-size: 1.4rem !important; 
        font-weight: 700 !important;
        margin-top: 2rem !important;
    }

    /* Section dividers */
    hr { 
        border: none; 
        border-top: 1px solid #2a2a2a; 
        margin: 2rem 0 !important; 
    }

    /* Sidebar metrics */
    [data-testid="stMetricValue"] { 
        font-size: 2.2rem !important; 
        font-weight: 800 !important; 
    }

    /* Sidebar metric labels */
    [data-testid="stMetricLabel"] { 
        font-size: 0.9rem !important;
        color: #888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Dataframe table */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* General text */
    body {
        font-family: 'Inter', sans-serif !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("CMAPSS Engine Health Monitor")

    # Run pipeline and cache data
    with st.spinner("Loading and processing engine data..."):
        train_rs, test_rs = load_pipeline_data()

    df = test_rs.copy()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FLEET RISK OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_fleet_overview(df)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENGINE LEVEL DETAILS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    selected_engine_id = render_engine_selector(df)
    engine_df = df[df["unit"] == selected_engine_id].copy()

    st.divider()
    render_hi_plot(engine_df, selected_engine_id)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_dynamics_plots(engine_df)
    with col2:
        render_risk_gauge(engine_df)
        render_rul_prediction(engine_df)

    st.divider()
    render_cluster_timeline(engine_df, selected_engine_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_model_evaluation()


if __name__ == "__main__":
    main()
