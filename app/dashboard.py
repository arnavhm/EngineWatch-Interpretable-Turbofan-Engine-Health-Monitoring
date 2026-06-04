import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import joblib

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="EngineWatch — Interpretable Turbofan Engine Health Monitoring",
    layout="wide",
)

from app.utils.data_loader import load_pipeline_data
from app.components.fleet_overview import render_fleet_overview
from app.components.engine_selector import render_engine_selector
from app.components.anomaly_panel import render_anomaly_panel
from app.components.hi_plot import render_hi_plot
from app.components.dynamics_plots import render_dynamics_plots
from app.components.risk_gauge import render_risk_gauge
from app.components.cluster_timeline import render_cluster_timeline
from app.components.rul_prediction import render_rul_prediction
from app.components.model_evaluation import render_model_evaluation
from app.components.aog_panel import render_aog_panel
from app.components.sensor_panel import render_sensor_panel
from app.utils.rul_artifacts import load_or_rebuild_rul_artifacts


@st.cache_resource
def load_artifacts(dataset_id: str) -> dict:
    """
    Load dataset-specific artifacts from models/{dataset_id}/ directory.

    Purpose:
        Ensure feature engineering artifacts (PCA, scalers, clustering, etc.)
        are loaded from the correct dataset-specific subdirectory.
        Cache key includes dataset_id so FD001 artifacts are never reused for
        FD002, FD003, or FD004.

        Note: RUL artifacts are loaded separately via load_or_rebuild_rul_artifacts()
        due to sklearn version compatibility issues with joblib pickle.

    Input:  dataset_id (FD001, FD002, FD003, or FD004)
    Output: dict with keys for feature engineering artifacts
    """
    base = Path(f"models/{dataset_id}")
    if not base.exists():
        st.error(f"No artifacts found for {dataset_id} in models/")
        st.stop()

    return {
        "fault_classifier": joblib.load(base / "fault_classifier.joblib"),
        "hi_pca_by_axis": joblib.load(base / "hi_pca_by_axis.joblib"),
        "hi_scaler_by_axis": joblib.load(base / "hi_scaler_by_axis.joblib"),
        "cluster_models_by_fault": joblib.load(base / "cluster_models_by_fault.joblib"),
        "risk_artifacts_by_fault": joblib.load(base / "risk_artifacts_by_fault.joblib"),
    }


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

    st.title("EngineWatch — Interpretable Turbofan Engine Health Monitoring")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATASET SELECTOR & SESSION STATE MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Detect dataset changes and clear stale session state
    if "last_dataset_id" not in st.session_state:
        st.session_state["last_dataset_id"] = "FD001"

    with st.sidebar:
        st.markdown("### 📊 Dataset Selection")
        selected_dataset = st.radio(
            "Choose dataset:",
            options=["FD001", "FD002", "FD003", "FD004"],
            index=0,
            horizontal=True,
        )

        if selected_dataset != "FD001":
            st.info(
                f"**{selected_dataset} selected** — "
                f"{'6 operating conditions' if selected_dataset in ['FD002', 'FD004'] else '1 condition'}, "
                f"{'2 fault modes (HPC + Fan)' if selected_dataset in ['FD003', 'FD004'] else '1 fault mode (HPC)'}. "
                f"Pipeline validated: regime-normalized artifacts in models/{selected_dataset}/. "
                f"Full dashboard visualization for multi-condition datasets is in active integration. "
                f"Switch to FD001 for the complete diagnostic view."
            )

    # If dataset changed, clear stale values from session state
    if selected_dataset != st.session_state["last_dataset_id"]:
        st.session_state["last_dataset_id"] = selected_dataset
        # Rerun to pick up new dataset throughout the dashboard
        st.rerun()

    # Run pipeline and cache data (cache key includes dataset_id)
    with st.spinner(f"Loading and processing {selected_dataset} engine data..."):
        train_rs, test_rs = load_pipeline_data(dataset_id=selected_dataset)

    # Load dataset-specific artifacts (cache key includes dataset_id)
    # Ensures FD001 artifacts are never reused for FD002/FD003/FD004
    artifacts = load_artifacts(dataset_id=selected_dataset)

    df = test_rs.copy()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FLEET RISK OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_fleet_overview(df)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENGINE LEVEL DETAILS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    selected_engine_id = render_engine_selector(df, dataset_id=selected_dataset)
    engine_df = df[df["unit"] == selected_engine_id].copy()

    st.divider()
    render_anomaly_panel(df, selected_engine_id=selected_engine_id)

    st.divider()
    render_hi_plot(engine_df, selected_engine_id)

    st.divider()
    render_sensor_panel(engine_df, selected_engine_id)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_dynamics_plots(engine_df)
    with col2:
        render_risk_gauge(engine_df)
        render_rul_prediction(engine_df, dataset_id=selected_dataset)

    st.divider()

    # AOG panel — full width
    from data.load import load_config

    config = load_config()
    last_row = engine_df.iloc[-1]

    # Handle dual risk scores for FD003/FD004 dual-fault datasets.
    # Use the more degraded axis so the AOG decision stays conservative.
    if "risk_score_hpc" in last_row.index and "risk_score_fan" in last_row.index:
        risk_score_hpc = float(last_row.get("risk_score_hpc", 0.0))
        risk_score_fan = float(last_row.get("risk_score_fan", 0.0))
        risk_score = max(risk_score_hpc, risk_score_fan)
    else:
        # Single risk_score column (FD001, FD002, or unified scoring)
        risk_score = float(last_row.get("risk_score", 0.0))

    risk_state = str(last_row.get("risk_state", "Healthy"))

    # Compute predicted RUL using the same model as render_rul_prediction
    FEATURE_COLUMNS = [
        "health_index",
        "HI_velocity",
        "HI_variability",
        "risk_score",
    ]
    _rul_error: str | None = None
    predicted_rul: int = 0

    try:
        # Load RUL artifacts separately to handle sklearn version compatibility
        rul_artifacts = load_or_rebuild_rul_artifacts(dataset_id=selected_dataset)
        model = rul_artifacts.best_model
        features = engine_df[FEATURE_COLUMNS].iloc[-1:].values
        predicted_rul = max(int(float(model.predict(features)[0])), 0)
    except KeyError as e:
        _rul_error = (
            f"Feature column {e} not found in pipeline output for {selected_dataset}. "
            f"Expected columns: {FEATURE_COLUMNS}. "
            f"Check that the RUL model artifact and pipeline output columns match."
        )
    except Exception as e:
        _rul_error = f"RUL prediction failed for {selected_dataset}: {e}"

    if _rul_error:
        st.markdown("### 💰 AOG Cost Impact Analysis")
        st.error(f"⚠️ AOG panel unavailable — {_rul_error}")
    else:
        render_aog_panel(
            risk_score=risk_score,
            rul_pred=predicted_rul,
            risk_state=risk_state,
            config=config,
        )

    st.divider()
    render_cluster_timeline(engine_df, selected_engine_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_model_evaluation()


if __name__ == "__main__":
    main()
