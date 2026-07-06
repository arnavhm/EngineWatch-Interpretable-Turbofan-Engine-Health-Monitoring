import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional

import joblib
import sys
if "miniforge3" in sys.executable or ".venvs/project-2" not in sys.executable:
    raise RuntimeError(
        f"Wrong interpreter: {sys.executable}\n"
        "Activate .venvs/project-2 before running the dashboard. "
        "Per charter Section 8, training/dashboard/API must share one interpreter."
    )

import numpy as np
import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="EngineWatch — Interpretable Turbofan Engine Health Monitoring",
    layout="wide",
)

from app.utils.theme import DATASET_LABELS, inject_css

inject_css()

from app.components.anomaly_panel import render_anomaly_panel
from app.components.aog_panel import render_aog_panel
from app.components.cluster_timeline import render_cluster_timeline
from app.components.csv_upload_panel import render_csv_upload_panel
from app.components.dynamics_plots import render_dynamics_plots
from app.components.engine_diagram import render_engine_diagram
from app.components.engine_selector import render_engine_selector
from app.components.fleet_overview import render_fleet_overview
from app.components.hi_plot import render_hi_plot
from app.components.model_evaluation import render_model_evaluation
from app.components.narration_panel import render_narration_panel
from app.components.risk_gauge import render_risk_gauge
from app.components.rul_prediction import render_rul_prediction
from app.components.sensor_panel import render_sensor_panel
from app.utils.data_loader import load_pipeline_data
from app.utils.nl_parser import handle_nl_query
from app.utils.rul_artifacts import load_or_rebuild_rul_artifacts
from data.load import load_config


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
            horizontal=False,
            format_func=lambda d: DATASET_LABELS.get(d, d),
        )

        if selected_dataset != "FD001":
            st.success(
                f"**{selected_dataset} selected** — "
                f"{'6 operating conditions' if selected_dataset in ['FD002', 'FD004'] else '1 condition'}, "
                f"{'2 fault modes (HPC + Fan)' if selected_dataset in ['FD003', 'FD004'] else '1 fault mode (HPC)'}. "
                f"Pipeline validated: regime-normalized artifacts in models/{selected_dataset}/. "
                f"Full dashboard visualization is now fully supported and working for all datasets!"
            )

    # If dataset changed, clear stale values from session state
    if selected_dataset != st.session_state["last_dataset_id"]:
        st.session_state["last_dataset_id"] = selected_dataset
        # Rerun to pick up new dataset throughout the dashboard
        st.rerun()

    # Run pipeline and cache data (cache key includes dataset_id)
    with st.spinner(f"Loading and processing {selected_dataset} engine data..."):
        train_rs, test_rs, scaler = load_pipeline_data(dataset_id=selected_dataset)

    # Load dataset-specific artifacts (cache key includes dataset_id)
    # Ensures FD001 artifacts are never reused for FD002/FD003/FD004
    artifacts = load_artifacts(dataset_id=selected_dataset)

    df = test_rs.copy()

    # Sidebar quick natural-language lookup (allows programmatic engine selection)
    with st.sidebar:
        st.markdown("### 🔎 Quick Natural-Language Lookup")

        def process_quick_query():
            q = st.session_state.get("nl_quick_query_input", "").strip()
            if not q:
                return
            ok, msg, selection = handle_nl_query(
                q, df, st.session_state, require_confirmation=False
            )
            if ok:
                st.session_state["nl_msg"] = ("success", msg)
            else:
                st.session_state["nl_msg"] = ("info", msg)

        with st.form("nl_quick_query_form", clear_on_submit=True):
            st.text_input(
                "Ask (examples: 'state of engine 14 in FD001', 'fleet query')",
                key="nl_quick_query_input",
            )
            st.form_submit_button("Run query", on_click=process_quick_query)

        if "nl_msg" in st.session_state:
            msg_type, msg_text = st.session_state["nl_msg"]
            if msg_type == "success":
                st.success(msg_text)
            else:
                st.info(msg_text)
            del st.session_state["nl_msg"]

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
    
    config = load_config()
    
    # Re-fetch raw data to join op_settings and inverse_transform sensors for the panel
    from app.utils.data_loader import get_cached_dataset
    from data.regime import resolve_regime_config
    
    sensor_config = resolve_regime_config(config.copy(), selected_dataset)
    sensor_config["dataset_id"] = selected_dataset
    sensor_config["dataset"]["name"] = selected_dataset
    sensor_config["dataset"]["train_file"] = f"train_{selected_dataset}.txt"
    sensor_config["dataset"]["test_file"] = f"test_{selected_dataset}.txt"
    sensor_config["dataset"]["rul_file"] = f"RUL_{selected_dataset}.txt"
    
    _, test_raw, _ = get_cached_dataset(selected_dataset, sensor_config)
    engine_raw = test_raw[test_raw["unit"] == selected_engine_id]
    
    setting_cols = sensor_config["regimes"]["setting_cols"]
    sensor_df = engine_df.merge(
        engine_raw[["unit", "cycle"] + setting_cols],
        on=["unit", "cycle"],
        how="left",
        validate="one_to_one"
    )
    fitted_sensor_cols = list(scaler._sensor_cols)
    sensor_df = scaler.inverse_transform_df(sensor_df, fitted_sensor_cols)
    render_sensor_panel(sensor_df, selected_engine_id)

    config = load_config()

    st.divider()
    render_engine_diagram(engine_df, artifacts, config, dataset_id=selected_dataset)

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        render_dynamics_plots(engine_df)
    with col2:
        render_risk_gauge(engine_df)
        render_rul_prediction(engine_df, dataset_id=selected_dataset)

    st.divider()

    # AOG panel — full width
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
    _rul_error: Optional[str] = None
    predicted_rul: int = 0
    predicted_rul_value: float = 0.0
    rul_ci: tuple[float, float] = (0.0, 0.0)

    try:
        # Load RUL artifacts separately to handle sklearn version compatibility
        rul_artifacts = load_or_rebuild_rul_artifacts(dataset_id=selected_dataset)
        model = rul_artifacts.best_model
        features = engine_df[FEATURE_COLUMNS].iloc[-1:].values
        predicted_rul_value = max(float(model.predict(features)[0]), 0.0)
        predicted_rul = max(int(predicted_rul_value), 0)

        rf_model = rul_artifacts.all_models.get("random_forest")
        if rf_model is not None and hasattr(rf_model, "estimators_"):
            tree_preds = np.array(
                [tree.predict(features)[0] for tree in rf_model.estimators_],
                dtype=float,
            )
            ci_std = float(tree_preds.std())
            rul_ci = (
                max(predicted_rul_value - ci_std, 0.0),
                predicted_rul_value + ci_std,
            )
        else:
            rul_ci = (predicted_rul_value, predicted_rul_value)
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

        render_narration_panel(
            engine_df=engine_df,
            config=config,
            predicted_rul=predicted_rul_value,
            rul_ci=rul_ci,
            artifacts=artifacts,
            fleet_df=df,
            dataset_id=selected_dataset,
        )

    st.divider()
    render_cluster_timeline(engine_df, selected_engine_id)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_model_evaluation(selected_dataset)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BATCH CSV PREDICTION (true hybrid — HTTP → API)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.divider()
    render_csv_upload_panel(config)


if __name__ == "__main__":
    main()
