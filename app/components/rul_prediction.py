import streamlit as st
import pandas as pd
from typing import Any

from data.load import load_config
from app.theme import STATE_COLORS, SECTION_TITLE_CSS
from app.utils.rul_artifacts import load_or_rebuild_rul_artifacts

FEATURE_COLUMNS = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


@st.cache_resource
def _load_rul_artifacts() -> Any:
    """Load pre-trained RUL artifacts for inference."""
    return load_or_rebuild_rul_artifacts()


@st.cache_data
def _load_rul_bands() -> tuple[float, float]:
    """Load dashboard RUL band thresholds from config."""
    config = load_config()
    bands = config.get("dashboard", {}).get("rul_bands", {})
    healthy_min = float(bands.get("healthy_min", 80))
    degrading_min = float(bands.get("degrading_min", 30))
    return healthy_min, degrading_min


def render_rul_prediction(df: pd.DataFrame) -> None:
    """Display predicted Remaining Useful Life for the selected engine."""
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Predicted Remaining Useful Life</p>',
        unsafe_allow_html=True,
    )

    artifacts = _load_rul_artifacts()
    model = artifacts.best_model
    metrics = artifacts.evaluation_metrics
    model_key = artifacts.best_model_name

    # Last cycle row for this engine
    last_row = df.iloc[[-1]]
    features = last_row[FEATURE_COLUMNS]

    predicted_rul = float(model.predict(features)[0])
    predicted_rul = max(predicted_rul, 0)  # floor at 0

    # Derive a readable model name and look up its RMSE in metrics dict.
    model_name = type(model).__name__

    model_metrics = metrics.get(model_key, {})
    rmse = model_metrics.get("rmse") if isinstance(model_metrics, dict) else None

    # --- Determine colour based on predicted RUL ---
    healthy_min, degrading_min = _load_rul_bands()
    if predicted_rul > healthy_min:
        rul_color = STATE_COLORS["Healthy"]
    elif predicted_rul > degrading_min:
        rul_color = STATE_COLORS["Degrading"]
    else:
        rul_color = STATE_COLORS["Critical"]

    # --- Build display strings ---
    rmse_text = f"{rmse:.2f}" if rmse is not None else "N/A"

    # --- Card layout ---
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {rul_color}22, {rul_color}11);
            border-left: 4px solid {rul_color};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.5rem;
        ">
            <div style="font-size: 2.4rem; font-weight: 700; color: {rul_color}; line-height: 1.1;">
                {predicted_rul:.0f} <span style="font-size: 1rem; font-weight: 400;">cycles</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #888;">
                Model&ensp;<b>{model_name}</b>&ensp;·&ensp;RMSE&ensp;<b>{rmse_text}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
