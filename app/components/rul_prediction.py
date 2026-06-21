"""
app/components/rul_prediction.py

Purpose:      Display predicted Remaining Useful Life for the selected engine.
              Point prediction from best model (GradientBoosting).
              Confidence interval from Random Forest tree variance.
"""


import pandas as pd
import streamlit as st

from app.theme import SECTION_TITLE_CSS
from app.utils.theme import STATE_COLORS
from data.load import load_config

FEATURE_COLUMNS = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


@st.cache_data
def _load_rul_bands() -> tuple[float, float]:
    """Load dashboard RUL band thresholds from config."""
    config = load_config()
    bands = config.get("dashboard", {}).get("rul_bands", {})
    healthy_min = float(bands.get("healthy_min", 80))
    degrading_min = float(bands.get("degrading_min", 30))
    return healthy_min, degrading_min


def render_rul_prediction(df: pd.DataFrame, dataset_id: str = "FD001") -> None:
    """
    Purpose:      Display predicted RUL with RF confidence interval for selected engine.
    Input:        df — single-engine DataFrame (all cycles); uses last row only
    Output:       None — renders into Streamlit context
    Assumptions:  FEATURE_COLUMNS are present in df
    Failure:      KeyError if feature columns missing; gracefully omits CI if RF unavailable
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Predicted Remaining Useful Life</p>',
        unsafe_allow_html=True,
    )

    from model.predict import predict_engine

    result = predict_engine(df, dataset_id=dataset_id)
    predicted_rul = result["rul_cycles"]
    ci_lower = result["ci_lower"]
    ci_upper = result["ci_upper"]
    ci_std = result.get("ci_std")

    model_name = result.get("model_name", "Unknown")
    rmse = result.get("rmse")

    # ── Colour by predicted RUL band ─────────────────────────────────
    healthy_min, degrading_min = _load_rul_bands()
    if predicted_rul > healthy_min:
        rul_color = STATE_COLORS["Healthy"]
    elif predicted_rul > degrading_min:
        rul_color = STATE_COLORS["Degrading"]
    else:
        rul_color = STATE_COLORS["Critical"]

    # ── Build display strings ─────────────────────────────────────────
    rmse_text = f"{rmse:.2f}" if rmse is not None else "N/A"

    headline = f"{predicted_rul:.0f}"

    if ci_std is not None:
        ci_line = (
            f'<div style="margin-top: 0.2rem; font-size: 1rem; font-weight: 400; color: #999;">'
            f"± {ci_std:.0f} cycles"
            f"</div>"
        )
        interval_text = f"[{ci_lower:.0f} – {ci_upper:.0f} cycles]"
        ci_subtitle = (
            f'<div style="margin-top: 0.3rem; font-size: 0.82rem; color: #aaa;">'
            f"RF interval&ensp;<b>{interval_text}</b>&ensp;·&ensp;"
            f"std&ensp;<b>{ci_std:.1f}</b>"
            f"</div>"
        )
    else:
        ci_line = ""
        ci_subtitle = ""

    # ── Card layout ───────────────────────────────────────────────────
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
                {headline} <span style="font-size: 1rem; font-weight: 400;">cycles</span>
            </div>
            {ci_line}
            {ci_subtitle}
            <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #888;">
                Model&ensp;<b>{model_name}</b>&ensp;·&ensp;RMSE&ensp;<b>{rmse_text}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
