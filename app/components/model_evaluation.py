import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from app.theme import STATE_COLORS, SECTION_TITLE_CSS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ARTIFACT_PATH = PROJECT_ROOT / "notebooks" / "models" / "rul_artifacts.joblib"
PLOT_PATH = PROJECT_ROOT / "reports" / "rul_evaluation_plots.png"

BEST_MODEL_KEY = "gradient_boosting"

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


@st.cache_resource
def _load_rul_artifacts():
    """Load pre-trained RUL model artifacts (cached across reruns)."""
    return joblib.load(ARTIFACT_PATH)


def render_model_evaluation():
    """Render the Model Evaluation panel at the bottom of the dashboard."""
    st.markdown(
        '<p style="font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem;">📊 Model Evaluation</p>',
        unsafe_allow_html=True,
    )

    artifacts = _load_rul_artifacts()
    metrics = artifacts.evaluation_metrics

    # ── Section 1: Model Comparison Table ──────────────────────────────
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Model Comparison</p>',
        unsafe_allow_html=True,
    )

    rows = []
    best_model_display = None
    for key, vals in metrics.items():
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        if key == BEST_MODEL_KEY:
            best_model_display = display_name
        rows.append({
            "Model": display_name,
            "RMSE": round(vals["rmse"], 2),
            "NASA Score": round(vals["nasa_score"], 2),
        })

    table_df = pd.DataFrame(rows)

    def _highlight_rows(row):
        if row["Model"] == best_model_display:
            return [f"background-color: {STATE_COLORS['Healthy']}33; font-weight: 700"] * len(row)
        # Non-winning rows get a subtle darker background for contrast
        return ["background-color: rgba(120,120,120,0.08)"] * len(row)

    styled = (
        table_df
        .style.apply(_highlight_rows, axis=1)
        .format({"RMSE": "{:.2f}", "NASA Score": "{:.2f}"})
        .hide(axis="index")
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

    # ── Section 2: Evaluation Plots ────────────────────────────────────
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Evaluation Plots</p>',
        unsafe_allow_html=True,
    )

    if PLOT_PATH.exists():
        st.image(
            str(PLOT_PATH),
            use_column_width=True,
            caption=(
                "Left: Predicted vs True RUL — "
                "Right: Prediction Error vs True RUL "
                "(red = late/dangerous, blue = early/safe)"
            ),
        )
    else:
        st.warning(f"Evaluation plot not found at `{PLOT_PATH}`.")

    # ── Section 3: Key Facts ───────────────────────────────────────────
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Key Facts</p>',
        unsafe_allow_html=True,
    )

    best_metrics = metrics[BEST_MODEL_KEY]

    # Styled metric cards with border and background
    _card_css = (
        "background: rgba(120,120,120,0.06); "
        "border: 1px solid rgba(150,150,150,0.2); "
        "border-radius: 8px; "
        "padding: 0.8rem 1rem; "
        "text-align: center;"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div style="{_card_css}">'
            f'<div style="font-size: 0.78rem; color: #888; margin-bottom: 0.3rem;">Best Model</div>'
            f'<div style="font-size: 1.05rem; font-weight: 700;">GradientBoosting</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="{_card_css}">'
            f'<div style="font-size: 0.78rem; color: #888; margin-bottom: 0.3rem;">RMSE</div>'
            f'<div style="font-size: 1.05rem; font-weight: 700;">{best_metrics["rmse"]:.2f} cycles</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="{_card_css}">'
            f'<div style="font-size: 0.78rem; color: #888; margin-bottom: 0.3rem;">Prediction Balance</div>'
            f'<div style="font-size: 1.05rem; font-weight: 700;">53 late / 47 early</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
