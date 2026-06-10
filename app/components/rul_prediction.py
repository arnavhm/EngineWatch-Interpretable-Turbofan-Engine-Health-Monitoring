"""
app/components/rul_prediction.py

Purpose:      Display predicted Remaining Useful Life for the selected engine.
              Point prediction from best model (GradientBoosting).
              Confidence interval from Random Forest tree variance.
"""

import numpy as np
import streamlit as st
import pandas as pd
from typing import Any

from data.load import load_config
from app.theme import STATE_COLORS, SECTION_TITLE_CSS
from app.utils.rul_artifacts import load_or_rebuild_rul_artifacts

FEATURE_COLUMNS = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


@st.cache_resource
def _load_rul_artifacts(dataset_id: str = "FD001") -> Any:
    """
    Purpose:      Load pre-trained RUL artifacts for inference. Cached per dataset.
    Input:        dataset_id — FD001/FD002/FD003/FD004
    Output:       RULArtifacts object
    Failure:      RuntimeError propagated if artifacts not found on disk
    """
    return load_or_rebuild_rul_artifacts(dataset_id=dataset_id)


@st.cache_data
def _load_rul_bands() -> tuple[float, float]:
    """Load dashboard RUL band thresholds from config."""
    config = load_config()
    bands = config.get("dashboard", {}).get("rul_bands", {})
    healthy_min = float(bands.get("healthy_min", 80))
    degrading_min = float(bands.get("degrading_min", 30))
    return healthy_min, degrading_min


def _compute_rf_ci(
    rf_model: Any,
    features: np.ndarray,
    point_pred: float,
) -> tuple[float, float, float]:
    """
    Purpose:      Compute confidence interval for one feature vector using RF tree variance.
                  Std is derived from individual tree predictions (model disagreement).
                  Bounds are expressed as point_pred ± std so the headline number
                  is always the best model (GB) prediction.
    Input:        rf_model — fitted RandomForestRegressor from artifacts.all_models
                  features — (1, 4) numpy array for the selected engine's last cycle
                  point_pred — GB point prediction (float), used as interval centre
    Output:       (ci_lower, ci_upper, ci_std) — all floats, ci_lower floored at 0
    Assumptions:  rf_model has .estimators_ attribute (sklearn RF always does)
    Failure:      AttributeError if rf_model is not a fitted RF — caller guards this
    """
    tree_preds = np.array([tree.predict(features)[0] for tree in rf_model.estimators_])
    ci_std = float(tree_preds.std())
    ci_lower = max(point_pred - ci_std, 0.0)
    ci_upper = point_pred + ci_std
    return ci_lower, ci_upper, ci_std


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

    artifacts = _load_rul_artifacts(dataset_id=dataset_id)
    model = artifacts.best_model
    metrics = artifacts.evaluation_metrics
    model_key = artifacts.best_model_name

    # Last cycle row for this engine
    last_row = df.iloc[[-1]]
    features = last_row[FEATURE_COLUMNS]

    predicted_rul = float(model.predict(features)[0])
    predicted_rul = max(predicted_rul, 0.0)

    # ── Confidence interval from RF tree variance ─────────────────────
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_std: float | None = None

    rf_model = artifacts.all_models.get("random_forest")
    if rf_model is not None and hasattr(rf_model, "estimators_"):
        ci_lower, ci_upper, ci_std = _compute_rf_ci(
            rf_model, features.values, predicted_rul
        )

    # ── Readable model name and RMSE ─────────────────────────────────
    model_name = type(model).__name__
    model_metrics = metrics.get(model_key, {})
    rmse = model_metrics.get("rmse") if isinstance(model_metrics, dict) else None

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
