"""Batch CSV Prediction panel — true hybrid architecture.

Uploads a raw CMAPSS-shaped CSV to the /predict/csv API endpoint over HTTP.
Config-driven base URL (env-first: API_BASE_URL env var → config.yaml fallback).
Graceful degradation when the API is unreachable (no crashes, no stack traces).
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from app.theme import STATE_COLORS, SECTION_TITLE_CSS

# ── Minimum cycles required by the pipeline for velocity/variability ────
_MIN_CYCLES_NOTE = 20


def _resolve_api_base(config: dict) -> str:
    """Return the API base URL.  Env-first, then config fallback.

    Priority:
        1. API_BASE_URL environment variable (for Streamlit Cloud → Render)
        2. dashboard.api_base_url in config.yaml (local dev default)
    """
    env = os.environ.get("API_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    return config.get("dashboard", {}).get("api_base_url", "http://localhost:8000").rstrip("/")


def render_csv_upload_panel(config: dict) -> None:
    """Render the Batch CSV Prediction section in the dashboard.

    Parameters
    ----------
    config : dict
        Loaded config/config.yaml (via ``load_config()``).
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">📤 Batch CSV Prediction</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload a raw CMAPSS-shaped sensor CSV to score every engine in one shot.  "
        "The file is sent to the **EngineWatch Inference API** over HTTP — "
        "the API must be running for this panel to work."
    )

    # ── controls ────────────────────────────────────────────────────────
    col_file, col_ds, col_btn = st.columns([3, 1, 1])

    with col_file:
        uploaded = st.file_uploader(
            "Sensor CSV",
            type=["csv", "txt"],
            help="Space- or comma-separated.  Must have columns: unit, cycle, op_setting_1–3, sensor_1–21.",
            key="csv_upload_file",
        )

    with col_ds:
        dataset_id = st.selectbox(
            "Dataset / model",
            options=["FD001", "FD002", "FD003", "FD004"],
            index=0,
            help="Selects which persisted transformers the API uses for scoring.",
            key="csv_upload_dataset",
        )

    with col_btn:
        st.write("")  # vertical spacer to align with the uploader
        predict_clicked = st.button("🚀 Predict", key="csv_upload_predict", use_container_width=True)

    if not predict_clicked or uploaded is None:
        return

    # ── call the API ────────────────────────────────────────────────────
    api_base = _resolve_api_base(config)
    url = f"{api_base}/predict/csv"

    with st.spinner("Sending CSV to prediction service…"):
        try:
            resp = requests.post(
                url,
                params={"dataset_id": dataset_id},
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=30,
            )
        except requests.exceptions.RequestException:
            st.error(
                f"⚠️ **Prediction service unavailable** — is the API running?  "
                f"(checked `{api_base}`)"
            )
            return

    # ── non-200 handling ────────────────────────────────────────────────
    if resp.status_code != 200:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail", "")
        except Exception:
            detail = resp.text[:300]
        st.error(
            f"⚠️ **API returned {resp.status_code}**"
            + (f" — {detail}" if detail else "")
        )
        return

    # ── 200 — parse and display ─────────────────────────────────────────
    data = resp.json()
    predictions: list[dict] = data.get("predictions", [])
    skipped: list[dict] = data.get("skipped", [])
    n_engines: int = data.get("n_engines", len(predictions) + len(skipped))

    st.success(
        f"**{n_engines}** engines processed: "
        f"**{len(predictions)}** predicted, **{len(skipped)}** skipped."
    )

    # ── risk-state distribution (small metrics row) ─────────────────────
    if predictions:
        state_counts: dict[str, int] = {}
        for p in predictions:
            s = p.get("risk_state", "Unknown")
            state_counts[s] = state_counts.get(s, 0) + 1
        cols = st.columns(len(state_counts))
        for col, (state, count) in zip(cols, state_counts.items()):
            color = STATE_COLORS.get(state, "#888")
            col.markdown(
                f"<span style='color:{color}; font-weight:700'>{state}</span>: {count}",
                unsafe_allow_html=True,
            )

    # ── skipped engines warning ─────────────────────────────────────────
    if skipped:
        with st.expander(f"⚠️ {len(skipped)} engine(s) skipped — insufficient cycles (≥{_MIN_CYCLES_NOTE} needed)", expanded=False):
            for s in skipped:
                reason = s.get("reason", "unknown reason")
                st.write(f"• **Engine {s.get('engine_id', '?')}** — {reason}")

    # ── predictions dataframe ───────────────────────────────────────────
    if predictions:
        df = pd.DataFrame(predictions)
        df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

        # Build display columns
        display_cols = ["engine_id", "risk_state", "risk_score", "rul_cycles"]

        # CI display: prefer ci_lower/ci_upper; fall back to ci_std if present
        has_ci_bounds = "ci_lower" in df.columns and "ci_upper" in df.columns
        has_ci_std = "ci_std" in df.columns

        if has_ci_bounds and df["ci_lower"].notna().any():
            df["CI (cycles)"] = df.apply(
                lambda r: f"{r['ci_lower']:.0f} – {r['ci_upper']:.0f}"
                if pd.notna(r.get("ci_lower")) else "—",
                axis=1,
            )
            display_cols.append("CI (cycles)")
        elif has_ci_std and df["ci_std"].notna().any():
            df["RUL ± σ"] = df.apply(
                lambda r: f"{r['rul_cycles']:.0f} ± {r['ci_std']:.1f}"
                if pd.notna(r.get("ci_std")) else f"{r['rul_cycles']:.0f}",
                axis=1,
            )
            display_cols.append("RUL ± σ")

        if "model_name" in df.columns:
            display_cols.append("model_name")

        rename = {
            "engine_id": "Engine",
            "risk_state": "Risk State",
            "risk_score": "Risk Score",
            "rul_cycles": "RUL (cycles)",
            "model_name": "Model",
        }

        df_display = df[display_cols].rename(columns=rename)

        def _color_state(val: str) -> str:
            color = STATE_COLORS.get(val, "white")
            return f"color: {color}; font-weight: bold"

        st.dataframe(
            df_display.style.map(_color_state, subset=["Risk State"]),
            hide_index=True,
            use_container_width=True,
        )
