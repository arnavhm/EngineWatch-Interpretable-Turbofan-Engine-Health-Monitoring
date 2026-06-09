"""
Agentic AI diagnostic narration panel powered by Gemini.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.theme import SECTION_TITLE_CSS
from app.utils.prompt_builder import build_gemini_chat_prompt
from app.utils.nl_parser import handle_nl_query
from evaluation.validation import detect_anomalous_engines


@st.cache_data(show_spinner=False)
def fetch_gemini_narration(prompt: str, model_name: str, api_key: str) -> str:
    """
    Purpose:       Call Gemini with a cached prompt so Streamlit widget reruns do not
                   trigger repeated API requests.
    Input:         prompt (str), model_name (str), api_key (str).
    Output:        str Gemini narrative text.
    Assumptions:   google-genai is installed and the API key is valid.
    Failure:       Raises RuntimeError when the SDK import fails, the API call fails,
                   or Gemini returns an empty response.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    narrative = getattr(response, "text", None)

    if narrative is None or not str(narrative).strip():
        raise RuntimeError("Gemini returned an empty narration payload.")

    return str(narrative).strip()


def _chat_session_key(dataset_name: str, unit_id: int) -> str:
    """
    Purpose:       Build a stable session-state key for one dataset/engine chat thread.
    Input:         dataset_name (str), unit_id (int).
    Output:        str session-state key.
    Assumptions:   Dataset names are short labels such as FD001.
    Failure:       None expected.
    """
    return f"gemini_chat::{dataset_name}::{unit_id}"


def _initial_chat_state() -> dict[str, Any]:
    """
    Purpose:       Create the default in-memory chat state structure.
    Input:         None.
    Output:        dict with open flag, identity fields, and message history.
    Assumptions:   State is stored in Streamlit session_state.
    Failure:       None expected.
    """
    return {"open": False, "dataset_name": None, "unit_id": None, "history": []}


def _chat_identity_changed(
    state: dict[str, Any], dataset_name: str, unit_id: int
) -> bool:
    """
    Purpose:       Detect when the active chat thread must be reset for a new engine.
    Input:         state (dict), dataset_name (str), unit_id (int).
    Output:        bool indicating whether the identity changed.
    Assumptions:   State may be empty or partially initialised.
    Failure:       None expected.
    """
    return state.get("dataset_name") != dataset_name or state.get("unit_id") != unit_id


def _build_engine_context(
    current_row: pd.Series,
    config: dict[str, Any],
    artifacts: dict[str, Any],
    fleet_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Purpose:       Extract the current engine context used by the assistant.
    Input:         current_row (pd.Series), config (dict), artifacts (dict), fleet_df (DataFrame | None).
    Output:        dict with scalar engine fields, sensor attribution, and anomaly context.
    Assumptions:   current_row is the last-cycle snapshot for one selected engine.
    Failure:       Raises ValueError/KeyError only if required context is missing.
    """
    unit_id = int(current_row["unit"])
    current_cycle = int(current_row["cycle"])

    if "risk_score_hpc" in current_row.index and "risk_score_fan" in current_row.index:
        risk_score = max(
            float(current_row.get("risk_score_hpc", 0.0)),
            float(current_row.get("risk_score_fan", 0.0)),
        )
    else:
        risk_score = float(current_row.get("risk_score", 0.0))

    risk_state = str(current_row.get("risk_state", "Healthy"))
    health_index = float(current_row.get("health_index", 0.0))
    velocity = float(current_row.get("HI_velocity", 0.0))
    variability = float(current_row.get("HI_variability", 0.0))

    hi_pca_by_axis = artifacts.get("hi_pca_by_axis", {})
    axis_name = next(iter(config.get("health_index", {}).get("axes", {}).keys()), None)
    if axis_name is None and hi_pca_by_axis:
        axis_name = next(iter(hi_pca_by_axis.keys()))

    if axis_name not in hi_pca_by_axis:
        raise KeyError("PCA attribution artifacts are missing.")

    pca = hi_pca_by_axis[axis_name]
    axis_cfg = config.get("health_index", {}).get("axes", {}).get(axis_name, {})
    sensor_cols = [
        sensor for sensor in axis_cfg.get("sensors", []) if sensor in current_row.index
    ]

    if not sensor_cols:
        raise KeyError("No matching sensor columns were found for attribution.")

    loadings = np.asarray(pca.components_[0], dtype=float)
    if len(loadings) != len(sensor_cols):
        raise ValueError("PCA loading shape does not match the sensor list.")

    sensor_values = current_row[sensor_cols].to_numpy(dtype=float)
    contributions = sensor_values * loadings
    abs_contributions = np.abs(contributions)
    contribution_total = float(abs_contributions.sum()) or 1.0

    top_indices = np.argsort(abs_contributions)[::-1][:3]
    top_sensors: dict[str, dict[str, float]] = {}
    for rank, sensor_index in enumerate(top_indices, start=1):
        sensor_name = sensor_cols[int(sensor_index)]
        signed_contribution = float(contributions[int(sensor_index)])
        loading = float(loadings[int(sensor_index)])
        sensor_value = float(sensor_values[int(sensor_index)])
        share_pct = float(
            abs_contributions[int(sensor_index)] / contribution_total * 100.0
        )
        top_sensors[sensor_name] = {
            "rank": int(rank),
            "sensor_value": round(sensor_value, 4),
            "loading": round(loading, 4),
            "signed_contribution": round(signed_contribution, 4),
            "abs_contribution_share_pct": round(share_pct, 1),
        }

    is_anomalous = False
    anomaly_reason = "Within normal fleet range"
    if fleet_df is not None and not fleet_df.empty:
        anomaly_df = detect_anomalous_engines(fleet_df)
        anomaly_row = anomaly_df[anomaly_df["unit"] == unit_id]
        if not anomaly_row.empty:
            is_anomalous = bool(anomaly_row["is_anomaly"].iloc[0])
            anomaly_reason = str(anomaly_row["anomaly_reason"].iloc[0])

    return {
        "unit_id": unit_id,
        "current_cycle": current_cycle,
        "risk_score": risk_score,
        "risk_state": risk_state,
        "health_index": health_index,
        "velocity": velocity,
        "variability": variability,
        "top_sensors": top_sensors,
        "is_anomalous": is_anomalous,
        "anomaly_reason": anomaly_reason,
    }


def _render_chat_messages(history: list[dict[str, str]]) -> None:
    """
    Purpose:       Render the stored conversation turns as chat bubbles.
    Input:         history (list[dict[str, str]]) containing user/assistant messages.
    Output:        None.
    Assumptions:   Each message dict has 'role' and 'content' keys.
    Failure:       None expected.
    """
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_narration_panel(
    engine_df: pd.DataFrame,
    config: dict[str, Any],
    predicted_rul: float,
    rul_ci: tuple[float, float],
    artifacts: dict[str, Any],
    fleet_df: pd.DataFrame | None = None,
) -> None:
    """
    Purpose:       Build the Gemini prompt from the current engine snapshot, fetch the
                   narrative, and render it inside a compact Streamlit expander.
    Input:         engine_df (pd.DataFrame) for one engine; config (dict) with dashboard
                   and health-index settings; predicted_rul (float); rul_ci (tuple[float, float]);
                   artifacts (dict) with hi_pca_by_axis; fleet_df (pd.DataFrame | None) for
                   anomaly lookup across the full monitored fleet.
    Output:        None. Renders a diagnostic expander into Streamlit.
    Assumptions:   engine_df contains the latest cycle rows and the expected feature columns;
                   config provides dashboard.gemini_model_name; GEMINI_API_KEY is set in the
                   local environment; artifacts contains the fitted PCA map for attribution.
    Failure:       No exceptions escape to the frontend; missing inputs, API failures, or
                   artifact issues are handled with Streamlit warnings or info messages.
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">🤖 Agentic AI Diagnostic Assistant</p>',
        unsafe_allow_html=True,
    )

    if engine_df.empty:
        st.info("Narration unavailable: no engine rows were supplied.")
        return

    # Prefer Streamlit secrets (for Streamlit Cloud / local .streamlit/secrets.toml),
    # then fall back to environment variables.
    api_key = ""
    try:
        api_key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
    except Exception:
        # st.secrets may not be available in some test contexts
        api_key = ""

    # Allow a session-scoped pasted key for quick local testing
    if "GEMINI_API_KEY_SESSION" in st.session_state:
        api_key = str(st.session_state.get("GEMINI_API_KEY_SESSION", "")).strip()

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        st.info(
            "Narration unavailable: GEMINI_API_KEY is not set.\n"
            "Options to enable narration:\n"
            '- Locally: `export GEMINI_API_KEY="your_key"` then run Streamlit.\n'
            "- Persist in shell: add the export to your `~/.zshrc` or `~/.bashrc`.\n"
            "- Streamlit Cloud: add `GEMINI_API_KEY` to the app secrets (Settings → Secrets).\n"
            '- Local dev: create a file `.streamlit/secrets.toml` with `GEMINI_API_KEY = "your_key"`.'
        )

        # Provide a temporary, session-scoped input so devs can paste a key into the UI
        with st.expander(
            "Paste GEMINI key for this session (temporary)", expanded=False
        ):
            pasted = st.text_input(
                "Paste GEMINI_API_KEY (hidden)",
                type="password",
                key="_gemini_key_paste",
            )
            if pasted:
                if st.button(
                    "Use pasted key for this session", key="_use_pasted_gemini"
                ):
                    st.session_state["GEMINI_API_KEY_SESSION"] = pasted
                    st.rerun()

        return

    ordered_engine = engine_df.sort_values("cycle")
    current_row = ordered_engine.iloc[-1]
    model_name = config["dashboard"]["gemini_model_name"]

    try:
        engine_context = _build_engine_context(current_row, config, artifacts, fleet_df)
    except (KeyError, ValueError) as exc:
        st.warning(f"Narration unavailable: {exc}")
        return

    session_key = _chat_session_key(
        config["dataset"]["name"], engine_context["unit_id"]
    )
    if session_key not in st.session_state:
        st.session_state[session_key] = _initial_chat_state()

    chat_state = st.session_state[session_key]
    if _chat_identity_changed(
        chat_state, config["dataset"]["name"], engine_context["unit_id"]
    ):
        st.session_state[session_key] = _initial_chat_state()
        chat_state = st.session_state[session_key]
        chat_state["dataset_name"] = config["dataset"]["name"]
        chat_state["unit_id"] = engine_context["unit_id"]

    chat_state["dataset_name"] = config["dataset"]["name"]
    chat_state["unit_id"] = engine_context["unit_id"]

    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.caption(
            f"Model: {model_name} · Engine {engine_context['unit_id']} · Cycle {engine_context['current_cycle']} · "
            f"RUL {predicted_rul:.1f} cycles [{rul_ci[0]:.1f}, {rul_ci[1]:.1f}]"
        )
    with c_right:
        if st.button(
            "Open assistant" if not chat_state["open"] else "Hide assistant",
            key=f"{session_key}::toggle",
            use_container_width=True,
        ):
            chat_state["open"] = not chat_state["open"]

    if not chat_state["open"]:
        with st.expander("🤖 Agentic AI Diagnostic Summary", expanded=False):
            st.info("Open the assistant to ask questions about this engine snapshot.")
            if engine_context["is_anomalous"]:
                st.warning(f"Anomaly flag: {engine_context['anomaly_reason']}")
            st.markdown(
                "This assistant explains the current engine state, the sensor attribution, and the RUL estimate. "
                "It cannot recalculate metrics or change the underlying model outputs."
            )
        return

    with st.expander("🤖 Agentic AI Diagnostic Summary", expanded=True):
        if engine_context["is_anomalous"]:
            st.warning(f"Anomaly flag: {engine_context['anomaly_reason']}")

        # Integrated quick natural-language lookup inside the assistant UI
        st.markdown("**Quick natural-language lookup**")
        nl_query_local = st.text_input(
            "Ask across the fleet (e.g., 'state of engine 14 in FD001')",
            key=f"{session_key}::nl_quick",
        )
        if (
            st.button("Run fleet query", key=f"{session_key}::nl_run")
            and nl_query_local
        ):
            # Use fleet_df for validating engine existence
            ok, msg, selection = handle_nl_query(
                nl_query_local,
                fleet_df if fleet_df is not None else engine_df,
                st.session_state,
                require_confirmation=True,
            )
            if not ok:
                st.info(msg)
            else:
                if selection is not None:
                    st.info(msg)
                    st.markdown("Confirm selection:")
                    if st.button("Confirm select", key=f"{session_key}::nl_confirm"):
                        ds, eng = selection
                        st.session_state["last_dataset_id"] = ds
                        st.session_state[f"select_engine_override_{ds}"] = eng
                        st.rerun()
                    if st.button("Undo", key=f"{session_key}::nl_undo"):
                        st.info("Selection canceled.")
                else:
                    st.success(msg)

        _render_chat_messages(chat_state["history"])

        user_query = st.chat_input(
            "Ask about this engine snapshot",
            key=f"{session_key}::input",
        )

        if user_query:
            chat_state["history"].append({"role": "user", "content": user_query})
            prompt = build_gemini_chat_prompt(
                unit_id=engine_context["unit_id"],
                current_cycle=engine_context["current_cycle"],
                health_index=engine_context["health_index"],
                velocity=engine_context["velocity"],
                variability=engine_context["variability"],
                risk_score=engine_context["risk_score"],
                risk_state=engine_context["risk_state"],
                predicted_rul=predicted_rul,
                rul_ci=rul_ci,
                top_sensors=engine_context["top_sensors"],
                is_anomalous=engine_context["is_anomalous"],
                anomaly_reason=engine_context["anomaly_reason"],
                user_query=user_query,
                chat_history=chat_state["history"],
            )

            try:
                with st.spinner("Gemini is thinking..."):
                    assistant_reply = fetch_gemini_narration(
                        prompt=prompt,
                        model_name=model_name,
                        api_key=api_key,
                    )
                chat_state["history"].append(
                    {"role": "assistant", "content": assistant_reply}
                )
                # Detect generic/limited-data replies and append a clarifying note
                low = assistant_reply.lower()
                generic_indicators = [
                    "i can only provide",
                    "i only have data",
                    "i only have information",
                    "i can only access data",
                ]
                if any(ind in low for ind in generic_indicators):
                    # Show available unit ids as context for the clarification
                    avail = []
                    try:
                        avail = (
                            sorted(fleet_df["unit"].unique())
                            if fleet_df is not None
                            else []
                        )
                    except Exception:
                        avail = []
                    avail_sample = avail[:5]
                    clar = (
                        "Clarification: the assistant appears to have limited access to dataset rows. "
                        "Available Unit IDs (sample): "
                        + str(avail_sample)
                        + ("..." if len(avail) > 5 else "")
                        + " — use the Quick Lookup to select a different engine or confirm selection."
                    )
                    chat_state["history"].append({"role": "assistant", "content": clar})
                st.rerun()
            except Exception as exc:
                chat_state["history"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Gemini narration unavailable: {exc}. "
                            "The dashboard data remain intact; this assistant is read-only."
                        ),
                    }
                )
                st.rerun()
