"""
AOG Cost Impact Panel — EngineWatch Dashboard Component
Native Streamlit only. No raw HTML. Called by app/dashboard.py.

Usage:
    from app.components.aog_panel import render_aog_panel
    render_aog_panel(risk_score, rul_pred, risk_state, config)
"""

import streamlit as st
from app.components.aog_cost_simulator import compute_maintenance_decision
from app.utils.theme import TOKENS

_URGENCY_COLOURS: dict[str, str] = {
    "CRITICAL": TOKENS["critical"],
    "HIGH": TOKENS["degrading"],
    "MODERATE": TOKENS["degrading"],  # Fallback to caution
    "LOW": TOKENS["healthy"],
}

_URGENCY_ICONS: dict[str, str] = {
    "CRITICAL": "🚨",
    "HIGH": "⚠️",
    "MODERATE": "🔔",
    "LOW": "✅",
}


def _cr(value: float) -> str:
    """Format Crores value for display."""
    return f"₹ {value:.2f} Cr"


def render_aog_panel(
    risk_score: float,
    rul_pred: int,
    risk_state: str,
    config: dict,
) -> None:
    """
    Purpose:      Render AOG Cost Impact panel using compact horizontal layout.
    Input:        risk_score float, rul_pred int, risk_state str, config dict
    Output:       None — renders into Streamlit context
    Failure:      Catches ValueError/KeyError, renders error box, does not crash.
    """
    st.markdown("---")
    st.markdown("### 💰 AOG Cost Impact Analysis")

    try:
        d = compute_maintenance_decision(
            risk_score=risk_score,
            rul_cycles=int(rul_pred),
            risk_state=risk_state,
            config=config,
        )
    except (ValueError, KeyError) as exc:
        st.error(f"AOG simulator error: {exc}")
        return

    # --- Decision summary row (tight horizontal layout) ---
    colour = _URGENCY_COLOURS.get(d["urgency_level"], TOKENS["muted"])
    icon = _URGENCY_ICONS.get(d["urgency_level"], "")

    c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.8, 1.8])

    with c1:
        st.markdown(
            f"<span style='background:{colour};color:white;padding:4px 10px;"
            f"border-radius:4px;font-weight:bold;font-size:0.9rem;display:inline-block;'>"
            f"{icon} {d['urgency_level']}</span>",
            unsafe_allow_html=True,
        )

    with c2:
        st.metric("Risk Score", f"{risk_score:.2f}", risk_state, delta_color="inverse")

    with c3:
        st.metric(
            "Predicted RUL",
            f"{rul_pred} cycles",
            f"~{d['failure_window_days']}d",
            delta_color="inverse",
        )

    with c4:
        st.metric("Failure Prob.", f"{d['failure_probability']:.1%}")

    st.markdown(f"**{d['recommendation']}**")

    # --- Cost comparison table (horizontal) ---
    import pandas as pd

    comparison_data = {
        "Scenario": ["Act Now (Preventive)", "Wait (Reactive AOG)", "Difference"],
        "Direct Cost": [
            _cr(d["preventive_cost_rs_cr"]),
            _cr(d["aog_direct_cost_rs_cr"]),
            _cr(d["aog_direct_cost_rs_cr"] - d["preventive_cost_rs_cr"]),
        ],
        "Revenue Impact": [
            "Minimal",
            _cr(d["revenue_loss_rs_cr"]),
            _cr(d["revenue_loss_rs_cr"]),
        ],
        "Other Costs": [
            "None",
            _cr(d["disruption_cost_rs_cr"]),
            _cr(d["disruption_cost_rs_cr"]),
        ],
        "Total Cost": [
            _cr(d["preventive_cost_rs_cr"]),
            _cr(d["expected_aog_cost_rs_cr"]),
            _cr(d["estimated_saving_rs_cr"])
            if d["act_now"]
            else _cr(d["expected_aog_cost_rs_cr"] - d["preventive_cost_rs_cr"]),
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # --- Decision & reasoning (horizontal) ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        if d["act_now"]:
            st.success(
                f"✅ **Act Now.** Save {_cr(d['estimated_saving_rs_cr'])} "
                f"by scheduling preventive maintenance."
            )
        else:
            st.info(
                f"⏳ **Continue Monitoring.** AOG risk does not yet justify "
                f"preventive action."
            )

    with col_right:
        with st.expander("ℹ️ Reasoning"):
            st.write(d["explanation"])
            st.caption(
                "Sources: Go First NCLT (2023), IATA MCX FY2024, "
                "BTS Form 41 P-5.2 FY2023, Eurocontrol Std Inputs Ed.10 (2024)."
            )
