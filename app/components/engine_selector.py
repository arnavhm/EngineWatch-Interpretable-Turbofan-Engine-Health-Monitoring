import streamlit as st
import pandas as pd
from app.theme import STATE_COLORS

def render_engine_selector(df: pd.DataFrame) -> int:
    with st.sidebar:
        st.markdown("### Engine Selection")
        engine_ids = sorted(df["unit"].unique())
        selected_engine = st.selectbox("Select Engine", options=engine_ids)
        
        # Get last state for selected engine
        engine_last_state = df[df["unit"] == selected_engine].iloc[-1]
        current_cycle = int(engine_last_state["cycle"])
        risk_score = engine_last_state["risk_score"]
        health_state = engine_last_state["risk_state"]
        
        st.metric("Current Cycle", current_cycle)
        st.metric("Risk Score", f"{risk_score:.2f}")
        
        # Badge for Health State — larger, bolder, full-width, rounded
        badge_color = STATE_COLORS.get(health_state, "gray")
        st.markdown(
            f"""
            <div style="
                background-color: {badge_color};
                color: white;
                padding: 0.75rem 1rem;
                border-radius: 10px;
                text-align: center;
                font-weight: 800;
                font-size: 1.25rem;
                letter-spacing: 0.03em;
                margin-top: 1rem;
                width: 100%;
                box-sizing: border-box;
            ">
                {health_state}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return selected_engine
