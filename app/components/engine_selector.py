import streamlit as st
import pandas as pd

COLORS = {
    "Healthy": "#2ecc71",
    "Degrading": "#f39c12",
    "Critical": "#e74c3c"
}

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
        
        # Badge for Health State
        badge_color = COLORS.get(health_state, "gray")
        st.markdown(
            f"""
            <div style="
                background-color: {badge_color};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                text-align: center;
                font-weight: bold;
                font-size: 1.2rem;
                margin-top: 1rem;
            ">
                {health_state}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return selected_engine
