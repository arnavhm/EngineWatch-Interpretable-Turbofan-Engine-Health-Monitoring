import streamlit as st
import pandas as pd
from app.theme import STATE_COLORS


def render_engine_selector(df: pd.DataFrame, dataset_id: str = "FD001") -> int:
    with st.sidebar:
        st.markdown("### Engine Selection")
        engine_ids = sorted(df["unit"].unique())
        # Use dataset_id as part of key to force selectbox reset on dataset change
        # Allow programmatic selection via session_state['select_engine_override']
        override_key = f"select_engine_override_{dataset_id}"
        initial_index = 0
        if override_key in st.session_state:
            try:
                override_val = int(st.session_state[override_key])
                if override_val in engine_ids:
                    initial_index = engine_ids.index(override_val)
            except Exception:
                # ignore invalid override
                pass

        selected_engine = st.selectbox(
            "Select Engine",
            options=engine_ids,
            index=initial_index,
            key=f"engine_selector_{dataset_id}",
        )

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
            unsafe_allow_html=True,
        )

        return selected_engine
