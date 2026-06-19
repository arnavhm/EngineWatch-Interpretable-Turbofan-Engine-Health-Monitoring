import streamlit as st
import pandas as pd
from app.utils.theme import state_chip


def render_engine_selector(df: pd.DataFrame, dataset_id: str = "FD001") -> int:
    with st.sidebar:
        st.markdown("### Engine Selection")
        engine_ids = sorted(df["unit"].unique())
        # Use dataset_id as part of key to force selectbox reset on dataset change
        # Allow programmatic selection via session_state['select_engine_override']
        override_key = f"select_engine_override_{dataset_id}"
        widget_key = f"engine_selector_{dataset_id}"
        
        if override_key in st.session_state:
            try:
                override_val = int(st.session_state[override_key])
                if override_val in engine_ids:
                    st.session_state[widget_key] = override_val
            except Exception:
                # ignore invalid override
                pass
            del st.session_state[override_key]

        selected_engine = st.selectbox(
            "Select Engine",
            options=engine_ids,
            key=widget_key,
        )

        # Get last state for selected engine
        engine_last_state = df[df["unit"] == selected_engine].iloc[-1]
        current_cycle = int(engine_last_state["cycle"])
        risk_score = engine_last_state["risk_score"]
        health_state = engine_last_state["risk_state"]

        st.metric("Current Cycle", current_cycle)
        st.metric("Risk Score", f"{risk_score:.2f}")

        # Badge for Health State
        st.markdown(state_chip(health_state), unsafe_allow_html=True)

        return selected_engine
