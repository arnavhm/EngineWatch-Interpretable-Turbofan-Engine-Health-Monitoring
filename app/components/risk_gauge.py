import streamlit as st
import pandas as pd
from app.theme import SECTION_TITLE_CSS
from app.utils.theme import risk_dial

def render_risk_gauge(df: pd.DataFrame):
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Current Risk Score</p>',
        unsafe_allow_html=True,
    )
    
    last_record = df.iloc[-1]
    risk_score = last_record["risk_score"]
    
    st.plotly_chart(risk_dial(risk_score), use_container_width=True)
