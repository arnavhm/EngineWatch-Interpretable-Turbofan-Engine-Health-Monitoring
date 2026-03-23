import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from app.theme import STATE_COLORS, SECTION_TITLE_CSS

def render_risk_gauge(df: pd.DataFrame):
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Current Risk Score</p>',
        unsafe_allow_html=True,
    )
    
    last_record = df.iloc[-1]
    risk_score = last_record["risk_score"]
    risk_state = last_record["risk_state"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"font": {"size": 48, "color": STATE_COLORS.get(risk_state, "black")}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{risk_state}</b>",
            "font": {"size": 18, "color": STATE_COLORS.get(risk_state, "black")},
        },
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': STATE_COLORS['Healthy']},
                {'range': [0.3, 0.7], 'color': STATE_COLORS['Degrading']},
                {'range': [0.7, 1.0], 'color': STATE_COLORS['Critical']},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score,
            },
        },
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    st.plotly_chart(fig, use_container_width=True)
