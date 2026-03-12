import streamlit as st
import plotly.graph_objects as go
import pandas as pd

COLORS = {
    "Healthy": "#2ecc71",
    "Degrading": "#f39c12",
    "Critical": "#e74c3c"
}

def render_risk_gauge(df: pd.DataFrame):
    st.markdown("#### Current Risk Score")
    
    last_record = df.iloc[-1]
    risk_score = last_record["risk_score"]
    risk_state = last_record["risk_state"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{risk_state}</b>", "font": {"size": 16, "color": COLORS.get(risk_state, "black")}},
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # hide the default bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': COLORS['Healthy']},
                {'range': [0.3, 0.7], 'color': COLORS['Degrading']},
                {'range': [0.7, 1.0], 'color': COLORS['Critical']}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
