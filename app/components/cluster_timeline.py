import streamlit as st
import plotly.express as px
import pandas as pd

COLORS = {
    "Healthy": "#2ecc71",
    "Degrading": "#f39c12",
    "Critical": "#e74c3c"
}

def render_cluster_timeline(df: pd.DataFrame, unit_id: int):
    st.markdown(f"#### Degradation State Timeline — Engine {unit_id}")
    
    # Ensure risk_state is ordered Categorical for consistent plotting
    category_order = ["Healthy", "Degrading", "Critical"]
    df = df.copy()
    df['risk_state'] = pd.Categorical(df['risk_state'], categories=category_order, ordered=True)
    
    fig = px.scatter(
        df,
        x="cycle",
        y="risk_state",
        color="risk_state",
        color_discrete_map=COLORS,
        category_orders={"risk_state": category_order},
        hover_data=["cycle", "risk_state", "risk_score", "health_index"]
    )
    
    fig.update_traces(marker=dict(size=10, symbol="square"))
    
    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Risk State",
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(type='category', categoryorder='array', categoryarray=category_order)
    )
    
    st.plotly_chart(fig, use_container_width=True)
