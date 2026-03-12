import streamlit as st
import plotly.graph_objects as go
import pandas as pd

COLORS = {
    "Healthy": "#2ecc71",
    "Degrading": "#f39c12",
    "Critical": "#e74c3c"
}

def render_hi_plot(df: pd.DataFrame, unit_id: int):
    st.markdown(f"#### Health Index Trajectory — Engine {unit_id}")
    
    # Calculate rolling mean
    df_plot = df.copy()
    df_plot["hi_rolling_mean"] = df_plot["health_index"].rolling(window=10, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Add actual HI line
    fig.add_trace(go.Scatter(
        x=df_plot["cycle"],
        y=df_plot["health_index"],
        mode="lines",
        name="Health Index",
        line=dict(color="#3498db", width=2),
        hovertemplate="Cycle: %{x}<br>Health Index: %{y:.3f}<extra></extra>"
    ))
    
    # Add rolling mean line (dashed)
    fig.add_trace(go.Scatter(
        x=df_plot["cycle"],
        y=df_plot["hi_rolling_mean"],
        mode="lines",
        name="10-Cycle Rolling Mean",
        line=dict(color="#2980b9", width=2, dash="dash"),
        hoverinfo="skip"
    ))
    
    # Add horizontal reference line at y=0.5
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color=COLORS["Critical"],
        annotation_text="Degradation Threshold",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Health Index",
        yaxis_range=[0, 1],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
