import streamlit as st
import plotly.graph_objects as go
import pandas as pd

COLORS = {
    "Healthy": "#2ecc71",
    "Degrading": "#f39c12",
    "Critical": "#e74c3c"
}

def render_dynamics_plots(df: pd.DataFrame):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Degradation Velocity")
        fig_vel = go.Figure()
        
        # Add velocity line
        fig_vel.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["HI_velocity"],
            mode="lines",
            name="Velocity",
            line=dict(color="#3498db", width=2),
            hovertemplate="Cycle: %{x}<br>Velocity: %{y:.4f}<extra></extra>"
        ))
        
        # Add horizontal reference line at y=0
        fig_vel.add_hline(
            y=0,
            line_dash="dot",
            line_color="black",
        )
        
        # Add shaded area for negative velocity
        fig_vel.add_trace(go.Scatter(
            x=df["cycle"],
            y=[0] * len(df),
            hoverinfo="skip",
            showlegend=False,
            line=dict(width=0)
        ))
        
        fig_vel.add_trace(go.Scatter(
            x=df["cycle"],
            y=[min(v, 0) for v in df["HI_velocity"]], # Mask positive values to 0
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.3)', # Red shading for negative values
            line=dict(width=0),
            name="Active Degradation",
            hoverinfo="skip",
            showlegend=False
        ))
        
        fig_vel.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Velocity",
            hovermode="x unified",
        )
        st.plotly_chart(fig_vel, use_container_width=True)
        
    with col2:
        st.markdown("#### Health Variability")
        fig_var = go.Figure()
        
        fig_var.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["HI_variability"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.4)", # Orange with 40% opacity
            line=dict(color="#f39c12", width=2),
            name="Variability",
            hovertemplate="Cycle: %{x}<br>Variability: %{y:.4f}<extra></extra>"
        ))
        
        fig_var.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Variability",
            hovermode="x unified",
        )
        st.plotly_chart(fig_var, use_container_width=True)
