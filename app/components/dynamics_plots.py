import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from app.theme import STATE_COLORS, SECTION_TITLE_CSS

def render_dynamics_plots(df: pd.DataFrame):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<p style="{SECTION_TITLE_CSS}">Degradation Velocity</p>',
            unsafe_allow_html=True,
        )
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
            y=[min(v, 0) for v in df["HI_velocity"]],
            fill='tonexty',
            fillcolor=f'rgba({int(STATE_COLORS["Critical"][1:3], 16)}, {int(STATE_COLORS["Critical"][3:5], 16)}, {int(STATE_COLORS["Critical"][5:7], 16)}, 0.3)',
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
        st.markdown(
            f'<p style="{SECTION_TITLE_CSS}">Health Variability</p>',
            unsafe_allow_html=True,
        )
        fig_var = go.Figure()
        
        # Derive RGBA from Degrading colour
        _dg = STATE_COLORS["Degrading"]
        _dg_rgba = f'rgba({int(_dg[1:3], 16)}, {int(_dg[3:5], 16)}, {int(_dg[5:7], 16)}, 0.4)'

        fig_var.add_trace(go.Scatter(
            x=df["cycle"],
            y=df["HI_variability"],
            mode="lines",
            fill="tozeroy",
            fillcolor=_dg_rgba,
            line=dict(color=STATE_COLORS["Degrading"], width=2),
            name="Variability",
            hovertemplate="Cycle: %{x}<br>Variability: %{y:.4f}<extra></extra>"
        ))
        
        fig_var.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Variability",
            hovermode="x unified",
        )
        st.plotly_chart(fig_var, use_container_width=True)
