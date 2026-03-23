import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.theme import STATE_COLORS, SECTION_TITLE_CSS

def render_fleet_overview(df: pd.DataFrame):
    st.markdown(
        f'<p style="font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem;">Fleet Risk Overview</p>',
        unsafe_allow_html=True,
    )
    
    # Snapshot of the last cycle for each engine
    df_last = df.groupby("unit").last().reset_index()
    
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Top 5", "Heatmap"])
    
    with tab1:
        st.markdown(
            f'<p style="{SECTION_TITLE_CSS}">Fleet Risk Distribution</p>',
            unsafe_allow_html=True,
        )
        fig_bar = px.bar(
            df_last,
            x="unit",
            y="risk_score",
            color="risk_state",
            color_discrete_map=STATE_COLORS,
            hover_data=["unit", "cycle", "risk_score", "risk_state"],
            labels={"unit": "Engine ID", "risk_score": "Risk Score", "risk_state": "Risk State"}
        )
        fig_bar.update_layout(xaxis_title="Engine ID", yaxis_title="Risk Score")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab2:
        st.markdown(
            f'<p style="{SECTION_TITLE_CSS}">Priority Maintenance List</p>',
            unsafe_allow_html=True,
        )
        # Top 5 by risk_score
        top_5 = df_last.sort_values(by="risk_score", ascending=False).head(5)
        top_5_display = top_5[["unit", "cycle", "risk_score", "risk_state"]].copy()
        top_5_display.insert(0, "Rank", range(1, 6))
        top_5_display.rename(columns={"unit": "Engine ID", "cycle": "Current Cycle", "risk_score": "Risk Score", "risk_state": "Risk State"}, inplace=True)
        
        # Format styling to color rows by Risk State
        def color_risk_state(val):
            color = STATE_COLORS.get(val, "white")
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(
            top_5_display.style.map(color_risk_state, subset=['Risk State']),
            hide_index=True,
            use_container_width=True
        )

    with tab3:
        st.markdown(
            f'<p style="{SECTION_TITLE_CSS}">Fleet Degradation Heatmap</p>',
            unsafe_allow_html=True,
        )
        
        # Pivot the full dataframe to get cycle as columns and unit as rows, with risk_score as values
        heatmap_data = df.pivot(index="unit", columns="cycle", values="risk_score")
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, STATE_COLORS['Healthy']], [0.5, STATE_COLORS['Degrading']], [1.0, STATE_COLORS['Critical']]],
            hoverongaps=False,
            hovertemplate="Engine: %{y}<br>Cycle: %{x}<br>Risk Score: %{z:.3f}<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Engine ID",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
