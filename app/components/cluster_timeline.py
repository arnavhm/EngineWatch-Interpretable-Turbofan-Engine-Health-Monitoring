import pandas as pd
import plotly.express as px
import streamlit as st

from app.theme import SECTION_TITLE_CSS
from app.utils.theme import STATE_COLORS, apply_plotly_theme


def render_cluster_timeline(df: pd.DataFrame, unit_id: int):
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Degradation State Timeline — Engine {unit_id}</p>',
        unsafe_allow_html=True,
    )

    # Ensure risk_state is ordered Categorical for consistent plotting
    category_order = ["Healthy", "Degrading", "Critical"]
    df = df.copy()
    df["risk_state"] = pd.Categorical(
        df["risk_state"], categories=category_order, ordered=True
    )

    fig = px.scatter(
        df,
        x="cycle",
        y="risk_state",
        color="risk_state",
        color_discrete_map=STATE_COLORS,
        category_orders={"risk_state": category_order},
        hover_data=["cycle", "risk_state", "risk_score", "health_index"],
    )

    fig.update_traces(marker=dict(size=10, symbol="square"))

    fig.update_layout(
        xaxis_title="Cycle",
        yaxis_title="Risk State",
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(
            type="category", categoryorder="array", categoryarray=category_order
        ),
    )

    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
