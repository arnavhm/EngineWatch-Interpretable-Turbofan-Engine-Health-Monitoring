"""
app/components/sensor_panel.py

Purpose:    Per-engine sensor trajectory panel with expandable info blocks.
            Two tabs — Degradation Indicators (rising = unhealthy) and
            Health Indicators (falling = unhealthy) — drawn from the 14
            active sensors in SENSOR_CATALOG.
            Confirmed Phase 1 correlations render as solid lines;
            informational signals render as dashed lines.
            Late-life shading covers the final 20% of the engine's cycles.
            A 10-cycle rolling mean reference line is overlaid on each chart.

Usage:
    from app.components.sensor_panel import render_sensor_panel
    render_sensor_panel(engine_df, selected_engine_id)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.utils.sensor_catalog import SENSOR_CATALOG
from app.theme import SECTION_TITLE_CSS
from app.utils.theme import TOKENS, apply_plotly_theme, rgba


def render_sensor_panel(engine_df: pd.DataFrame, unit_id: int) -> None:
    """
    Purpose:    Render per-sensor trajectory panel for a single engine.
    Input:      engine_df — DataFrame filtered to one engine unit; must contain
                             'cycle' column and sensor columns matching
                             SENSOR_CATALOG keys.
                unit_id   — integer engine unit identifier for display labels.
    Output:     None — side-effects render into active Streamlit context.
    Assumptions: engine_df is sorted by cycle ascending (standard pipeline output).
    Failure:    Sensor columns absent from engine_df are skipped silently.
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">📡 Sensor Trajectories — Engine {unit_id}</p>',
        unsafe_allow_html=True,
    )

    degradation_sensors = [
        k for k, v in SENSOR_CATALOG.items() if v["signal_direction"] == "rising"
    ]
    health_sensors = [
        k for k, v in SENSOR_CATALOG.items() if v["signal_direction"] == "falling"
    ]

    tab1, tab2 = st.tabs(["📈 Degradation Indicators", "📉 Health Indicators"])

    with tab1:
        _render_sensor_tab(engine_df, degradation_sensors)

    with tab2:
        _render_sensor_tab(engine_df, health_sensors)


def _render_sensor_tab(engine_df: pd.DataFrame, sensors: list[str]) -> None:
    """
    Purpose:    Render compact plots and expandable info blocks for a sensor list.
    Input:      engine_df — single-engine DataFrame; 'cycle' column required.
                sensors   — ordered list of SENSOR_CATALOG keys to render.
    Output:     None — renders into active Streamlit context.
    Assumptions: 'cycle' column is present in engine_df.
    Failure:    Sensors absent from engine_df are silently excluded;
                if no sensors are available an info message is shown.
    """
    available = [s for s in sensors if s in engine_df.columns]
    if not available:
        st.info("No sensor data available for this engine.")
        return

    max_cycle = float(engine_df["cycle"].max())
    late_life_start = max_cycle * 0.80

    cols = st.columns(2)
    for i, sensor in enumerate(available):
        meta = SENSOR_CATALOG[sensor]
        with cols[i % 2]:
            _render_single_sensor(engine_df, sensor, meta, late_life_start, max_cycle)


def _render_single_sensor(
    engine_df: pd.DataFrame,
    sensor: str,
    meta: dict,
    late_life_start: float,
    max_cycle: float,
) -> None:
    """
    Purpose:    Render one sensor's trajectory chart and its expandable info block.
    Input:      engine_df       — single-engine DataFrame.
                sensor          — sensor column name (must be in engine_df.columns).
                meta            — SensorMeta dict from SENSOR_CATALOG.
                late_life_start — cycle at which late-life shading begins (80% of max).
                max_cycle       — final observed cycle for the engine.
    Output:     None — renders chart + expander into active Streamlit context.
    Assumptions: sensor exists in engine_df.columns.
    Failure:    plotly rendering errors propagate to Streamlit's error boundary.
    """
    line_dash = "solid" if meta["confirmed"] else "dash"
    rolling_mean = engine_df[sensor].rolling(window=10, min_periods=1).mean()

    fig = go.Figure()

    # Late-life shading — last 20 % of cycles
    fig.add_vrect(
        x0=late_life_start,
        x1=max_cycle,
        fillcolor=rgba(TOKENS["critical"], 0.08),
        layer="below",
        line_width=0,
        annotation_text="Late-life",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=TOKENS["critical"],
    )

    # Sensor signal
    units_suffix = f" {meta['units']}" if meta["units"] else ""
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[sensor],
            mode="lines",
            name=meta["symbol"],
            line=dict(color=TOKENS["accent"], width=1.5, dash=line_dash),
            hovertemplate=(
                f"Cycle: %{{x}}<br>{meta['symbol']}: %{{y:.3f}}{units_suffix}"
                "<extra></extra>"
            ),
        )
    )

    # 10-cycle rolling mean reference
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=rolling_mean,
            mode="lines",
            name="10-cycle mean",
            line=dict(color=TOKENS["muted"], width=1.2, dash="dot"),
            hoverinfo="skip",
        )
    )

    units_axis = f" ({meta['units']})" if meta["units"] else ""
    fig.update_layout(
        title=dict(
            text=f"{meta['symbol']} — {meta['description']}",
            font=dict(size=12),
            x=0,
        ),
        xaxis_title="Cycle",
        yaxis_title=f"{meta['symbol']}{units_axis}",
        height=220,
        margin=dict(t=45, b=35, l=55, r=15),
        showlegend=False,
    )

    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    _render_sensor_reference(sensor, meta)


def _render_sensor_reference(sensor: str, meta: dict) -> None:
    """
    Purpose:    Render expandable sensor info block with physical metadata and
                plain English explanation.
    Input:      sensor — sensor column name (used as stable expander key).
                meta   — SensorMeta dict from SENSOR_CATALOG.
    Output:     None — renders expander into active Streamlit context.
    Assumptions: meta contains keys: symbol, confirmed, module, units,
                 signal_direction.  'explanation' may be absent — falls back
                 to a default string.
    Failure:    KeyError on required fields propagates; explanation missing
                is handled gracefully via dict.get().
    """
    badge = "✅ Confirmed" if meta["confirmed"] else "ℹ️ Informational"
    direction_label = (
        "Rising with degradation ↑"
        if meta["signal_direction"] == "rising"
        else "Falling with degradation ↓"
    )

    with st.expander(f"{meta['symbol']} — {badge} · {meta['module']}"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Symbol:** {meta['symbol']}")
            st.markdown(f"**Module:** {meta['module']}")
            st.markdown(f"**Units:** {meta['units'] if meta['units'] else '—'}")
        with c2:
            st.markdown(f"**Signal:** {direction_label}")
            st.markdown(f"**Status:** {badge}")
        st.write(meta.get("explanation", "No detail available for this sensor."))
