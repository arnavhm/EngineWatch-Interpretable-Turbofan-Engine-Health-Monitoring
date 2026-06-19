"""
EngineWatch dashboard theme — Mission Control (dark / EICAS-grounded).

Place at: app/utils/theme.py

Centralises all presentation tokens and styling so the dashboard reads as a
single coherent instrument panel. Health-state colours follow the real aviation
crew-alerting convention (EICAS): green = nominal, amber = caution, red = warning.
This is a domain choice, not decoration — the colour *is* the alert level.

Public API:
    TOKENS, STATE_COLORS          - palette + state->colour map (single source of truth)
    inject_css()                  - call once near the top of dashboard.py
    apply_plotly_theme(fig)       - dark-style any Plotly figure in place
    risk_dial(risk_score, ...)    - cockpit gauge (go.Indicator) for the risk panel
    state_chip(state)             - HTML pill for a Healthy/Degrading/Critical badge

No ML, no pipeline logic, no global mutable state. Pure presentation.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Tokens — the single source of truth for the whole dashboard's look.
# (Kept here rather than config.yaml because these are presentation-only and
#  are consumed by CSS/Plotly, not the ML pipeline. Move to config.yaml later
#  if strict charter parity is wanted.)
# --------------------------------------------------------------------------- #
TOKENS: dict[str, str] = {
    "bg":        "#0B1014",
    "panel":     "#121A21",
    "panel_2":   "#18222B",
    "border":    "#243039",
    "text":      "#E6EDF3",
    "muted":     "#8896A3",
    "faint":     "#5C6975",
    "healthy":   "#2DD4A7",  # EICAS nominal (green)
    "degrading": "#F5A524",  # EICAS caution (amber)
    "critical":  "#FF5A5F",  # EICAS warning (red)
    "accent":    "#4DA3FF",  # neutral interactive data (kept off alarm colours)
    "sans":      "'IBM Plex Sans', system-ui, sans-serif",
    "mono":      "'IBM Plex Mono', monospace",
}

# Canonical pipeline state labels -> colour. Keys match risk_state values.
STATE_COLORS: dict[str, str] = {
    "Healthy":   TOKENS["healthy"],
    "Degrading": TOKENS["degrading"],
    "Critical":  TOKENS["critical"],
}

DATASET_LABELS: dict[str, str] = {
    "FD001": "FD001 — Baseline · 1 condition, 1 fault (HPC)",
    "FD002": "FD002 — Multi-regime · 6 conditions, 1 fault (HPC)",
    "FD003": "FD003 — Dual-fault · 1 condition, 2 faults (HPC + Fan)",
    "FD004": "FD004 — Complex · 6 conditions, 2 faults (HPC + Fan)",
}


def inject_css() -> None:
    """
    Purpose:      Load IBM Plex and apply mission-control styling to Streamlit's
                  native chrome (app bg, headers, metrics, sidebar, inputs, chips).
    Input:        None. Reads from module-level TOKENS.
    Output:       None. Writes one <style> block via st.markdown.
    Assumptions:  Called once, after st.set_page_config, before widgets render.
                  config.toml has already set base="dark" + base palette.
    Failure:      None raised. If Streamlit's internal data-testid names change
                  across versions, some selectors no-op silently (layout still works).
    """
    t = TOKENS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        /* base typography */
        html, body, [class*="css"], .stApp {{
            font-family: {t['sans']};
            color: {t['text']};
        }}
        .stApp {{ background: {t['bg']}; }}

        /* headers — IBM Plex Sans, tight, sentence case */
        h1, h2, h3, h4 {{
            font-family: {t['sans']};
            font-weight: 600;
            letter-spacing: -0.2px;
            color: {t['text']};
        }}
        h1 {{ font-size: 22px; }}
        h2 {{ font-size: 18px; }}
        h3 {{ font-size: 15px; color: {t['muted']}; text-transform: uppercase;
              letter-spacing: 0.7px; font-weight: 500; }}

        /* metric cards — st.metric */
        [data-testid="stMetric"] {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: 10px;
            padding: 16px 18px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {t['muted']};
            text-transform: uppercase;
            letter-spacing: 0.7px;
            font-size: 11px;
        }}
        [data-testid="stMetricValue"] {{
            font-family: {t['mono']};
            font-weight: 600;
            font-size: 28px;
            letter-spacing: -0.5px;
        }}
        [data-testid="stMetricDelta"] {{ font-family: {t['mono']}; font-size: 12px; }}

        /* sidebar */
        [data-testid="stSidebar"] {{
            background: {t['panel']};
            border-right: 1px solid {t['border']};
        }}

        /* inputs / selects */
        [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
            background: {t['panel_2']};
            border: 1px solid {t['border']};
            border-radius: 8px;
            font-family: {t['mono']};
            color: {t['text']};
        }}

        /* buttons */
        .stButton > button {{
            background: {t['panel_2']};
            border: 1px solid {t['border']};
            border-radius: 8px;
            color: {t['text']};
            font-family: {t['mono']};
            transition: border-color .15s ease;
        }}
        .stButton > button:hover {{ border-color: {t['healthy']}; color: {t['healthy']}; }}

        /* tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {t['border']}; }}
        .stTabs [data-baseweb="tab"] {{ font-family: {t['mono']}; font-size: 13px; color: {t['muted']}; }}
        .stTabs [aria-selected="true"] {{ color: {t['text']}; }}

        /* code / monospace blocks */
        code, pre, .stCode {{ font-family: {t['mono']}; }}

        /* dividers */
        hr {{ border-color: {t['border']}; }}

        /* mono numerals utility class (for st.markdown readouts) */
        .ew-readout {{ font-family: {t['mono']}; font-weight: 600; letter-spacing: -0.5px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: "go.Figure") -> "go.Figure":
    """
    Purpose:      Restyle any Plotly figure to the mission-control dark theme so
                  every chart shares one visual language (bg, grid, font, colours).
    Input:        fig - a plotly.graph_objects.Figure (any traces already added).
    Output:       The same figure, mutated in place and also returned for chaining.
    Assumptions:  Trace colours that should encode health state are set by the
                  caller using STATE_COLORS; this only sets layout-level styling.
    Failure:      AttributeError if `fig` is not a Plotly Figure.
    """
    t = TOKENS
    fig.update_layout(
        paper_bgcolor=t["bg"],
        plot_bgcolor=t["panel"],
        font=dict(family="IBM Plex Sans, sans-serif", color=t["text"], size=12),
        margin=dict(l=48, r=20, t=36, b=36),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono, monospace", size=11, color=t["muted"]),
        ),
        colorway=[t["healthy"], t["accent"], t["degrading"], t["critical"]],
    )
    axis_kwargs = dict(
        gridcolor=t["border"],
        zerolinecolor=t["border"],
        linecolor=t["border"],
        tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=t["faint"]),
        title_font=dict(family="IBM Plex Sans, sans-serif", size=12, color=t["muted"]),
    )
    fig.update_xaxes(**axis_kwargs)
    fig.update_yaxes(**axis_kwargs)
    return fig


def risk_dial(
    risk_score: float,
    healthy_max: float = 0.46,
    critical_min: float = 0.66,
    title: str = "RISK",
) -> "go.Figure":
    """
    Purpose:      Cockpit-style risk gauge with EICAS green/amber/red zones and a
                  needle at the current risk score. The dashboard's signature element.
    Input:        risk_score   - float in [0, 1] (pipeline risk_score for latest cycle).
                  healthy_max  - upper bound of green zone (default = FD001 Healthy
                                 centroid risk 0.46).
                  critical_min - lower bound of red zone (default = FD001 Critical
                                 centroid risk 0.66; amber spans healthy_max..critical_min).
                  title        - small uppercase label under the number.
    Output:       plotly.graph_objects.Figure (go.Indicator gauge), already themed.
    Assumptions:  risk_score is on the same [0,1] scale as the cluster centroids.
                  Zone bounds default to FD001 centroids — pass dataset-specific
                  bounds for FD002-FD004 if you want exact per-dataset zones.
    Failure:      Renders with the needle clamped at 0 or 1 if risk_score is out of
                  range; does not raise.
    """
    t = TOKENS
    value = max(0.0, min(1.0, float(risk_score)))

    if value >= critical_min:
        num_color = t["critical"]
    elif value >= healthy_max:
        num_color = t["degrading"]
    else:
        num_color = t["healthy"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(
                font=dict(family="IBM Plex Mono, monospace", size=40, color=num_color),
                valueformat=".2f",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 1],
                    tickwidth=1,
                    tickcolor=t["faint"],
                    tickfont=dict(family="IBM Plex Mono, monospace", size=9, color=t["faint"]),
                ),
                bar=dict(color="rgba(0,0,0,0)"),  # hide default bar; needle = threshold
                bgcolor=t["panel"],
                borderwidth=0,
                steps=[
                    dict(range=[0, healthy_max], color=t["healthy"]),
                    dict(range=[healthy_max, critical_min], color=t["degrading"]),
                    dict(range=[critical_min, 1], color=t["critical"]),
                ],
                threshold=dict(
                    line=dict(color=t["text"], width=3),
                    thickness=0.85,
                    value=value,
                ),
            ),
            title=dict(
                text=title,
                font=dict(family="IBM Plex Mono, monospace", size=11, color=t["muted"]),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=t["bg"],
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def state_chip(state: str) -> str:
    """
    Purpose:      Render a colour-coded health-state pill (Healthy/Degrading/Critical)
                  using the EICAS palette. Returned as HTML for st.markdown.
    Input:        state - one of "Healthy", "Degrading", "Critical".
    Output:       HTML string. Pass to st.markdown(..., unsafe_allow_html=True).
    Assumptions:  state matches a key in STATE_COLORS.
    Failure:      Unknown state falls back to muted grey (no raise) so a bad label
                  never crashes the panel.
    """
    color = STATE_COLORS.get(state, TOKENS["muted"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;'
        f'padding:5px 12px;border-radius:6px;font-family:{TOKENS["mono"]};'
        f'font-size:12px;font-weight:500;letter-spacing:.4px;color:{color};'
        f'background:{color}1F;border:1px solid {color}59;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{color};"></span>'
        f'{state.upper()}</span>'
    )


def rgba(hex_color: str, alpha: float) -> str:
    """
    Purpose:  Convert a 6-digit hex colour to an rgba() string Plotly accepts.
              Plotly's fillcolor rejects 8-digit hex (#RRGGBBAA) — use this.
    Input:    hex_color like "#FF5A5F"; alpha in [0,1].
    Output:   "rgba(r,g,b,alpha)" string.
    Failure:  ValueError if hex_color is not a 6-digit hex.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
