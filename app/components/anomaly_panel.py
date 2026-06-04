"""
app/components/anomaly_panel.py

Purpose:      Fleet anomaly detection scatter panel.
              Calls detect_anomalous_engines() on the full test fleet and renders
              a HI vs HI_velocity scatter. Anomalous engines are overlaid as red
              diamonds. Selected engine is ringed in white.

Usage:
    from app.components.anomaly_panel import render_anomaly_panel
    render_anomaly_panel(df, selected_engine_id=selected_engine_id)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from evaluation.validation import detect_anomalous_engines
from app.theme import STATE_COLORS, SECTION_TITLE_CSS


@st.cache_data
def _run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:      Cached wrapper around detect_anomalous_engines.
    Input:        Full test DataFrame — unit, cycle, health_index, HI_velocity,
                  HI_variability required. Mahalanobis uses first three.
    Output:       DataFrame — one row per engine, columns from detect_anomalous_engines.
    Failure:      ValueError propagated if < 10 engines.
    """
    return detect_anomalous_engines(df)


def render_anomaly_panel(
    df: pd.DataFrame,
    selected_engine_id: int | None = None,
) -> None:
    """
    Purpose:      Render Mahalanobis anomaly scatter for the full engine fleet.
    Input:        df — full test pipeline DataFrame (all engines, all cycles).
                  selected_engine_id — optional, rings selected engine in white.
    Output:       None — renders into Streamlit context.
    Failure:      ValueError (< 10 engines) renders st.info and exits cleanly.
                  All other exceptions render st.error and exit cleanly.
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">🔍 Fleet Anomaly Detection</p>',
        unsafe_allow_html=True,
    )

    try:
        anomaly_df = _run_anomaly_detection(df)
    except ValueError as exc:
        st.info(f"Anomaly detection unavailable: {exc}")
        return
    except Exception as exc:
        st.error(f"Anomaly detection failed unexpectedly: {exc}")
        return

    n_engines = len(anomaly_df)
    n_anomalous = int(anomaly_df["is_anomaly"].sum())
    threshold = float(anomaly_df["chi_square_threshold"].iloc[0])
    p_value = float(anomaly_df["chi_square_p_value"].iloc[0])

    # ── Summary metrics row ───────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Engines Monitored", n_engines)
    c2.metric(
        "Anomalous Engines",
        n_anomalous,
        delta="Requires caution" if n_anomalous > 0 else None,
        delta_color="inverse",
    )
    c3.metric("χ² Threshold", f"{threshold:.2f}", f"p = {p_value:.2f}")

    # ── Merge last-cycle snapshot with anomaly results ────────────────
    df_last = df.sort_values("cycle").groupby("unit").last().reset_index()
    df_last = df_last.merge(
        anomaly_df[["unit", "mahalanobis_distance", "is_anomaly"]],
        on="unit",
        how="left",
    )

    normal_df = df_last[~df_last["is_anomaly"]]
    anomaly_plot_df = df_last[df_last["is_anomaly"]]

    # ── Scatter plot ──────────────────────────────────────────────────
    fig = go.Figure()

    # Normal engines — one trace per risk state for consistent legend colours
    for state in ["Healthy", "Degrading", "Critical"]:
        subset = normal_df[normal_df["risk_state"] == state]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["health_index"],
                y=subset["HI_velocity"],
                mode="markers",
                name=state,
                marker=dict(
                    color=STATE_COLORS.get(state, "#888"),
                    size=8,
                    opacity=0.75,
                ),
                customdata=subset[["unit", "mahalanobis_distance"]].values,
                hovertemplate=(
                    "Engine %{customdata[0]}<br>"
                    "HI: %{x:.3f}<br>"
                    "Velocity: %{y:.5f}<br>"
                    "Mahalanobis: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    # Anomalous engines — red diamonds, labelled with engine ID
    if not anomaly_plot_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_plot_df["health_index"],
                y=anomaly_plot_df["HI_velocity"],
                mode="markers+text",
                name="Anomalous",
                marker=dict(
                    color="#FF4B4B",
                    size=14,
                    symbol="diamond",
                    line=dict(color="white", width=1.5),
                ),
                text=anomaly_plot_df["unit"].astype(str),
                textposition="top center",
                textfont=dict(size=10, color="#FF4B4B"),
                customdata=anomaly_plot_df[["unit", "mahalanobis_distance"]].values,
                hovertemplate=(
                    "⚠️ Engine %{customdata[0]} — ANOMALOUS<br>"
                    "HI: %{x:.3f}<br>"
                    "Velocity: %{y:.5f}<br>"
                    "Mahalanobis: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    # Selected engine — white ring, renders on top
    if selected_engine_id is not None:
        sel = df_last[df_last["unit"] == selected_engine_id]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel["health_index"],
                    y=sel["HI_velocity"],
                    mode="markers",
                    name=f"Engine {selected_engine_id} (selected)",
                    marker=dict(
                        color="rgba(0,0,0,0)",
                        size=18,
                        line=dict(color="white", width=2.5),
                    ),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis_title="Health Index (last cycle)",
        yaxis_title="HI Velocity (last cycle)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=420,
        margin=dict(t=40, b=40, l=40, r=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Flagged engine details ────────────────────────────────────────
    if n_anomalous > 0:
        with st.expander(f"⚠️ {n_anomalous} flagged engine(s) — details"):
            flagged = (
                anomaly_df[anomaly_df["is_anomaly"]][
                    [
                        "unit",
                        "mahalanobis_distance",
                        "mahalanobis_distance_sq",
                        "chi_square_threshold",
                        "anomaly_reason",
                    ]
                ]
                .copy()
                .rename(
                    columns={
                        "unit": "Engine ID",
                        "mahalanobis_distance": "Distance",
                        "mahalanobis_distance_sq": "Distance²",
                        "chi_square_threshold": "χ² Threshold",
                        "anomaly_reason": "Reason",
                    }
                )
            )
            st.dataframe(flagged, hide_index=True, use_container_width=True)
            st.caption(
                "RUL predictions for flagged engines should be treated with caution — "
                "their degradation pattern was not seen during training."
            )
    else:
        st.success("✅ All engines within normal fleet degradation bounds.")
