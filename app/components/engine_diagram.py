import re
import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

from app.theme import SECTION_TITLE_CSS
from features.health_index import (
    compute_sensor_contributions,
    aggregate_module_contributions,
)

# Hardcoded display labels from Saxena 2008 Table 2
SENSOR_LABELS = {
    "s1": "T2", "s2": "T24", "s3": "T30", "s4": "T50", "s5": "P2",
    "s6": "P15", "s7": "P30", "s8": "Nf", "s9": "Nc", "s10": "epr",
    "s11": "Ps30", "s12": "phi", "s13": "NRf", "s14": "NRc", "s15": "BPR",
    "s16": "farB", "s17": "htBleed", "s18": "Nf_dmd", "s19": "PCNfR_dmd",
    "s20": "W31", "s21": "W32"
}

def render_engine_diagram(engine_df: pd.DataFrame, artifacts: dict, config: dict, dataset_id: str) -> None:
    """
    Renders the Engine Health Map diagram.
    """
    st.markdown(
        f'<p style="{SECTION_TITLE_CSS}">Engine Health Map</p>',
        unsafe_allow_html=True,
    )
    
    diagram_axis_cfg = config.get("diagram_axis", "hpc")
    if isinstance(diagram_axis_cfg, str):
        if diagram_axis_cfg in ["dual", "both", "all"]:
            axes_to_render = list(config.get("health_index", {}).get("axes", {}).keys())
        else:
            axes_to_render = [diagram_axis_cfg]
    elif isinstance(diagram_axis_cfg, list):
        axes_to_render = diagram_axis_cfg
    else:
        axes_to_render = ["hpc"]

    sensor_contributions = {}
    
    for axis in axes_to_render:
        try:
            pca = artifacts["hi_pca_by_axis"][axis]
        except KeyError:
            st.warning(f"Missing PCA artifact for axis '{axis}'. Skipping.")
            continue
            
        axis_cfg = config.get("health_index", {}).get("axes", {}).get(axis, {})
        axis_sensors = axis_cfg.get("by_dataset", {}).get(dataset_id, axis_cfg.get("sensors", []))
        axis_sensors = [s for s in axis_sensors if s in engine_df.columns]
        
        if not axis_sensors:
            continue
            
        df_contrib = compute_sensor_contributions(engine_df.iloc[[-1]], pca, axis_sensors)
        last_row = df_contrib.iloc[-1]
        
        for s in axis_sensors:
            col_name = f"{s}_contribution"
            if col_name in last_row:
                s_num = s.split("_")[-1]
                s_key = f"s{s_num}"
                sensor_contributions[s_key] = sensor_contributions.get(s_key, 0.0) + float(last_row[col_name])

    if not sensor_contributions:
        st.error("No valid sensor contributions could be calculated for the diagram.")
        return

    try:
        module_heat = aggregate_module_contributions(sensor_contributions, config)
    except Exception as e:
        st.error(f"Failed to aggregate module contributions: {e}")
        return

    # Load SVG asset
    svg_path = Path(__file__).resolve().parent.parent / "assets" / "engine_diagram.svg"
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
    except FileNotFoundError:
        st.error("Engine diagram SVG not found at app/assets/engine_diagram.svg.")
        return

    # Map directions to CSS classes
    direction_classes = {
        "healthy": "nominal",
        "critical": "crit",  # default critical, overrides based on magnitude later
        "inactive": "inactive"
    }

    # Rank modules by magnitude
    ranked_modules = sorted(
        [m for m, data in module_heat.items() if data["is_active"]],
        key=lambda m: module_heat[m]["magnitude"],
        reverse=True
    )
    
    # Process each module and update the SVG
    for module_key, data in module_heat.items():
        base_match = f'<g id="module-{module_key}" data-module="{module_key}" class="module">'
        if base_match not in svg_content:
            continue
            
        direction = data["direction"]
        norm_mag = data["norm_magnitude"]
        
        # Determine CSS class
        css_class = "inactive"
        opacity = 1.0
        
        if direction == "healthy":
            css_class = "nominal"
            opacity = 0.25 + 0.75 * norm_mag
        elif direction == "critical":
            if norm_mag >= 0.6:
                css_class = "crit"
            else:
                css_class = "caution"
            opacity = 0.25 + 0.75 * norm_mag
            
        # Build tooltip title
        tooltip_lines = [module_key.upper()]
        
        if not data["is_active"]:
            tooltip_lines.append("No active sensors")
        else:
            for s_id, contrib in data["active_sensors"].items():
                label = SENSOR_LABELS.get(s_id, "Unknown")
                sign = "+" if contrib > 0 else ""
                tooltip_lines.append(f"{s_id} · {label} · {sign}{contrib:.2f}")
                
            # Footer
            if norm_mag == 1.0:
                rank_str = "dominant driver"
            else:
                try:
                    rank_idx = ranked_modules.index(module_key) + 1
                    rank_str = f"rank {rank_idx}"
                except ValueError:
                    rank_str = ""
            
            footer = f"{direction} ({rank_str})"
            tooltip_lines.append("")
            tooltip_lines.append(footer)
            
        tooltip_text = "&#10;".join(tooltip_lines)
        
        # Replace the <g> tag with the styled version + <title>
        new_g_tag = (
            f'<g id="module-{module_key}" data-module="{module_key}" '
            f'class="module {css_class}" style="opacity:{opacity:.3f}">\n'
            f'<title>{tooltip_text}</title>'
        )
        
        svg_content = svg_content.replace(base_match, new_g_tag)

    # Render via Streamlit Components iframe
    components.html(svg_content, height=400)
    
    st.caption("relative PC1 attribution; absolute state on the risk gauge.")
