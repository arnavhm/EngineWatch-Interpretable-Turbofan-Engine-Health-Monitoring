"""
Shared visual theme for the CMAPSS Engine Health Monitor dashboard.
Import these constants in every component to ensure visual consistency.
"""

# ── Health state colours (canonical) ────────────────────────────────
STATE_COLORS = {
    "Healthy":   "#2ECC71",
    "Degrading":  "#F39C12",
    "Critical":   "#E74C3C",
}

# ── Section title helper ────────────────────────────────────────────
SECTION_TITLE_CSS = (
    "font-size: 1.15rem; font-weight: 700; margin-bottom: 0.25rem;"
)

# ── Styled section divider (subtle rule with breathing room) ────────
SECTION_DIVIDER = (
    '<hr style="border: none; border-top: 1px solid rgba(150,150,150,0.25); '
    'margin: 1.6rem 0;">'
)
