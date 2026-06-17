"""
Shared UI components — Classic Blue styled HTML helpers.

Imported by app.py and pages/*.py so the visual language lives in one place.
These functions only BUILD html strings; they never call st.* (so importing this
module is side-effect free and safe before st.set_page_config).
"""
import pandas as pd

from theme import (
    TEXT, MUTED, GREEN, RED, ORANGE, BLUE, SURFACE, SUBTLE_BG,
    BORDER_FAINT, BORDER_LIGHT, BORDER_MEDIUM,
)

# ── Color presets for cb_table ───────────────────────────────────────────────
# Each preset maps a (color, font-weight) to the substrings that should trigger
# it. Different tabs flag different keywords, so each gets its own preset rather
# than forcing one shared keyword set.
_WEIGHT_BOLD = "500"
_WEIGHT_REG  = "400"

CB_PRESET_MACRO = {
    (GREEN, _WEIGHT_BOLD):  ["✅", "Leading", "Positive", "Clear", "Open", "Rising",
                             "OK", "🟢", "Phase 2", "Confirmed", "GREEN"],
    (RED, _WEIGHT_BOLD):    ["❌", "Lagging", "Negative", "FLAG", "Closed",
                             "Declining", "🔴", "OVERRIDE", "ACTIVE", "Critical"],
    (ORANGE, _WEIGHT_BOLD): ["⚠️", "Mixed", "Weakening", "Elevated", "pressure", "🟡"],
    (BLUE, _WEIGHT_BOLD):   ["🔵", "Improving", "Phase 1", "Early"],
}

CB_PRESET_PORTFOLIO = {
    (GREEN, _WEIGHT_BOLD):  ["✅", "Target", "Core", "FULL", "🟢", "GREEN", "Open", "Rising"],
    (RED, _WEIGHT_BOLD):    ["❌", "Stop", "🔴", "RED", "loss"],
    (ORANGE, _WEIGHT_BOLD): ["⚠️", "Rule-based", "Mixed", "🟡", "HALF"],
    (BLUE, _WEIGHT_BOLD):   ["🔵", "Tactical"],
}

_DEFAULT_CELL = (TEXT, _WEIGHT_REG)


def cb_table(
    df: pd.DataFrame,
    max_height: int | None = None,
    bordered: bool = True,
    preset: dict | None = None,
    font_size: int = 14,
) -> str:
    """Render a DataFrame as a Classic Blue styled HTML table.

    bordered=False omits the outer container — use when the table sits inside a card().
    preset selects which keyword→color map to use (defaults to the macro preset).
    """
    color_map = preset if preset is not None else CB_PRESET_MACRO

    def _color(val) -> tuple[str, str]:
        s = str(val)
        for style, keywords in color_map.items():
            if any(k in s for k in keywords):
                return style
        return _DEFAULT_CELL

    th = (f"padding: 7px 12px; font-size: 11px; font-weight: 500; color: {MUTED};"
          " text-align: left; white-space: nowrap;")
    td_base = (f"padding: 8px 12px; font-size: {font_size}px;"
               f" border-top: 0.5px solid {BORDER_FAINT};")

    cols   = list(df.columns)
    header = "".join(f'<th style="{th}">{c}</th>' for c in cols)

    rows_html = ""
    for _, row in df.iterrows():
        cells = ""
        for col in cols:
            val = row[col]
            color, weight = _color(val)
            cells += (f'<td style="{td_base} color: {color}; font-weight: {weight};">'
                      f'{val}</td>')
        rows_html += f"<tr>{cells}</tr>"

    inner = (
        f'<table style="width: 100%; border-collapse: collapse; background: {SURFACE};">'
        f'<thead><tr style="background: {SUBTLE_BG};">{header}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )
    if not bordered:
        return inner
    scroll = f"max-height: {max_height}px; overflow-y: auto;" if max_height else ""
    return (
        f'<div style="border-radius: 9px; overflow: hidden;'
        f' border: 1px solid {BORDER_LIGHT}; {scroll}">'
        f'{inner}</div><div style="margin-bottom:8px"></div>'
    )


def card(heading: str, inner_html: str, pill: str = "") -> str:
    """White card with uppercase heading, optional pill label, and arbitrary inner HTML."""
    pill_html = (
        f'<span style="background:{SUBTLE_BG}; color:{MUTED}; font-size:10px; font-weight:500;'
        f' padding:2px 8px; border-radius:4px; border:0.5px solid {BORDER_MEDIUM};">{pill}</span>'
        if pill else ""
    )
    return (
        f'<div style="background:{SURFACE}; border-radius:12px; border:0.5px solid {BORDER_LIGHT};'
        f' padding:15px 17px; margin-bottom:10px; overflow:hidden;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between;'
        f' margin-bottom:10px; padding-bottom:7px; border-bottom:0.5px solid {BORDER_FAINT};">'
        f'<span style="font-size:11px; font-weight:500; color:{MUTED}; text-transform:uppercase;'
        f' letter-spacing:0.04em;">{heading}</span>{pill_html}</div>'
        f'{inner_html}</div>'
    )


def gate_bar_html(perm: str, text: str) -> str:
    """Render the permission state gate bar."""
    from theme import GATE
    bg, border, dot_c, text_c = GATE.get(perm, GATE["Green"])
    return (
        f'<div style="background:{bg}; border-radius:9px; border:0.5px solid {border};'
        f' padding:10px 16px; display:flex; align-items:center; gap:10px; margin-bottom:10px;">'
        f'<div style="width:9px; height:9px; border-radius:50%; background:{dot_c}; flex-shrink:0;"></div>'
        f'<span style="font-size:13px; font-weight:500; color:{text_c};">{text}</span>'
        f'</div>'
    )


def tile(label: str, value: str, signal: str = "", signal_color: str = MUTED) -> str:
    """Single metric tile for the header row."""
    sig_html = (
        f'<div style="font-size:11px; font-weight:500; color:{signal_color}; margin-top:3px;">{signal}</div>'
        if signal else ""
    )
    return (
        f'<div style="background:{SUBTLE_BG}; border-radius:9px; border:0.5px solid {BORDER_MEDIUM};'
        f' padding:10px 12px;">'
        f'<div style="font-size:11px; font-weight:400; color:{MUTED}; margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:17px; font-weight:500; color:{TEXT};">{value}</div>'
        f'{sig_html}</div>'
    )
