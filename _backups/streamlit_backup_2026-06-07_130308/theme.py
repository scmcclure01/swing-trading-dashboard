"""
Theme — single source of truth for the Classic Blue palette.

Every color used across app.py and pages/*.py lives here. Change a color once
and it propagates everywhere. Use THEME["text"] etc. in new code; the bare
module-level constants exist for terse f-string interpolation.
"""

# ── Core palette ─────────────────────────────────────────────────────────────
TEXT        = "#103766"   # primary dark text
MUTED       = "#5A7BAA"   # secondary / label text
GREEN       = "#27500A"   # positive / leading
RED         = "#CC1111"   # negative / override / critical
ORANGE      = "#E07800"   # caution / mixed
BLUE        = "#288CFA"   # info / improving / accent
SURFACE     = "#FFFFFF"   # card background
SUBTLE_BG   = "#EEF3FA"   # tile / header background

# ── Gate-bar tints (background, border, dot, text) per permission state ───────
GATE = {
    "Green":  ("#D6F0D6", "rgba(29,122,42,0.30)", "#1D7A2A", "#173404"),
    "Yellow": ("#FFF3D6", "rgba(224,120,0,0.30)", "#E07800", "#412402"),
    "Red":    ("#FFE4E4", "rgba(204,17,17,0.30)", "#CC1111", "#501313"),
}

# ── Common borders ───────────────────────────────────────────────────────────
BORDER_FAINT  = "rgba(16,55,102,0.09)"
BORDER_LIGHT  = "rgba(16,55,102,0.12)"
BORDER_MEDIUM = "rgba(16,55,102,0.15)"

# ── Convenience dict for new code ────────────────────────────────────────────
THEME = {
    "text": TEXT, "muted": MUTED, "green": GREEN, "red": RED,
    "orange": ORANGE, "blue": BLUE, "surface": SURFACE, "subtle_bg": SUBTLE_BG,
    "border_faint": BORDER_FAINT, "border_light": BORDER_LIGHT,
    "border_medium": BORDER_MEDIUM,
}


def pnl_color(v: float) -> str:
    """Green for gains, red for losses, muted for flat."""
    if v > 0:
        return GREEN
    if v < 0:
        return RED
    return MUTED
