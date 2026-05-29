# Excel Style Guide — Scot's Preferred Formatting

This document defines the visual style to use for **any Excel (.xlsx) file** created in this project. Reference this whenever generating a spreadsheet. The style is based on a blue/white palette with yellow input cells and italic grey notes.

---

## Color palette (exact hex values)

| Element | Hex | RGB | Usage |
|---|---|---|---|
| **Master header band** | `#1D3B93` | 29, 59, 147 | Top-level title band (e.g., "MODEL INPUTS") — dark navy |
| **Master header text** | `#FFFFFF` | 255, 255, 255 | White text on master header |
| **Section header band** | `#1D3B93` | 29, 59, 147 | Section dividers (e.g., "TIER CONTRACT", "GLOBAL ASSUMPTIONS") — same dark navy |
| **Section header text** | `#FFFFFF` or `#93B6FA` | — | White or light-blue text on section header bands |
| **Column header row fill** | `#93B6FA` | 147, 182, 250 | Sub-header rows with field labels (Parameter / Value / Units / Notes) |
| **Column header text** | `#1D3B93` | 29, 59, 147 | Dark navy text on light-blue column header |
| **Input cell fill (yellow)** | `#FFFECE` | 255, 254, 206 | Editable user-input cells — pale yellow |
| **Input cell text** | `#000000` | 0, 0, 0 | Black, bold-not-required |
| **Data row fill** | `#FFFFFF` | 255, 255, 255 | Standard white for read-only rows |
| **Data row text** | `#000000` | 0, 0, 0 | Black |
| **Notes column text (italic)** | `#000000` | 0, 0, 0 | Black italic for explanatory notes |
| **Gridlines / cell borders** | `#D8E6FF` | 216, 230, 255 | Very light blue — subtle separation |

---

## Layout conventions

**Worksheet margins:**

- **Row 1** is always left blank. Content starts on row 2.
- **Column A** is always left blank. Content starts in column B.

**Row hierarchy (top to bottom):**

1. **Master header band** — full-width row with the sheet/section title in white bold text on `#1D3B93` navy. Font size ~12–14pt.
2. **Section header band** — full-width row introducing a sub-group, same navy fill, may include an inline instruction like "Edit yellow cells" in lighter blue italics.
3. **Column header row** — light-blue fill `#93B6FA`, navy `#1D3B93` bold text, labels like: Parameter | Value | Units | Notes.
4. **Data rows** — white background, black text. Yellow `#FFFECE` fill on cells the user is meant to edit.

**Standard columns (for input/assumption sheets):**

| Parameter | Value | Units | Notes |
|---|---|---|---|
| Label of the input | The editable number (yellow cell) | The unit string | Italic explanatory text |

**Styling rules:**

- **Yellow cells = user inputs.** Any cell a user should edit gets the pale yellow `#FFFECE` fill. Nothing else.
- **Notes column is italic**, left-aligned, in plain black.
- **Numeric formatting:** use appropriate formats — currency `$#,##0`, percentages `0%` or `0.00%`, decimals matched to precision needed.
- **Gridlines:** thin `#D8E6FF` borders around every cell in the data area. Turn off the default Excel grid display if possible.
- **Font:** Arial or Calibri, 10–11pt for data rows, 11–12pt bold for headers, 12–14pt bold for master header.
- **Column widths:** auto-fit to content but maintain readability — Parameter column wider than Value/Units.
- **Row heights:** slightly taller than default (~18–20pt) for legibility.

---

## Implementation notes (openpyxl)

```python
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

# Fills
NAVY_FILL = PatternFill(start_color="1D3B93", end_color="1D3B93", fill_type="solid")
LIGHT_BLUE_FILL = PatternFill(start_color="93B6FA", end_color="93B6FA", fill_type="solid")
YELLOW_FILL = PatternFill(start_color="FFFECE", end_color="FFFECE", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

# Fonts
MASTER_HEADER_FONT = Font(name="Arial", size=12, bold=True, color="FFFFFF")
SECTION_HEADER_FONT = Font(name="Arial", size=11, bold=True, color="FFFFFF")
COLUMN_HEADER_FONT = Font(name="Arial", size=11, bold=True, color="1D3B93")
DATA_FONT = Font(name="Arial", size=10, color="000000")
NOTES_FONT = Font(name="Arial", size=10, italic=True, color="000000")

# Borders
thin_blue = Side(border_style="thin", color="D8E6FF")
CELL_BORDER = Border(top=thin_blue, bottom=thin_blue, left=thin_blue, right=thin_blue)
```

---

## Key principles

- Yellow = editable. Everything else = display/computed.
- Navy headers with white text establish strong section boundaries.
- Light blue column headers provide a softer secondary tier.
- Italic notes always appear to the right of values — never inline with data.
- Keep it clean: no extra colors, no gradients, no conditional formatting unless specifically requested.

---

*Reference this style guide anytime Scot asks for an Excel file to be created in this project.*
