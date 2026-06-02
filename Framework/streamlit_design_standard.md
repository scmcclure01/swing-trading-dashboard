# Streamlit Design Standard — Swing Trading Framework

Classic Blue light theme is the active theme as of May 2026. All new development targets this theme. The dark theme spec is preserved below for reference only.

---

## Color system

### Classic Blue — active theme

| Element | Hex | Usage |
|---|---|---|
| Page background | `#EEF3FA` | Outermost surface, tile panel fill |
| Sidebar | `#D6E8FA` | Sidebar background |
| Card (outer) | `#FFFFFF` | White card containers wrapping each section |
| Accent | `#288CFA` | Tab underline, active elements, Improving signal |
| Primary text | `#103766` | Headings, metric values, table body text |
| Secondary text | `#5A7BAA` | Card section headings, table column headers, labels, captions |
| Row divider | `rgba(16,55,102,0.09)` | Between table rows |
| Card border | `rgba(16,55,102,0.12)` | Card outer border, table outer border |
| Tile border | `rgba(16,55,102,0.15)` | Metric tile border |

**config.toml**
```toml
[theme]
base = "light"
backgroundColor = "#EEF3FA"
secondaryBackgroundColor = "#FFFFFF"
primaryColor = "#288CFA"
textColor = "#103766"
font = "sans serif"
```

### Dark theme — reference only (retired May 2026)

| Element | Hex |
|---|---|
| Page background | `#060C1C` |
| Card | `#1D3B93` |
| Panel | `#D8E6FF` |
| Accent | `#93B6FA` |
| Primary text (on card) | `#FFFFFF` |
| Secondary text (on card) | `#93B6FA` |

---

## Signal / conditional text colors

Applied to both themes. Used in table cells, tile signal lines, and gate bars.

| Signal | Hex | Usage |
|---|---|---|
| Positive / bullish | `#27500A` | Green — Leading, Positive, Clear, Open |
| Positive (lighter) | `#1D7A2A` | Gate bar dot (green state) |
| Neutral / mixed | `#E07800` | Orange — Mixed, Half signal, Weakening |
| Negative / bearish | `#CC1111` | Red — Lagging, Negative, Override Active |
| Improving / accent | `#288CFA` | Blue — Improving quadrant, GIL accent |

Signal text auto-detection in `cb_table()` matches these keywords:

- **Green** (`#27500A`): ✅ Leading Positive Clear Open Rising OK 🟢 Confirmed GREEN
- **Red** (`#CC1111`): ❌ Lagging Negative FLAG Closed Declining 🔴 OVERRIDE ACTIVE Critical
- **Orange** (`#E07800`): ⚠️ Mixed Weakening Elevated pressure 🟡
- **Blue** (`#288CFA`): 🔵 Improving Phase 1 Early

---

## Borders

| Context | Rule |
|---|---|
| Card outer border | `0.5px solid rgba(16,55,102,0.12)` |
| Card section heading divider | `0.5px solid rgba(16,55,102,0.09)` |
| Metric tile border | `0.5px solid rgba(16,55,102,0.15)` |
| Table row divider | `0.5px solid rgba(16,55,102,0.09)` (top of each row) |
| Table outer (standalone) | `1px solid rgba(16,55,102,0.12)` |
| Gate bar border (green) | `0.5px solid rgba(29,122,42,0.30)` |
| Gate bar border (yellow) | `0.5px solid rgba(224,120,0,0.30)` |
| Gate bar border (red) | `0.5px solid rgba(204,17,17,0.30)` |
| Sidebar border-right | `1px solid rgba(16,55,102,0.15)` |

---

## Typography

| Role | Size | Weight | Color |
|---|---|---|---|
| Page title | 28px | 500 | `#103766` |
| Page date | 12px | 400 | `#5A7BAA` |
| Card section heading | 11px | 500 | `#5A7BAA` (uppercase, 0.04em tracking) |
| Card pill label | 10px | 500 | `#5A7BAA` on `#EEF3FA` |
| Tile label | 11px | 400 | `#5A7BAA` |
| Tile value | 17px | 500 | `#103766` |
| Tile signal line | 11px | 500 | Signal color |
| Table header | 11px | 500 | `#5A7BAA` |
| Table body text | 13px | 400 | `#103766` |
| Table signal text | 13px | 500 | Signal color |
| Gate bar text | 13px | 500 | State text color (see gate bar spec) |

Two weights only: **400** (regular) and **500** (medium). Never 600/700.

---

## Border radius

| Element | Radius |
|---|---|
| Card (outer container) | 12px |
| Metric tile, gate bar, standalone table | 9px |
| Pill label | 4px |

---

## Component patterns

### `_tile(label, value, signal, signal_color)` — metric tile

```
background: #EEF3FA
border-radius: 9px
border: 0.5px solid rgba(16,55,102,0.15)
padding: 10px 12px

label:  font-size 11px, weight 400, color #5A7BAA, margin-bottom 2px
value:  font-size 17px, weight 500, color #103766
signal: font-size 11px, weight 500, color <signal_color>, margin-top 3px
```

Tile rows sit inside a white outer card (`#FFFFFF`, 12px radius, 15px 17px padding).
Grid: `grid-template-columns: repeat(N, 1fr); gap: 9px`

---

### `_gate_bar_html(perm, text)` — permission / alert bar

| State | Background | Border | Dot color | Text color |
|---|---|---|---|---|
| Green | `#D6F0D6` | `rgba(29,122,42,0.30)` | `#1D7A2A` | `#173404` |
| Yellow | `#FFF3D6` | `rgba(224,120,0,0.30)` | `#E07800` | `#412402` |
| Red | `#FFE4E4` | `rgba(204,17,17,0.30)` | `#CC1111` | `#501313` |

```
border-radius: 9px
padding: 10px 16px
display: flex; align-items: center; gap: 10px
dot: 9px circle, flex-shrink 0
text: font-size 13px, weight 500
```

Replaces `st.success / st.warning / st.error` throughout the app.

---

### `_card(heading, inner_html, pill)` — section container

```
background: #FFFFFF
border-radius: 12px
border: 0.5px solid rgba(16,55,102,0.12)
padding: 15px 17px
margin-bottom: 10px

heading row:
  font-size 11px, weight 500, color #5A7BAA
  text-transform uppercase, letter-spacing 0.04em
  border-bottom: 0.5px solid rgba(16,55,102,0.09)
  padding-bottom 7px, margin-bottom 10px

pill (optional):
  background #EEF3FA, color #5A7BAA
  font-size 10px, weight 500
  padding 2px 8px, border-radius 4px
  border: 0.5px solid rgba(16,55,102,0.15)
```

Tables rendered inside a card use `cb_table(df, bordered=False)` — no outer border, just the inner table markup.

---

### `cb_table(df, bordered, max_height)` — data table

```
Table header row:
  background: #EEF3FA
  th: font-size 11px, weight 500, color #5A7BAA, padding 7px 12px

Table body rows:
  background: #FFFFFF
  td: font-size 13px, padding 8px 12px
  border-top: 0.5px solid rgba(16,55,102,0.09)
  color/weight: auto-detected from signal keywords (see Signal section)

bordered=True (standalone):
  outer div: border-radius 9px, overflow hidden
  border: 1px solid rgba(16,55,102,0.12)

bordered=False (inside a _card):
  no outer wrapper — table fills the card directly
```

---

## Layout patterns

### Two-column tab layout (L0/L2)

```
display: grid
grid-template-columns: 1fr 1fr
gap: 12px
```

Each column contains stacked `_card()` components.

### Stats tile row (L4 Screener)

White outer card wrapping a tile grid:
```
background: #FFFFFF
border-radius: 12px
border: 0.5px solid rgba(16,55,102,0.12)
padding: 15px 17px

inner grid: repeat(4, 1fr), gap 9px
subtitle line above tiles: font-size 11px, color #5A7BAA
```

---

## Plotly chart theme

```python
paper_bgcolor = "#EEF3FA"
plot_bgcolor  = "#FFFFFF"
font color    = "#103766"
gridcolor     = "rgba(16,55,102,0.10)"
```

---

## What NOT to use

The following Streamlit native components are replaced by custom HTML in this app:

| Retired | Replaced by |
|---|---|
| `st.metric` | `_tile()` inside a white card |
| `st.success / st.warning / st.error` | `_gate_bar_html()` |
| `st.expander` | `_card()` |
| `st.dataframe` (in L0/L2/L3/L4) | `cb_table()` inside `_card()` |
| `st.subheader` (section headers) | Card heading inside `_card()` |
| `st.divider` | Not used — card spacing handles separation |
