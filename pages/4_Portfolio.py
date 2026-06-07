"""
Portfolio Tracker — Swing Trading Framework
Open positions (live prices via yfinance) + closed positions + performance summary.
Data stored in portfolio.json at the framework root.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
import sys
from datetime import datetime, date, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_PATH = os.path.join(_HERE, "portfolio.json")

# Ensure the framework root (parent of pages/) is importable for shared modules.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE HELPERS  (imported from ui_components / theme — single source)
# ─────────────────────────────────────────────────────────────────────────────

from ui_components import (
    card as _card,
    tile as _tile,
    cb_table as _cb_table_base,
    CB_PRESET_PORTFOLIO,
)
from theme import pnl_color as _pnl_color


def cb_table(df: pd.DataFrame, max_height: int | None = None, bordered: bool = True) -> str:
    """Classic Blue HTML table using the portfolio color preset (see ui_components)."""
    return _cb_table_base(df, max_height=max_height, bordered=bordered,
                          preset=CB_PRESET_PORTFOLIO, font_size=13)


def pct_fmt(v: float, decimals: int = 1) -> str:
    return f"{v*100:+.{decimals}f}%"


def dollar_fmt(v: float) -> str:
    return f"${v:+,.0f}" if v != 0 else "$0"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_PATH):
        return {"account_size": 100000, "open_positions": [], "closed_positions": []}
    with open(PORTFOLIO_PATH, "r") as f:
        return json.load(f)


def save_portfolio(data: dict) -> None:
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_prices(tickers_key: str) -> dict:
    """Fetch last close for a comma-separated list of tickers. Cached 5 min."""
    tickers = [t.strip() for t in tickers_key.split(",") if t.strip()]
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        closes = raw["Close"] if len(tickers) > 1 else raw["Close"].rename(tickers[0])
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        return {t: float(closes[t].dropna().iloc[-1]) for t in tickers if t in closes.columns}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# OPEN POSITIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_open_table(positions: list, prices: dict, account_size: float) -> tuple:
    """Returns (display_df, total_market_value, total_cost_basis, total_unrealized_pnl, total_risk_dollars)."""
    rows = []
    total_mv    = 0.0
    total_cb    = 0.0
    total_upnl  = 0.0
    total_risk  = 0.0

    for p in positions:
        ticker     = p["ticker"]
        entry_px   = p["entry_price"]
        shares     = p["shares"]
        stop_px    = p["stop_price"]
        entry_date = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
        days_held  = (date.today() - entry_date).days
        layer      = p.get("layer", "Tactical")

        cur_px     = prices.get(ticker, p.get("current_price") or entry_px)
        cost_basis = entry_px * shares
        mkt_val    = cur_px * shares
        upnl_d     = mkt_val - cost_basis
        upnl_pct   = upnl_d / cost_basis if cost_basis else 0.0
        risk_d     = (entry_px - stop_px) * shares

        total_mv   += mkt_val
        total_cb   += cost_basis
        total_upnl += upnl_d
        total_risk += risk_d

        rows.append({
            "Ticker":     ticker,
            "Layer":      layer,
            "Entry Date": str(entry_date),
            "Entry":      f"${entry_px:.2f}",
            "Current":    f"${cur_px:.2f}",
            "Shares":     shares,
            "Cost Basis": f"${cost_basis:,.0f}",
            "Mkt Value":  f"${mkt_val:,.0f}",
            "P&L $":      dollar_fmt(upnl_d),
            "P&L %":      pct_fmt(upnl_pct),
            "Days":       days_held,
            "Stop":       f"${stop_px:.2f}",
            "Risk $":     f"${risk_d:,.0f}",
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Ticker","Layer","Entry Date","Entry","Current","Shares",
                 "Cost Basis","Mkt Value","P&L $","P&L %","Days","Stop","Risk $"])
    return df, total_mv, total_cb, total_upnl, total_risk


# ─────────────────────────────────────────────────────────────────────────────
# CLOSED POSITIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_closed_table(positions: list) -> pd.DataFrame:
    rows = []
    for p in positions:
        entry_px  = p["entry_price"]
        exit_px   = p["exit_price"]
        shares    = p["shares"]
        pnl_d     = (exit_px - entry_px) * shares
        pnl_pct   = pnl_d / (entry_px * shares) if entry_px else 0.0
        entry_dt  = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
        exit_dt   = datetime.strptime(p["exit_date"],  "%Y-%m-%d").date()
        hold_days = (exit_dt - entry_dt).days

        rows.append({
            "Ticker":     p["ticker"],
            "Layer":      p.get("layer", "Tactical"),
            "Entry Date": str(entry_dt),
            "Exit Date":  str(exit_dt),
            "Entry":      f"${entry_px:.2f}",
            "Exit":       f"${exit_px:.2f}",
            "Shares":     shares,
            "P&L $":      dollar_fmt(pnl_d),
            "P&L %":      pct_fmt(pnl_pct),
            "Hold Days":  hold_days,
            "Exit Reason":p.get("exit_reason", "—"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Ticker","Layer","Entry Date","Exit Date","Entry","Exit",
                 "Shares","P&L $","P&L %","Hold Days","Exit Reason"])


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE STATS
# ─────────────────────────────────────────────────────────────────────────────

def calc_performance(closed: list, start_date: date, end_date: date) -> dict:
    """Calculate performance stats for closed trades in [start_date, end_date]."""
    filtered = [
        p for p in closed
        if start_date <= datetime.strptime(p["exit_date"], "%Y-%m-%d").date() <= end_date
    ]
    if not filtered:
        return {"count": 0, "realized_pnl": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "trades": []}

    pnls = []
    for p in filtered:
        pnl = (p["exit_price"] - p["entry_price"]) * p["shares"]
        pct = (p["exit_price"] / p["entry_price"] - 1) if p["entry_price"] else 0
        pnls.append({"pnl": pnl, "pct": pct})

    wins   = [x for x in pnls if x["pnl"] > 0]
    losses = [x for x in pnls if x["pnl"] <= 0]

    realized_pnl  = sum(x["pnl"] for x in pnls)
    win_rate      = len(wins) / len(pnls) if pnls else 0
    avg_win       = sum(x["pct"] for x in wins) / len(wins) if wins else 0
    avg_loss      = sum(x["pct"] for x in losses) / len(losses) if losses else 0
    gross_wins    = sum(x["pnl"] for x in wins)
    gross_losses  = abs(sum(x["pnl"] for x in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "count":         len(filtered),
        "realized_pnl":  realized_pnl,
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "trades":         filtered,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FORMS
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar_add_position(data: dict) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<span style="font-size:11px; font-weight:500; color:#5A7BAA; text-transform:uppercase;'
        ' letter-spacing:0.04em;">Add Open Position</span>',
        unsafe_allow_html=True,
    )
    with st.sidebar.form("add_position_form", clear_on_submit=True):
        ticker     = st.text_input("Ticker").upper().strip()
        layer      = st.selectbox("Layer", ["Tactical", "Core"])
        entry_date = st.date_input("Entry Date", value=date.today())
        entry_px   = st.number_input("Entry Price", min_value=0.01, step=0.01, format="%.2f")
        shares     = st.number_input("Shares", min_value=1, step=1)
        stop_px    = st.number_input("Stop Price", min_value=0.01, step=0.01, format="%.2f")
        notes      = st.text_input("Notes (optional)")
        submitted  = st.form_submit_button("Add Position")
        if submitted and ticker and entry_px > 0 and shares > 0 and stop_px > 0:
            data["open_positions"].append({
                "ticker":      ticker,
                "layer":       layer,
                "entry_date":  str(entry_date),
                "entry_price": entry_px,
                "shares":      int(shares),
                "stop_price":  stop_px,
                "current_price": None,
                "notes":       notes,
            })
            save_portfolio(data)
            st.cache_data.clear()
            st.rerun()


def _sidebar_close_position(data: dict) -> None:
    open_tickers = [p["ticker"] for p in data["open_positions"]]
    if not open_tickers:
        return
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<span style="font-size:11px; font-weight:500; color:#5A7BAA; text-transform:uppercase;'
        ' letter-spacing:0.04em;">Close Position</span>',
        unsafe_allow_html=True,
    )
    with st.sidebar.form("close_position_form", clear_on_submit=True):
        ticker      = st.selectbox("Ticker", open_tickers)
        exit_date   = st.date_input("Exit Date", value=date.today())
        exit_px     = st.number_input("Exit Price", min_value=0.01, step=0.01, format="%.2f")
        exit_reason = st.selectbox("Exit Reason", ["Target", "Stop", "Rule-based"])
        notes       = st.text_input("Notes (optional)")
        submitted   = st.form_submit_button("Close Position")
        if submitted and ticker and exit_px > 0:
            # Move from open to closed
            pos = next((p for p in data["open_positions"] if p["ticker"] == ticker), None)
            if pos:
                closed_entry = {
                    "ticker":      pos["ticker"],
                    "layer":       pos.get("layer", "Tactical"),
                    "entry_date":  pos["entry_date"],
                    "exit_date":   str(exit_date),
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_px,
                    "shares":      pos["shares"],
                    "exit_reason": exit_reason,
                    "notes":       notes,
                }
                data["closed_positions"].insert(0, closed_entry)
                data["open_positions"] = [p for p in data["open_positions"] if p["ticker"] != ticker]
                save_portfolio(data)
                st.cache_data.clear()
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
<style>
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .main > div { background-color: #EEF3FA !important; }
.block-container {
    padding-top: 2rem !important;
    background-color: #EEF3FA !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #D6E8FA !important;
    border-right: 1px solid rgba(16,55,102,0.15) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #5A7BAA !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3   { color: #103766 !important; }
h2 { color: #103766 !important; font-weight: 500 !important; font-size: 18px !important; }
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] { color: #5A7BAA !important; font-size: 11px !important; }
[data-baseweb="tab-list"] { background-color: transparent !important; border-bottom: 1px solid rgba(16,55,102,0.15) !important; }
[data-baseweb="tab"]      { color: #5A7BAA !important; font-weight: 400 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #103766 !important;
    border-bottom: 2px solid #288CFA !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] > div > div { border-color: rgba(16,55,102,0.20) !important; }
[data-testid="stNumberInput"] > div > div > input { border-color: rgba(16,55,102,0.20) !important; }
hr { border-color: rgba(16,55,102,0.15) !important; margin: 0.75rem 0 !important; }
p  { color: #103766 !important; }
</style>
"""


def main():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    today = date.today()

    # ── Load data ─────────────────────────────────────────────────────────────
    data         = load_portfolio()
    account_size = data.get("account_size", 100000)
    open_pos     = data.get("open_positions", [])
    closed_pos   = data.get("closed_positions", [])

    # ── Fetch live prices ─────────────────────────────────────────────────────
    open_tickers = [p["ticker"] for p in open_pos]
    prices = {}
    if open_tickers:
        with st.spinner("Fetching live prices…"):
            prices = fetch_current_prices(",".join(open_tickers))

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="font-size:11px; font-weight:500; color:#5A7BAA; text-transform:uppercase;'
        ' letter-spacing:0.04em; margin-bottom:6px;">Account Size</div>',
        unsafe_allow_html=True,
    )
    new_acct = st.sidebar.number_input(
        "Account Size ($)", value=account_size, min_value=1000, step=1000,
        label_visibility="collapsed"
    )
    if new_acct != account_size:
        data["account_size"] = new_acct
        save_portfolio(data)
        account_size = new_acct

    _sidebar_add_position(data)
    _sidebar_close_position(data)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:16px;">'
        f'<span style="font-size:28px; font-weight:500; color:#103766;">Portfolio Tracker</span>'
        f'<span style="font-size:12px; color:#5A7BAA; margin-left:12px;">{today.strftime("%B %d, %Y")}'
        f' &nbsp;·&nbsp; Account ${account_size:,.0f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — PERFORMANCE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    # Period selector
    perf_col1, perf_col2 = st.columns([2, 5])
    with perf_col1:
        perf_period = st.selectbox(
            "Performance Period",
            ["YTD", "1M", "3M", "6M", "1Y", "Custom"],
            index=0,
            label_visibility="collapsed",
        )

    # Resolve date range
    if perf_period == "YTD":
        p_start = date(today.year, 1, 1)
        p_end   = today
        period_label = f"YTD {today.year}"
    elif perf_period == "1M":
        p_start = today - timedelta(days=30)
        p_end   = today
        period_label = "Last 30 Days"
    elif perf_period == "3M":
        p_start = today - timedelta(days=91)
        p_end   = today
        period_label = "Last 3 Months"
    elif perf_period == "6M":
        p_start = today - timedelta(days=182)
        p_end   = today
        period_label = "Last 6 Months"
    elif perf_period == "1Y":
        p_start = today - timedelta(days=365)
        p_end   = today
        period_label = "Last 12 Months"
    else:  # Custom
        with perf_col2:
            custom_range = st.date_input(
                "Date Range", value=(date(today.year, 1, 1), today),
                min_value=date(2020, 1, 1), max_value=today,
                label_visibility="collapsed",
            )
        p_start = custom_range[0] if isinstance(custom_range, (list, tuple)) else date(today.year, 1, 1)
        p_end   = custom_range[1] if isinstance(custom_range, (list, tuple)) and len(custom_range) > 1 else today
        period_label = f"{p_start} → {p_end}"

    perf = calc_performance(closed_pos, p_start, p_end)

    # Build open-position live stats
    _, total_mv, total_cb, total_upnl, total_risk = build_open_table(open_pos, prices, account_size)
    deployed_pct = total_cb / account_size if account_size else 0
    heat_pct     = total_risk / account_size if account_size else 0

    # Profit factor display
    pf_val = perf["profit_factor"]
    pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"

    # Realized P&L color
    rpnl_color = _pnl_color(perf["realized_pnl"])
    upnl_color = _pnl_color(total_upnl)

    summary_html = (
        f'<div style="background:#FFFFFF; border-radius:12px; border:0.5px solid rgba(16,55,102,0.12);'
        f' padding:15px 17px; margin-bottom:10px;">'
        f'<div style="font-size:11px; color:#5A7BAA; margin-bottom:10px; font-weight:500;">'
        f'PERFORMANCE SUMMARY &nbsp;·&nbsp; {period_label}</div>'
        f'<div style="display:grid; grid-template-columns:repeat(8,1fr); gap:9px;">'
        + _tile("Realized P&L",   dollar_fmt(perf["realized_pnl"]), f"{perf['count']} closed trades", rpnl_color)
        + _tile("Unrealized P&L", dollar_fmt(total_upnl), f"{len(open_pos)} open positions", upnl_color)
        + _tile("Win Rate",       f"{perf['win_rate']*100:.0f}%", "" if perf["count"] == 0 else f"{sum(1 for p in perf['trades'] if (p['exit_price']-p['entry_price'])*p['shares']>0)}W / {sum(1 for p in perf['trades'] if (p['exit_price']-p['entry_price'])*p['shares']<=0)}L")
        + _tile("Avg Win",        pct_fmt(perf["avg_win"]) if perf["avg_win"] else "—", "", "#27500A" if perf["avg_win"] > 0 else "#5A7BAA")
        + _tile("Avg Loss",       pct_fmt(perf["avg_loss"]) if perf["avg_loss"] else "—", "", "#CC1111" if perf["avg_loss"] < 0 else "#5A7BAA")
        + _tile("Profit Factor",  pf_str, "≥ 1.5 target", "#27500A" if pf_val >= 1.5 else "#E07800" if pf_val >= 1.0 else "#CC1111")
        + _tile("Deployed",       f"{deployed_pct*100:.1f}%", f"${total_cb:,.0f} cost basis", "#5A7BAA")
        + _tile("Portfolio Heat", f"{heat_pct*100:.1f}%", "risk $ / account", "#CC1111" if heat_pct > 0.15 else "#E07800" if heat_pct > 0.08 else "#27500A")
        + '</div></div>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — OPEN POSITIONS
    # ─────────────────────────────────────────────────────────────────────────

    open_df, _, _, _, _ = build_open_table(open_pos, prices, account_size)

    n_core     = sum(1 for p in open_pos if p.get("layer") == "Core")
    n_tactical = sum(1 for p in open_pos if p.get("layer") == "Tactical")
    open_pill  = f"{len(open_pos)} positions · {n_core} Core / {n_tactical} Tactical"

    if not open_df.empty:
        st.markdown(
            _card("Open Positions", cb_table(open_df, bordered=False), pill=open_pill),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _card("Open Positions",
                  '<p style="font-size:13px; color:#5A7BAA;">No open positions. Add one using the sidebar.</p>',
                  pill="0 positions"),
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — CLOSED POSITIONS
    # ─────────────────────────────────────────────────────────────────────────

    # Year selector — derive available years from data, default = current year
    closed_years = sorted(
        set(datetime.strptime(p["exit_date"], "%Y-%m-%d").year for p in closed_pos),
        reverse=True,
    )
    if not closed_years:
        closed_years = [today.year]

    year_options = [str(y) for y in closed_years]
    default_idx  = year_options.index(str(today.year)) if str(today.year) in year_options else 0

    selected_year_str = st.selectbox(
        "Closed Positions — Year",
        options=year_options,
        index=default_idx,
        label_visibility="collapsed",
    )
    selected_year = int(selected_year_str)

    filtered_closed = [
        p for p in closed_pos
        if datetime.strptime(p["exit_date"], "%Y-%m-%d").year == selected_year
    ]
    closed_df = build_closed_table(filtered_closed)

    year_pnl = sum(
        (p["exit_price"] - p["entry_price"]) * p["shares"] for p in filtered_closed
    )
    year_pnl_color = _pnl_color(year_pnl)
    closed_pill = (
        f"{selected_year} · {len(filtered_closed)} trades · "
        f'<span style="color:{year_pnl_color}; font-weight:500;">{dollar_fmt(year_pnl)}</span>'
    )

    if not closed_df.empty:
        st.markdown(
            _card(f"Closed Positions — {selected_year}",
                  cb_table(closed_df, bordered=False),
                  pill=f"{selected_year} · {len(filtered_closed)} trades · {dollar_fmt(year_pnl)}"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _card(f"Closed Positions — {selected_year}",
                  f'<p style="font-size:13px; color:#5A7BAA;">No closed trades in {selected_year}.</p>',
                  pill=f"{selected_year}"),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
