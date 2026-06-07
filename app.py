"""
Swing Trading Framework — Streamlit Dashboard
Automates Layers 0, 1, 1.5, 2, and 3.

Architecture:
  Constants → Data fetching → Layer calculations → Chart builders →
  Helpers → Tab render functions → main()
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
import json
import os
import warnings
warnings.filterwarnings("ignore")

# FRED_AVAILABLE is imported below from data.fred (the fredapi import lives there).


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Trading Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS: style st.data_editor to match Classic Blue theme ─────────────
# st.data_editor renders as a glide-data-grid <canvas>, so th/td selectors don't
# reach it — the grid is themed through its CSS custom properties (--gdg-*).
st.markdown("""
<style>
/* Glide-data-grid theme vars — blue header, white data rows.
   The grid reads --gdg-* from its own scroll container (.gdg-* / .dvn-scroller),
   the .glideDataEditor element, and the stDataFrameResizable wrapper — set on all
   of them so the canvas-drawn header picks up the blue regardless of version. */
[data-testid="stDataEditor"],
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"],
[data-testid="stDataEditor"] .glideDataEditor,
[data-testid="stDataEditor"] .dvn-scroller,
[data-testid="stDataEditorResizable"],
.glideDataEditor,
.gdg-wmyidgi {
    --gdg-accent-color: #288CFA !important;
    --gdg-bg-header: #EEF3FA !important;
    --gdg-bg-header-has-focus: #EEF3FA !important;
    --gdg-bg-header-hovered: #EEF3FA !important;
    --gdg-text-header: #5A7BAA !important;
    --gdg-text-header-selected: #5A7BAA !important;
    --gdg-bg-cell: #FFFFFF !important;
    --gdg-bg-cell-medium: #FFFFFF !important;
    --gdg-text-dark: #103766 !important;
    --gdg-font-family: sans-serif !important;
    --gdg-header-font-style: 500 11px !important;
    --gdg-bg-icon-header: #5A7BAA !important;
    --gdg-fg-icon-header: #EEF3FA !important;
}
/* Fallback for any non-canvas (HTML) rendering of the grid */
[data-testid="stDataEditor"] th,
[data-testid="stDataEditor"] [role="columnheader"] {
    background-color: #EEF3FA !important;
    color: #5A7BAA !important;
    font-size: 11px !important;
    font-weight: 500 !important;
}
[data-testid="stDataEditor"] td,
[data-testid="stDataEditor"] [role="gridcell"] {
    background-color: #FFFFFF !important;
    color: #103766 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — moved to config.py
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    LB_1M, LB_3M, LB_6M,
    RS_NEW_HI_THRESHOLD, VELOCITY_THRESHOLD,
    SECTOR_ETFS, SECTOR_TICKERS, ALL_SECTORS,
    RISK_ON_SECTORS, REFLATION_SECTORS, DEFENSIVE_SECTORS,
    SECTOR_COLORS, SECTOR_DASH, SECTOR_DASH_BY_SECTOR,
    PERM_LIMITS, SETUP_STYLE,
    FLOW_SIZE_MAP, PHASE_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

# Data-fetching functions moved to the data/ package. Imported under their
# original names so existing call sites are unchanged.
from data.market import (
    fetch_macro_data,
    fetch_screener_data,
    fetch_portfolio_prices,
    fetch_earnings_dates,
)
from data.fred import (
    fetch_fred_data,
    fetch_fed_net_liquidity,
    fetch_mispricing_rates,
    fetch_tbill_rate,
    FRED_AVAILABLE,
)
from data.flows import fetch_etf_fund_flows
from trading_logic import (
    compute_entry_stop,
    period_to_range,
    add_open_position,
    close_position,
)


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER V3 — AUTONOMOUS, BUTTON-TRIGGERED
# ─────────────────────────────────────────────────────────────────────────────

# Larger universe per sector ETF (~100-200 liquid names each)
SCREENER_UNIVERSE = {
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","AMD","CSCO","ACN","ADBE",
        "IBM","INTU","TXN","QCOM","AMAT","NOW","PANW","ADI","LRCX","KLAC",
        "SNPS","CDNS","CRWD","MSI","APH","MCHP","FTNT","ROP","TEL","NXPI",
        "ADSK","MPWR","ON","KEYS","CDW","FSLR","IT","HPE","HPQ",
        "ZBRA","TYL","EPAM","AKAM","SWKS","TER","NTAP","PTC","TRMB",
        "GEN","WDC","STX","SMCI","DELL","PLTR","NET","DDOG","ZS","MDB",
        "SNOW","TEAM","HUBS","WDAY","VEEV","OKTA","ZM","DOCN","PATH",
        "S","ESTC","MNDY","BILL","GTLB","IOT","AI","APP",
        "FICO","ANET","MRVL","ARM","UBER","DASH","SHOP","TTD","COIN",
    ],
    "Semiconductors": [
        "NVDA","AMD","AVGO","QCOM","TXN","AMAT","ADI","LRCX","KLAC","MCHP",
        "NXPI","ON","MPWR","MRVL","SWKS","TER","MKSI","ENTG","AMKR",
        "CRUS","ONTO","WOLF","RMBS","SMTC","ACLS","AOSL","ALGM","SITM","POWI",
        "DIOD","AMBA","LSCC","MTSI","PI","COHR","MU",
        "INTC","TSM","ASML","ARM","SMCI","GFS","UMC","ASX",
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","ABNB",
        "CMG","ORLY","AZO","ROST","DHI","LEN","PHM","NVR","GPC","GRMN",
        "POOL","BBY","DRI","YUM","ULTA","LULU","DECK","TPR","RL",
        "HAS","MAT","WYNN","LVS","MGM","CZR","RCL","CCL","NCLH","HLT",
        "MAR","H","EXPE","LKQ","KMX","AN","LAD","CVNA","DPZ",
        "WING","TXRH","EAT","SHAK","BROS","CAVA",
        "BIRK","ONON","CROX","COLM","VFC","PVH","ETSY",
        "W","RH","WSM","FIVE","OLLI","DLTR","DG","COST","WMT","TGT",
    ],
    "Financials": [
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","SPGI","BLK",
        "SCHW","C","AXP","CB","AON","PGR","ICE","CME","MCO",
        "MSCI","AJG","AFL","TRV","MET","AIG","PRU","ALL","COF",
        "SYF","USB","PNC","TFC","FITB","MTB","HBAN","CFG","KEY",
        "RF","ZION","NDAQ","CBOE","FDS","MKTX","VIRT","HOOD","IBKR",
        "RJF","LPLA","EVR","HLI","PJT","SF","PIPR","SEIC","TROW",
        "BEN","IVZ","WBS","FNB","EWBC","WAL","OZK","SSB","BOKF","GBCI","PNFP","WTFC",
    ],
    "Industrials": [
        "GE","CAT","RTX","HON","UNP","UPS","DE","LMT","BA","ADP",
        "ETN","ITW","GD","NOC","WM","RSG","CSX","NSC","PCAR","EMR",
        "JCI","TT","CARR","OTIS","ROK","FAST","SWK","GWW","IR",
        "PH","DOV","FTV","AME","XYL","IEX","RBC","GNRC","PWR","HUBB",
        "BLDR","TTC","WAB","AGCO","AXON","TDG","HWM","HEI","TRMB",
        "VRSK","CPRT","PAYX","CTAS","ODFL","SAIA","XPO","JBHT","CHRW",
        "LSTR","EXPD","KEX","WERN","SNDR","MATX","GXO","ALLE","AOS",
        "WSO","RRX","MIDD","MAS","AAON","SPXC","RHI",
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY",
        "HES","DVN","FANG","HAL","BKR","CTRA","MRO","APA","CHRD","OVV",
        "EQT","RRC","AR","PR","MTDR","CRGY","SM","MGY","NOG","VTLE",
        "CRC","TRGP","WMB","KMI","OKE","ET","MPLX",
        "AM","DTM","DINO","PBF","PARR","CVI",
    ],
    "Materials": [
        "LIN","SHW","APD","ECL","FCX","NEM","NUE","VMC","MLM","DOW",
        "DD","PPG","CE","EMN","ALB","IFF","FMC","CF","MOS","RPM",
        "BALL","PKG","IP","WRK","SEE","SON","AVY","AXTA",
        "HUN","OLN","TROX","CC","IOSP","KWR","CBT","GEF",
    ],
    "Health Care": [
        "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","PFE","AMGN",
        "BMY","MDT","ISRG","SYK","BSX","GILD","VRTX","REGN","ZTS","BDX",
        "CI","ELV","HCA","MCK","COR","HUM","CNC","MOH","GEHC","EW",
        "DXCM","IDXX","RMD","HOLX","BAX","A","IQV","MTD","WAT",
    ],
    "Consumer Staples": [
        "PG","PEP","KO","COST","WMT","PM","MO","MDLZ","CL","EL",
        "KMB","GIS","SJM","K","HSY","MKC","HRL","CAG","CPB","TSN",
        "BG","ADM","MNST","KDP","STZ","TAP",
        "WBA","KR","SYY","USFD","PFGC","CHD","CLX","COTY",
    ],
    "Utilities": [
        "NEE","SO","DUK","D","SRE","AEP","EXC","XEL","ED","WEC",
        "ES","EIX","AWK","ATO","CMS","DTE","ETR","FE","PPL","CEG",
        "AES","LNT","EVRG","NI","PNW","OGE","NRG","VST","CWEN",
    ],
}

# Sector ETF name → SCREENER_UNIVERSE key mapping
# (L0 uses SECTOR_ETFS keys like "Technology", "Financials" etc. which
#  match SCREENER_UNIVERSE keys directly, except we also have "Semiconductors")
# SMH sectors show up as "Technology" in L0 — we always include Semis when Tech is leading.
SECTOR_TO_SCREEN = {
    "Technology":             ["Technology", "Semiconductors"],
    "Financials":             ["Financials"],
    "Consumer Discretionary": ["Consumer Discretionary"],
    "Industrials":            ["Industrials"],
    "Energy":                 ["Energy"],
    "Materials":              ["Materials"],
    "Health Care":            ["Health Care"],
    "Consumer Staples":       ["Consumer Staples"],
    "Utilities":              ["Utilities"],
}


@st.cache_data(ttl=86400, show_spinner=False)
def run_screener_v3(
    regime: str,
    leading_sectors: list,
    mixed_sectors: list,
    perm: str,
    accel_sectors_key: str,
    rec_flags: int,
    tbill_rate_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Autonomous screener v3 — scans L0 Leading/Mixed sectors through L4 filters,
    then applies L5 entry trigger assessment (trigger type, entry/stop, verdict).
    Cached for 24 hours. Button-triggered.
    Returns a DataFrame of all passing candidates with L5 verdicts, ranked by RS.
    """
    accel_secs = set(accel_sectors_key.split(",")) if accel_sectors_key else set()
    rec_override = rec_flags >= 4
    today = datetime.today().date()

    tbill_rate = tbill_rate_pct if tbill_rate_pct > 0 else None

    # Use actual L0 sector readings — only scan what's Leading or Mixed
    l0_sectors = leading_sectors + mixed_sectors
    screen_sectors = []
    for sec in l0_sectors:
        for mapped in SECTOR_TO_SCREEN.get(sec, []):
            if mapped not in screen_sectors:
                screen_sectors.append(mapped)

    # Fallback if L0 has nothing (shouldn't happen, but safety)
    if not screen_sectors:
        screen_sectors = list(SCREENER_UNIVERSE.keys())

    # Build ticker universe
    tickers = []
    ticker_sector = {}
    for sec in screen_sectors:
        for t in SCREENER_UNIVERSE.get(sec, []):
            if t not in ticker_sector:
                tickers.append(t)
                ticker_sector[t] = sec

    if not tickers:
        return pd.DataFrame()

    # Batch download
    all_tickers = tickers + ["SPY"]
    frames = {}
    batch_size = 200

    # Diagnostics — so the UI can tell "quiet market" apart from "data failed".
    requested      = len(all_tickers)
    failed_batches = 0           # whole batches that errored (likely rate-limit/network)
    last_error     = ""

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            data = yf.download(batch, period="6mo", group_by="ticker",
                               threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        df = data[t][["Close", "Volume"]].dropna()
                        if len(df) >= 50:
                            frames[t] = df
                    except (KeyError, TypeError):
                        # Individual ticker missing from the response — expected for
                        # thinly traded / delisted names; counted via fetched total.
                        pass
            elif len(batch) == 1 and len(data) >= 50:
                frames[batch[0]] = data[["Close", "Volume"]].dropna()
        except Exception as e:
            # A whole batch failing usually means a network or rate-limit problem,
            # not a quiet market — record it so the caller can warn the user.
            failed_batches += 1
            last_error = str(e)
        if i + batch_size < len(all_tickers):
            import time
            time.sleep(2)

    # fetched excludes SPY (the benchmark, not a candidate)
    fetched = len([t for t in frames if t != "SPY"])
    diagnostics = {
        "requested":      requested - 1,   # exclude SPY from the candidate count
        "fetched":        fetched,
        "failed_batches": failed_batches,
        "last_error":     last_error,
        "spy_missing":    "SPY" not in frames,
    }

    spy_data = frames.get("SPY")
    if spy_data is None:
        # Benchmark itself failed — can't compute relative strength. Return an
        # empty frame but carry the reason so the UI shows a data-error warning.
        empty = pd.DataFrame()
        empty.attrs["screener_diagnostics"] = diagnostics
        return empty
    spy_close = spy_data["Close"].squeeze()

    # Compute signals
    results = []
    for ticker, df in frames.items():
        if ticker == "SPY":
            continue
        try:
            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            if len(close) < 50:
                continue

            price = float(close.iloc[-1])
            if price < 5.0:
                continue

            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if price < ma20 or price < ma50:
                continue

            roc_1m = (price / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
            roc_3m = (price / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None
            if roc_1m is None:
                continue
            if roc_3m is not None:
                if roc_1m > 0 and roc_3m > 0:
                    two_speed = "FULL"
                elif roc_1m > 0 or roc_3m > 0:
                    two_speed = "HALF"
                else:
                    continue
            else:
                two_speed = "HALF" if roc_1m > 0 else None
                if two_speed is None:
                    continue

            avg_vol = float(volume.tail(20).mean())
            avg_dollar_vol = avg_vol * price
            if avg_dollar_vol < 50000:
                continue

            common = close.index.intersection(spy_close.index)
            if len(common) < 21:
                continue
            rs = close.reindex(common) / spy_close.reindex(common)
            rs_chg = (float(rs.iloc[-1]) / float(rs.iloc[-21]) - 1) * 100
            if rs_chg <= 0:
                continue

            dist_ma20 = ((price - ma20) / ma20) * 100
            pct_above_20 = (price / ma20 - 1)

            # RS new high (within 2% of 3-month peak)
            rs_line = close.reindex(common) / spy_close.reindex(common)
            rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-min(63, len(rs_line)):].max() * 0.98)
            rs_rising = bool(len(rs_line) >= 21 and rs_line.iloc[-1] > rs_line.iloc[-21])

            # MACD crossover
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            histogram = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
            macd_cross = bool(len(histogram) >= 2 and float(histogram.iloc[-2]) < 0 and float(histogram.iloc[-1]) > 0)

            # Volume ratio
            recent_vol = float(volume.tail(5).mean())
            vol_ratio = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 0

            # Entry zone
            if dist_ma20 <= 3.0:
                entry_zone = "Near MA (pullback)"
            elif dist_ma20 <= 6.0:
                entry_zone = "Normal"
            elif dist_ma20 <= 15.0:
                entry_zone = "Extended (accel only)"
            else:
                entry_zone = "Too extended"

            # Passes = full L4 criteria + entry zone filter
            # "Too extended" never passes. "Extended (accel only)" requires sector accelerating.
            fundamentals_ok = (two_speed == "FULL" and rs_new_hi and rs_rising
                               and avg_dollar_vol >= 10_000_000)
            if entry_zone == "Too extended":
                zone_ok = False
            elif entry_zone == "Extended (accel only)":
                zone_ok = sec in accel_secs
            else:
                zone_ok = True
            passes = fundamentals_ok and zone_ok

            # ── L5 Entry Trigger Assessment ──────────────────────────────────
            sec = ticker_sector.get(ticker, "Unknown")
            ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
            pct_vs_50 = (price / ma50 - 1) if ma50 > 0 else 0

            # 6-week base range
            base_px = close.iloc[-30:]
            base_high = float(base_px.max())
            base_low = float(base_px.min())

            # Last day volume ratio (for breakout confirmation)
            last_vol = float(volume.iloc[-1]) if len(volume) > 0 else 0
            last_vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0

            # Earnings check + forward P/E (reuse single Ticker object)
            yf_ticker = yf.Ticker(ticker)
            try:
                cal = yf_ticker.calendar
                next_ed = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed is not None:
                        for d in (ed if isinstance(ed, list) else [ed]):
                            try:
                                dt = pd.Timestamp(d).date()
                                if dt >= today:
                                    next_ed = dt
                                    break
                            except Exception:
                                pass
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    for col in cal.columns:
                        try:
                            dt = pd.Timestamp(cal.at["Earnings Date", col]).date()
                            if dt >= today:
                                next_ed = dt
                                break
                        except Exception:
                            pass
            except Exception:
                next_ed = None

            days_to_ed = (next_ed - today).days if next_ed else None
            earnings_flag = (days_to_ed is not None and days_to_ed <= 14)

            # Earnings Carry (L4 Step 4): Forward EY − T-Bill Rate
            carry_spread = None
            carry_label = "N/A"
            try:
                fwd_pe = yf_ticker.info.get("forwardPE")
                if fwd_pe and fwd_pe > 0 and tbill_rate is not None:
                    fwd_ey = (1.0 / fwd_pe) * 100  # forward earnings yield %
                    carry_spread = round(fwd_ey - tbill_rate, 2)
                    if carry_spread > 3:
                        carry_label = "Strong"
                    elif carry_spread >= 0:
                        carry_label = "Positive"
                    else:
                        carry_label = "Negative"
            except Exception:
                pass

            # Trigger type routing
            if perm == "Red":
                trigger_type = "None"
            elif sec in accel_secs:
                trigger_type = "Accelerating"
            elif rec_override or regime in ("Deflation", "Stagflation"):
                trigger_type = "Pullback"
            elif regime == "Risk-on" or (perm == "Green" and regime == "Reflation"):
                trigger_type = "Breakout"
            else:
                trigger_type = "Pullback"

            # Verdict + entry/stop
            entry_px_suggest = None
            stop_px_suggest = None
            verdict = "⬜ NOT READY"
            notes = []

            if earnings_flag:
                verdict = "❌ EARNINGS"
                notes.append(f"Earnings in {days_to_ed}d")
            elif trigger_type == "None":
                verdict = "❌ RED"
                notes.append("No new entries")
            elif trigger_type == "Breakout":
                # Entry = top of base (pivot). Stop = just below bottom of base.
                near_top = pct_above_20 <= 0.08 and price <= base_high * 1.05
                vol_ok = last_vol_ratio >= 1.4
                entry_px_suggest = round(base_high, 2)
                stop_px_suggest = round(base_low * 0.99, 2)
                if near_top and vol_ok and rs_rising:
                    verdict = "🟢 ENTRY READY"
                elif near_top and last_vol_ratio >= 1.0:
                    verdict = "🟡 WATCH"
                    notes.append(f"Vol {last_vol_ratio:.1f}x — needs 1.4x")
                else:
                    if not near_top:
                        notes.append("Not at pivot")
            elif trigger_type == "Pullback":
                # Entry = at the 20d or 50d MA. Stop = 1-2% below that MA.
                near_20 = abs(pct_above_20) <= 0.03
                near_50 = abs(pct_vs_50) <= 0.03
                vol_decl = last_vol_ratio <= 1.0
                ref_ma = ma20 if (near_20 or not near_50) else ma50
                entry_px_suggest = round(ref_ma, 2)
                stop_px_suggest = round(ref_ma * 0.98, 2)
                if (near_20 or near_50) and vol_decl:
                    verdict = "🟢 ENTRY READY"
                    notes.append("At 20d MA" if near_20 else "At 50d MA")
                elif near_20 or near_50:
                    verdict = "🟡 WATCH"
                    notes.append(f"Vol {last_vol_ratio:.1f}x — need declining")
                else:
                    pass  # remains NOT READY
            elif trigger_type == "Accelerating":
                in_range = 0 <= pct_above_20 <= 0.15
                vol_ok = last_vol_ratio >= 1.0
                if in_range and vol_ok:
                    verdict = "🟢 ENTRY READY"
                    entry_px_suggest = round(price, 2)
                    stop_px_suggest = round(ema10 * 0.99, 2)
                    notes.append("Half size · 10d EMA stop · 6wk")
                elif in_range:
                    verdict = "🟡 WATCH"
                    entry_px_suggest = round(price, 2)
                    stop_px_suggest = round(ema10 * 0.99, 2)
                    notes.append(f"Vol {last_vol_ratio:.1f}x · Half size")
                else:
                    entry_px_suggest = round(price, 2)
                    stop_px_suggest = round(ema10 * 0.99, 2)

            if next_ed and not earnings_flag:
                try:
                    notes.append(f"Earnings: {next_ed.strftime('%b %-d')}")
                except Exception:
                    notes.append(f"Earnings: {next_ed}")

            # Earnings carry note (negative = size reduction flag)
            if carry_label == "Negative" and carry_spread is not None:
                notes.append(f"Carry {carry_spread:+.1f}% — reduce 25%")

            # Monitoring = strong fundamentals but too extended to trade
            monitoring = fundamentals_ok and not zone_ok

            results.append({
                "Ticker": ticker,
                "Sector": sec,
                "Price": round(price, 2),
                "% > 20MA": round(pct_above_20, 4),
                "Avg $Vol(M)": round(avg_dollar_vol / 1e6, 1),
                "2-Speed": two_speed,
                "PASS": passes,
                "MONITORING": monitoring,
                "RS_vs_SPY_21d": round(rs_chg, 2),
                "Dist_MA20_pct": round(dist_ma20, 1),
                "Entry_Zone": entry_zone,
                "MACD_Crossover": macd_cross,
                "Vol_Ratio_5d": vol_ratio,
                "Carry_Spread": carry_spread,
                "Carry_Label": carry_label,
                "Trigger": trigger_type,
                "Entry": entry_px_suggest,
                "Stop": stop_px_suggest,
                "Verdict": verdict,
                "Notes": " · ".join(notes) if notes else "",
            })
        except Exception:
            continue

    if not results:
        empty = pd.DataFrame()
        empty.attrs["screener_diagnostics"] = diagnostics
        return empty

    df_out = pd.DataFrame(results).sort_values("RS_vs_SPY_21d", ascending=False).reset_index(drop=True)
    df_out.attrs["screener_diagnostics"] = diagnostics
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 — MACRO REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────

# Layer calculations moved to layers.py. Imported under their original names so
# existing call sites (and calc_layer0's internal Fed-liquidity call) are unchanged.
from layers import (
    calc_layer0,
    score_recession_composite,
    score_layer1_mispricing,
    drawdown_tier,
    is_etf_position,
    etf_exit_signals,
    calc_layer2,
    calc_layer3,
    calc_layer4,
    calc_layer5,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def build_chart(ticker: str):
    """
    Build a 5-panel Plotly chart for a single ticker:
    Price + MAs / Volume / RS vs SPY / RSI(14) / MACD.
    Returns None if data is unavailable.
    """
    hist     = yf.Ticker(ticker).history(period="6mo")
    spy_hist = yf.Ticker("SPY").history(period="6mo")
    if hist.empty:
        return None

    px  = hist["Close"].dropna()
    op  = hist["Open"].reindex(px.index)
    hi  = hist["High"].reindex(px.index)
    lo  = hist["Low"].reindex(px.index)
    vol = hist["Volume"].reindex(px.index)

    ma20   = px.rolling(20).mean()
    ma50   = px.rolling(50).mean()
    ema10  = px.ewm(span=10, adjust=False).mean()  # 10d EMA for Accelerating Protocol (v4)
    spy_a  = spy_hist["Close"].reindex(px.index).ffill()
    rs    = px / spy_a
    rsi   = RSIIndicator(close=px, window=14).rsi()
    mo    = MACDIndicator(close=px, window_slow=26, window_fast=12, window_sign=9)
    ml, mg, mh = mo.macd(), mo.macd_signal(), mo.macd_diff()

    def tl(series):
        return [None if pd.isna(v) else float(v) for v in series]

    dates = [str(d)[:10] for d in px.index]

    PANEL_SEP    = [0.633, 0.500, 0.313, 0.144]
    PANEL_LABELS = [
        (f"{ticker} — Price", 0.993, 0.13),
        ("Volume",            0.621, 0.01),
        ("RS vs SPY",         0.488, 0.01),
        ("RSI (14)",          0.301, 0.01),
        ("MACD",              0.132, 0.01),
    ]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.38, 0.12, 0.18, 0.16, 0.16],
        vertical_spacing=0.025,
    )
    fig.add_trace(go.Candlestick(
        x=dates, open=tl(op), high=tl(hi), low=tl(lo), close=tl(px),
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(ema10), name="10d EMA", line=dict(color="#ef4444", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(ma20), name="20d MA",  line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(ma50), name="50d MA",  line=dict(color="#3b82f6", width=1.5)), row=1, col=1)

    vol_c = ["#22c55e" if (c or 0) >= (o or 0) else "#ef4444" for c, o in zip(tl(px), tl(op))]
    fig.add_trace(go.Bar(x=dates, y=tl(vol), marker_color=vol_c, opacity=0.7, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(rs),  line=dict(color="#a78bfa", width=1.5), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(rsi), line=dict(color="#38bdf8", width=1.5), showlegend=False), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=4, col=1)

    mh_c = ["#22c55e" if (v or 0) >= 0 else "#ef4444" for v in tl(mh)]
    fig.add_trace(go.Bar(x=dates, y=tl(mh), marker_color=mh_c, showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(ml), name="MACD",   line=dict(color="#f59e0b", width=1.5), showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(mg), name="Signal", line=dict(color="#a78bfa", width=1.5), showlegend=False), row=5, col=1)

    for y in PANEL_SEP:
        fig.add_shape(
            type="line", xref="paper", yref="paper",
            x0=0, x1=1, y0=y, y1=y,
            line=dict(color="#6b7280", width=2),
        )
    for label, y, x in PANEL_LABELS:
        fig.add_annotation(
            text=f"<b>{label}</b>", xref="paper", yref="paper",
            x=x, y=y, showarrow=False,
            font=dict(color="#5A7BAA", size=11),
            xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        height=750, paper_bgcolor="#EEF3FA", plot_bgcolor="#FFFFFF",
        font=dict(color="#103766", size=11), margin=dict(l=50, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.01, bgcolor="rgba(0,0,0,0)"),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="rgba(16,55,102,0.10)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(16,55,102,0.10)", row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    return fig


def build_rrg_chart(l3_data: list) -> go.Figure:
    """
    Build the Relative Rotation Graph scatter chart.
    Each sector gets a unique color (SECTOR_COLORS) and a smooth spline trail
    showing the last 8 weekly positions.
    Returns None if data is empty.
    """
    if not l3_data:
        return None

    all_x = [v for d in l3_data for v in d["trail_x"]]
    all_y = [v for d in l3_data for v in d["trail_y"]]
    pad   = 0.8
    xlo   = min(min(all_x) - pad, 98.5)
    xhi   = max(max(all_x) + pad, 101.5)
    ylo   = min(min(all_y) - pad, 98.5)
    yhi   = max(max(all_y) + pad, 101.5)

    quad_label_colors = {
        "LEADING":   "#27500A",
        "IMPROVING": "#288CFA",
        "WEAKENING": "#E07800",
        "LAGGING":   "#CC1111",
    }
    fill_map = {
        "Leading":   "rgba(39,80,10,0.15)",
        "Improving": "rgba(37,80,200,0.15)",
        "Weakening": "rgba(224,120,0,0.15)",
        "Lagging":   "rgba(204,17,17,0.15)",
    }

    shapes = [
        dict(type="rect", xref="x", yref="y", x0=100, x1=xhi, y0=100, y1=yhi, fillcolor=fill_map["Leading"],   line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=xlo, x1=100, y0=100, y1=yhi, fillcolor=fill_map["Improving"], line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=100, x1=xhi, y0=ylo, y1=100, fillcolor=fill_map["Weakening"], line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=xlo, x1=100, y0=ylo, y1=100, fillcolor=fill_map["Lagging"],   line_width=0, layer="below"),
        dict(type="line", xref="x", yref="y", x0=xlo, x1=xhi, y0=100, y1=100, line=dict(color="#6b7280", width=1, dash="dot")),
        dict(type="line", xref="x", yref="y", x0=100, x1=100, y0=ylo, y1=yhi, line=dict(color="#6b7280", width=1, dash="dot")),
    ]

    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    for d in l3_data:
        color      = SECTOR_COLORS.get(d["sector"], "#475569")
        dash_style = SECTOR_DASH_BY_SECTOR.get(d["sector"], "solid")
        tx, ty     = d["trail_x"], d["trail_y"]

        # Spline trail — faded, markers on historical positions
        fig.add_trace(go.Scatter(
            x=tx, y=ty,
            mode="lines+markers",
            line=dict(color=color, width=1.5, shape="spline", smoothing=1.3, dash=dash_style),
            marker=dict(size=[4] * (len(tx) - 1) + [0], color=color, opacity=0.45),
            showlegend=False, hoverinfo="skip",
        ))

        # Current position — large labeled dot
        fig.add_trace(go.Scatter(
            x=[d["rs_ratio"]], y=[d["rs_momentum"]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(color="#EEF3FA", width=1.5)),
            text=[d["etf"]],
            textposition="top center",
            textfont=dict(color="#103766", size=10),
            name=f"{d['etf']} — {d['quadrant']}",
            hovertemplate=(
                f"<b>{d['sector']} ({d['etf']})</b><br>"
                f"RS-Ratio: %{{x:.2f}}<br>"
                f"RS-Momentum: %{{y:.2f}}<br>"
                f"Phase: {d['phase']}<br>"
                f"Action: {d['action']}<extra></extra>"
            ),
        ))

    for text, x, y, xa, ya in [
        ("LEADING",   xhi - 0.05, yhi - 0.05, "right", "top"),
        ("IMPROVING", xlo + 0.05, yhi - 0.05, "left",  "top"),
        ("WEAKENING", xhi - 0.05, ylo + 0.05, "right", "bottom"),
        ("LAGGING",   xlo + 0.05, ylo + 0.05, "left",  "bottom"),
    ]:
        fig.add_annotation(
            text=f"<b>{text}</b>", x=x, y=y,
            xanchor=xa, yanchor=ya, showarrow=False,
            font=dict(color=quad_label_colors[text], size=11),
        )

    fig.update_layout(
        height=600,
        paper_bgcolor="#EEF3FA", plot_bgcolor="#FFFFFF",
        font=dict(color="#103766", size=11),
        # Extra bottom room so the x-axis title and the legend don't collide.
        margin=dict(l=55, r=20, t=30, b=110),
        xaxis=dict(
            title=dict(text="RS-Ratio  →  (stronger relative performance)", standoff=14),
            gridcolor="rgba(16,55,102,0.10)", zeroline=False, range=[xlo, xhi],
        ),
        yaxis=dict(
            title="RS-Momentum  ↑  (accelerating)",
            gridcolor="rgba(16,55,102,0.10)", zeroline=False, range=[ylo, yhi],
        ),
        showlegend=True,
        # Legend sits well below the axis title (more negative y) and is centred.
        legend=dict(
            orientation="h", xanchor="center", x=0.5, y=-0.22,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pct(v):  return f"{v*100:+.1f}%"
def icon(v): return "✅" if v else "❌"
def macd(v): return "▲" if v else "▼"


# cb_table / _card / _gate_bar_html / _tile moved to ui_components.py.
# Imported under the original names so existing call sites are unchanged.
# app.py uses the macro color preset for cb_table.
from ui_components import (
    card as _card,
    gate_bar_html as _gate_bar_html,
    tile as _tile,
    cb_table as _cb_table_base,
    CB_PRESET_MACRO,
)


def cb_table(df: pd.DataFrame, max_height: int | None = None, bordered: bool = True) -> str:
    """Classic Blue HTML table using the macro color preset (see ui_components)."""
    return _cb_table_base(df, max_height=max_height, bordered=bordered,
                          preset=CB_PRESET_MACRO, font_size=14)


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _staleness(date_str: str, stale_days: int = 45) -> tuple:
    """Return (color, age_label) for a FRED as-of date string (YYYY-MM-DD).

    Layer 1 is a monthly check, so 'fresh' is generous: green within ~1 month,
    amber up to stale_days, red beyond (a CPI/PCE cycle has likely passed).
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "#5A7BAA", "—"
    age = (date.today() - d).days
    if age <= 30:
        return "#27500A", f"{age}d ago"
    if age <= stale_days:
        return "#E07800", f"{age}d ago"
    return "#CC1111", f"{age}d ago — STALE"


def _layer1_body_html() -> str:
    """Build the Layer 1 monthly-mispricing card BODY html (no card chrome).

    Pulls FRED real/nominal 10y + SPY forward EY, scores the composite, and
    color-codes by data staleness. Returned body is wrapped by _card() or
    _macro_card() at the call site."""
    rates   = fetch_mispricing_rates()
    # Forward earnings yield is entered manually (no free forward-EY data source).
    _ey_val = st.session_state.get("spy_fwd_ey", 0.0)
    fwd_ey  = _ey_val if _ey_val and _ey_val > 0 else None

    if "error" in rates:
        return (f'<p style="font-size:12px; color:#CC1111; margin:0;">'
                f'Rate data unavailable: {rates["error"]}</p>')

    gdp_signal = st.session_state.get("gdpnow_signal", "Not set")
    result = score_layer1_mispricing(
        fwd_earnings_yield=fwd_ey,
        tips_10y=rates.get("tips_10y"),
        nominal_10y=rates.get("nominal_10y"),
        gdpnow_signal=gdp_signal,
    )

    # Staleness — drive the card's accent off the oldest FRED input.
    tips_color, tips_age = _staleness(rates.get("tips_date", ""))
    nom_color, nom_age   = _staleness(rates.get("nominal_date", ""))
    # worst (reddest) of the two governs the header note
    order = {"#27500A": 0, "#E07800": 1, "#CC1111": 2, "#5A7BAA": 0}
    hdr_color = tips_color if order[tips_color] >= order[nom_color] else nom_color

    # Composite verdict color
    comp = result["composite"]
    vcolor = "#27500A" if comp >= 2 else ("#CC1111" if comp <= -2 else "#5A7BAA")

    ey_str = f"{fwd_ey:.2f}%" if fwd_ey is not None else "N/A"
    rows = ""
    for c in result["checks"]:
        sc = c["score"]
        sc_color = "#27500A" if sc > 0 else ("#CC1111" if sc < 0 else "#5A7BAA")
        rows += (
            f'<tr>'
            f'<td style="padding:5px 8px; font-size:12px; color:#103766;">{c["name"]}</td>'
            f'<td style="padding:5px 8px; font-size:12px; color:#103766; text-align:right;">{c["value"]}</td>'
            f'<td style="padding:5px 8px; font-size:12px; color:{sc_color}; text-align:right;">{c["label"]}</td>'
            f'</tr>'
        )

    body = (
        f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">'
        f'<span style="font-size:13px; font-weight:600; color:{vcolor};">{result["verdict"]} '
        f'(composite {comp:+d})</span>'
        f'<span style="font-size:10px; color:{hdr_color};">FRED rates: TIPS {tips_age} · 10y {nom_age}</span>'
        f'</div>'
        f'<p style="font-size:11px; color:#5A7BAA; margin:0 0 8px 0;">S&amp;P fwd earnings yield: {ey_str} '
        f'· TIPS {rates.get("tips_10y")}% · 10y {rates.get("nominal_10y")}%</p>'
        f'<table style="width:100%; border-collapse:collapse;">{rows}</table>'
        f'<p style="font-size:11px; color:#5A7BAA; margin:8px 0 0 0;">{result["sizing"]}</p>'
        + (
            f'<p style="font-size:10px; color:#E07800; margin:4px 0 0 0;">'
            f'Enter S&amp;P forward earnings yield in the sidebar to compute checks 1 &amp; 3.</p>'
            if fwd_ey is None else
            f'<p style="font-size:10px; color:#5A7BAA; margin:4px 0 0 0;">'
            f'Run after CPI/PCE. Forward EY &amp; GDPNow leg set manually in sidebar.</p>'
        )
    )
    return body


# ── Macro-tab redesign helpers (2026-06-07) ──────────────────────────────────
# Tooltip copy is sourced from Framework/swing_trading_framework_v4.md so the
# on-screen explanation stays in sync with the framework document.
_MACRO_TIPS = {
    "spy":       ("Two-speed trend", "SPY 1-month and 6-month rate of change. Both positive opens the gate (green); both negative forces red; mixed pulls toward yellow. This is the primary permission signal."),
    "recession": ("Recession composite", "Five macro indicators (Chauvet-Piger, Sahm, yield curve, claims, LEI). 2–3 flags push the state to yellow; 4+ forces red regardless of SPY."),
    "liquidity": ("The override", "HYG/IEF credit spread, Fed net liquidity, TLT vs SPY. Rapidly tightening liquidity forces red and overrides every other signal — so it sits with the gate inputs."),
    "mispricing":("Valuation governor", "Earnings yield vs real rates, equity risk premium, GDPNow gap. Run monthly after CPI/PCE. Doesn't change the gate — it cuts max risk per trade when equities are rich vs macro."),
    "velocity":  ("Acceleration detector", "Sector ROC 21 above +15% flags a parabolic move and switches that sector to the alternate entry protocol — the standard pullback entry never comes in a melt-up."),
    "sectorrs":  ("Regime flavor", "Which sectors lead SPY reveals the regime: tech/discretionary/financials = risk-on; energy/materials = reflation; staples/utilities = defensive. Guides where to hunt setups."),
    "flow":      ("Flow preview", "Sustained directional ETF inflows, checked weekly alongside RS. Feeds the Sector Rotation tab and Core deployment — Phase 1 is early/improving, Phase 2 is confirmed leading."),
    "core_candidates": ("Core entry candidates", "Phase 2 (confirmed-leading) sectors from the rotation graph. These are eligible for Core deployment — enter on the standard trigger, stop at the 20-day MA shown."),
    "core_holdings": ("Core holdings", "Your persistent regime exposure (sector ETFs). Exit triggers: a close below the 20-day MA, a move into the weakening/lagging quadrant, or a RED permission state (which exits all Core to cash)."),
    "rot_candidates": ("Entry candidates", "Sectors in the Improving (early) or Leading (confirmed) quadrants of the rotation graph. Flow strength sets the size: stronger sustained inflows justify a fuller position."),
    "rot_exit": ("Exit watch", "Sectors that have rotated into the Weakening quadrant — momentum is fading even if still above SPY. Review stops; outflows confirm the exit."),
}


def _info_hover(key: str) -> str:
    """Return an inline info-icon + CSS tooltip for a macro card heading."""
    title, body = _MACRO_TIPS.get(key, ("", ""))
    if not title:
        return ""
    return (
        '<span class="mt-info" tabindex="0">ⓘ'
        f'<span class="mt-tip"><b>{title}.</b> {body}</span>'
        '</span>'
    )


def _macro_card(heading: str, inner_html: str, tip_key: str = "",
                accent: str = "", quiet: bool = False) -> str:
    """Card variant for the macro tab: optional left accent + info hover.

    accent: hex color for a 3px left border (gate/modifier tiering). "" = none.
    quiet:  reference-tier styling (muted fill, fainter border).
    """
    info = _info_hover(tip_key)
    bg     = "#F6F8FC" if quiet else "#FFFFFF"
    border = "rgba(16,55,102,0.07)" if quiet else "rgba(16,55,102,0.12)"
    accent_css = (f"border-left:3px solid {accent}; border-radius:0 8px 8px 0;"
                  if accent else "")
    head_color = "#5A7BAA"
    return (
        f'<div style="background:{bg}; border:0.5px solid {border}; {accent_css}'
        f' border-radius:8px; padding:9px 11px; margin-bottom:8px; overflow:visible;">'
        f'<div style="display:flex; align-items:center; justify-content:space-between;'
        f' gap:6px; font-size:10px; font-weight:500; color:{head_color};'
        f' text-transform:uppercase; letter-spacing:0.04em;'
        f' border-bottom:0.5px solid rgba(16,55,102,0.09); padding-bottom:5px; margin-bottom:6px;">'
        f'<span>{heading}</span>{info}</div>'
        f'{inner_html}</div>'
    )


# CSS injected once per tab render — powers the info-hover tooltips.
_MACRO_TIP_CSS = """
<style>
.mt-info{position:relative; display:inline-flex; cursor:help; color:#85B7EB; font-size:12px; flex-shrink:0;}
.mt-info .mt-tip{visibility:hidden; opacity:0; transition:opacity .12s; position:absolute; right:0; top:18px;
 width:215px; background:#103766; color:#EAF1FB; font-size:11px; font-weight:400; line-height:1.45;
 text-transform:none; letter-spacing:0; padding:8px 10px; border-radius:7px; z-index:50;
 box-shadow:0 2px 10px rgba(16,55,102,.25); text-align:left;}
.mt-info .mt-tip b{color:#fff; font-weight:500;}
.mt-info:hover .mt-tip, .mt-info:focus .mt-tip{visibility:visible; opacity:1;}
.mt-group{font-size:11px; font-weight:500; color:#103766; margin:12px 0 6px;}
.mt-group span{color:#5A7BAA; font-weight:400;}
</style>
"""


def _render_layer0_2_tab(l0: dict, fred_data: dict, rec_indicators: list,
                         rec_flags: int, rec_total: int, perm: str,
                         limits: dict, l3_data: list, regime: str = "") -> None:
    """Render the combined Layer 0 & 2 — Macro Regime + Permission State tab.

    Layout (redesigned 2026-06-07): verdict hero + ranked tiles on top, then
    cards grouped by framework role — "what sets the state" (gate-deciding),
    "what scales the risk" (sizing/entry modifiers), "where to look" (sector
    reference). No visible layer codes; each card carries an info-hover that
    explains its framework tie. See Dashboards/UX_Audit_2026-06-07.md."""

    roc1_pos      = l0["spy_ret_1m"] > 0
    roc6_pos      = l0["spy_ret_6m"] > 0
    spy_signal    = "GREEN" if (roc1_pos and roc6_pos) else ("RED" if (not roc1_pos and not roc6_pos) else "MIXED")
    liq_override  = bool(l0.get("liquidity_tighten")) or l0.get("fnl_signal") == "OVERRIDE ACTIVE"
    tlt_dir       = l0.get("tlt_direction", "N/A")
    vix_v         = l0.get("vix")
    hyg_r         = l0.get("hyg_ief_ratio")
    eps_signal    = st.session_state.eps_signal
    eps_declining = "Declining" in eps_signal

    # ── Build Sector RS table ─────────────────────────────────────────────────
    sr = l0.get("sector_rs", {})
    if sr:
        sr_rows = []
        for sec, v in sr.items():
            emoji = "🟢" if v["trend"] == "Leading" else "🟡" if v["trend"] == "Mixed" else "🔴"
            sr_rows.append({
                "Sector": sec, "ETF": v["etf"],
                "Price":  f"${v['price']:.2f}",
                "1M RS":  pct(v["rs_1m"]),
                "3M RS":  pct(v["rs_3m"]),
                "RS Hi":  icon(v["rs_new_hi"]),
                "Status": f"{emoji} {v['trend']}",
                "_sort":  v["rs_3m"],
            })
        sr_df = pd.DataFrame(sr_rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
        sector_rs_html = cb_table(sr_df, bordered=False)
    else:
        sector_rs_html = "<p style='color:#5A7BAA; font-size:13px;'>Sector data unavailable.</p>"

    # ── Build Velocity Flag table (v4) ───────────────────────────────────────
    if sr:
        vf_rows = []
        for sec, v in sorted(sr.items(), key=lambda x: x[1].get("roc_21", 0), reverse=True):
            vel = v.get("velocity", "N/A")
            roc = v.get("roc_21", 0)
            vel_icon = "🔥" if vel == "ACCELERATING" else ("✅" if vel == "NORMAL" else "⬜")
            vf_rows.append({
                "Sector": sec, "ETF": v["etf"],
                "ROC 21": pct(roc),
                "Status": f"{vel_icon} {vel}",
            })
        velocity_html = cb_table(pd.DataFrame(vf_rows), bordered=False)
        accel_list = l0.get("accelerating", [])
        if accel_list:
            velocity_html += (f'<p style="font-size:11px; color:#CC1111; font-weight:500; margin-top:6px;">'
                              f'Accelerating Protocol eligible: {", ".join(accel_list)}</p>')
    else:
        velocity_html = "<p style='color:#5A7BAA; font-size:13px;'>Velocity data unavailable.</p>"

    # ── Build Bond & Liquidity table ──────────────────────────────────────────
    bond_rows = [
        {"Signal": "TLT Direction (4-week)",
         "Value": f"{tlt_dir}  ({pct(l0['tlt_ret_1m'])} 1M)" if l0.get("tlt_ret_1m") is not None else tlt_dir},
        {"Signal": "TLT vs 50d MA",
         "Value": ("✅ Above" if l0.get("tlt_above_50") else "❌ Below") if l0.get("tlt_above_50") is not None else "N/A"},
        {"Signal": "Bond/SPY Regime Signal",
         "Value": l0.get("tlt_spy_signal", "N/A")},
        {"Signal": "HYG/IEF Credit Spread",
         "Value": f"{hyg_r:.3f}  ({'⬇️ Tightening' if l0.get('liquidity_tighten') else '✅ Stable'})" if hyg_r else "N/A"},
        {"Signal": "VIX",
         "Value": f"{vix_v:.1f}  ({'⚠️ Elevated' if l0.get('vix_elevated') else '✅ Normal'})" if vix_v else "N/A"},
        {"Signal": f"Fed Net Liquidity ({l0.get('fnl_as_of', '')})",
         "Value": (
             f"${l0['fnl_current']:,.1f}B  |  4-week: ${l0['fnl_change_4w']:+.1f}B  |  "
             f"{'🔴 OVERRIDE ACTIVE' if l0['fnl_signal'] == 'OVERRIDE ACTIVE' else ('⚠️ Declining' if l0['fnl_signal'] == 'DECLINING' else '✅ Rising')}"
         ) if l0.get("fnl_current") is not None else "⚪ Unavailable — data pull failed, retry"},
    ]
    bond_html = cb_table(pd.DataFrame(bond_rows), bordered=False)

    # ── Build Sector Flow preview table ──────────────────────────────────────
    if l3_data:
        improving = [d for d in l3_data if d["quadrant"] == "Improving"]
        leading   = [d for d in l3_data if d["quadrant"] == "Leading"]
        flow_rows = [
            {"Phase": "🔵 Phase 1 — Early",     "ETFs": ", ".join(d["etf"] for d in improving) or "None"},
            {"Phase": "🟢 Phase 2 — Confirmed", "ETFs": ", ".join(d["etf"] for d in leading)   or "None"},
        ]
        flow_html = cb_table(pd.DataFrame(flow_rows), bordered=False)
        flow_html += '<p style="font-size:11px; color:#5A7BAA; margin-top:6px;">Full detail in the Layer 3 tab.</p>'
    else:
        flow_html = "<p style='color:#5A7BAA; font-size:13px;'>RRG data unavailable.</p>"

    # ── Build SPY Two-Speed table ─────────────────────────────────────────────
    gate_status = ("✅ Both positive — gate open" if spy_signal == "GREEN"
                   else "❌ Both negative — Red state" if spy_signal == "RED"
                   else "⚠️ Mixed — Yellow pressure")
    spy_rows = [
        {"Signal": "ROC 21 (1-Month)",  "Value": pct(l0["spy_ret_1m"]),  "Status": "✅ Positive" if roc1_pos else "❌ Negative"},
        {"Signal": "ROC 126 (6-Month)", "Value": pct(l0["spy_ret_6m"]),  "Status": "✅ Positive" if roc6_pos else "❌ Negative"},
        {"Signal": "Gate",              "Value": "",                       "Status": gate_status},
    ]
    spy_html = cb_table(pd.DataFrame(spy_rows), bordered=False)

    # ── Build Recession Composite table ──────────────────────────────────────
    if "error" in fred_data:
        rec_html = f"<p style='color:#5A7BAA; font-size:13px;'>{fred_data['error']}</p>"
    else:
        rec_rows = [{
            "Indicator": ind["name"],
            "Value":     ind["value"],
            "As of":     ind["as_of"],
            "Threshold": ind["threshold"],
            "Status":    "✅ OK" if ind["ok"] else "⚠️ FLAG",
        } for ind in rec_indicators]
        flag_text = ("Clear — full risk operations." if rec_flags == 0
                     else f"{rec_flags}/{rec_total} indicator(s) flagging.")
        flag_color_css = "#27500A" if rec_flags == 0 else ("#E07800" if rec_flags <= 2 else "#CC1111")
        rec_html = (cb_table(pd.DataFrame(rec_rows), bordered=False)
                    + f'<p style="font-size:11px; color:{flag_color_css}; font-weight:500; margin-top:6px;">{flag_text}</p>')

    # ── Build Gate Summary table ──────────────────────────────────────────────
    gate_rows = [
        {"Gate": "SPY both positive",  "Status": "✅ Open" if (roc1_pos and roc6_pos) else ("❌ Closed" if spy_signal == "RED" else "⚠️ Mixed"),    "Effect": "Required for Green"},
        {"Gate": "Liquidity override", "Status": "🔴 ACTIVE" if liq_override else "✅ Clear",                                                        "Effect": "Forces Red if active"},
        {"Gate": "Recession composite","Status": "🔴 Critical" if rec_flags >= 4 else ("⚠️ Elevated" if rec_flags >= 2 else "✅ Clear"),             "Effect": "≥4 flags → Red; 2–3 → Yellow"},
        {"Gate": "EPS Revisions",      "Status": f"{'⚠️' if eps_declining else '✅'} {eps_signal}",                                                  "Effect": "Declining → Yellow pressure"},
        {"Gate": "Taylor Rule",        "Status": st.session_state.taylor_rule,                                                                        "Effect": "Informational"},
        {"Gate": "Drawdown from Peak", "Status": st.session_state.drawdown_state,                                                                     "Effect": "Informs risk scaling"},
    ]
    gate_sum_html = cb_table(pd.DataFrame(gate_rows), bordered=False)

    # ── Verdict hero + ranked tiles ───────────────────────────────────────────
    # Palette per permission state (bg, border, accent, text).
    _HERO = {
        "Green":  ("#D6F0D6", "rgba(29,122,42,.35)", "#27500A", "#173404", "#3B6D11"),
        "Yellow": ("#FFF3D6", "rgba(224,120,0,.35)", "#854F0B", "#412402", "#854F0B"),
        "Red":    ("#FFE4E4", "rgba(204,17,17,.35)", "#A32D2D", "#501313", "#A32D2D"),
    }
    h_bg, h_bd, h_lbl, h_val, h_sub = _HERO.get(perm, _HERO["Green"])
    risk_str = (f"{limits['risk_lo']}–{limits['risk_hi']}% risk"
                if limits.get("risk_hi", 0) > 0 else "No new trades")
    core_target = {"Green": 40, "Yellow": 20, "Red": 0}.get(perm, 0)
    rule_str = (f"Up to {limits['max_pos_label']} positions · {risk_str} · "
                f"Core {core_target}%")

    # Gate one-liner — the three inputs that decided the state.
    spy_ok = "✓" if spy_signal == "GREEN" else ("✗" if spy_signal == "RED" else "~")
    liq_ok = "✗" if liq_override else "✓"
    rec_ok = "✓" if rec_flags < 2 else ("~" if rec_flags < 4 else "✗")
    gate_line = (f"SPY both positive {spy_ok} · Liquidity clear {liq_ok} · "
                 f"Recession {rec_flags}/{rec_total} {rec_ok}")

    regime_lbl = regime or l0.get("regime", "")
    spy_tile_color = "#27500A" if l0["spy_ret_1m"] > 0 else "#CC1111"
    dd_pct   = st.session_state.get("drawdown_pct", 0.0)
    dd_label = st.session_state.get("drawdown_label", "At peak")
    dd_color = st.session_state.get("drawdown_color", "#27500A")
    core_pct = st.session_state.core_pct_deployed

    hero_html = (
        '<div style="display:grid; grid-template-columns:1.3fr 1fr; gap:9px; margin-bottom:4px;">'
        f'<div style="background:{h_bg}; border:1px solid {h_bd}; border-radius:10px; padding:11px 13px;">'
        f'<div style="font-size:13px; font-weight:500; color:{h_lbl}; letter-spacing:.01em;'
        f' margin-bottom:4px;">Permission state — {regime_lbl} regime</div>'
        f'<div style="font-size:30px; font-weight:500; color:{h_val}; line-height:1.1; margin:3px 0 2px;">'
        f'{perm.upper()}</div>'
        f'<div style="font-size:12px; color:{h_lbl}; font-weight:500;">{rule_str}</div>'
        f'<div style="font-size:11px; color:{h_sub}; margin-top:5px;">Gate: {gate_line}</div>'
        '</div>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">'
        + _tile("SPY trend", f"{pct(l0['spy_ret_1m'])} / {pct(l0['spy_ret_6m'])}", "", spy_tile_color)
        + _tile("Drawdown", f"{dd_pct:+.1f}%", dd_label, dd_color)
        + _tile("Core deployed", f"{core_pct:.0f}%", f"Target {core_target}%")
        + _tile("Max heat", f"{limits['heat']}%", "")
        + '</div>'
        '</div>'
    )

    # ── Grouped, framework-honest card layout ─────────────────────────────────
    # Accent colors: green = gate-deciding & clear; amber = watch / modifier.
    GATE_GREEN, MOD_AMBER = "#1D7A2A", "#E07800"
    liq_accent = MOD_AMBER if not liq_override else "#CC1111"

    full_html = (
        _MACRO_TIP_CSS
        + hero_html

        # ── What sets the state ───────────────────────────────────────────
        + '<div class="mt-group">What sets the state '
          '<span>— these decide green, yellow, or red</span></div>'
        + '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;">'
        + _macro_card("SPY Two-Speed Trend", spy_html, "spy", accent=GATE_GREEN)
        + _macro_card("Recession Composite", rec_html, "recession", accent=GATE_GREEN)
        + _macro_card("Bond &amp; Liquidity", bond_html, "liquidity", accent=liq_accent)
        + '</div>'

        # ── What scales the risk ──────────────────────────────────────────
        + '<div class="mt-group">What scales the risk '
          '<span>— these adjust sizing &amp; entry inside the state</span></div>'
        + '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
        + _macro_card("Monthly Mispricing", _layer1_body_html(), "mispricing", accent=MOD_AMBER)
        + _macro_card("Velocity Flag", velocity_html, "velocity", accent=MOD_AMBER)
        + '</div>'

        # ── Where to look ─────────────────────────────────────────────────
        + '<div class="mt-group">Where to look '
          '<span>— sector context for stock selection</span></div>'
        + '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
        + _macro_card("Sector Relative Strength", sector_rs_html, "sectorrs", quiet=True)
        + _macro_card("Sector Flow Momentum", flow_html, "flow", quiet=True)
        + '</div>'

        # ── Full gate detail (reference) ──────────────────────────────────
        + '<div class="mt-group">Gate detail '
          '<span>— every input and its effect on the state</span></div>'
        + _macro_card("Gate Summary", gate_sum_html, "", quiet=True)
    )
    st.markdown(full_html, unsafe_allow_html=True)


def _render_layer3_tab(l3_data: list) -> None:
    """
    Render the Sector Rotation tab.
    Where-to-act hero (tickers by action) + quadrant counts, then the RRG chart
    as centerpiece, then entry candidates and exit-watch as quiet reference.
    Flow-strength and All-Sectors detail live in expanders.
    """
    # ── Quadrant membership (drives the hero + counts) ────────────────────────
    leading_t   = [d["etf"] for d in l3_data if d["quadrant"] == "Leading"]
    improving_t = [d["etf"] for d in l3_data if d["quadrant"] == "Improving"]
    weakening_t = [d["etf"] for d in l3_data if d["quadrant"] == "Weakening"]
    n_leading, n_improving = len(leading_t), len(improving_t)
    n_weakening = len(weakening_t)
    n_lagging   = sum(1 for d in l3_data if d["quadrant"] == "Lagging")

    def _act_row(dot, label, tickers):
        names = ", ".join(tickers) if tickers else "None"
        return (f'<div><span style="color:{dot}">●</span> '
                f'<strong style="font-weight:500">{label}:</strong> {names}</div>')

    # Hero recolors: green if anything Leading, amber if only weakening to act on.
    if n_leading > 0:
        h_bg, h_bd, h_lbl, h_txt = "#D6F0D6", "rgba(29,122,42,.35)", "#27500A", "#173404"
    elif n_improving > 0:
        h_bg, h_bd, h_lbl, h_txt = "#E6F1FB", "rgba(40,140,250,.35)", "#185FA5", "#042C53"
    else:
        h_bg, h_bd, h_lbl, h_txt = "#FFF3D6", "rgba(224,120,0,.35)", "#854F0B", "#412402"

    rotation_hero = (
        '<div style="display:grid; grid-template-columns:1.3fr 1fr; gap:9px; margin-bottom:4px;">'
        f'<div style="background:{h_bg}; border:1px solid {h_bd}; border-radius:10px; padding:11px 13px;">'
        f'<div style="font-size:13px; font-weight:500; color:{h_lbl}; letter-spacing:.01em;'
        f' margin-bottom:4px;">Where to act this week</div>'
        f'<div style="font-size:12px; color:{h_txt}; margin-top:6px; line-height:1.7;">'
        + _act_row("#27500A", "Enter / hold (Leading)", leading_t)
        + _act_row("#288CFA", "Watch (Improving)", improving_t)
        + _act_row("#E07800", "Review stops (Weakening)", weakening_t)
        + '</div></div>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">'
        + _tile("Leading", str(n_leading), "Confirmed", "#27500A")
        + _tile("Improving", str(n_improving), "Early", "#288CFA")
        + _tile("Weakening", str(n_weakening), "Exiting", "#E07800")
        + _tile("Lagging", str(n_lagging), "No trade", "#CC1111")
        + '</div>'
        '</div>'
    )
    st.markdown(_MACRO_TIP_CSS + rotation_hero, unsafe_allow_html=True)

    # ── The rotation graph (centerpiece) ──────────────────────────────────────
    st.markdown('<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
                'The rotation <span style="font-weight:400; color:#5A7BAA;">'
                '— sector ETFs vs SPY, trailing 8 weeks · clockwise = normal cycle progression</span></div>',
                unsafe_allow_html=True)

    # Sector selector — a row of checkboxes (cleaner than the dropdown).
    st.markdown('<div style="font-size:11px; color:#5A7BAA; margin-bottom:2px;">Show sectors</div>',
                unsafe_allow_html=True)
    _sel_cols = st.columns(5)
    chart_sectors = []
    for i, sec in enumerate(ALL_SECTORS):
        with _sel_cols[i % 5]:
            if st.checkbox(sec, value=True, key=f"rrg_show_{sec}"):
                chart_sectors.append(sec)
    chart_data = [d for d in l3_data if d["sector"] in chart_sectors] if chart_sectors else l3_data

    with st.spinner("Building RRG chart..."):
        rrg_fig = build_rrg_chart(chart_data)
    if rrg_fig:
        st.plotly_chart(rrg_fig, use_container_width=True)
    else:
        st.markdown(_gate_bar_html("Yellow", "Insufficient data to build RRG chart."), unsafe_allow_html=True)

    # Flow strength — auto-scraped from etfdb.com, feeds the Sizing/Risk columns
    # below directly (no manual override panel; the data drives sizing on its own).
    auto_flows = fetch_etf_fund_flows()

    def _flow_of(etf: str) -> str:
        """Live flow-strength label for an ETF from the etfdb scrape."""
        return auto_flows.get(etf, {}).get("flow_strength", "Not set")

    _fl_as_of = next((auto_flows[e].get("as_of") for e in auto_flows
                      if auto_flows.get(e, {}).get("as_of")), "")
    _fl1, _fl2 = st.columns([4, 1])
    with _fl1:
        st.caption(f"Flow strength (Sizing / Risk columns) is live from etfdb.com"
                   + (f" · as of {_fl_as_of}" if _fl_as_of else ""))
    with _fl2:
        if st.button("🔄 Refresh flows", key="refresh_flows"):
            fetch_etf_fund_flows.clear()
            st.rerun()

    # ── ETF Entry Candidates / Weakening — two-column layout ────────────────
    candidates = [d for d in l3_data if d["quadrant"] in ("Improving", "Leading")]
    weakening = [d for d in l3_data if d["quadrant"] == "Weakening"]

    if candidates:
        cand_rows = []
        for d in candidates:
            flow_strength = _flow_of(d["etf"])
            if flow_strength in FLOW_SIZE_MAP:
                sizing, risk_pct, _ = FLOW_SIZE_MAP[flow_strength]
            else:
                sizing, risk_pct = d["sizing"], d["risk_pct"]
            q_icon = "🔵" if d["quadrant"] == "Improving" else "🟢"
            cand_rows.append({
                "ETF":         d["etf"],
                "Sector":      d["sector"],
                "Phase":       f"{q_icon} {d['phase']}",
                "RS Ratio":    d["rs_ratio"],
                "RS Mom":      d["rs_momentum"],
                "vs 20MA":     "✅" if d["above_20"] else "❌",
                "Price":       f"${d['price']:.2f}",
                "Stop (20MA)": f"${d['ma20']:.2f}",
                "Flow Signal": flow_strength,
                "Sizing":      sizing,
                "Risk %":      risk_pct,
            })
        cand_html = cb_table(pd.DataFrame(cand_rows), bordered=False)
    else:
        cand_html = '<p style="font-size:13px; color:#5A7BAA;">No Improving or Leading sectors currently.</p>'

    if weakening:
        weak_rows = []
        for d in weakening:
            flow_strength = _flow_of(d["etf"])
            weak_rows.append({
                "ETF":         d["etf"],
                "Sector":      d["sector"],
                "RS Ratio":    d["rs_ratio"],
                "RS Mom":      d["rs_momentum"],
                "vs 20MA":     "✅" if d["above_20"] else "❌",
                "Price":       f"${d['price']:.2f}",
                "Stop (20MA)": f"${d['ma20']:.2f}",
                "Flow Signal": flow_strength,
                "Action":      "Exit review — outflows" if flow_strength == "Outflows" else d["action"],
            })
        weak_html = cb_table(pd.DataFrame(weak_rows), bordered=False)
    else:
        weak_html = '<p style="font-size:13px; color:#27500A;">No sectors in Weakening quadrant.</p>'

    st.markdown(
        '<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
        'Entry candidates <span style="font-weight:400; color:#5A7BAA;">'
        '— Improving + Leading, with flow-based sizing</span></div>'
        + _macro_card(f"{len(candidates)} eligible", cand_html, "rot_candidates", quiet=True)
        + '<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
          'Exit watch <span style="font-weight:400; color:#5A7BAA;">'
          '— Weakening quadrant, review stops</span></div>'
        + _macro_card(f"{len(weakening)} sector{'s' if len(weakening) != 1 else ''}",
                      weak_html, "rot_exit", quiet=True),
        unsafe_allow_html=True,
    )

    # Full sector table
    st.markdown('<div style="margin-top:4px;"></div>', unsafe_allow_html=True)
    with st.expander("All Sectors", expanded=False):
        q_icons = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}
        all_rows = []
        for d in l3_data:
            flow_strength = _flow_of(d["etf"])
            if flow_strength in FLOW_SIZE_MAP:
                sizing, risk_pct, _ = FLOW_SIZE_MAP[flow_strength]
            else:
                sizing, risk_pct = d["sizing"], d["risk_pct"]
            all_rows.append({
                "Sector":      d["sector"],
                "ETF":         d["etf"],
                "Phase":       f"{q_icons[d['quadrant']]} {d['phase']}",
                "RS Ratio":    d["rs_ratio"],
                "RS Mom":      d["rs_momentum"],
                "Price":       f"${d['price']:.2f}",
                "vs 20MA":     "✅" if d["above_20"] else "❌",
                "Flow Signal": flow_strength,
                "Sizing":      sizing,
                "Action":      d["action"],
            })
        st.markdown(cb_table(pd.DataFrame(all_rows)), unsafe_allow_html=True)


def _render_screener_data_warning(results_df) -> None:
    """Surface a warning bar if the screener's data fetch was incomplete.

    Reads diagnostics attached to the result DataFrame by run_screener_v3.
    Silent when every requested ticker came back (the normal case).
    """
    if results_df is None:
        return
    diag = results_df.attrs.get("screener_diagnostics", {})
    if not diag:
        return
    requested = diag.get("requested", 0)
    fetched   = diag.get("fetched", 0)
    failed    = diag.get("failed_batches", 0)

    # Warn if whole batches failed, or if we got back noticeably fewer names
    # than we asked for (a sign of throttling rather than a quiet tape).
    incomplete = requested > 0 and fetched < requested * 0.8
    if failed or incomplete:
        bits = []
        if failed:
            bits.append(f"{failed} download batch(es) failed")
        if incomplete:
            bits.append(f"only {fetched}/{requested} stocks returned data")
        st.markdown(
            _gate_bar_html(
                "Yellow",
                "Incomplete data — " + "; ".join(bits) +
                ". Results may be partial; wait a minute and re-run to refresh.",
            ),
            unsafe_allow_html=True,
        )


def _render_layer4_tab(perm: str, regime: str, l0: dict) -> None:
    """Render the Layer 4 — Screener tab (v3, button-triggered with 24h cache)."""

    if perm == "Red":
        st.markdown(
            _gate_bar_html("Red", "RED STATE — No new entries. Results shown for reference only."),
            unsafe_allow_html=True,
        )

    # ── Run Screener button ──────────────────────────────────────────────────
    leading = l0.get("leading_sectors", [])
    mixed   = l0.get("mixed_sectors", [])
    l0_sectors = leading + mixed
    screen_sectors = []
    for sec in l0_sectors:
        for mapped in SECTOR_TO_SCREEN.get(sec, []):
            if mapped not in screen_sectors:
                screen_sectors.append(mapped)
    if not screen_sectors:
        screen_sectors = list(SCREENER_UNIVERSE.keys())
    universe_size = sum(len(SCREENER_UNIVERSE.get(s, [])) for s in screen_sectors)

    # Show which L0 sectors are driving the scan
    leading_str = ", ".join(leading) if leading else "None"
    mixed_str   = ", ".join(mixed) if mixed else "None"

    col_btn, col_info, col_ts = st.columns([1, 3, 1])
    with col_btn:
        run_clicked = st.button("🔍 Run Screener", use_container_width=True, type="primary")
    with col_info:
        st.caption(
            f"L0 Leading: {leading_str} · Mixed: {mixed_str} → "
            f"Scanning: {', '.join(screen_sectors)} ({universe_size} stocks). Cached 24h."
        )
    with col_ts:
        last_run = st.session_state.get("screener_last_run")
        if last_run:
            cst_time = last_run.astimezone(ZoneInfo("America/Chicago"))
            st.caption(f"Last scanned: {cst_time.strftime('%b %d, %I:%M %p')} CST")
        else:
            st.caption("Not yet scanned")

    # Run or load cached
    if run_clicked:
        st.cache_data.clear()  # nuke all data caches
        _tbill = fetch_tbill_rate()
        with st.spinner(f"Scanning {universe_size} stocks across {len(screen_sectors)} sectors..."):
            accel_key = ",".join(l0.get("accelerating", []))
            rec_indicators = score_recession_composite(
                fetch_fred_data(), st.session_state.get("lei_signal", "Not set"))
            rec_flags_local = sum(1 for i in rec_indicators if not i["ok"])
            results_df = run_screener_v3(regime, leading, mixed, perm, accel_key, rec_flags_local, _tbill)
        st.session_state["screener_results"] = results_df
        st.session_state["screener_regime"] = regime
        st.session_state["screener_last_run"] = datetime.now(ZoneInfo("UTC"))
        n_pass = len(results_df[results_df["PASS"]]) if results_df is not None and not results_df.empty else 0
        st.toast(f"✅ Screener complete — {n_pass} passes found", icon="🔍")
    else:
        results_df = st.session_state.get("screener_results")
        # Try loading from cache on first visit
        if results_df is None:
            accel_key = ",".join(l0.get("accelerating", []))
            rec_indicators = score_recession_composite(
                fetch_fred_data(), st.session_state.get("lei_signal", "Not set"))
            rec_flags_local = sum(1 for i in rec_indicators if not i["ok"])
            _tbill2 = fetch_tbill_rate()
            results_df = run_screener_v3(regime, leading, mixed, perm, accel_key, rec_flags_local, _tbill2)
            if results_df is not None and not results_df.empty:
                st.session_state["screener_results"] = results_df
                st.session_state["screener_regime"] = regime
                st.session_state["screener_last_run"] = datetime.now(ZoneInfo("UTC"))

    _render_screener_data_warning(results_df)

    if results_df is None or results_df.empty:
        # An empty frame can mean "data fetch failed" or "ran fine, nothing passed".
        diag = results_df.attrs.get("screener_diagnostics", {}) if results_df is not None else {}
        if diag.get("spy_missing") or diag.get("failed_batches"):
            msg = "Screener could not fetch market data (likely a rate limit). Wait a minute and re-run."
        elif diag:
            msg = "Screener ran, but no candidates passed the filters in the scanned sectors."
        else:
            msg = "Click <b>Run Screener</b> to scan the universe. First run takes ~30 seconds."
        st.markdown(
            f'<p style="font-size:14px; color:#5A7BAA; text-align:center; padding:40px;">{msg}</p>',
            unsafe_allow_html=True,
        )
        return

    # Check if regime changed since last run
    cached_regime = st.session_state.get("screener_regime", "")
    if cached_regime and cached_regime != regime:
        st.markdown(
            _gate_bar_html("Yellow", f"Screener was run for {cached_regime} regime. Current regime is {regime}. Re-run to update."),
            unsafe_allow_html=True,
        )

    # ── Split into passes / half ─────────────────────────────────────────────
    passes_df = results_df[results_df["PASS"]].copy()
    half_df   = results_df[(results_df["2-Speed"] == "HALF") & (~results_df["PASS"])].copy()
    no_trade  = len(results_df) - len(passes_df) - len(half_df)

    # Store for L5 consumption
    st.session_state["screener_passes"] = passes_df
    st.session_state["screener_half"]   = half_df

    # ── Stats tiles ──────────────────────────────────────────────────────────
    full_ready = int((passes_df["Verdict"] == "🟢 ENTRY READY").sum()) if not passes_df.empty else 0
    full_watch = int((passes_df["Verdict"] == "🟡 WATCH").sum()) if not passes_df.empty else 0
    half_ready_watch = int(half_df["Verdict"].isin(["🟢 ENTRY READY", "🟡 WATCH"]).sum()) if not half_df.empty else 0
    stats_html = (
        f'<div style="background:#FFFFFF; border-radius:12px; border:0.5px solid rgba(16,55,102,0.12);'
        f' padding:15px 17px; margin-bottom:10px;">'
        f'<div style="font-size:11px; color:#5A7BAA; margin-bottom:10px;">'
        f'Sectors: {", ".join(screen_sectors)} &nbsp;·&nbsp; Scanned: {len(results_df)} passing L4 filters</div>'
        f'<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:9px;">'
        + _tile("Entry Ready", str(full_ready), "Full signal + trigger confirmed", "#27500A")
        + _tile("Watch", str(full_watch), "Full signal, trigger pending", "#E07800")
        + _tile("Half Signal", str(half_ready_watch), "Mixed two-speed, actionable", "#288CFA")
        + _tile("Scanned", str(len(results_df)), "Total passing L4 filters")
        + '</div></div>'
    )
    st.markdown(stats_html, unsafe_allow_html=True)

    # ── Helper: render a selectable table with checkboxes ────────────────────
    def _selectable_table(df_raw, display_cols, section_key, heading, pill=""):
        """
        Render a table with checkbox selection (st.data_editor). Selected tickers
        are stored in session state and pushed to the Position Sizer tab.
        df_raw must have Ticker, Entry, Stop, Trigger columns for sizing.
        display_cols is a dict of {display_name: series}.
        """
        disp = pd.DataFrame(display_cols)
        disp.insert(0, "Size", False)
        disp = disp.reset_index(drop=True)

        st.markdown(
            f'<div style="background:#FFFFFF; border-radius:12px; border:0.5px solid rgba(16,55,102,0.12);'
            f' padding:15px 17px; margin-bottom:4px;">'
            f'<div style="display:flex; align-items:center; justify-content:space-between;'
            f' margin-bottom:10px; padding-bottom:7px; border-bottom:0.5px solid rgba(16,55,102,0.09);">'
            f'<span style="font-size:11px; font-weight:500; color:#5A7BAA; text-transform:uppercase;'
            f' letter-spacing:0.04em;">{heading}</span>'
            + (f'<span style="background:#EEF3FA; color:#5A7BAA; font-size:10px; font-weight:500;'
               f' padding:2px 8px; border-radius:4px; border:0.5px solid rgba(16,55,102,0.15);">{pill}</span>'
               if pill else "")
            + f'</div></div>',
            unsafe_allow_html=True,
        )

        edited = st.data_editor(
            disp,
            column_config={
                "Size": st.column_config.CheckboxColumn("Size", default=False, width="small"),
            },
            column_order=list(disp.columns),
            disabled=[c for c in disp.columns if c != "Size"],
            hide_index=True,
            use_container_width=True,
            key=f"sel3_{section_key}",
        )

        # Collect selected tickers and push to session state
        if edited is not None and "Size" in edited.columns:
            selected_rows = edited[edited["Size"] == True]
            if not selected_rows.empty:
                selected_tickers = selected_rows["Ticker"].tolist()
                existing = st.session_state.get("sizer_queue", [])
                for t in selected_tickers:
                    match = df_raw[df_raw["Ticker"] == t]
                    if not match.empty:
                        row = match.iloc[0]
                        entry = {
                            "ticker": t,
                            "entry": float(row["Entry"]) if pd.notna(row.get("Entry")) else 0,
                            "stop": float(row["Stop"]) if pd.notna(row.get("Stop")) else 0,
                            "trigger": row.get("Trigger", "Breakout"),
                            "price": float(row["Price"]),
                            "sector": row.get("Sector", ""),
                            "verdict": row.get("Verdict", ""),
                            "carry_spread": float(row["Carry_Spread"]) if "Carry_Spread" in row.index and pd.notna(row.get("Carry_Spread")) else None,
                            "carry_label": row.get("Carry_Label", "N/A") if "Carry_Label" in row.index else "N/A",
                        }
                        if not any(e["ticker"] == t for e in existing):
                            existing.append(entry)
                st.session_state["sizer_queue"] = existing

        st.markdown('<div style="margin-bottom:10px"></div>', unsafe_allow_html=True)

    # Initialize sizer queue
    if "sizer_queue" not in st.session_state:
        st.session_state["sizer_queue"] = []

    # Clear queue button
    if st.session_state.get("sizer_queue"):
        queue_tickers = [e["ticker"] for e in st.session_state["sizer_queue"]]
        st.markdown(
            _gate_bar_html("Green", f"Sizing queue: {', '.join(queue_tickers)} → go to Position Sizer tab"),
            unsafe_allow_html=True,
        )
        if st.button("Clear sizing queue", key="clear_queue"):
            st.session_state["sizer_queue"] = []
            st.rerun()

    # ── Standard column set for both tables ────────────────────────────────
    def _std_cols(df):
        return {
            "Ticker":     df["Ticker"].values,
            "Sector":     df["Sector"].values,
            "Price":      df["Price"].apply(lambda x: f"${x:.2f}").values,
            "Trigger":    df["Trigger"].values,
            "Entry":      df["Entry"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—").values,
            "Stop":       df["Stop"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—").values,
            "Verdict":    df["Verdict"].values,
            "Carry":      (df["Carry_Spread"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—") if "Carry_Spread" in df.columns else pd.Series(["—"] * len(df))).values,
            "RS vs SPY":  df["RS_vs_SPY_21d"].apply(lambda x: f"+{x:.1f}%").values,
            "vs 20MA":    df["Dist_MA20_pct"].apply(lambda x: f"+{x:.1f}%").values,
            "Vol 5d":     df["Vol_Ratio_5d"].apply(lambda x: f"{x:.1f}x").values,
            "MACD":       df["MACD_Crossover"].apply(lambda x: "✓" if x else "").values,
            "Notes":      df["Notes"].values,
        }

    # ── Table 1: Actionable Setups (Full signal, ENTRY READY + WATCH) ─────
    actionable_mask = (
        passes_df["Verdict"].isin(["🟢 ENTRY READY", "🟡 WATCH"])
    ) if not passes_df.empty else pd.Series(dtype=bool)
    actionable_df = passes_df[actionable_mask].copy() if not passes_df.empty else pd.DataFrame()

    if not actionable_df.empty:
        # Sort: ENTRY READY first, then by RS strength
        actionable_df["_sort"] = actionable_df["Verdict"].map({"🟢 ENTRY READY": 0, "🟡 WATCH": 1}).fillna(2)
        actionable_df = actionable_df.sort_values(["_sort", "RS_vs_SPY_21d"], ascending=[True, False]).drop(columns="_sort")
        actionable_show = actionable_df.head(50)

        ready_n = int((actionable_show["Verdict"] == "🟢 ENTRY READY").sum())
        watch_n = int((actionable_show["Verdict"] == "🟡 WATCH").sum())
        showing = f"top 50 of {len(actionable_df)}" if len(actionable_df) > 50 else str(len(actionable_df))

        _selectable_table(
            actionable_show,
            _std_cols(actionable_show),
            "actionable",
            f"Actionable Setups — Full Signal — {showing}",
            pill=f"✅ {ready_n} ready · 🟡 {watch_n} watch",
        )
        # ticker copy box removed per user request
    else:
        st.markdown(
            '<p style="font-size:13px; color:#5A7BAA; text-align:center; padding:20px;">'
            'No actionable full-signal setups in this scan.</p>',
            unsafe_allow_html=True,
        )

    # ── Table 2: Watchlist (Half signal, ENTRY READY + WATCH) ─────────────
    watchlist_mask = (
        half_df["Verdict"].isin(["🟢 ENTRY READY", "🟡 WATCH"])
    ) if not half_df.empty else pd.Series(dtype=bool)
    watchlist_df = half_df[watchlist_mask].copy() if not half_df.empty else pd.DataFrame()

    if not watchlist_df.empty:
        watchlist_df["_sort"] = watchlist_df["Verdict"].map({"🟢 ENTRY READY": 0, "🟡 WATCH": 1}).fillna(2)
        watchlist_df = watchlist_df.sort_values(["_sort", "RS_vs_SPY_21d"], ascending=[True, False]).drop(columns="_sort")
        watchlist_show = watchlist_df.head(20)

        _selectable_table(
            watchlist_show,
            _std_cols(watchlist_show),
            "watchlist",
            f"Watchlist — Half Signal — {len(watchlist_df)} candidates (half-size eligible)",
            pill="⚠️ Half",
        )

    # ── Table 3: Monitoring (strong fundamentals but too extended to trade) ──
    monitoring_df = results_df[results_df.get("MONITORING", pd.Series(dtype=bool)) == True].copy() if "MONITORING" in results_df.columns else pd.DataFrame()

    if not monitoring_df.empty:
        monitoring_df = monitoring_df.sort_values("RS_vs_SPY_21d", ascending=False).head(25)
        mon_cols = {
            "Ticker":     monitoring_df["Ticker"].values,
            "Sector":     monitoring_df["Sector"].values,
            "Price":      monitoring_df["Price"].apply(lambda x: f"${x:.2f}").values,
            "vs 20MA":    monitoring_df["Dist_MA20_pct"].apply(lambda x: f"+{x:.1f}%").values,
            "Entry Zone": monitoring_df["Entry_Zone"].values,
            "RS vs SPY":  monitoring_df["RS_vs_SPY_21d"].apply(lambda x: f"+{x:.1f}%").values,
            "Vol 5d":     monitoring_df["Vol_Ratio_5d"].apply(lambda x: f"{x:.1f}x").values,
            "MACD":       monitoring_df["MACD_Crossover"].apply(lambda x: "✓" if x else "").values,
            "Notes":      monitoring_df["Notes"].values,
        }
        st.markdown(
            _card(
                f"Monitoring Only — {len(monitoring_df)} extended stocks (do not enter)",
                cb_table(pd.DataFrame(mon_cols), bordered=False),
                pill="⚠️ Too extended",
            ),
            unsafe_allow_html=True,
        )


def _render_position_sizer_tab(
    candidates_df: pd.DataFrame,
    perm: str,
    l0: dict,
) -> None:
    """
    Render the Position Sizer tab — standalone.
    Reads selected ticker from screener or allows custom entry.
    Computes shares, risk, and shows profit targets.
    """
    st.markdown(
        _card(
            "Layer 6 — Position Sizer",
            '<p style="font-size:12px; color:#5A7BAA; margin:0;">'
            'Select a candidate from the screener or enter custom values. '
            'Risk % auto-adjusts to permission state and drawdown tier.</p>',
            pill="v4",
        ),
        unsafe_allow_html=True,
    )

    account = st.session_state.account_value
    drawdown_state = st.session_state.drawdown_state

    # ── Build candidate list from sizer_queue (selected on Screener tab) ─────
    queue = st.session_state.get("sizer_queue", [])
    candidates = {}

    # Only show queued tickers (selected via checkboxes on Screener tab)
    for item in queue:
        label = f"{item['ticker']}  —  {item.get('verdict', '')}  ({item.get('trigger', 'Breakout')})  ${item.get('price', 0):.2f}"
        candidates[label] = item

    st.session_state["_sizer_candidates"] = candidates

    def _on_candidate_change():
        sel = st.session_state.get("sizer_select", "Custom entry")
        cands = st.session_state.get("_sizer_candidates", {})
        if sel == "Custom entry":
            st.session_state["sizer_entry"] = 0.0
            st.session_state["sizer_stop"] = 0.0
            st.session_state["sizer_live_quote"] = None
        elif sel in cands:
            c = cands[sel]
            trigger_opts = ["Breakout", "Pullback", "Accelerating"]
            if c.get("trigger") in trigger_opts:
                st.session_state["sizer_trigger"] = c["trigger"]
            trigger = c.get("trigger", "Breakout")

            # Fetch live data and compute entry/stop per framework L5 rules
            try:
                tkr = yf.Ticker(c["ticker"])
                hist = tkr.history(period="6mo")
                close = hist["Close"].squeeze()
                volume = hist["Volume"].squeeze()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                price = round(float(close.iloc[-1]), 2)
                ma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
                ma50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
                ema10 = round(float(close.ewm(span=10, adjust=False).mean().iloc[-1]), 2)

                # 6-week base range
                base_px = close.iloc[-30:]
                base_high = round(float(base_px.max()), 2)
                base_low = round(float(base_px.min()), 2)

                entry, stop = compute_entry_stop(trigger, {
                    "price": price, "ma20": ma20, "ema10": ema10,
                    "base_high": base_high, "base_low": base_low,
                })
                st.session_state["sizer_entry"] = entry
                st.session_state["sizer_stop"] = stop
                st.session_state["sizer_live_quote"] = {
                    "price": price,
                    "ticker": c["ticker"],
                    "ma20": ma20,
                    "ma50": ma50,
                    "ema10": ema10,
                    "base_high": base_high,
                    "base_low": base_low,
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Fallback to screener values
                st.session_state["sizer_entry"] = c.get("entry", 0.0)
                st.session_state["sizer_stop"] = c.get("stop", 0.0)
                st.session_state["sizer_live_quote"] = None

    if "sizer_entry" not in st.session_state:
        st.session_state["sizer_entry"] = 0.0
    if "sizer_stop" not in st.session_state:
        st.session_state["sizer_stop"] = 0.0

    # Show queue status
    if queue:
        queue_tickers = [e["ticker"] for e in queue]
        st.markdown(
            _gate_bar_html("Green", f"From screener: {', '.join(queue_tickers)}"),
            unsafe_allow_html=True,
        )

    options = ["Custom entry"] + list(candidates.keys())
    st.selectbox(
        "Select candidate",
        options,
        key="sizer_select",
        on_change=_on_candidate_change,
    )

    # ── Live quote & reference levels ────────────────────────────────────────
    live = st.session_state.get("sizer_live_quote")
    if live:
        sel_key = st.session_state.get("sizer_select", "Custom entry")
        cands_ref = st.session_state.get("_sizer_candidates", {})
        screener_px = cands_ref.get(sel_key, {}).get("price", 0)
        chg = live["price"] - screener_px if screener_px else 0
        chg_pct = (chg / screener_px * 100) if screener_px else 0
        chg_color = "#22C55E" if chg >= 0 else "#EF4444"
        ref_parts = []
        if live.get("ma20"):
            ref_parts.append(f'20d MA: ${live["ma20"]:.2f}')
        if live.get("ma50"):
            ref_parts.append(f'50d MA: ${live["ma50"]:.2f}')
        if live.get("ema10"):
            ref_parts.append(f'10d EMA: ${live["ema10"]:.2f}')
        if live.get("base_high"):
            ref_parts.append(f'Base: ${live["base_low"]:.2f}–${live["base_high"]:.2f}')
        ref_str = " · ".join(ref_parts)
        st.markdown(
            f'<p style="font-size:13px; margin:4px 0 2px 0;">'
            f'<b>{live["ticker"]}</b> live: <b>${live["price"]:.2f}</b> '
            f'<span style="color:{chg_color};">({chg:+.2f} / {chg_pct:+.1f}% vs screener)</span></p>'
            f'<p style="font-size:11px; color:#5A7BAA; margin:0 0 8px 0;">{ref_str}</p>',
            unsafe_allow_html=True,
        )

    # ── Recompute entry/stop when trigger type changes ──────────────────────
    def _on_trigger_change():
        live = st.session_state.get("sizer_live_quote")
        if not live:
            return
        trig = st.session_state.get("sizer_trigger", "Breakout")
        entry, stop = compute_entry_stop(trig, live)
        st.session_state["sizer_entry"] = entry
        st.session_state["sizer_stop"] = stop

    # ── Input columns ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        entry_px = st.number_input("Entry $", step=0.01, format="%.2f", key="sizer_entry")
    with c2:
        stop_px = st.number_input("Stop $", step=0.01, format="%.2f", key="sizer_stop")
    with c3:
        trigger_type = st.selectbox(
            "Trigger type",
            ["Breakout", "Pullback", "Accelerating"],
            key="sizer_trigger",
            on_change=_on_trigger_change,
        )
    with c4:
        acct_override = st.number_input("Account $", value=account, step=1000, format="%d", key="sizer_acct")

    # ── Compute sizing ────────────────────────────────────────────────────────
    if entry_px > 0 and stop_px > 0 and entry_px > stop_px:
        # Base risk % from permission state
        risk_pct_map = {"Green": 0.0075, "Yellow": 0.005, "Red": 0.0}
        base_risk_pct = risk_pct_map.get(perm, 0.0075)

        # Drawdown adjustment (Layer 9)
        if "7–10%" in drawdown_state or "Tier 2" in drawdown_state:
            drawdown_mult = 0.50
            dd_note = "Tier 2 drawdown — risk reduced 50%"
        elif "10–15%" in drawdown_state or "Tier 3" in drawdown_state:
            drawdown_mult = 0.0
            dd_note = "Tier 3 drawdown — no new Tactical entries"
        elif ">15%" in drawdown_state:
            drawdown_mult = 0.0
            dd_note = "Emergency — no new positions"
        else:
            drawdown_mult = 1.0
            dd_note = ""

        adj_risk_pct = base_risk_pct * drawdown_mult

        # Earnings carry adjustment (L4 Step 4): negative carry → reduce 25%
        carry_mult = 1.0
        carry_note = ""
        sel_key_carry = st.session_state.get("sizer_select", "Custom entry")
        cands_carry = st.session_state.get("_sizer_candidates", {})
        if sel_key_carry in cands_carry:
            c_carry = cands_carry[sel_key_carry]
            if c_carry.get("carry_label") == "Negative":
                carry_mult = 0.75
                carry_spread_val = c_carry.get("carry_spread", 0)
                carry_note = f"Negative carry ({carry_spread_val:+.1f}%) — size reduced 25%"

        adj_risk_pct = adj_risk_pct * carry_mult
        is_accel = trigger_type == "Accelerating"

        risk_dollars = round(acct_override * adj_risk_pct)
        risk_per_share = entry_px - stop_px
        shares = int(risk_dollars / risk_per_share) if risk_per_share > 0 else 0

        if is_accel:
            shares = shares // 2

        pos_value = round(shares * entry_px)
        max_pos_value = round(acct_override * 0.10)
        capped = False
        if pos_value > max_pos_value:
            shares = int(max_pos_value / entry_px)
            pos_value = round(shares * entry_px)
            capped = True

        actual_risk = round(shares * risk_per_share)
        actual_risk_pct = (actual_risk / acct_override * 100) if acct_override > 0 else 0

        # T1 / T2 targets
        if is_accel:
            t1_pct, t2_pct = 0.09, 0.175
            t1_label, t2_label = "+8-10%", "+15-20%"
            max_hold = "6 weeks"
        else:
            t1_pct, t2_pct = 0.10, 0.225
            t1_label, t2_label = "+8-12%", "+20-25%"
            max_hold = "12 weeks"

        t1_price = round(entry_px * (1 + t1_pct), 2)
        t2_price = round(entry_px * (1 + t2_pct), 2)
        breakeven_trigger = round(entry_px * 1.05, 2)

        # Build result rows
        result_rows = [
            {"Metric": "Shares",           "Value": f"{shares:,}"},
            {"Metric": "Position Value",   "Value": f"${pos_value:,}  ({pos_value/acct_override*100:.1f}% of account)"},
            {"Metric": "Risk Dollars",     "Value": f"${actual_risk:,}  ({actual_risk_pct:.2f}% of account)"},
            {"Metric": "Risk / Share",     "Value": f"${risk_per_share:.2f}"},
            {"Metric": "Risk %",           "Value": f"{adj_risk_pct*100:.3f}%  ({perm} state{', drawdown adjusted' if drawdown_mult < 1 else ''}{', carry adjusted' if carry_mult < 1 else ''})"},
        ]

        target_rows = [
            {"Level": "Breakeven stop",  "Price": f"${breakeven_trigger:.2f}", "Action": "Move stop to breakeven at +5%"},
            {"Level": f"T1 ({t1_label})", "Price": f"${t1_price:.2f}",         "Action": "Sell 1/3, stop → breakeven"},
            {"Level": f"T2 ({t2_label})", "Price": f"${t2_price:.2f}",         "Action": "Sell 1/3, trail at 20d MA"},
            {"Level": "Max hold",         "Price": max_hold,                    "Action": "Exit if insufficient progress"},
        ]

        # Sector concentration check (25% cap per framework L7)
        sector_conc_warn = ""
        sel_key = st.session_state.get("sizer_select", "Custom entry")
        cands_ref = st.session_state.get("_sizer_candidates", {})
        candidate_sector = cands_ref.get(sel_key, {}).get("sector", "")
        if candidate_sector:
            port_data = _portfolio_load()
            existing_in_sector = 0.0
            for p in port_data.get("open_positions", []):
                p_sec = ""
                for s, tickers in SECTOR_TICKERS.items():
                    if p["ticker"] in tickers:
                        p_sec = s
                        break
                # Also check SECTOR_ETFS for Core ETF positions
                if not p_sec:
                    for s, etf in SECTOR_ETFS.items():
                        if p["ticker"] == etf:
                            p_sec = s
                            break
                if p_sec == candidate_sector:
                    existing_in_sector += p["entry_price"] * p["shares"]
            new_total = existing_in_sector + pos_value
            sector_pct = new_total / acct_override * 100 if acct_override > 0 else 0
            if sector_pct > 25:
                sector_conc_warn = f"⚠️ Sector concentration: {candidate_sector} would be {sector_pct:.0f}% of account (${new_total:,.0f}) — exceeds 25% cap"

        # Warnings
        warnings = []
        if is_accel:
            warnings.append("🔥 Accelerating Protocol — half-sized, 10d EMA stop, 6-week hold")
        if capped:
            warnings.append(f"⚠️ Position capped at 10% of account (${max_pos_value:,})")
        if sector_conc_warn:
            warnings.append(sector_conc_warn)
        if carry_note:
            warnings.append(f"📉 {carry_note}")
        if dd_note:
            warnings.append(f"⚠️ {dd_note}")
        if perm == "Red":
            warnings.append("❌ RED state — no new positions allowed")

        warn_html = ""
        if warnings:
            warn_html = '<div style="margin-bottom:8px;">' + "".join(
                f'<p style="font-size:12px; font-weight:500; color:#CC1111; margin:2px 0;">{w}</p>'
                for w in warnings
            ) + '</div>'

        sizing_html = warn_html + cb_table(pd.DataFrame(result_rows), bordered=False)
        targets_html = cb_table(pd.DataFrame(target_rows), bordered=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                _card("Position Size", sizing_html, pill=f"{shares:,} shares"),
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                _card("Profit Targets & Management", targets_html, pill=trigger_type),
                unsafe_allow_html=True,
            )
    else:
        if entry_px > 0 and stop_px > 0 and entry_px <= stop_px:
            st.warning("Entry price must be greater than stop price.")
        else:
            st.markdown(
                '<p style="font-size:13px; color:#5A7BAA; text-align:center; padding:20px;">'
                'Select a candidate above or enter entry/stop prices to calculate position size.</p>',
                unsafe_allow_html=True,
            )




def _render_core_tab(l0: dict, l3_data: list, perm: str) -> None:
    """Render the Core Allocation tab (v4)."""

    account   = st.session_state.account_value
    core_pct  = st.session_state.core_pct_deployed
    core_tickers = [t.strip().upper() for t in st.session_state.core_positions.split(",") if t.strip()]

    # Deployment floor targets
    floor_map  = {"Green": (40, 60), "Yellow": (20, 35), "Red": (0, 0)}
    floor_lo, floor_hi = floor_map.get(perm, (0, 0))
    core_target = {"Green": 40, "Yellow": 20, "Red": 0}.get(perm, 0)

    # Core status
    core_value   = account * core_pct / 100
    slots_used   = len(core_tickers)
    slots_max    = 3
    target_value = account * core_target / 100
    gap_value    = target_value - core_value  # +ve = room to deploy

    # Phase 2 candidates from RRG
    phase2 = [d for d in l3_data if d["quadrant"] == "Leading"]
    if phase2:
        p2_rows = []
        for d in phase2:
            already_held = d["etf"] in core_tickers
            sr = l0.get("sector_rs", {}).get(d["sector"], {})
            vel = sr.get("velocity", "N/A")
            p2_rows.append({
                "ETF":        d["etf"],
                "Sector":     d["sector"],
                "Phase":      "🟢 Phase 2 — Confirmed",
                "Price":      f"${d['price']:.2f}",
                "Stop (20MA)": f"${d['ma20']:.2f}",
                "Velocity":   f"{'🔥' if vel == 'ACCELERATING' else ''} {vel}",
                "Held?":      "✅ Yes" if already_held else "—",
            })
        p2_html = cb_table(pd.DataFrame(p2_rows), bordered=False)
    else:
        p2_html = "<p style='color:#5A7BAA; font-size:13px;'>No sectors in Phase 2 currently.</p>"

    # Current Core positions detail (if any)
    sr = l0.get("sector_rs", {})
    if core_tickers:
        held_rows = []
        for etf in core_tickers:
            # Find sector for this ETF
            sec_name = next((sec for sec, e in SECTOR_ETFS.items() if e == etf), "Unknown")
            sr_data = sr.get(sec_name, {})
            price  = sr_data.get("price", "N/A")
            vel    = sr_data.get("velocity", "N/A")
            trend  = sr_data.get("trend", "N/A")
            # Check RRG quadrant
            rrg_match = next((d for d in l3_data if d["etf"] == etf), None)
            phase = rrg_match["phase"] if rrg_match else "N/A"
            above_20 = rrg_match["above_20"] if rrg_match else None
            ma20 = rrg_match["ma20"] if rrg_match else "N/A"

            held_rows.append({
                "ETF":        etf,
                "Sector":     sec_name,
                "Price":      f"${price}" if isinstance(price, (int, float)) else price,
                "Stop (20MA)": f"${ma20}" if isinstance(ma20, (int, float)) else ma20,
                "vs 20MA":    ("✅" if above_20 else "❌") if above_20 is not None else "N/A",
                "RS Trend":   trend,
                "Velocity":   vel,
                "Phase":      phase,
            })
        held_html = cb_table(pd.DataFrame(held_rows), bordered=False)
    else:
        held_html = "<p style='color:#5A7BAA; font-size:13px;'>No Core positions entered. Set in sidebar.</p>"

    # Exit signals
    exit_signals = []
    for etf in core_tickers:
        rrg_match = next((d for d in l3_data if d["etf"] == etf), None)
        if rrg_match and not rrg_match["above_20"]:
            exit_signals.append(f"⚠️ {etf} — below 20d MA → exit review")
        if rrg_match and rrg_match["quadrant"] in ("Weakening", "Lagging"):
            exit_signals.append(f"⚠️ {etf} — {rrg_match['quadrant']} quadrant → exit review")
    if perm == "Red":
        exit_signals.append("🔴 RED state — exit all Core positions")

    exit_html = ""
    if exit_signals:
        exit_html = '<div style="margin-top:6px;">' + "".join(
            f'<p style="font-size:12px; color:#CC1111; font-weight:500; margin:3px 0;">{s}</p>'
            for s in exit_signals
        ) + '</div>'

    # ── Deployment hero ───────────────────────────────────────────────────────
    # State palette: RED regime = exit (red); under floor = act (amber);
    # at/above target = on-target (green).
    if perm == "Red":
        hb_bg, hb_bd, hb_lbl, hb_txt = "#FFE4E4", "rgba(204,17,17,.35)", "#A32D2D", "#501313"
        bar_color = "#CC1111"
    elif core_pct < floor_lo:
        hb_bg, hb_bd, hb_lbl, hb_txt = "#FFF3D6", "rgba(224,120,0,.35)", "#854F0B", "#412402"
        bar_color = "#E07800"
    else:
        hb_bg, hb_bd, hb_lbl, hb_txt = "#D6F0D6", "rgba(29,122,42,.35)", "#27500A", "#173404"
        bar_color = "#27500A"

    # Gauge scale: target sits at 75% of the track so there's headroom to show
    # over-deployment, and the labeled target marker is well clear of the right edge.
    track_max  = (core_target / 0.75) if core_target > 0 else max(floor_hi, 1)
    fill_pct   = min(core_pct / track_max * 100, 100) if track_max else 0
    target_mark = min(core_target / track_max * 100, 100) if track_max else 0

    if perm == "Red":
        action_line = '<i class="ti ti-arrow-down" style="font-size:12px"></i> RED state — exit all Core positions to cash'
    elif gap_value > account * 0.005:  # meaningfully under target
        n_sec = 1 if gap_value < account * 0.20 else 2
        action_line = (f'<i class="ti ti-arrow-up" style="font-size:12px"></i> '
                       f'Under-deployed by {core_target - core_pct:.0f}% — deploy ~${gap_value:,.0f} '
                       f'into {n_sec} Phase 2 sector{"s" if n_sec > 1 else ""}')
    elif core_pct > floor_hi:
        action_line = '<i class="ti ti-alert-triangle" style="font-size:12px"></i> Above deployment ceiling — trim toward target'
    else:
        action_line = '<i class="ti ti-check" style="font-size:12px"></i> On target — hold and manage'

    core_hero = (
        '<div style="display:grid; grid-template-columns:1.3fr 1fr; gap:9px; margin-bottom:4px;">'
        f'<div style="background:{hb_bg}; border:1px solid {hb_bd}; border-radius:10px; padding:11px 13px;">'
        f'<div style="font-size:13px; font-weight:500; color:{hb_lbl}; letter-spacing:.01em;'
        f' margin-bottom:4px;">Core deployment · {perm.upper()} regime</div>'
        f'<div style="display:flex; align-items:baseline; gap:8px; margin:4px 0 2px;">'
        f'<span style="font-size:30px; font-weight:500; color:{hb_txt}; line-height:1;">{core_pct:.0f}%</span>'
        f'<span style="font-size:13px; color:{hb_lbl};">'
        + (f'of {core_target}% target' if core_target > 0 else 'deployed · target 0% in RED')
        + '</span></div>'
        # ── Deployment gauge with a prominent, labeled target marker ──────────
        f'<div style="position:relative; margin:8px 0 3px; padding-top:2px;">'
        # the bar track + fill
        f'<div style="height:11px; background:#fff; border-radius:6px; overflow:hidden;'
        f' border:0.5px solid {hb_bd}; position:relative;">'
        f'<div style="position:absolute; left:0; top:0; bottom:0; width:{fill_pct:.0f}%; background:{bar_color};"></div>'
        + (  # target marker — bold full-height line through the bar
            f'<div style="position:absolute; left:{target_mark:.0f}%; top:-3px; bottom:-3px;'
            f' width:2.5px; background:#103766; border-radius:1px;"></div>'
            if core_target > 0 else ""
        )
        + '</div>'
        # target label pinned under the marker (its own row so it never clips)
        + (
            f'<div style="position:relative; height:14px; margin-top:1px;">'
            f'<div style="position:absolute; left:{target_mark:.0f}%; transform:translateX(-50%);'
            f' font-size:10px; font-weight:500; color:#103766; white-space:nowrap;">'
            f'▲ {core_target}% target</div></div>'
            if core_target > 0 else ""
        )
        + '</div>'
        f'<div style="font-size:11px; color:{hb_txt}; margin-top:3px;">{action_line}</div>'
        '</div>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">'
        + _tile("Core $ deployed", f"${core_value:,.0f}", "", "#103766")
        + _tile("Room to target", f"${gap_value:,.0f}" if gap_value > 0 else "At/over",
                "", hb_lbl if gap_value > 0 else "#27500A")
        + _tile("Slots used", f"{slots_used} / {slots_max}", "")
        + _tile("Floor", f"{floor_lo}–{floor_hi}%", "",
                "#CC1111" if core_pct < floor_lo else "#27500A")
        + '</div>'
        '</div>'
    )

    # ── Reference groups: candidates (to close gap) then holdings ─────────────
    n_cand = len(phase2)
    full_html = (
        _MACRO_TIP_CSS  # enables the info-hover tooltips on this tab
        + core_hero
        + '<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
          'How to close the gap <span style="font-weight:400; color:#5A7BAA;">'
          '— Phase 2 sectors eligible for Core entry</span></div>'
        + _macro_card(f"Entry candidates · {n_cand} eligible", p2_html, "core_candidates", quiet=True)
        + '<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
          'Holdings <span style="font-weight:400; color:#5A7BAA;">'
          '— current Core positions &amp; exit watch</span></div>'
        + _macro_card(f"Held · {slots_used} position{'s' if slots_used != 1 else ''}",
                      held_html + exit_html, "core_holdings", quiet=True)
    )
    st.markdown(full_html, unsafe_allow_html=True)


def _render_charts_tab(passes_df: pd.DataFrame) -> None:
    """Render the Charts tab — individual stock charts for Full Signal candidates."""

    if passes_df.empty:
        st.info("Run the screener first to populate charts.")
        return

    sel = st.selectbox(
        "Select ticker",
        passes_df["Ticker"].tolist(),
        format_func=lambda t: f"{t}  —  {passes_df[passes_df['Ticker']==t]['Sector'].iloc[0]}",
    )
    if sel:
        row = passes_df[passes_df["Ticker"] == sel].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Price",    f"${row['Price']:.2f}")
        c2.metric("RS vs SPY", f"+{row.get('RS_vs_SPY_21d', 0):.1f}%")
        c3.metric("vs 20MA",   f"+{row.get('Dist_MA20_pct', 0):.1f}%")
        with st.spinner(f"Building {sel} chart..."):
            fig = build_chart(sel)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not load chart data for {sel}.")

    if st.checkbox("Show all Full Signal charts"):
        for _, row in passes_df.head(20).iterrows():
            t = row["Ticker"]
            rs = row.get("RS_vs_SPY_21d", 0)
            with st.expander(f"{t}  —  {row['Sector']}  |  RS: +{rs:.1f}%", expanded=False):
                with st.spinner(f"Loading {t}..."):
                    fig = build_chart(t)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO TRACKER
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")


def _portfolio_load() -> dict:
    if not os.path.exists(PORTFOLIO_PATH):
        return {"cash_balance": 100000, "open_positions": [], "closed_positions": []}
    with open(PORTFOLIO_PATH, "r") as f:
        data = json.load(f)
    # Migrate legacy field name
    if "account_size" in data and "cash_balance" not in data:
        open_cb = sum(p["entry_price"] * p["shares"] for p in data.get("open_positions", []))
        data["cash_balance"] = data.pop("account_size") - open_cb
    return data


def _compute_account_value(data: dict, live_prices: dict) -> tuple:
    """
    Compute total account value dynamically.
    Returns (account_value, cash_balance, total_market_value, total_cost_basis).
    account_value = cash_balance + sum(open shares × live price)
    """
    cash = data.get("cash_balance", 100000)
    open_pos = data.get("open_positions", [])
    total_mv = 0.0
    total_cb = 0.0
    for p in open_pos:
        shares = p["shares"]
        entry = p["entry_price"]
        cur_px = live_prices.get(p["ticker"], p.get("current_price") or entry)
        total_mv += shares * cur_px
        total_cb += shares * entry
    account_value = cash + total_mv
    return account_value, cash, total_mv, total_cb


def _portfolio_save(data: dict) -> None:
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(data, f, indent=2)


# fetch_portfolio_prices moved to data/market.py (imported at top of file).


def _dollar_fmt(v: float) -> str:
    return f"${v:+,.0f}" if v != 0 else "$0"


def _pct_fmt(v: float) -> str:
    return f"{v*100:+.1f}%"


def _pnl_color(v: float) -> str:
    if v > 0: return "#27500A"
    if v < 0: return "#CC1111"
    return "#5A7BAA"


def _build_open_table(positions: list, prices: dict, account_size: float):
    rows = []
    total_mv = total_cb = total_upnl = total_risk = 0.0
    sector_exposure = {}  # ticker → cost basis for concentration calc
    for p in positions:
        ticker     = p["ticker"]
        entry_px   = p["entry_price"]
        shares     = p["shares"]
        stop_px    = p["stop_price"]  # may be None if stop not yet set
        entry_dt   = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
        days_held  = (date.today() - entry_dt).days
        layer      = p.get("layer", "Tactical")
        cur_px     = prices.get(ticker, p.get("current_price") or entry_px)
        cost_basis = entry_px * shares
        mkt_val    = cur_px * shares
        upnl_d     = mkt_val - cost_basis
        upnl_pct   = upnl_d / cost_basis if cost_basis else 0.0
        risk_d     = (entry_px - stop_px) * shares if stop_px is not None else 0.0
        total_mv   += mkt_val
        total_cb   += cost_basis
        total_upnl += upnl_d
        total_risk += risk_d

        # L8: Determine if Accelerating based on notes
        is_accel = "Accelerating" in (p.get("notes") or "")

        # L8: Profit targets
        if is_accel:
            t1_px = round(entry_px * 1.09, 2)   # +8-10%
            t2_px = round(entry_px * 1.175, 2)  # +15-20%
            max_hold_weeks = 6
        elif layer == "Core":
            t1_px = None  # Core doesn't use T1/T2
            t2_px = None
            max_hold_weeks = None
        else:
            t1_px = round(entry_px * 1.10, 2)   # +8-12%
            t2_px = round(entry_px * 1.225, 2)  # +20-25%
            max_hold_weeks = 12

        be_px = round(entry_px * 1.05, 2)  # Breakeven stop at +5%

        # Max hold date
        if max_hold_weeks:
            max_hold_date = entry_dt + timedelta(weeks=max_hold_weeks)
            days_remaining = (max_hold_date - date.today()).days
            if days_remaining <= 0:
                hold_status = "⚠️ EXPIRED"
            elif days_remaining <= 7:
                hold_status = f"⚠️ {days_remaining}d left"
            else:
                hold_status = f"{days_remaining}d left"
        else:
            max_hold_date = None
            hold_status = "No limit" if layer == "Core" else "—"

        # L8: Trade management status
        if cur_px >= (t2_px or 999999):
            mgmt_status = "🟢 Past T2 — trail 20MA"
        elif cur_px >= (t1_px or 999999):
            mgmt_status = "🟡 Past T1 — sell 1/3, BE stop"
        elif cur_px >= be_px:
            mgmt_status = "Move stop → BE"
        else:
            mgmt_status = "Hold — below +5%"

        if layer == "Core":
            mgmt_status = "Trail 20d MA"

        rows.append({
            "Ticker":     ticker,
            "Layer":      layer,
            "Entry":      f"${entry_px:.2f}",
            "Current":    f"${cur_px:.2f}",
            "Shares":     shares,
            "P&L $":      _dollar_fmt(upnl_d),
            "P&L %":      _pct_fmt(upnl_pct),
            "Stop":       f"${stop_px:.2f}" if stop_px is not None else "TBD",
            "BE":         f"${be_px:.2f}",
            "T1":         f"${t1_px:.2f}" if t1_px else "—",
            "T2":         f"${t2_px:.2f}" if t2_px else "—",
            "Days":       days_held,
            "Hold":       hold_status,
            "Status":     mgmt_status,
            "Risk $":     f"${risk_d:,.0f}",
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Ticker","Layer","Entry","Current","Shares","P&L $","P&L %",
                 "Stop","BE","T1","T2","Days","Hold","Status","Risk $"])
    return df, total_mv, total_cb, total_upnl, total_risk


def _build_closed_table(positions: list) -> pd.DataFrame:
    rows = []
    for p in positions:
        entry_px = p["entry_price"]
        exit_px  = p["exit_price"]
        shares   = p["shares"]
        pnl_d    = (exit_px - entry_px) * shares
        pnl_pct  = pnl_d / (entry_px * shares) if entry_px else 0.0
        entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
        exit_dt  = datetime.strptime(p["exit_date"],  "%Y-%m-%d").date()
        rows.append({
            "Ticker":      p["ticker"],
            "Layer":       p.get("layer", "Tactical"),
            "Entry Date":  str(entry_dt),
            "Exit Date":   str(exit_dt),
            "Entry":       f"${entry_px:.2f}",
            "Exit":        f"${exit_px:.2f}",
            "Shares":      shares,
            "P&L $":       _dollar_fmt(pnl_d),
            "P&L %":       _pct_fmt(pnl_pct),
            "Hold Days":   (exit_dt - entry_dt).days,
            "Exit Reason": p.get("exit_reason", "—"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Ticker","Layer","Entry Date","Exit Date","Entry","Exit",
                 "Shares","P&L $","P&L %","Hold Days","Exit Reason"])


def _calc_performance(closed: list, start_dt: date, end_dt: date) -> dict:
    filtered = [
        p for p in closed
        if start_dt <= datetime.strptime(p["exit_date"], "%Y-%m-%d").date() <= end_dt
    ]
    if not filtered:
        return {"count": 0, "realized_pnl": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "trades": []}
    pnls = [{"pnl": (p["exit_price"]-p["entry_price"])*p["shares"],
             "pct": p["exit_price"]/p["entry_price"]-1} for p in filtered]
    wins   = [x for x in pnls if x["pnl"] > 0]
    losses = [x for x in pnls if x["pnl"] <= 0]
    gross_wins   = sum(x["pnl"] for x in wins)
    gross_losses = abs(sum(x["pnl"] for x in losses))
    return {
        "count":        len(filtered),
        "realized_pnl": sum(x["pnl"] for x in pnls),
        "win_rate":     len(wins) / len(pnls),
        "avg_win":      sum(x["pct"] for x in wins)   / len(wins)   if wins   else 0,
        "avg_loss":     sum(x["pct"] for x in losses) / len(losses) if losses else 0,
        "profit_factor":gross_wins / gross_losses if gross_losses > 0 else float("inf"),
        "trades":       filtered,
        "n_wins":       len(wins),
        "n_losses":     len(losses),
    }


def _collect_etf_exit_signals(open_positions: list) -> list:
    """Return ETF/Core exit-trigger signals as a list of dicts:
    {ticker, trigger, detail, severity}. No rendering — feeds the action hero."""
    etf_positions = [p for p in open_positions if is_etf_position(p)]
    if not etf_positions:
        return []

    etf_to_sector = {etf: name for name, etf in SECTOR_ETFS.items()}
    flows = fetch_etf_fund_flows()
    macro_close = fetch_macro_data()
    l0 = calc_layer0(macro_close)
    sector_rs = l0.get("sector_rs", {}) if isinstance(l0, dict) else {}

    out = []
    for p in etf_positions:
        tkr = p["ticker"]
        price = ma20 = None
        if tkr in macro_close.columns:
            series = macro_close[tkr].dropna()
            if len(series) >= 20:
                price = float(series.iloc[-1])
                ma20  = float(series.rolling(20).mean().iloc[-1])
        flow_strength = flows.get(tkr, {}).get("flow_strength")
        sector_name = etf_to_sector.get(tkr)
        rs_trend = sector_rs.get(sector_name, {}).get("trend") if sector_name else None
        for s in etf_exit_signals(tkr, price, ma20, flow_strength, rs_trend):
            out.append({"ticker": tkr, **s})
    return out


def _mgmt_action_items(open_pos: list, prices: dict, account_size: float) -> list:
    """Scan open positions for L8 management actions that need doing now
    (stop→BE, sell 1/3 at T1, trail at T2, hold expired). Returns a list of
    {severity, icon, text} dicts. Mirrors the mgmt_status logic in
    _build_open_table without rendering a table."""
    items = []
    today = date.today()
    for p in open_pos:
        tkr     = p["ticker"]
        layer   = p.get("layer", "Tactical")
        entry   = p["entry_price"]
        cur     = prices.get(tkr, p.get("current_price") or entry)
        is_accel = "Accelerating" in (p.get("notes") or "")
        if is_accel:
            t1, t2 = round(entry*1.09, 2), round(entry*1.175, 2)
            weeks = 6
        elif layer == "Core":
            t1 = t2 = None; weeks = None
        else:
            t1, t2 = round(entry*1.10, 2), round(entry*1.225, 2)
            weeks = 12
        be = round(entry*1.05, 2)

        if layer != "Core":
            if t2 and cur >= t2:
                items.append({"severity": "info", "icon": "arrow-up-right",
                              "text": f"{tkr} past T2 — trail remainder at 20d MA"})
            elif t1 and cur >= t1:
                items.append({"severity": "info", "icon": "arrow-up",
                              "text": f"{tkr} past T1 — sell 1/3, move stop to breakeven"})
            elif cur >= be:
                items.append({"severity": "watch", "icon": "shield",
                              "text": f"{tkr} above +5% — move stop to breakeven"})
        # Max-hold expiry
        if weeks:
            entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d").date()
            days_left = ((entry_dt + timedelta(weeks=weeks)) - today).days
            if days_left <= 0:
                items.append({"severity": "watch", "icon": "clock",
                              "text": f"{tkr} hit max hold ({weeks}w) — review for exit"})
            elif days_left <= 7:
                items.append({"severity": "info", "icon": "clock",
                              "text": f"{tkr} {days_left}d to max hold"})
    return items


def _render_portfolio_tab() -> None:
    """Render the Portfolio Tracker tab."""
    today = date.today()
    data         = _portfolio_load()
    open_pos     = data.get("open_positions", [])
    closed_pos   = data.get("closed_positions", [])

    # ── Live prices ───────────────────────────────────────────────────────────
    open_tickers = [p["ticker"] for p in open_pos]
    prices = {}
    if open_tickers:
        with st.spinner("Fetching live prices…"):
            prices = fetch_portfolio_prices(",".join(open_tickers))

    # ── Dynamic account value ─────────────────────────────────────────────────
    account_value, cash_balance, total_mv, total_cb_live = _compute_account_value(data, prices)
    account_size = account_value  # alias for compatibility with downstream calcs

    # Push to session state so position sizer and Core tab can read it
    st.session_state["account_value"] = round(account_value)

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Portfolio")
        st.markdown(
            f'<div style="background:#EEF3FA; border-radius:9px; padding:10px 12px; margin-bottom:8px;">'
            f'<div style="font-size:11px; color:#5A7BAA;">Account Value (live)</div>'
            f'<div style="font-size:17px; font-weight:500; color:#103766;">${account_value:,.0f}</div>'
            f'<div style="font-size:11px; color:#5A7BAA;">Cash: ${cash_balance:,.0f} · Positions: ${total_mv:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Add position form
        with st.expander("➕ Add Open Position"):
            with st.form("add_pos_form", clear_on_submit=True):
                ticker     = st.text_input("Ticker").upper().strip()
                layer      = st.selectbox("Layer", ["Tactical", "Core"], key="add_layer")
                entry_date = st.date_input("Entry Date", value=today, key="add_edate")
                entry_px   = st.number_input("Entry Price", min_value=0.01, step=0.01, format="%.2f", key="add_epx")
                shares     = st.number_input("Shares", min_value=1, step=1, key="add_shares")
                stop_px    = st.number_input("Stop Price", min_value=0.01, step=0.01, format="%.2f", key="add_stop")
                notes      = st.text_input("Notes", key="add_notes")
                if st.form_submit_button("Add") and ticker and entry_px > 0 and shares > 0:
                    add_open_position(
                        data, ticker=ticker, layer=layer,
                        entry_date=str(entry_date), entry_price=entry_px,
                        shares=int(shares), stop_price=stop_px, notes=notes,
                    )
                    _portfolio_save(data)
                    st.cache_data.clear()
                    st.rerun()

        # Close position form
        if open_tickers:
            with st.expander("✅ Close Position"):
                with st.form("close_pos_form", clear_on_submit=True):
                    sel_ticker  = st.selectbox("Ticker", open_tickers, key="close_ticker")
                    exit_date   = st.date_input("Exit Date", value=today, key="close_edate")
                    exit_px     = st.number_input("Exit Price", min_value=0.01, step=0.01, format="%.2f", key="close_px")
                    exit_reason = st.selectbox("Exit Reason", ["Target","Stop","Rule-based"], key="close_reason")
                    close_notes = st.text_input("Notes", key="close_notes")
                    if st.form_submit_button("Close") and exit_px > 0:
                        close_position(
                            data, ticker=sel_ticker, exit_date=str(exit_date),
                            exit_price=exit_px, exit_reason=exit_reason, notes=close_notes,
                        )
                        _portfolio_save(data)
                        st.cache_data.clear()
                        st.rerun()

    # ── Performance period selector ───────────────────────────────────────────
    perf_col1, perf_col2 = st.columns([2, 5])
    with perf_col1:
        perf_period = st.selectbox(
            "Performance Period",
            ["YTD", "1M", "3M", "6M", "1Y", "Custom"],
            index=0, label_visibility="collapsed", key="port_period",
        )
    custom_range = None
    if perf_period == "Custom":
        with perf_col2:
            custom_range = st.date_input(
                "Range", value=(date(today.year,1,1), today),
                min_value=date(2020,1,1), max_value=today,
                label_visibility="collapsed", key="port_custom_range",
            )
        if not isinstance(custom_range, (list, tuple)):
            custom_range = (custom_range, today)
    p_start, p_end, period_label = period_to_range(perf_period, today, custom_range)

    perf = _calc_performance(closed_pos, p_start, p_end)
    _, total_mv, total_cb, total_upnl, total_risk = _build_open_table(open_pos, prices, account_size)

    deployed_pct = total_cb / account_size if account_size else 0
    heat_pct     = total_risk / account_size if account_size else 0
    pf_val       = perf["profit_factor"]
    pf_str       = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"
    rpnl_color   = _pnl_color(perf["realized_pnl"])
    upnl_color   = _pnl_color(total_upnl)

    # ── Drawdown (needed by hero health tiles) — computed early ───────────────
    _peak = data.get("peak_equity", account_value)
    _dd_pct_hero = ((account_value - _peak) / _peak * 100) if _peak > 0 else 0.0

    # ── Action items: exit triggers + L8 mgmt + (compliance added below) ──────
    action_items = []
    for s in _collect_etf_exit_signals(open_pos):
        sev = "alert" if s.get("severity") == "high" else "watch"
        icon = "bell" if sev == "alert" else "alert-triangle"
        action_items.append({"severity": sev, "icon": icon,
                             "text": f"{s['ticker']} — {s['trigger']}: {s['detail']}"})
    action_items += _mgmt_action_items(open_pos, prices, account_size)

    # ── Performance period & health summary ───────────────────────────────────
    # Hero renders just below (after compliance warnings are computed) so it can
    # include breached caps. Build the demoted performance strip here.
    perf_strip_html = (
        f'<div style="font-size:11px; color:#5A7BAA; margin:11px 0 6px; font-weight:500;">'
        f'How you’re doing <span style="font-weight:400;">— realized performance · {period_label}</span></div>'
        f'<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:6px;">'
        + _tile("Realized P&L", _dollar_fmt(perf["realized_pnl"]),
                f"{perf['count']} closed", rpnl_color)
        + _tile("Win Rate", f"{perf['win_rate']*100:.0f}%" if perf["count"] else "—",
                f"{perf.get('n_wins',0)}W / {perf.get('n_losses',0)}L" if perf["count"] else "")
        + _tile("Profit Factor", pf_str,
                "≥1.5 target", "#27500A" if pf_val >= 1.5 else "#E07800" if pf_val >= 1.0 else "#CC1111")
        + _tile("Avg Win / Loss",
                f"{_pct_fmt(perf['avg_win'])} / {_pct_fmt(perf['avg_loss'])}" if perf["count"] else "—", "")
        + '</div>'
    )

    # ── L7 / L9 Compliance checks ───────────────────────────────────────────
    perm_state = st.session_state.get("_current_perm", "Green")  # set by main()
    perm_limits = PERM_LIMITS.get(perm_state, PERM_LIMITS["Green"])
    max_pos = perm_limits["max_pos"]
    max_heat_pct = perm_limits["heat"] / 100
    pos_count = len(open_pos)
    n_core     = sum(1 for p in open_pos if p.get("layer") == "Core")
    n_tactical = pos_count - n_core

    # Deployment floor
    floor_map = {"Green": (40, 60), "Yellow": (20, 35), "Red": (0, 0)}
    floor_lo, floor_hi = floor_map.get(perm_state, (0, 0))

    # Sector concentration (by ticker → sector from SECTOR_TICKERS lookup)
    sector_cost = {}
    for p in open_pos:
        sec = "Unknown"
        for s, tickers in SECTOR_TICKERS.items():
            if p["ticker"] in tickers:
                sec = s
                break
        cb = p["entry_price"] * p["shares"]
        sector_cost[sec] = sector_cost.get(sec, 0) + cb
    max_sector_pct = max(v / account_size for v in sector_cost.values()) * 100 if sector_cost else 0
    max_sector_name = max(sector_cost, key=sector_cost.get) if sector_cost else "—"

    # Drawdown from peak — track high-water mark in portfolio.json
    current_equity = account_value  # already computed dynamically above
    peak_equity = data.get("peak_equity", current_equity)
    if current_equity > peak_equity:
        peak_equity = current_equity
        data["peak_equity"] = round(peak_equity, 2)
        _portfolio_save(data)
    drawdown_pct = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

    if drawdown_pct >= 0:
        dd_tier = "At peak"
        dd_color = "#27500A"
    elif drawdown_pct > -7:
        dd_tier = "Tier 1: Normal"
        dd_color = "#27500A"
    elif drawdown_pct > -10:
        dd_tier = "Tier 2: Reduce risk 50%"
        dd_color = "#E07800"
    elif drawdown_pct > -15:
        dd_tier = "Tier 3: Defensive"
        dd_color = "#CC1111"
    else:
        dd_tier = "Emergency: 100% cash"
        dd_color = "#CC1111"

    # Build compliance rows
    compliance_rows = [
        {"Check": "Position Count",
         "Current": f"{pos_count} ({n_core}C / {n_tactical}T)",
         "Limit": f"Max {perm_limits['max_pos_label']} ({perm_state})",
         "Status": "✅ OK" if pos_count <= max_pos else "❌ Over limit"},
        {"Check": "Portfolio Heat",
         "Current": f"{heat_pct*100:.1f}%",
         "Limit": f"Max {perm_limits['heat']}%",
         "Status": "✅ OK" if heat_pct <= max_heat_pct else "❌ Over limit"},
        {"Check": "Deployed Capital",
         "Current": f"{deployed_pct*100:.1f}%",
         "Limit": f"Floor {floor_lo}–{floor_hi}%",
         "Status": "✅ OK" if deployed_pct * 100 >= floor_lo else "⚠️ Below floor"},
        {"Check": "Max Sector",
         "Current": f"{max_sector_name}: {max_sector_pct:.0f}%",
         "Limit": "Max 25%",
         "Status": "✅ OK" if max_sector_pct <= 25 else "❌ Over concentrated"},
        {"Check": "Drawdown",
         "Current": f"{drawdown_pct:+.1f}%",
         "Limit": dd_tier,
         "Status": f"{'✅ OK' if drawdown_pct > -7 else '⚠️ Caution' if drawdown_pct > -10 else '❌ Action required'}"},
    ]
    compliance_html = cb_table(pd.DataFrame(compliance_rows), bordered=False)

    # Compliance warnings
    warnings = []
    if pos_count > max_pos:
        warnings.append(f"🔴 Over position limit: {pos_count} vs max {max_pos} in {perm_state}")
    if heat_pct > max_heat_pct:
        warnings.append(f"🔴 Portfolio heat {heat_pct*100:.1f}% exceeds {perm_limits['heat']}% cap")
    if deployed_pct * 100 < floor_lo and floor_lo > 0:
        warnings.append(f"⚠️ Below deployment floor: {deployed_pct*100:.1f}% vs {floor_lo}% minimum")
    if drawdown_pct <= -7:
        warnings.append(f"⚠️ Drawdown {drawdown_pct:.1f}% — {dd_tier}")

    # Add breached compliance caps to the action hero (already-detailed warnings).
    for w in warnings:
        sev = "alert" if w.startswith("🔴") else "watch"
        clean = w.lstrip("🔴⚠️ ").strip()
        action_items.append({"severity": sev,
                             "icon": "alert-triangle" if sev == "watch" else "alert-octagon",
                             "text": clean})

    # ── Render: ACTION HERO + health tiles + performance strip ────────────────
    _ICON = {"bell": "🔔", "alert-triangle": "⚠️", "alert-octagon": "⛔",
             "arrow-up": "↑", "arrow-up-right": "↗", "shield": "🛡️", "clock": "⏱️"}
    _SEV_COLOR = {"alert": "#CC1111", "watch": "#E07800", "info": "#27500A"}
    # Hero palette keyed to the worst severity present.
    if any(a["severity"] == "alert" for a in action_items):
        hb_bg, hb_bd, hb_lbl, hb_txt = "#FFE4E4", "rgba(204,17,17,.35)", "#A32D2D", "#501313"
    elif any(a["severity"] == "watch" for a in action_items):
        hb_bg, hb_bd, hb_lbl, hb_txt = "#FFF3D6", "rgba(224,120,0,.35)", "#854F0B", "#412402"
    else:
        hb_bg, hb_bd, hb_lbl, hb_txt = "#D6F0D6", "rgba(29,122,42,.35)", "#27500A", "#173404"

    if action_items:
        items_html = "".join(
            f'<div style="margin:2px 0;"><span style="color:{_SEV_COLOR[a["severity"]]};">'
            f'{_ICON.get(a["icon"], "•")}</span> {a["text"]}</div>'
            for a in action_items
        )
        hero_title = f"Needs action today · {len(action_items)} item{'s' if len(action_items)!=1 else ''}"
    else:
        items_html = '<div style="margin-top:4px;">All positions holding — nothing needs a decision today.</div>'
        hero_title = "Nothing needs action"

    dd_color_hero = ("#27500A" if _dd_pct_hero >= -7 else "#E07800" if _dd_pct_hero > -10 else "#CC1111")
    portfolio_hero = (
        '<div style="display:grid; grid-template-columns:1.25fr 1fr; gap:9px; margin-bottom:4px;">'
        f'<div style="background:{hb_bg}; border:1px solid {hb_bd}; border-radius:10px; padding:11px 13px;">'
        f'<div style="font-size:13px; font-weight:500; color:{hb_lbl}; letter-spacing:.01em;'
        f' margin-bottom:4px;">{hero_title}</div>'
        f'<div style="font-size:12px; color:{hb_txt}; margin-top:6px; line-height:1.6;">{items_html}</div>'
        '</div>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">'
        + _tile("Account value", f"${account_value:,.0f}", "", "#103766")
        + _tile("Open P&L", _dollar_fmt(total_upnl), f"{len(open_pos)} open", upnl_color)
        + _tile("Portfolio heat", f"{heat_pct*100:.1f}%", "",
                "#CC1111" if heat_pct > 0.15 else "#E07800" if heat_pct > 0.08 else "#27500A")
        + _tile("Drawdown", f"{_dd_pct_hero:+.1f}%", "", dd_color_hero)
        + '</div>'
        '</div>'
    )
    st.markdown(portfolio_hero + perf_strip_html, unsafe_allow_html=True)

    # ── Open positions (reference) ────────────────────────────────────────────
    open_df, _, _, _, _ = _build_open_table(open_pos, prices, account_size)
    open_pill  = f"{pos_count} positions · {n_core} Core / {n_tactical} Tactical"

    st.markdown('<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
                'Positions <span style="font-weight:400; color:#5A7BAA;">— open holdings &amp; management state</span></div>',
                unsafe_allow_html=True)
    if not open_df.empty:
        st.markdown(
            _macro_card("Open Positions", cb_table(open_df, bordered=False), "", quiet=True),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _macro_card("Open Positions",
                  '<p style="font-size:13px; color:#5A7BAA;">No open positions. Add one in the sidebar.</p>',
                  "", quiet=True),
            unsafe_allow_html=True,
        )

    # ── Closed positions ──────────────────────────────────────────────────────
    closed_years = sorted(
        set(datetime.strptime(p["exit_date"], "%Y-%m-%d").year for p in closed_pos),
        reverse=True,
    ) if closed_pos else [today.year]

    year_options = [str(y) for y in closed_years]
    default_idx  = year_options.index(str(today.year)) if str(today.year) in year_options else 0

    selected_year = int(st.selectbox(
        "Year", year_options, index=default_idx,
        label_visibility="collapsed", key="port_year",
    ))

    filtered_closed = [
        p for p in closed_pos
        if datetime.strptime(p["exit_date"], "%Y-%m-%d").year == selected_year
    ]
    closed_df  = _build_closed_table(filtered_closed)
    year_pnl   = sum((p["exit_price"]-p["entry_price"])*p["shares"] for p in filtered_closed)
    closed_pill = f"{selected_year} · {len(filtered_closed)} trades · {_dollar_fmt(year_pnl)}"

    if not closed_df.empty:
        st.markdown(
            _macro_card(f"Closed — {selected_year} · {closed_pill}",
                        cb_table(closed_df, bordered=False), "", quiet=True),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _macro_card(f"Closed — {selected_year}",
                  f'<p style="font-size:13px; color:#5A7BAA;">No closed trades in {selected_year}.</p>',
                  "", quiet=True),
            unsafe_allow_html=True,
        )

    # ── Full compliance detail (reference) — breaches surfaced in hero above ──
    comp_clear = "All clear" if not warnings else f"{len(warnings)} issue(s)"
    st.markdown('<div style="font-size:11px; font-weight:500; color:#103766; margin:11px 0 6px;">'
                'Risk &amp; compliance detail <span style="font-weight:400; color:#5A7BAA;">'
                '— every cap &amp; floor · ' + comp_clear + '</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        _macro_card("Compliance — all checks", compliance_html, "", quiet=True),
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
<style>
/* ── PAGE BACKGROUND ───────────────────────────────────────────────────── */
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

/* ── SIDEBAR ───────────────────────────────────────────────────────────── */
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

/* ── HEADINGS ──────────────────────────────────────────────────────────── */
h2 { color: #103766 !important; font-weight: 500 !important; font-size: 18px !important; }
h3 { color: #103766 !important; font-weight: 500 !important; }
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] { color: #5A7BAA !important; font-size: 11px !important; }
h4 {
    color: #5A7BAA !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(16,55,102,0.15) !important;
    padding-bottom: 7px !important;
    margin-top: 2px !important;
    margin-bottom: 12px !important;
}

/* ── METRIC PANELS ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(16,55,102,0.15) !important;
    border-radius: 9px !important;
    padding: 10px 12px !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    color: #5A7BAA !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 17px !important;
    font-weight: 500 !important;
    color: #103766 !important;
}
[data-testid="stMetricDelta"] { color: #1D7A2A !important; font-size: 11px !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── EXPANDER CARDS ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(16,55,102,0.15) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 1px rgba(16,55,102,0.06) !important;
}
[data-testid="stExpander"] details summary p {
    color: #5A7BAA !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stExpanderToggleIcon"] svg { color: #5A7BAA !important; }
[data-testid="stExpander"] details summary {
    border-bottom: 1px solid rgba(16,55,102,0.12) !important;
    padding-bottom: 8px !important;
}

/* ── DATAFRAMES / TABLES ────────────────────────────────────────────────── */
[data-testid="stDataFrame"]       { border-radius: 9px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] > div { border-radius: 9px !important; }

/* ── ALERT BOXES ────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    border: 1px solid rgba(16,55,102,0.15) !important;
}

/* ── TABS ───────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"]           { background-color: transparent !important; border-bottom: 1px solid rgba(16,55,102,0.15) !important; }
[data-baseweb="tab"]                { color: #5A7BAA !important; font-weight: 400 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #103766 !important;
    border-bottom: 2px solid #288CFA !important;
    font-weight: 500 !important;
}

/* ── MISC ───────────────────────────────────────────────────────────────── */
hr { border-color: rgba(16,55,102,0.15) !important; margin: 0.75rem 0 !important; }
p  { color: #103766 !important; }

/* ── METRIC TEXT ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] p,
[data-testid="stMetric"] label,
[data-testid="stMetric"] span    { color: #5A7BAA !important; }
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p  { color: #5A7BAA !important; }
[data-testid="stMetricValue"] > div { color: #103766 !important; }

/* ── SIDEBAR WIDGETS ────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div     { border-color: rgba(16,55,102,0.20) !important; }
[data-testid="stNumberInput"] > div > div > input { border-color: rgba(16,55,102,0.20) !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # ── SESSION STATE (with URL persistence for manual signals) ─────────────
    # Manual signals (set once per weekly review, persisted in URL query params)
    defaults = {
        "eps_signal":    "Not set",
        "lei_signal":    "Not set",
        "taylor_rule":   "Not set",
        "gdpnow_signal": "Not set",
        "spy_fwd_ey":    0.0,
    }

    # Keys that persist across refreshes via URL query params
    _persist_keys = ["eps_signal", "lei_signal", "taylor_rule", "gdpnow_signal", "spy_fwd_ey"]

    # Load persisted values from URL on first visit
    qp = st.query_params
    for k, v in defaults.items():
        if k not in st.session_state:
            if k in _persist_keys and k in qp:
                # Restore from URL
                url_val = qp[k]
                if isinstance(v, float):
                    try:
                        st.session_state[k] = float(url_val)
                    except (ValueError, TypeError):
                        st.session_state[k] = v
                elif isinstance(v, int):
                    try:
                        st.session_state[k] = int(url_val)
                    except (ValueError, TypeError):
                        st.session_state[k] = v
                else:
                    st.session_state[k] = url_val
            else:
                st.session_state[k] = v

    def _save_manual_signals():
        """Persist manual signal values to URL query params."""
        params = {}
        for k in _persist_keys:
            val = st.session_state.get(k)
            if val is not None and val != defaults.get(k):
                params[k] = str(val)
        st.query_params.update(params)

    # ── AUTO DRAWDOWN TIER (computed before sidebar so the dropdown can show it) ─
    # Uses cached portfolio load + prices, so this is cheap. Tracks the high-water
    # mark in portfolio.json and maps current vs peak equity to a Layer 9 tier.
    _dd_data    = _portfolio_load()
    _dd_open    = _dd_data.get("open_positions", [])
    _dd_tickers = [p["ticker"] for p in _dd_open]
    _dd_prices  = fetch_portfolio_prices(",".join(_dd_tickers)) if _dd_tickers else {}
    _dd_equity, _, _, _ = _compute_account_value(_dd_data, _dd_prices)
    _dd_peak = _dd_data.get("peak_equity", _dd_equity)
    if _dd_equity > _dd_peak:
        _dd_peak = _dd_equity
        _dd_data["peak_equity"] = round(_dd_peak, 2)
        _portfolio_save(_dd_data)
    _dd_tier = drawdown_tier(_dd_equity, _dd_peak)
    st.session_state["drawdown_state"]     = _dd_tier["state"]   # consumed by sizing/gate
    st.session_state["drawdown_pct"]       = _dd_tier["pct"]
    st.session_state["drawdown_label"]     = _dd_tier["label"]
    st.session_state["drawdown_color"]     = _dd_tier["color"]

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 Controls")
        st.caption(f"Loaded: {datetime.now().strftime('%b %d, %Y  %I:%M %p')}")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("Manual Signals")
        st.caption("Set once at the start of each weekly review.")

        _eps_opts = ["Not set", "↑ Rising 3+ weeks ✅", "Flat", "↓ Declining ⚠️"]
        st.session_state.eps_signal = st.selectbox(
            "EPS Revisions (FactSet)", _eps_opts,
            index=_eps_opts.index(st.session_state.eps_signal),
            on_change=_save_manual_signals,
        )
        # Drawdown is auto-computed from peak equity (see above) and shown in the
        # header Drawdown tile — no manual control needed here.
        _lei_opts = ["Not set", "Rising ✅", "Flat", "6mo declining ⚠️"]
        st.session_state.lei_signal = st.selectbox(
            "Conference Board LEI", _lei_opts,
            index=_lei_opts.index(st.session_state.lei_signal),
            on_change=_save_manual_signals,
        )
        _taylor_opts = [
            "Not set",
            "Positive >1% — Fed too loose ⚠️",
            "Near zero (−1% to +1%) — neutral",
            "Negative <−1% — Fed too tight ✅",
        ]
        st.session_state.taylor_rule = st.selectbox(
            "Taylor Rule Deviation (monthly)", _taylor_opts,
            index=_taylor_opts.index(st.session_state.taylor_rule),
            on_change=_save_manual_signals,
        )
        st.session_state.spy_fwd_ey = st.number_input(
            "S&P 500 Forward Earnings Yield % (L1, monthly)",
            min_value=0.0, max_value=20.0, step=0.01, format="%.2f",
            value=float(st.session_state.spy_fwd_ey),
            on_change=_save_manual_signals,
            help="Forward earnings yield = 100 / forward P/E. Source: FactSet Earnings "
                 "Insight or multpl.com (forward). Leave 0 if not set. Feeds L1 checks 1 & 3.",
        )
        _gdpnow_opts = ["Not set", "Bullish", "Neutral", "Bearish"]
        st.session_state.gdpnow_signal = st.selectbox(
            "GDPNow vs Analyst EPS (L1, monthly)", _gdpnow_opts,
            index=_gdpnow_opts.index(st.session_state.gdpnow_signal),
            on_change=_save_manual_signals,
            help="Atlanta Fed GDPNow vs analyst EPS direction. Bullish = GDP accelerating "
                 "while estimates lag; Bearish = GDP falling while estimates haven't caught down.",
        )

        st.divider()
        st.subheader("Core Allocation (v4)")
        _core_pos_str = st.session_state.get("core_positions", "")
        _core_pct_val = st.session_state.get("core_pct_deployed", 0.0)
        st.markdown(
            f'<div style="background:#EEF3FA; border-radius:9px; padding:10px 12px; margin-bottom:8px;">'
            f'<div style="font-size:11px; color:#5A7BAA;">Core Deployed (live)</div>'
            f'<div style="font-size:17px; font-weight:500; color:#103766;">{_core_pct_val:.1f}%</div>'
            f'<div style="font-size:11px; color:#5A7BAA;">{_core_pos_str if _core_pos_str else "No Core positions"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Overrides")
        regime_ov = st.selectbox("Regime",           ["Auto", "Risk-on", "Reflation", "Deflation", "Stagflation"])
        perm_ov   = st.selectbox("Permission State", ["Auto", "Green", "Yellow", "Red"])

        st.divider()
        st.caption("Screener runs on-demand from the Screener tab.")

    # ── DATA LOADING ──────────────────────────────────────────────────────────
    # ── PORTFOLIO VALUE (computed before tabs so all tabs can use it) ────────
    _port_data = _portfolio_load()
    _port_open = _port_data.get("open_positions", [])
    _port_tickers = [p["ticker"] for p in _port_open]
    _port_prices = fetch_portfolio_prices(",".join(_port_tickers)) if _port_tickers else {}
    _acct_val, _, _, _ = _compute_account_value(_port_data, _port_prices)
    st.session_state["account_value"] = round(_acct_val)

    # Compute Core % deployed from open positions
    _core_mv = sum(
        p["shares"] * _port_prices.get(p["ticker"], p.get("current_price") or p["entry_price"])
        for p in _port_open if p.get("layer") == "Core"
    )
    st.session_state["core_pct_deployed"] = round(_core_mv / _acct_val * 100, 1) if _acct_val > 0 else 0.0
    _core_tickers = [p["ticker"] for p in _port_open if p.get("layer") == "Core"]
    st.session_state["core_positions"] = ", ".join(_core_tickers)

    with st.spinner("Loading market data..."):
        macro_close = fetch_macro_data()
        l0          = calc_layer0(macro_close)

    if "error" in l0:
        st.error(l0["error"])
        return

    with st.spinner("Loading FRED indicators..."):
        fred_data = fetch_fred_data()

    with st.spinner("Computing sector rotation (RRG)..."):
        l3_data = calc_layer3(macro_close)

    # ── REGIME AND PERMISSION STATE ───────────────────────────────────────────
    regime = regime_ov if regime_ov != "Auto" else l0["regime"]

    rec_indicators = score_recession_composite(fred_data, st.session_state.lei_signal)
    rec_flags      = sum(1 for i in rec_indicators if not i["ok"])
    rec_total      = len(rec_indicators)

    perm, limits = calc_layer2(l0, rec_flags, st.session_state.eps_signal, perm_ov)
    st.session_state["_current_perm"] = perm

    # ── GLOBAL ALERTS ─────────────────────────────────────────────────────────
    # The big title + 8-tile global strip were removed (June 2026): per-tab heroes
    # carry that context, so the bottom block was pure redundancy. Only the
    # cross-tab warning bars (liquidity override / recession) render globally —
    # and only when something is actually wrong, so a clean state shows nothing.
    liq_override = l0.get("liquidity_tighten") or l0.get("fnl_signal") == "OVERRIDE ACTIVE"
    core_pct     = st.session_state.core_pct_deployed  # used by downstream tabs

    warn_bars = ""
    if liq_override:
        reasons = []
        if l0.get("liquidity_tighten"):      reasons.append("HYG/IEF declining")
        if l0.get("fnl_signal") == "OVERRIDE ACTIVE": reasons.append(f"Fed Net Liquidity −${abs(l0.get('fnl_change_4w', 0)):.0f}B (4-week)")
        warn_bars += _gate_bar_html("Red", f"Liquidity override active — {'; '.join(reasons)}. Reduce exposure immediately.")
    if rec_flags >= 3:
        warn_bars += _gate_bar_html("Red",    f"Recession composite elevated: {rec_flags}/{rec_total} indicators flagging.")
    elif rec_flags >= 1:
        warn_bars += _gate_bar_html("Yellow", f"Recession composite: {rec_flags}/{rec_total} indicator(s) flagging — monitor.")

    # Global alerts render below the nav, above tab content (only if non-empty).
    header_html = warn_bars

    # ── TABS — rendered at the very top, styled as webpage nav links ──────────
    st.markdown("""
    <style>
    /* Pull the tab nav to the very top of the page */
    div[data-testid="stTabs"] { margin-top: -0.5rem; }
    /* Nav row — sits flush at top, thin baseline rule like a site header */
    div[data-testid="stTabs"] > div[role="tablist"] {
        border-bottom: 1px solid rgba(16,55,102,0.12);
        gap: 4px;
        margin-bottom: 1rem;
    }
    /* Tabs as text hyperlinks */
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 14px;
        font-weight: 400;
        color: #288CFA;                     /* link blue */
        padding: 6px 14px;
        border: none;
        background: transparent;
        text-decoration: underline;
        text-underline-offset: 4px;
        text-decoration-thickness: 1px;
        text-decoration-color: rgba(40,140,250,0.45);
        transition: color 0.12s, text-decoration-color 0.12s;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        font-size: 14px; font-weight: 400;  /* override Streamlit inner <p> */
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #103766;
        text-decoration-color: #103766;     /* darken underline on hover */
    }
    /* Active tab — visited/current link: darker, no underline, bold */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #103766;
        font-weight: 500;
        text-decoration: none;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        font-weight: 500;
    }
    /* Hide the sliding red indicator bar Streamlit draws under tabs */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Macro & Permission",
        "Sector Rotation",
        "Core Allocation",
        "Screener",
        "Position Sizer",
        "Charts",
        "Portfolio",
    ])

    # Global alert bars render here (below nav, above tab content) — only if any.
    if header_html:
        st.markdown(header_html, unsafe_allow_html=True)

    with tab1:
        _render_layer0_2_tab(l0, fred_data, rec_indicators, rec_flags, rec_total, perm, limits, l3_data, regime)

    with tab2:
        _render_layer3_tab(l3_data)

    with tab3:
        _render_core_tab(l0, l3_data, perm)

    with tab4:
        _render_layer4_tab(perm, regime, l0)

    with tab5:
        # Position Sizer — standalone tab, reads selected ticker from screener
        results_df = st.session_state.get("screener_results")
        if results_df is not None and not results_df.empty:
            # Build candidate list from screener results that have entry/stop
            sizer_candidates = results_df[
                results_df["Entry"].notna() & results_df["Stop"].notna()
            ].copy()
            _render_position_sizer_tab(sizer_candidates, perm, l0)
        else:
            st.markdown(
                '<p style="font-size:14px; color:#5A7BAA; text-align:center; padding:40px;">'
                'Run the screener first (Screener tab) to populate candidates for sizing.</p>',
                unsafe_allow_html=True,
            )

    with tab6:
        passes_df = st.session_state.get("screener_passes", pd.DataFrame())
        _render_charts_tab(passes_df)

    with tab7:
        _render_portfolio_tab()


if __name__ == "__main__":
    main()
