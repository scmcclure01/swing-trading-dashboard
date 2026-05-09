"""
Swing Trading Framework — Streamlit Dashboard
Automates Layers 0, 1, and 2
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
import warnings
warnings.filterwarnings("ignore")

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swing Trading Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Industrials":            "XLI",
    "Technology":             "XLK",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Utilities":              "XLU",
    "Health Care":            "XLV",
}

SECTOR_TICKERS = {
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "DVN","HAL","BKR","FANG","MRO","APA","EQT","CTRA","TRGP","OKE",
        "KMI","WMB","LNG","CVI","MGY",
    ],
    "Materials": [
        "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","ALB",
        "DD","EMN","IFF","PPG","RPM","FMC","MOS","CF","BALL","IP",
        "PKG","SEE","CCK","AVY","SON","AMCR","CE","DOW","LYB","WLK",
    ],
    "Industrials": [
        "RTX","HON","UNP","UPS","BA","LMT","GE","CAT","DE","MMM",
        "ITW","EMR","ETN","PH","ROK","FDX","CSX","NSC","WM","RSG",
        "CTAS","CPRT","GWW","AME","TT","IR","CARR","OTIS","PWR","URI",
        "MAS","JCI","XYL","AXON","TDG","HWM","NOC","GD","LHX","LDOS",
        "HUBB","FTV","RRX","GNRC","SAIA","ODFL","JBHT","EXPD","TXT",
    ],
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","AMD","QCOM","TXN",
        "AMAT","LRCX","KLAC","MU","ADI","MCHP","CDNS","SNPS","FTNT",
        "PANW","CRWD","NOW","ZS","DDOG","NET",
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG",
        "ROST","ORLY","AZO","DHI","LEN","PHM","ULTA","YUM","DRI",
    ],
    "Financials": [
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW",
        "AXP","C","USB","PNC","TFC","COF","ICE","CME","SPGI","MCO",
    ],
    "Consumer Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","GIS","K",
        "SJM","HRL","CAG","CPB","MKC","CHD","CLX","KMB","MDLZ",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","ETR","ED",
        "XEL","WEC","ES","AWK","DTE","FE","PPL","AEE","CMS","NI",
    ],
    "Health Care": [
        "LLY","UNH","JNJ","ABT","TMO","DHR","BMY","AMGN","ISRG","MDT",
        "SYK","BSX","EW","BDX","IDXX","DXCM",
    ],
}

ALL_SECTORS       = list(SECTOR_ETFS.keys())
RISK_ON_SECTORS   = {"Technology", "Financials", "Consumer Discretionary", "Industrials"}
REFLATION_SECTORS = {"Energy", "Materials"}
DEFENSIVE_SECTORS = {"Consumer Staples", "Utilities", "Health Care"}

PERM_LIMITS = {
    "Green":  {"max_pos": 20, "risk_lo": 0.75, "risk_hi": 1.00, "heat": 15},
    "Yellow": {"max_pos": 10, "risk_lo": 0.25, "risk_hi": 0.50, "heat": 8},
    "Red":    {"max_pos":  5, "risk_lo": 0.00, "risk_hi": 0.00, "heat": 3},
}

SETUP_STYLE = {
    "Green":  "Momentum breakouts",
    "Yellow": "Pullbacks to 20d / 50d MA",
    "Red":    "No new entries — protect capital",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data():
    """SPY, sector ETFs, TLT, HYG, IEF — 1 year."""
    tickers = ["SPY", "TLT", "HYG", "IEF"] + list(SECTOR_ETFS.values())
    raw   = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    close = raw["Close"]
    try:
        vix = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
        close["^VIX"] = vix["Close"].squeeze()
    except Exception:
        pass
    return close


@st.cache_data(ttl=86400, show_spinner=False)   # FRED updates daily/monthly — cache 24h
def fetch_fred_data():
    """Pull recession composite indicators from FRED."""
    if not FRED_AVAILABLE:
        return {"error": "fredapi package not installed"}
    try:
        api_key = st.secrets.get("FRED_API_KEY", "")
        if not api_key:
            return {"error": "Add FRED_API_KEY to Streamlit Secrets to enable recession composite"}
        fred = Fred(api_key=api_key)

        def latest(series_id, start="2023-01-01"):
            s = fred.get_series(series_id, observation_start=start).dropna()
            return float(s.iloc[-1]), str(s.index[-1].date())

        data = {}
        data["sahm"],    data["sahm_date"]    = latest("SAHMREALTIME")
        data["cfnai"],   data["cfnai_date"]   = latest("CFNAIMA3")
        data["t10y3m"],  data["t10y3m_date"]  = latest("T10Y3M")
        data["recprob"], data["recprob_date"] = latest("RECPROUSM156N")
        return data
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(sectors_key: str):
    """Download 1-year OHLCV for all tickers in the given sectors."""
    sectors = sectors_key.split(",")
    all_t   = [t for s in sectors for t in SECTOR_TICKERS.get(s, [])]
    raw    = yf.download(all_t + ["SPY"], period="1y", auto_adjust=True, progress=False)
    close  = raw["Close"]
    volume = raw["Volume"]
    return close, volume, all_t


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0
# ─────────────────────────────────────────────────────────────────────────────
def calc_layer0(close: pd.DataFrame) -> dict:
    r = {}

    def s(name):
        return close[name].dropna() if name in close.columns else pd.Series(dtype=float)

    spy = s("SPY")
    if len(spy) < 126:
        return {"error": "Insufficient SPY data — try refreshing."}

    # SPY two-speed (ROC 21 = 1 month, ROC 126 = 6 month)
    r["spy_price"]    = float(spy.iloc[-1])
    r["spy_ma20"]     = float(spy.rolling(20).mean().iloc[-1])
    r["spy_ma50"]     = float(spy.rolling(50).mean().iloc[-1])
    r["spy_above_20"] = bool(spy.iloc[-1] > spy.rolling(20).mean().iloc[-1])
    r["spy_above_50"] = bool(spy.iloc[-1] > spy.rolling(50).mean().iloc[-1])
    r["spy_ret_1m"]   = float(spy.iloc[-1] / spy.iloc[-21]  - 1)
    r["spy_ret_3m"]   = float(spy.iloc[-1] / spy.iloc[-63]  - 1)
    r["spy_ret_6m"]   = float(spy.iloc[-1] / spy.iloc[-126] - 1)

    # TLT
    tlt = s("TLT")
    if len(tlt) >= 21:
        r["tlt_above_50"] = bool(tlt.iloc[-1] > tlt.rolling(50).mean().iloc[-1]) if len(tlt) >= 50 else None
        r["tlt_ret_1m"]   = float(tlt.iloc[-1] / tlt.iloc[-21] - 1)
        # 4-week direction: flat if |ret| < 1%
        if   r["tlt_ret_1m"] >  0.01: r["tlt_direction"] = "Rising"
        elif r["tlt_ret_1m"] < -0.01: r["tlt_direction"] = "Declining"
        else:                          r["tlt_direction"] = "Flat"

        # TLT + SPY combined regime confirmation (Signal 3)
        tlt_rising = r["tlt_direction"] == "Rising"
        spy_rising = r.get("spy_ret_1m", 0) > 0
        if   tlt_rising and spy_rising:       r["tlt_spy_signal"] = "TLT ↑ + SPY ↑ → Risk-on confirmed"
        elif not tlt_rising and spy_rising:   r["tlt_spy_signal"] = "TLT ↓ + SPY ↑ → Reflation regime"
        elif tlt_rising and not spy_rising:   r["tlt_spy_signal"] = "TLT ↑ + SPY ↓ → Deflationary — reduce significantly"
        else:                                 r["tlt_spy_signal"] = "TLT ↓ + SPY ↓ → Stagflation / liquidity crisis"
    else:
        r["tlt_above_50"]  = None
        r["tlt_ret_1m"]    = None
        r["tlt_direction"] = "N/A"
        r["tlt_spy_signal"] = "N/A"

    # Liquidity: HYG / IEF ratio
    hyg, ief = s("HYG"), s("IEF")
    if len(hyg) >= 21 and len(ief) >= 21:
        ief_a  = ief.reindex(hyg.index).ffill()
        ratio  = (hyg / ief_a).dropna()
        r["hyg_ief_ratio"]    = float(ratio.iloc[-1])
        r["hyg_ief_1m_ago"]   = float(ratio.iloc[-21])
        r["liquidity_tighten"] = bool(ratio.iloc[-1] < ratio.iloc[-21])
    else:
        r["hyg_ief_ratio"]    = None
        r["hyg_ief_1m_ago"]   = None
        r["liquidity_tighten"] = False

    # VIX
    vix = s("^VIX")
    r["vix"]          = float(vix.iloc[-1]) if len(vix) > 0 else None
    r["vix_elevated"]  = bool(r["vix"] > 25) if r["vix"] else False

    # Sector RS vs SPY
    sector_rs = {}
    for sector, etf in SECTOR_ETFS.items():
        ep = s(etf)
        if len(ep) < 63:
            continue
        idx  = ep.index.intersection(spy.index)
        ep_a = ep.loc[idx]
        sa   = spy.loc[idx]
        if len(ep_a) < 63:
            continue

        rs_line   = ep_a / sa
        rs_1m     = float(ep_a.iloc[-1]/ep_a.iloc[-21] - 1) - float(sa.iloc[-1]/sa.iloc[-21] - 1)
        rs_3m     = float(ep_a.iloc[-1]/ep_a.iloc[-63] - 1) - float(sa.iloc[-1]/sa.iloc[-63] - 1)
        rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-63:].max() * 0.98)

        if   rs_1m > 0 and rs_3m > 0: trend = "Leading"
        elif rs_1m > 0 or  rs_3m > 0: trend = "Mixed"
        else:                          trend = "Lagging"

        sector_rs[sector] = {
            "etf": etf, "price": round(float(ep_a.iloc[-1]), 2),
            "ret_1m": float(ep_a.iloc[-1]/ep_a.iloc[-21] - 1),
            "ret_3m": float(ep_a.iloc[-1]/ep_a.iloc[-63] - 1),
            "rs_1m": rs_1m, "rs_3m": rs_3m,
            "rs_new_hi": rs_new_hi, "trend": trend,
        }

    r["sector_rs"]       = sector_rs
    r["leading_sectors"] = [s for s, v in sector_rs.items() if v["trend"] == "Leading"]
    r["mixed_sectors"]   = [s for s, v in sector_rs.items() if v["trend"] == "Mixed"]

    # Regime
    leading = set(r["leading_sectors"])
    ro = len(leading & RISK_ON_SECTORS)
    re = len(leading & REFLATION_SECTORS)
    de = len(leading & DEFENSIVE_SECTORS)

    if   re >= 2 and de >= 1: r["regime"] = "Stagflation"
    elif ro >= 2:              r["regime"] = "Risk-on"
    elif re >= 2:              r["regime"] = "Reflation"
    elif de >= 2:              r["regime"] = "Deflation"
    else:                      r["regime"] = "Mixed"

    return r


# ─────────────────────────────────────────────────────────────────────────────
# RECESSION COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────
def score_recession_composite(fred: dict, lei_manual: str) -> list:
    """
    Returns a list of indicator dicts with keys:
    name, value, as_of, threshold, ok (bool), note
    """
    indicators = []

    if "error" not in fred:
        # Chauvet-Piger — flag if > 50%
        cp = fred.get("recprob")
        indicators.append({
            "name":      "Chauvet-Piger Recession Prob",
            "value":     f"{cp:.1f}%" if cp is not None else "N/A",
            "as_of":     fred.get("recprob_date", ""),
            "threshold": "< 50%",
            "ok":        (cp < 50) if cp is not None else True,
        })
        # Sahm Rule — flag if ≥ 0.50
        sahm = fred.get("sahm")
        indicators.append({
            "name":      "Sahm Rule",
            "value":     f"{sahm:.2f}" if sahm is not None else "N/A",
            "as_of":     fred.get("sahm_date", ""),
            "threshold": "< 0.50",
            "ok":        (sahm < 0.50) if sahm is not None else True,
        })
        # CFNAI 3-mo MA — flag if ≤ -0.70
        cfnai = fred.get("cfnai")
        indicators.append({
            "name":      "CFNAI 3-mo MA",
            "value":     f"{cfnai:.2f}" if cfnai is not None else "N/A",
            "as_of":     fred.get("cfnai_date", ""),
            "threshold": "> -0.70",
            "ok":        (cfnai > -0.70) if cfnai is not None else True,
        })
        # 10Y-3M Yield Curve — flag if < 0
        t10y3m = fred.get("t10y3m")
        indicators.append({
            "name":      "10Y-3M Yield Curve",
            "value":     f"{t10y3m:+.2f}%" if t10y3m is not None else "N/A",
            "as_of":     fred.get("t10y3m_date", ""),
            "threshold": "> 0% (uninverted)",
            "ok":        (t10y3m >= 0) if t10y3m is not None else True,
        })

    # Conference Board LEI — manual
    # "Not set" = unknown/skip (don't count as flag). Only flag if explicitly declining.
    lei_ok = lei_manual != "6mo declining ⚠️"
    indicators.append({
        "name":      "Conference Board LEI",
        "value":     lei_manual,
        "as_of":     "manual",
        "threshold": "Not declining 6mo",
        "ok":        lei_ok,
    })

    return indicators


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1
# ─────────────────────────────────────────────────────────────────────────────
def calc_layer1(l0: dict, rec_flags: int, eps_signal: str, override: str = "Auto") -> tuple:
    """
    Permission state per framework v3:
      Green:  SPY both positive + rec composite < 2/5 + EPS not declining + no liquidity override
      Yellow: SPY mixed OR rec composite 2-3/5 OR EPS declining
      Red:    SPY both negative OR liquidity override OR rec composite 4+/5
    """
    if override != "Auto":
        return override, PERM_LIMITS[override]

    roc1_pos = l0.get("spy_ret_1m", 0) > 0
    roc6_pos = l0.get("spy_ret_6m", 0) > 0
    liq_override = l0.get("liquidity_tighten") and l0.get("vix_elevated")
    eps_declining = "Declining" in eps_signal

    # Red — hard stops
    if liq_override or (not roc1_pos and not roc6_pos) or rec_flags >= 4:
        perm = "Red"
    # Yellow — caution conditions
    elif rec_flags >= 2 or eps_declining or not roc1_pos or not roc6_pos:
        perm = "Yellow"
    # Green — all clear
    else:
        perm = "Green"

    return perm, PERM_LIMITS[perm]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 SCREENER
# ─────────────────────────────────────────────────────────────────────────────
def run_screener(
    close: pd.DataFrame, volume: pd.DataFrame,
    tickers: list, ticker_sector: dict, spy: pd.Series,
    min_dollar_vol: float = 10_000_000, rs_lookback: int = 63,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        px  = close[t].dropna()
        vol = volume[t].dropna() if t in volume.columns else pd.Series(dtype=float)
        if len(px) < 63:
            continue

        price         = float(px.iloc[-1])
        ma20          = float(px.rolling(20).mean().iloc[-1])
        ma50          = float(px.rolling(50).mean().iloc[-1])
        avg_share_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0.0
        avg_dollar_vol = price * avg_share_vol   # dollar volume = price × shares
        ret_1m        = float(px.iloc[-1] / px.iloc[-21] - 1)
        ret_3m        = float(px.iloc[-1] / px.iloc[-63] - 1)

        spy_a     = spy.reindex(px.index).ffill()
        rs_line   = px / spy_a
        rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-rs_lookback:].max() * 0.98)
        # RS direction: is RS line higher now than 21 days ago?
        rs_rising = bool(len(rs_line) >= 21 and rs_line.iloc[-1] > rs_line.iloc[-21])

        try:
            rsi_v  = float(RSIIndicator(close=px, window=14).rsi().iloc[-1])
            mo     = MACDIndicator(close=px, window_slow=26, window_fast=12, window_sign=9)
            m_hist = float(mo.macd_diff().iloc[-1])
            m_bull = bool(mo.macd().iloc[-1] > mo.macd_signal().iloc[-1])
        except Exception:
            rsi_v, m_hist, m_bull = 50.0, 0.0, False

        above_20 = price > ma20
        above_50 = price > ma50
        vol_ok   = avg_dollar_vol >= min_dollar_vol

        if   ret_1m > 0 and ret_3m > 0: two_spd = "FULL"
        elif ret_1m > 0 or  ret_3m > 0: two_spd = "HALF"
        else:                            two_spd = "NO TRADE"

        passes = above_20 and above_50 and vol_ok and rs_new_hi and rs_rising and two_spd == "FULL"

        rows.append({
            "Ticker": t, "Sector": ticker_sector.get(t, ""),
            "Price": round(price, 2),
            "vs 20MA": above_20, "vs 50MA": above_50,
            "1M Ret": ret_1m, "3M Ret": ret_3m,
            "RS Hi": rs_new_hi, "RS ↑": rs_rising, "RSI": round(rsi_v, 1),
            "MACD": m_bull, "MACD Hist": round(m_hist, 4),
            "Avg $Vol(M)": round(avg_dollar_vol / 1e6, 1),
            "2-Speed": two_spd, "PASS": passes,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def build_chart(ticker: str):
    hist     = yf.Ticker(ticker).history(period="6mo")
    spy_hist = yf.Ticker("SPY").history(period="6mo")
    if hist.empty:
        return None

    px  = hist["Close"].dropna()
    op  = hist["Open"].reindex(px.index)
    hi  = hist["High"].reindex(px.index)
    lo  = hist["Low"].reindex(px.index)
    vol = hist["Volume"].reindex(px.index)

    ma20 = px.rolling(20).mean()
    ma50 = px.rolling(50).mean()
    spy_a = spy_hist["Close"].reindex(px.index).ffill()
    rs    = px / spy_a
    rsi   = RSIIndicator(close=px, window=14).rsi()
    mo    = MACDIndicator(close=px, window_slow=26, window_fast=12, window_sign=9)
    ml, mg, mh = mo.macd(), mo.macd_signal(), mo.macd_diff()

    def tl(series):
        return [None if pd.isna(v) else float(v) for v in series]

    dates = [str(d)[:10] for d in px.index]

    # row_heights=[0.38,0.12,0.18,0.16,0.16], vertical_spacing=0.025
    # Panel domain tops (paper coords, bottom→top): MACD=0.144, RSI=0.313, RS=0.500, Vol=0.633, Price=1.000
    PANEL_SEP    = [0.633, 0.500, 0.313, 0.144]   # separator line y positions
    PANEL_LABELS = [                               # (label_text, y_position, x_anchor)
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
    fig.add_trace(go.Scatter(x=dates, y=tl(ma20), name="20d MA", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=tl(ma50), name="50d MA", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)

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

    # Separator lines between panels
    for y in PANEL_SEP:
        fig.add_shape(
            type="line", xref="paper", yref="paper",
            x0=0, x1=1, y0=y, y1=y,
            line=dict(color="#6b7280", width=2),
        )

    # Panel labels — placed just inside the top of each panel
    for label, y, x in PANEL_LABELS:
        fig.add_annotation(
            text=f"<b>{label}</b>",
            xref="paper", yref="paper",
            x=x, y=y,
            showarrow=False,
            font=dict(color="#d1d5db", size=11),
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        height=750, paper_bgcolor="#111827", plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db", size=11), margin=dict(l=50, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.01, bgcolor="rgba(0,0,0,0)"),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="#374151", row=i, col=1)
        fig.update_yaxes(gridcolor="#374151", row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pct(v):  return f"{v*100:+.1f}%"
def icon(v): return "✅" if v else "❌"
def macd(v): return "▲" if v else "▼"

def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["1M Ret", "3M Ret"]:
        if col in d.columns: d[col] = d[col].apply(pct)
    for col in ["vs 20MA", "vs 50MA", "RS Hi", "RS ↑"]:
        if col in d.columns: d[col] = d[col].apply(icon)
    if "MACD"        in d.columns: d["MACD"]        = d["MACD"].apply(macd)
    if "Avg $Vol(M)" in d.columns: d["Avg $Vol(M)"] = d["Avg $Vol(M)"].apply(lambda x: f"${x:.1f}M")
    return d.drop(columns=["PASS", "MACD Hist"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── SESSION STATE (manual signals) ────────────────────────────────────────
    defaults = {
        "eps_signal":    "Not set",
        "fed_liquidity": "Not set",
        "drawdown_state": "At or near peak — full risk",
        "lei_signal":    "Not set",
        "taylor_rule":   "Not set",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── CSS ───────────────────────────────────────────────────────────────────
    # Design standard: Page #060C1C · Card #1D3B93 · Panel #D8E6FF · Accent #93B6FA
    # Border: 1px solid rgba(255,255,255,0.28) applied to all shaped elements
    # Signal colors: Positive #27500A · Neutral #E07800 · Negative #CC1111
    st.markdown("""
    <style>
    /* ── PAGE BACKGROUND ───────────────────────────────────────────────────── */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main, .main > div {
        background-color: #060C1C !important;
    }
    .block-container {
        padding-top: 1.25rem !important;
        background-color: #060C1C !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ───────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #0D1B42 !important;
        border-right: 1px solid rgba(147,182,250,0.30) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: #93B6FA !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

    /* ── HEADINGS ──────────────────────────────────────────────────────────── */
    h2 { color: #FFFFFF !important; font-weight: 500 !important; font-size: 18px !important; }
    h3 { color: #FFFFFF !important; font-weight: 500 !important; }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] { color: #5A78C0 !important; font-size: 11px !important; }
    h4 {
        color: #93B6FA !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid rgba(147,182,250,0.30) !important;
        padding-bottom: 7px !important;
        margin-top: 2px !important;
        margin-bottom: 12px !important;
    }

    /* ── METRIC PANELS (#D8E6FF) ───────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #D8E6FF !important;
        border: 1px solid rgba(147,182,250,0.50) !important;
        border-radius: 9px !important;
        padding: 10px 12px !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        font-size: 11px !important;
        color: #3A5EAA !important;
        font-weight: 400 !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 17px !important;
        font-weight: 500 !important;
        color: #1D3B93 !important;
    }
    [data-testid="stMetricDelta"] { color: #3B6D11 !important; font-size: 11px !important; }
    [data-testid="stMetricDelta"] svg { display: none; }

    /* ── SECTION CARDS (#1D3B93) ───────────────────────────────────────────── */
    /* Primary selector */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(147,182,250,0.40) !important;
        padding: 15px 17px !important;
        background-color: #1D3B93 !important;
        box-shadow: 0 0 0 1px rgba(147,182,250,0.15) !important;
    }
    /* Fallback: Streamlit 1.30+ renamed the wrapper */
    div[class*="stVerticalBlock"][class*="Border"],
    div[class*="withBorder"] {
        border-radius: 12px !important;
        border: 1px solid rgba(147,182,250,0.40) !important;
        background-color: #1D3B93 !important;
    }
    /* Inner content area of bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: #1D3B93 !important;
    }

    /* ── DATAFRAMES / TABLES ────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 9px !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] > div { border-radius: 9px !important; }

    /* ── ALERT BOXES ────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 9px !important;
        border: 1px solid rgba(147,182,250,0.30) !important;
    }

    /* ── TABS ───────────────────────────────────────────────────────────────── */
    [data-baseweb="tab-list"] { background-color: transparent !important; border-bottom: 1px solid rgba(147,182,250,0.20) !important; }
    [data-baseweb="tab"] { color: #93B6FA !important; font-weight: 400 !important; }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #93B6FA !important;
        font-weight: 500 !important;
    }

    /* ── MISC ───────────────────────────────────────────────────────────────── */
    hr { border-color: rgba(147,182,250,0.25) !important; margin: 0.75rem 0 !important; }
    p { color: #C8D8F8 !important; }

    /* Sidebar widgets */
    [data-testid="stSelectbox"] > div > div { border-color: rgba(147,182,250,0.30) !important; }
    [data-testid="stNumberInput"] > div > div > input { border-color: rgba(147,182,250,0.30) !important; }
    </style>
    """, unsafe_allow_html=True)

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

        st.session_state.eps_signal = st.selectbox(
            "EPS Revisions (FactSet)",
            ["Not set", "↑ Rising 3+ weeks ✅", "Flat", "↓ Declining ⚠️"],
            index=["Not set", "↑ Rising 3+ weeks ✅", "Flat", "↓ Declining ⚠️"].index(
                st.session_state.eps_signal
            ),
        )
        st.session_state.fed_liquidity = st.selectbox(
            "Fed Net Liquidity (TV: jlb05013)",
            ["Not set", "Expanding ✅", "Flat / Neutral", "Contracting ⚠️"],
            index=["Not set", "Expanding ✅", "Flat / Neutral", "Contracting ⚠️"].index(
                st.session_state.fed_liquidity
            ),
        )
        st.session_state.drawdown_state = st.selectbox(
            "Drawdown from Peak Equity",
            ["At or near peak — full risk", "5–10% drawdown — reduce risk",
             "10–15% drawdown — defensive", ">15% drawdown — cash"],
            index=["At or near peak — full risk", "5–10% drawdown — reduce risk",
                   "10–15% drawdown — defensive", ">15% drawdown — cash"].index(
                st.session_state.drawdown_state
            ),
        )
        st.session_state.lei_signal = st.selectbox(
            "Conference Board LEI",
            ["Not set", "Rising ✅", "Flat", "6mo declining ⚠️"],
            index=["Not set", "Rising ✅", "Flat", "6mo declining ⚠️"].index(
                st.session_state.lei_signal
            ),
        )

        _taylor_opts = [
            "Not set",
            "Positive >1% — Fed too loose ⚠️",
            "Near zero (−1% to +1%) — neutral",
            "Negative <−1% — Fed too tight ✅",
        ]
        st.session_state.taylor_rule = st.selectbox(
            "Taylor Rule Deviation (monthly)",
            _taylor_opts,
            index=_taylor_opts.index(st.session_state.taylor_rule),
        )

        st.divider()
        st.subheader("Overrides")
        regime_ov = st.selectbox("Regime", ["Auto","Risk-on","Reflation","Deflation","Stagflation","Mixed"])
        perm_ov   = st.selectbox("Permission State", ["Auto","Green","Yellow","Red"])

        st.divider()
        st.subheader("Screener Settings")
        min_vol   = st.number_input("Min Avg Dollar Volume ($M)", value=10, step=5, format="%d") * 1_000_000
        rs_lb     = st.slider("RS Lookback (days)", 21, 126, 63)
        show_half = st.checkbox("Show Half Signal watchlist", value=True)
        show_all  = st.checkbox("Show full universe table",   value=False)

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        macro_close = fetch_macro_data()
        l0 = calc_layer0(macro_close)

    if "error" in l0:
        st.error(l0["error"])
        return

    with st.spinner("Loading FRED indicators..."):
        fred_data = fetch_fred_data()

    # ── REGIME / PERMISSION ───────────────────────────────────────────────────
    regime = regime_ov if regime_ov != "Auto" else l0["regime"]

    # Recession composite must be scored before Layer 1 (feeds permission state)
    rec_indicators = score_recession_composite(fred_data, st.session_state.lei_signal)
    rec_flags      = sum(1 for i in rec_indicators if not i["ok"])
    rec_total      = len(rec_indicators)

    perm, limits = calc_layer1(l0, rec_flags, st.session_state.eps_signal, perm_ov)

    # Sectors to screen
    regime_sectors = {
        "Risk-on":     list(RISK_ON_SECTORS),                                           # includes Industrials
        "Reflation":   list(REFLATION_SECTORS | {"Industrials"}),                       # XLI in both per Layer 2 Step 1
        "Deflation":   list(DEFENSIVE_SECTORS),
        "Stagflation": list(REFLATION_SECTORS | {"Industrials", "Consumer Staples", "Utilities"}),
    }
    # Mixed: use Leading + Mixed trend sectors for a broader but still filtered universe
    mixed_auto = l0.get("leading_sectors", []) + l0.get("mixed_sectors", [])
    auto_sectors = regime_sectors.get(regime, mixed_auto or ALL_SECTORS)

    with st.sidebar:
        st.divider()
        st.subheader("Sectors to Screen")
        selected_sectors = st.multiselect(
            "Override if needed",
            ALL_SECTORS,
            default=auto_sectors,
            help="Auto-set from regime. Adjust to add/remove sectors.",
        )
    active_sectors = selected_sectors if selected_sectors else auto_sectors

    # ── LOAD SCREENER DATA ────────────────────────────────────────────────────
    universe_size = sum(len(SECTOR_TICKERS.get(s, [])) for s in active_sectors)
    with st.spinner(f"Loading {universe_size} stocks..."):
        sectors_key             = ",".join(sorted(active_sectors))
        close_sc, vol_sc, all_t = fetch_screener_data(sectors_key)

    ticker_sector = {t: sec for sec in active_sectors for t in SECTOR_TICKERS.get(sec, [])}
    spy_sc        = close_sc["SPY"] if "SPY" in close_sc.columns else pd.Series(dtype=float)

    with st.spinner("Running screener..."):
        results_df = run_screener(close_sc, vol_sc, all_t, ticker_sector, spy_sc, min_vol, rs_lb)  # min_vol already in dollars

    if results_df.empty:
        st.error("Screener returned no results. Check your internet connection and refresh.")
        return

    passes_df = results_df[results_df["PASS"]].sort_values(["Sector","3M Ret"], ascending=[True, False])
    half_df   = results_df[(results_df["2-Speed"] == "HALF") & (~results_df["PASS"])].sort_values("3M Ret", ascending=False)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("## Swing Trading Framework")
    st.caption(f"{datetime.now().strftime('%A, %B %d, %Y')}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime",        regime)
    c2.metric("Permission",    perm)
    c3.metric("SPY",           f"${l0['spy_price']:.2f}", f"{l0['spy_ret_1m']*100:+.1f}% 1M")
    c4.metric("Max Positions", limits["max_pos"])
    c5.metric("Risk / Trade",  f"{limits['risk_lo']}–{limits['risk_hi']}%" if limits["risk_hi"] > 0 else "No new trades")
    c6.metric("Max Heat",      f"{limits['heat']}%")

    # Permission banner
    if   perm == "Green":  st.success(f"🟢 **GREEN STATE** — Full deployment. {SETUP_STYLE['Green']}. Up to {limits['max_pos']} positions.")
    elif perm == "Yellow": st.warning(f"🟡 **YELLOW STATE** — Selective. {SETUP_STYLE['Yellow']}. {limits['max_pos']} positions max.")
    else:                  st.error(  f"🔴 **RED STATE** — {SETUP_STYLE['Red']}. {limits['max_pos']} positions max.")

    if l0.get("liquidity_tighten"):
        st.warning("⚠️ Liquidity tightening (HYG/IEF ratio declining). Override in effect — reduce exposure.")
    if rec_flags >= 3:
        st.error(f"⚠️ Recession composite elevated: {rec_flags}/{rec_total} indicators flagging.")
    elif rec_flags >= 1:
        st.warning(f"⚠️ Recession composite: {rec_flags}/{rec_total} indicator(s) flagging — monitor.")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Layer 0 — Macro", "🔍 Layer 2 — Screener", "📈 Charts"])

    # ── TAB 1: MACRO ──────────────────────────────────────────────────────────
    with tab1:
        roc1_pos   = l0["spy_ret_1m"] > 0
        roc6_pos   = l0["spy_ret_6m"] > 0
        spy_signal = "GREEN" if (roc1_pos and roc6_pos) else ("RED" if (not roc1_pos and not roc6_pos) else "MIXED")
        tlt_dir    = l0.get("tlt_direction", "N/A")
        vix_v      = l0.get("vix")
        hyg_r      = l0.get("hyg_ief_ratio")
        hyg_ago    = l0.get("hyg_ief_1m_ago")

        col_l, col_r = st.columns([1, 1], gap="medium")

        # ── LEFT COLUMN ───────────────────────────────────────────────────────
        with col_l:

            # SPY Two-Speed card
            with st.container(border=True):
                st.markdown("#### SPY Two-Speed Trend")
                s1, s2 = st.columns(2)
                with s1:
                    st.metric(
                        "1-Month (ROC 21)", pct(l0["spy_ret_1m"]),
                        delta="Positive ✅" if roc1_pos else "Negative ❌",
                        delta_color="normal" if roc1_pos else "inverse",
                    )
                with s2:
                    st.metric(
                        "6-Month (ROC 126)", pct(l0["spy_ret_6m"]),
                        delta="Positive ✅" if roc6_pos else "Negative ❌",
                        delta_color="normal" if roc6_pos else "inverse",
                    )
                if   spy_signal == "GREEN": st.success("✅ Both positive → **GREEN signal**")
                elif spy_signal == "RED":   st.error(  "❌ Both negative → **RED signal**")
                else:                       st.warning("⚠️ Mixed → **YELLOW signal**")

            st.write("")

            # Bond & Liquidity card
            with st.container(border=True):
                st.markdown("#### Bond & Liquidity")
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
                ]
                st.dataframe(pd.DataFrame(bond_rows), hide_index=True, use_container_width=True)

            st.write("")

            # Manual signals card
            with st.container(border=True):
                st.markdown("#### Signals — Set in Sidebar")
                sig_rows = [
                    {"Signal": "EPS Revisions (FactSet)",     "Value": st.session_state.eps_signal},
                    {"Signal": "Fed Net Liquidity (jlb05013)", "Value": st.session_state.fed_liquidity},
                    {"Signal": "Taylor Rule Deviation",        "Value": st.session_state.taylor_rule},
                    {"Signal": "Drawdown from Peak Equity",    "Value": st.session_state.drawdown_state},
                ]
                st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

        # ── RIGHT COLUMN ──────────────────────────────────────────────────────
        with col_r:

            # Recession Composite card
            flag_color = "🟢" if rec_flags == 0 else ("🟡" if rec_flags <= 2 else "🔴")
            with st.container(border=True):
                st.markdown(f"#### Recession Composite &nbsp; {flag_color} {rec_flags} / {rec_total}")
                if "error" in fred_data:
                    st.info(f"ℹ️ {fred_data['error']}")
                else:
                    rec_rows = [{
                        "Indicator": ind["name"],
                        "Value":     ind["value"],
                        "As of":     ind["as_of"],
                        "Threshold": ind["threshold"],
                        "Status":    "✅ OK" if ind["ok"] else "⚠️ FLAG",
                    } for ind in rec_indicators]
                    st.dataframe(pd.DataFrame(rec_rows), hide_index=True, use_container_width=True)
                    if rec_flags == 0:
                        st.success("Normal — full risk operations.")
                    elif rec_flags <= 2:
                        st.warning(f"{rec_flags} indicator(s) flagging — elevated caution.")
                    else:
                        st.error(f"{rec_flags} indicators flagging — reduce risk.")

            st.write("")

            # Sector RS card
            with st.container(border=True):
                st.markdown("#### Sector Relative Strength vs SPY")
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
                    st.dataframe(sr_df, hide_index=True, use_container_width=True, height=360)

        # ── THIS WEEK'S RULES ─────────────────────────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown("#### This Week's Rules")
            rules_text = (
                f"**Max Positions:** {limits['max_pos']}  &nbsp;|&nbsp;  "
                f"**Risk / Trade:** {limits['risk_lo']}–{limits['risk_hi']}%  &nbsp;|&nbsp;  "
                f"**Max Portfolio Heat:** {limits['heat']}%  &nbsp;|&nbsp;  "
                f"**Setup Style:** {SETUP_STYLE[perm]}"
            )
            if   perm == "Green":  st.success(rules_text)
            elif perm == "Yellow": st.warning(rules_text)
            else:                  st.error(rules_text)

    # ── TAB 2: SCREENER ───────────────────────────────────────────────────────
    with tab2:
        if perm == "Red":
            st.error("🔴 RED STATE — No new entries. Results shown for reference only.")

        st.caption(f"Sectors: {', '.join(active_sectors)}  ·  Universe: {len(results_df)} stocks")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Screened",       len(results_df))
        c2.metric("✅ Full Signal",  len(passes_df))
        c3.metric("⚠️ Half Signal", len(half_df))
        c4.metric("❌ No Trade",    len(results_df[results_df["2-Speed"] == "NO TRADE"]))

        st.divider()

        st.subheader(f"✅ Full Signal — {len(passes_df)} candidates")
        if not passes_df.empty:
            st.dataframe(fmt_df(passes_df), hide_index=True, use_container_width=True)
        else:
            st.info("No stocks passing all Layer 2 filters in the current regime.")

        if show_half:
            st.subheader("⚠️ Half Signal — Top 15 Watch List")
            if not half_df.empty:
                st.dataframe(fmt_df(half_df.head(15)), hide_index=True, use_container_width=True)
            else:
                st.info("No half-signal candidates.")

        if show_all:
            with st.expander(f"Full Universe — {len(results_df)} stocks"):
                all_s = results_df.sort_values(["PASS","2-Speed","3M Ret"], ascending=[False,True,False])
                st.dataframe(fmt_df(all_s), hide_index=True, use_container_width=True)

    # ── TAB 3: CHARTS ─────────────────────────────────────────────────────────
    with tab3:
        if passes_df.empty:
            st.info("No Full Signal candidates to chart.")
        else:
            sel = st.selectbox(
                "Select ticker",
                passes_df["Ticker"].tolist(),
                format_func=lambda t: f"{t}  —  {passes_df[passes_df['Ticker']==t]['Sector'].iloc[0]}",
            )
            if sel:
                row = passes_df[passes_df["Ticker"] == sel].iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price",  f"${row['Price']:.2f}")
                c2.metric("1M Ret", pct(row["1M Ret"]))
                c3.metric("3M Ret", pct(row["3M Ret"]))
                c4.metric("RSI",    f"{row['RSI']:.1f}")
                with st.spinner(f"Building {sel} chart..."):
                    fig = build_chart(sel)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Could not load chart data for {sel}.")

            if st.checkbox("Show all Full Signal charts"):
                for _, row in passes_df.iterrows():
                    t = row["Ticker"]
                    with st.expander(f"{t}  —  {row['Sector']}  |  1M: {pct(row['1M Ret'])}  |  3M: {pct(row['3M Ret'])}", expanded=False):
                        with st.spinner(f"Loading {t}..."):
                            fig = build_chart(t)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
