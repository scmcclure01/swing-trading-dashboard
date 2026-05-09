"""
Swing Trading Framework — Streamlit Dashboard
Automates Layers 0, 1, 1.5, and 2.

Architecture:
  Constants → Data fetching → Layer calculations → Chart builders →
  Helpers → Tab render functions → main()
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

# Lookback periods (trading days)
LB_1M = 21    # ~1 calendar month
LB_3M = 63    # ~3 calendar months
LB_6M = 126   # ~6 calendar months

# RS new-high threshold: within 2% of the recent peak = new high
RS_NEW_HI_THRESHOLD = 0.98

# Sector ETF universe
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

ALL_SECTORS = list(SECTOR_ETFS.keys())

# Regime classification sets.
# Industrials appears in both Risk-on and Reflation per framework v3.
RISK_ON_SECTORS   = {"Technology", "Financials", "Consumer Discretionary", "Industrials"}
REFLATION_SECTORS = {"Energy", "Materials", "Industrials"}
DEFENSIVE_SECTORS = {"Consumer Staples", "Utilities", "Health Care"}

# Per-sector display colors for the RRG chart
SECTOR_COLORS = {
    "Energy":                 "#F59E0B",   # amber
    "Materials":              "#10B981",   # emerald
    "Industrials":            "#3B82F6",   # blue
    "Technology":             "#A78BFA",   # violet
    "Financials":             "#EC4899",   # pink
    "Consumer Discretionary": "#F97316",   # orange
    "Consumer Staples":       "#06B6D4",   # cyan
    "Utilities":              "#EF4444",   # red
    "Health Care":            "#84CC16",   # lime
}
SECTOR_DASH = ["solid", "dash", "dot", "dashdot"]   # line-style overflow fallback

# Permission state limits and display labels
PERM_LIMITS = {
    "Green":  {"max_pos": 20, "max_pos_label": "Up to 20", "risk_lo": 0.75, "risk_hi": 1.00, "heat": 15},
    "Yellow": {"max_pos": 10, "max_pos_label": "8–12",     "risk_lo": 0.25, "risk_hi": 0.50, "heat": 8},
    "Red":    {"max_pos":  5, "max_pos_label": "3–5",      "risk_lo": 0.00, "risk_hi": 0.00, "heat": 3},
}

SETUP_STYLE = {
    "Green":  "Momentum breakouts",
    "Yellow": "Pullbacks to 20d / 50d MA",
    "Red":    "No new entries — protect capital",
}

# Layer 1.5 flow strength options and sizing map (framework v3, Table: Position Sizing)
FLOW_OPTS = ["Not set", "Weak", "Moderate", "Strong", "Outflows"]

FLOW_SIZE_MAP = {
    "Weak":     ("Quarter",  "0.19%", "1 week of modest inflows — watch closely"),
    "Moderate": ("Half",     "0.38%", "1–2 weeks consistent inflows — enter"),
    "Strong":   ("Full",     "0.75%", "2+ weeks accelerating inflows — full size"),
    "Outflows": ("Exit",     "—",     "Flow reversal — exit review immediately"),
}

# Layer 1.5 phase labels mapped from RRG quadrant
PHASE_MAP = {
    "Improving": "Phase 1 — Early",
    "Leading":   "Phase 2 — Confirmed",
    "Weakening": "Exiting",
    "Lagging":   "No Trade",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data() -> pd.DataFrame:
    """Fetch 1-year daily close for SPY, sector ETFs, TLT, HYG, IEF, and VIX."""
    tickers = ["SPY", "TLT", "HYG", "IEF"] + list(SECTOR_ETFS.values())
    raw   = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    close = raw["Close"]
    try:
        vix = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
        close["^VIX"] = vix["Close"].squeeze()
    except Exception:
        pass
    return close


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred_data() -> dict:
    """Fetch recession composite indicators from FRED (cached 24h — updates daily/monthly)."""
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fed_net_liquidity() -> dict:
    """
    Fetch Fed Net Liquidity from FRED public CSV (no API key required).
    Formula: WALCL - WTREGEN - RRPONTSYD
    Signal: 4-week change <= -$200B → Override Active
    Updates Thursdays at 4:30 PM ET with the H.4.1 release.
    """
    import urllib.request
    import csv
    from io import StringIO

    def fetch_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read().decode()
        rows = list(csv.reader(StringIO(data)))[1:]
        return {row[0]: float(row[1]) for row in rows if len(row) > 1 and row[1] not in ('.', '', 'NA')}

    try:
        walcl = fetch_series("WALCL")
        tga   = fetch_series("WTREGEN")
        rrp   = fetch_series("RRPONTSYD")

        dates = sorted(set(walcl) & set(tga) & set(rrp))
        if len(dates) < 5:
            return {"error": "Insufficient FRED data"}

        net_liq = {d: walcl[d] - tga[d] - rrp[d] for d in dates}
        recent      = sorted(net_liq.keys())
        latest_date = recent[-1]
        prior_date  = recent[-5]  # 4 weeks back

        current  = net_liq[latest_date]
        prior    = net_liq[prior_date]
        change_b = (current - prior) / 1000  # convert $M → $B

        if change_b <= -200:
            signal = "OVERRIDE ACTIVE"
        elif change_b < 0:
            signal = "DECLINING"
        else:
            signal = "RISING"

        return {
            "as_of":        latest_date,
            "current_b":    round(current / 1000, 1),
            "prior_b":      round(prior / 1000, 1),
            "prior_date":   prior_date,
            "change_4w_b":  round(change_b, 1),
            "signal":       signal,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(sectors_key: str) -> tuple:
    """
    Fetch 1-year OHLCV for all tickers in the given sectors (comma-separated key).
    Returns (close_df, volume_df, tickers_list) or raises on network failure.
    """
    try:
        sectors = sectors_key.split(",")
        all_t   = [t for s in sectors for t in SECTOR_TICKERS.get(s, [])]
        raw     = yf.download(all_t + ["SPY"], period="1y", auto_adjust=True, progress=False)
        return raw["Close"], raw["Volume"], all_t
    except Exception as e:
        raise RuntimeError(f"Screener data fetch failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 — MACRO REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────

def calc_layer0(close: pd.DataFrame) -> dict:
    """
    Compute all Layer 0 signals from daily price data:
      - SPY two-speed trend (Signal 1)
      - Sector RS vs SPY → regime classification (Signal 2)
      - TLT direction + TLT/SPY combined signal (Signal 3)
      - HYG/IEF liquidity check (Signal 4)
      - VIX level
    Returns a flat dict of readings; includes 'error' key if data is insufficient.
    """
    r = {}

    def s(name):
        return close[name].dropna() if name in close.columns else pd.Series(dtype=float)

    spy = s("SPY")
    if len(spy) < LB_6M:
        return {"error": "Insufficient SPY data — try refreshing."}

    # ── Signal 1: SPY two-speed trend ────────────────────────────────────────
    r["spy_price"]    = float(spy.iloc[-1])
    r["spy_ma20"]     = float(spy.rolling(20).mean().iloc[-1])
    r["spy_ma50"]     = float(spy.rolling(50).mean().iloc[-1])
    r["spy_above_20"] = bool(spy.iloc[-1] > spy.rolling(20).mean().iloc[-1])
    r["spy_above_50"] = bool(spy.iloc[-1] > spy.rolling(50).mean().iloc[-1])
    r["spy_ret_1m"]   = float(spy.iloc[-1] / spy.iloc[-LB_1M]  - 1)
    r["spy_ret_3m"]   = float(spy.iloc[-1] / spy.iloc[-LB_3M]  - 1)
    r["spy_ret_6m"]   = float(spy.iloc[-1] / spy.iloc[-LB_6M]  - 1)

    # ── Signal 3: Bond market ─────────────────────────────────────────────────
    tlt = s("TLT")
    if len(tlt) >= LB_1M:
        r["tlt_above_50"] = bool(tlt.iloc[-1] > tlt.rolling(50).mean().iloc[-1]) if len(tlt) >= 50 else None
        r["tlt_ret_1m"]   = float(tlt.iloc[-1] / tlt.iloc[-LB_1M] - 1)
        if   r["tlt_ret_1m"] >  0.01: r["tlt_direction"] = "Rising"
        elif r["tlt_ret_1m"] < -0.01: r["tlt_direction"] = "Declining"
        else:                          r["tlt_direction"] = "Flat"

        tlt_rising = r["tlt_direction"] == "Rising"
        spy_rising = r.get("spy_ret_1m", 0) > 0
        if   tlt_rising and spy_rising:       r["tlt_spy_signal"] = "TLT ↑ + SPY ↑ → Risk-on confirmed"
        elif not tlt_rising and spy_rising:   r["tlt_spy_signal"] = "TLT ↓ + SPY ↑ → Reflation regime"
        elif tlt_rising and not spy_rising:   r["tlt_spy_signal"] = "TLT ↑ + SPY ↓ → Deflationary — reduce significantly"
        else:                                 r["tlt_spy_signal"] = "TLT ↓ + SPY ↓ → Stagflation / liquidity crisis"
    else:
        r["tlt_above_50"]   = None
        r["tlt_ret_1m"]     = None
        r["tlt_direction"]  = "N/A"
        r["tlt_spy_signal"] = "N/A"

    # ── Signal 4: Liquidity — HYG/IEF credit spread ───────────────────────────
    # Override triggered by HYG/IEF declining over the past month (credit spreads widening).
    hyg, ief = s("HYG"), s("IEF")
    if len(hyg) >= LB_1M and len(ief) >= LB_1M:
        ief_a  = ief.reindex(hyg.index).ffill()
        ratio  = (hyg / ief_a).dropna()
        r["hyg_ief_ratio"]     = float(ratio.iloc[-1])
        r["hyg_ief_1m_ago"]    = float(ratio.iloc[-LB_1M])
        r["liquidity_tighten"] = bool(ratio.iloc[-1] < ratio.iloc[-LB_1M])
    else:
        r["hyg_ief_ratio"]     = None
        r["hyg_ief_1m_ago"]    = None
        r["liquidity_tighten"] = False

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix = s("^VIX")
    r["vix"]          = float(vix.iloc[-1]) if len(vix) > 0 else None
    r["vix_elevated"] = bool(r["vix"] > 25) if r["vix"] else False

    # ── Signal 6: Fed Net Liquidity (WALCL - TGA - RRP) ──────────────────────
    fnl = fetch_fed_net_liquidity()
    if "error" not in fnl:
        r["fnl_signal"]    = fnl["signal"]       # "RISING", "DECLINING", "OVERRIDE ACTIVE"
        r["fnl_change_4w"] = fnl["change_4w_b"]  # $B, 4-week change
        r["fnl_current"]   = fnl["current_b"]    # $B, latest reading
        r["fnl_as_of"]     = fnl["as_of"]        # date string
    else:
        r["fnl_signal"]    = "N/A"
        r["fnl_change_4w"] = None
        r["fnl_current"]   = None
        r["fnl_as_of"]     = ""
        r["fnl_error"]     = fnl["error"]

    # ── Signal 2: Sector RS vs SPY → regime flavor ───────────────────────────
    sector_rs = {}
    for sector, etf in SECTOR_ETFS.items():
        ep = s(etf)
        if len(ep) < LB_3M:
            continue
        idx  = ep.index.intersection(spy.index)
        ep_a = ep.loc[idx]
        sa   = spy.loc[idx]
        if len(ep_a) < LB_3M:
            continue

        rs_line   = ep_a / sa
        rs_1m     = float(ep_a.iloc[-1] / ep_a.iloc[-LB_1M] - 1) - float(sa.iloc[-1] / sa.iloc[-LB_1M] - 1)
        rs_3m     = float(ep_a.iloc[-1] / ep_a.iloc[-LB_3M] - 1) - float(sa.iloc[-1] / sa.iloc[-LB_3M] - 1)
        rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-LB_3M:].max() * RS_NEW_HI_THRESHOLD)

        if   rs_1m > 0 and rs_3m > 0: trend = "Leading"
        elif rs_1m > 0 or  rs_3m > 0: trend = "Mixed"
        else:                          trend = "Lagging"

        sector_rs[sector] = {
            "etf":      etf,
            "price":    round(float(ep_a.iloc[-1]), 2),
            "ret_1m":   float(ep_a.iloc[-1] / ep_a.iloc[-LB_1M] - 1),
            "ret_3m":   float(ep_a.iloc[-1] / ep_a.iloc[-LB_3M] - 1),
            "rs_1m":    rs_1m,
            "rs_3m":    rs_3m,
            "rs_new_hi": rs_new_hi,
            "trend":    trend,
        }

    r["sector_rs"]       = sector_rs
    r["leading_sectors"] = [sec for sec, v in sector_rs.items() if v["trend"] == "Leading"]
    r["mixed_sectors"]   = [sec for sec, v in sector_rs.items() if v["trend"] == "Mixed"]

    # Regime classification — priority order matters: Stagflation → Risk-on → Reflation → Deflation → Mixed
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
    Score the 5-model recession composite (framework Signal 5).
    Four models auto-fetched from FRED; one (Conference Board LEI) is manual
    because the source is paywalled.
    Returns a list of indicator dicts: name, value, as_of, threshold, ok (bool).
    """
    indicators = []

    if "error" not in fred:
        cp = fred.get("recprob")
        indicators.append({
            "name": "Chauvet-Piger Recession Prob", "value": f"{cp:.1f}%" if cp is not None else "N/A",
            "as_of": fred.get("recprob_date", ""), "threshold": "< 50%",
            "ok": (cp < 50) if cp is not None else True,
        })
        sahm = fred.get("sahm")
        indicators.append({
            "name": "Sahm Rule", "value": f"{sahm:.2f}" if sahm is not None else "N/A",
            "as_of": fred.get("sahm_date", ""), "threshold": "< 0.50",
            "ok": (sahm < 0.50) if sahm is not None else True,
        })
        cfnai = fred.get("cfnai")
        indicators.append({
            "name": "CFNAI 3-mo MA", "value": f"{cfnai:.2f}" if cfnai is not None else "N/A",
            "as_of": fred.get("cfnai_date", ""), "threshold": "> -0.70",
            "ok": (cfnai > -0.70) if cfnai is not None else True,
        })
        t10y3m = fred.get("t10y3m")
        indicators.append({
            "name": "10Y-3M Yield Curve", "value": f"{t10y3m:+.2f}%" if t10y3m is not None else "N/A",
            "as_of": fred.get("t10y3m_date", ""), "threshold": "> 0% (uninverted)",
            "ok": (t10y3m >= 0) if t10y3m is not None else True,
        })

    # Conference Board LEI — manual; only flags if explicitly set to declining
    indicators.append({
        "name": "Conference Board LEI", "value": lei_manual,
        "as_of": "manual", "threshold": "Not declining 6mo",
        "ok": lei_manual != "6mo declining ⚠️",
    })

    return indicators


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — MARKET PERMISSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def calc_layer1(l0: dict, rec_flags: int, eps_signal: str, override: str = "Auto") -> tuple:
    """
    Determine permission state (Green / Yellow / Red) per framework v3:
      Green:  SPY both positive + rec composite < 2/5 + EPS not declining + no liquidity override
      Yellow: SPY mixed OR rec composite 2–3/5 OR EPS declining
      Red:    SPY both negative OR liquidity override OR rec composite 4+/5

    Liquidity override: triggered by HYG/IEF ratio declining (credit spreads widening).
    VIX is informational only and does not gate the override.
    """
    if override != "Auto":
        return override, PERM_LIMITS[override]

    roc1_pos     = l0.get("spy_ret_1m", 0) > 0
    roc6_pos     = l0.get("spy_ret_6m", 0) > 0
    liq_override = (
        bool(l0.get("liquidity_tighten")) or          # HYG/IEF credit spread widening
        l0.get("fnl_signal") == "OVERRIDE ACTIVE"     # Fed Net Liquidity 4-week drop > $200B
    )
    eps_declining = "Declining" in eps_signal

    if liq_override or (not roc1_pos and not roc6_pos) or rec_flags >= 4:
        perm = "Red"
    elif rec_flags >= 2 or eps_declining or not roc1_pos or not roc6_pos:
        perm = "Yellow"
    else:
        perm = "Green"

    return perm, PERM_LIMITS[perm]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1.5 — SECTOR ROTATION (RRG)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def calc_layer15(close: pd.DataFrame) -> list:
    """
    Compute the Relative Rotation Graph (RRG) for all sector ETFs vs SPY.
    Uses weekly resampled data; trailing 8 weeks shown per sector.

    Methodology (JdK RS-Ratio / RS-Momentum):
      RS-Ratio   = (weekly RS / SMA10 of weekly RS) × 100  — centered at 100
      RS-Momentum = (RS-Ratio / SMA4 of RS-Ratio) × 100   — centered at 100

    Quadrant → framework phase mapping:
      Improving (RS < 100, Momentum > 100) → Phase 1 Early
      Leading   (RS > 100, Momentum > 100) → Phase 2 Confirmed
      Weakening (RS > 100, Momentum < 100) → Exiting
      Lagging   (RS < 100, Momentum < 100) → No Trade

    Flow-strength sizing is applied in the tab render function using session state,
    allowing user override without invalidating this cached computation.
    """
    spy = close["SPY"].dropna() if "SPY" in close.columns else pd.Series(dtype=float)
    if len(spy) < 60:
        return []

    results = []
    for sector, etf in SECTOR_ETFS.items():
        if etf not in close.columns:
            continue
        px  = close[etf].dropna()
        idx = px.index.intersection(spy.index)
        if len(idx) < 60:
            continue

        px_a  = px.loc[idx]
        spy_a = spy.loc[idx]

        rs_daily = px_a / spy_a
        rs_w     = rs_daily.resample("W-FRI").last().dropna()
        if len(rs_w) < 15:
            continue

        sma10       = rs_w.rolling(10).mean()
        rs_ratio    = (rs_w / sma10 * 100).dropna()

        sma4        = rs_ratio.rolling(4).mean()
        rs_momentum = (rs_ratio / sma4 * 100).dropna()

        common  = rs_ratio.index.intersection(rs_momentum.index)
        trail_x = rs_ratio.loc[common].iloc[-8:].tolist()
        trail_y = rs_momentum.loc[common].iloc[-8:].tolist()
        if not trail_x or not trail_y:
            continue

        cur_x, cur_y = trail_x[-1], trail_y[-1]

        if   cur_x >= 100 and cur_y >= 100: quadrant = "Leading"
        elif cur_x <  100 and cur_y >= 100: quadrant = "Improving"
        elif cur_x >= 100 and cur_y <  100: quadrant = "Weakening"
        else:                               quadrant = "Lagging"

        # Default sizing from quadrant — overridden by user flow-strength input in the tab
        default_size_map = {
            "Improving": ("Quarter → Half", "0.19–0.38%", "Early rotation — watch closely"),
            "Leading":   ("Half → Full",    "0.38–0.75%", "Confirmed — enter or hold"),
            "Weakening": ("Tighten stop",   "—",          "Tighten to 20d MA"),
            "Lagging":   ("No trade",       "—",          "Avoid"),
        }
        sizing, risk_pct, action = default_size_map[quadrant]

        price    = float(px_a.iloc[-1])
        ma20     = float(px_a.rolling(20).mean().iloc[-1])
        above_20 = price > ma20

        results.append({
            "sector":      sector,
            "etf":         etf,
            "rs_ratio":    round(cur_x, 2),
            "rs_momentum": round(cur_y, 2),
            "trail_x":     [round(v, 2) for v in trail_x],
            "trail_y":     [round(v, 2) for v in trail_y],
            "quadrant":    quadrant,
            "phase":       PHASE_MAP[quadrant],
            "sizing":      sizing,
            "risk_pct":    risk_pct,
            "action":      action,
            "price":       round(price, 2),
            "ma20":        round(ma20, 2),
            "above_20":    above_20,
        })

    order = {"Improving": 0, "Leading": 1, "Weakening": 2, "Lagging": 3}
    results.sort(key=lambda x: order[x["quadrant"]])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — STOCK SCREENER
# ─────────────────────────────────────────────────────────────────────────────

def calc_layer2(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    tickers: list,
    ticker_sector: dict,
    spy: pd.Series,
    min_dollar_vol: float = 10_000_000,
    rs_lookback: int = LB_3M,
) -> pd.DataFrame:
    """
    Screen individual stocks against Layer 2 criteria:
      - Price above 20d and 50d MA
      - Adequate average dollar volume
      - RS line at/near new highs vs SPY
      - RS line rising vs 1 month ago
      - Two-speed trend signal (1M + 3M returns)
      - RSI and MACD for additional context

    Note: Earnings Carry (Step 4) and Energy Three-Layer (Step 5) are not
    automated — both require external data sources not available via yfinance.
    """
    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        px  = close[t].dropna()
        vol = volume[t].dropna() if t in volume.columns else pd.Series(dtype=float)
        if len(px) < LB_3M:
            continue

        price          = float(px.iloc[-1])
        ma20           = float(px.rolling(20).mean().iloc[-1])
        ma50           = float(px.rolling(50).mean().iloc[-1])
        avg_share_vol  = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0.0
        avg_dollar_vol = price * avg_share_vol
        ret_1m         = float(px.iloc[-1] / px.iloc[-LB_1M] - 1)
        ret_3m         = float(px.iloc[-1] / px.iloc[-LB_3M] - 1)

        spy_a     = spy.reindex(px.index).ffill()
        rs_line   = px / spy_a
        rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-rs_lookback:].max() * RS_NEW_HI_THRESHOLD)
        rs_rising = bool(len(rs_line) >= LB_1M and rs_line.iloc[-1] > rs_line.iloc[-LB_1M])

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
            "Ticker":     t,
            "Sector":     ticker_sector.get(t, ""),
            "Price":      round(price, 2),
            "vs 20MA":    above_20,
            "vs 50MA":    above_50,
            "1M Ret":     ret_1m,
            "3M Ret":     ret_3m,
            "RS Hi":      rs_new_hi,
            "RS ↑":       rs_rising,
            "RSI":        round(rsi_v, 1),
            "MACD":       m_bull,
            "MACD Hist":  round(m_hist, 4),
            "Avg $Vol(M)": round(avg_dollar_vol / 1e6, 1),
            "2-Speed":    two_spd,
            "PASS":       passes,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


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

    ma20  = px.rolling(20).mean()
    ma50  = px.rolling(50).mean()
    spy_a = spy_hist["Close"].reindex(px.index).ffill()
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
            font=dict(color="#d1d5db", size=11),
            xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)",
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


def build_rrg_chart(l15_data: list) -> go.Figure:
    """
    Build the Relative Rotation Graph scatter chart.
    Each sector gets a unique color (SECTOR_COLORS) and a smooth spline trail
    showing the last 8 weekly positions.
    Returns None if data is empty.
    """
    if not l15_data:
        return None

    all_x = [v for d in l15_data for v in d["trail_x"]]
    all_y = [v for d in l15_data for v in d["trail_y"]]
    pad   = 0.8
    xlo   = min(min(all_x) - pad, 98.5)
    xhi   = max(max(all_x) + pad, 101.5)
    ylo   = min(min(all_y) - pad, 98.5)
    yhi   = max(max(all_y) + pad, 101.5)

    quad_label_colors = {
        "LEADING":   "#27500A",
        "IMPROVING": "#93B6FA",
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

    all_sector_keys = list(SECTOR_COLORS.keys())
    for d in l15_data:
        idx        = all_sector_keys.index(d["sector"]) if d["sector"] in all_sector_keys else 0
        color      = list(SECTOR_COLORS.values())[idx % len(SECTOR_COLORS)]
        dash_style = SECTOR_DASH[idx // len(SECTOR_COLORS)]
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
            marker=dict(size=14, color=color, line=dict(color="#FFFFFF", width=1.5)),
            text=[d["etf"]],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=10),
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
        height=560,
        paper_bgcolor="#111827", plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db", size=11),
        margin=dict(l=55, r=20, t=30, b=55),
        xaxis=dict(title="RS-Ratio  →  (stronger relative performance)", gridcolor="#374151", zeroline=False, range=[xlo, xhi]),
        yaxis=dict(title="RS-Momentum  ↑  (accelerating)",               gridcolor="#374151", zeroline=False, range=[ylo, yhi]),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.15, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pct(v):  return f"{v*100:+.1f}%"
def icon(v): return "✅" if v else "❌"
def macd(v): return "▲" if v else "▼"


def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a screener DataFrame for display:
      - Percentage columns formatted with sign and 1 decimal
      - Boolean columns converted to ✅ / ❌
      - MACD converted to ▲ / ▼
      - Dollar volume formatted as $XM
    Drops internal columns PASS and MACD Hist (not intended for display).
    """
    d = df.copy()
    for col in ["1M Ret", "3M Ret"]:
        if col in d.columns: d[col] = d[col].apply(pct)
    for col in ["vs 20MA", "vs 50MA", "RS Hi", "RS ↑"]:
        if col in d.columns: d[col] = d[col].apply(icon)
    if "MACD"         in d.columns: d["MACD"]         = d["MACD"].apply(macd)
    if "Avg $Vol(M)"  in d.columns: d["Avg $Vol(M)"]  = d["Avg $Vol(M)"].apply(lambda x: f"${x:.1f}M")
    return d.drop(columns=["PASS", "MACD Hist"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _render_layer0_tab(l0: dict, fred_data: dict, rec_indicators: list,
                       rec_flags: int, rec_total: int, perm: str,
                       limits: dict, l15_data: list) -> None:
    """Render the Layer 0 — Macro tab."""

    roc1_pos   = l0["spy_ret_1m"] > 0
    roc6_pos   = l0["spy_ret_6m"] > 0
    spy_signal = "GREEN" if (roc1_pos and roc6_pos) else ("RED" if (not roc1_pos and not roc6_pos) else "MIXED")
    tlt_dir    = l0.get("tlt_direction", "N/A")
    vix_v      = l0.get("vix")
    hyg_r      = l0.get("hyg_ief_ratio")

    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        # SPY Two-Speed
        with st.expander("SPY Two-Speed Trend", expanded=True):
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

        # Bond & Liquidity
        with st.expander("Bond & Liquidity", expanded=True):
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
                 ) if l0.get("fnl_current") is not None else l0.get("fnl_error", "N/A")},
            ]
            st.dataframe(pd.DataFrame(bond_rows), hide_index=True, use_container_width=True)

        st.write("")

        # Manual signals
        with st.expander("Signals — Set in Sidebar", expanded=True):
            sig_rows = [
                {"Signal": "EPS Revisions (FactSet)",      "Value": st.session_state.eps_signal},
                {"Signal": "Taylor Rule Deviation",         "Value": st.session_state.taylor_rule},
                {"Signal": "Drawdown from Peak Equity",     "Value": st.session_state.drawdown_state},
            ]
            st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

    with col_r:
        # Recession composite
        flag_color = "🟢" if rec_flags == 0 else ("🟡" if rec_flags <= 2 else "🔴")
        with st.expander(f"Recession Composite  {flag_color} {rec_flags} / {rec_total}", expanded=True):
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
                if   rec_flags == 0: st.success("Normal — full risk operations.")
                elif rec_flags <= 2: st.warning(f"{rec_flags} indicator(s) flagging — elevated caution.")
                else:                st.error(  f"{rec_flags} indicators flagging — reduce risk.")

        st.write("")

        # Sector RS
        with st.expander("Sector Relative Strength vs SPY", expanded=True):
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

    # Signal 9: Sector flow momentum summary (Layer 1.5 preview)
    st.write("")
    with st.expander("Signal 9 — Sector Flow Momentum (Layer 1.5 Preview)", expanded=True):
        if l15_data:
            improving = [d for d in l15_data if d["quadrant"] == "Improving"]
            leading   = [d for d in l15_data if d["quadrant"] == "Leading"]
            early_str = ", ".join(f"{d['etf']}" for d in improving) if improving else "None"
            conf_str  = ", ".join(f"{d['etf']}" for d in leading)   if leading   else "None"
            flow_rows = [
                {"Phase": "🔵 Phase 1 — Early rotation",      "Sectors": early_str},
                {"Phase": "🟢 Phase 2 — Confirmed rotation",  "Sectors": conf_str},
            ]
            st.dataframe(pd.DataFrame(flow_rows), hide_index=True, use_container_width=True)
            st.caption("Full RRG detail and ETF sizing in the Layer 1.5 tab.")
        else:
            st.info("RRG data unavailable.")

    # This week's rules
    st.write("")
    with st.expander("This Week's Rules", expanded=True):
        rules_text = (
            f"**Max Positions:** {limits['max_pos_label']}  &nbsp;|&nbsp;  "
            f"**Risk / Trade:** {limits['risk_lo']}–{limits['risk_hi']}%  &nbsp;|&nbsp;  "
            f"**Max Portfolio Heat:** {limits['heat']}%  &nbsp;|&nbsp;  "
            f"**Setup Style:** {SETUP_STYLE[perm]}"
        )
        if   perm == "Green":  st.success(rules_text)
        elif perm == "Yellow": st.warning(rules_text)
        else:                  st.error(rules_text)


def _render_layer15_tab(l15_data: list) -> None:
    """
    Render the Layer 1.5 — Sector Rotation tab.
    Includes the RRG chart, quadrant summary, ETF entry candidates with
    phase labels, flow-strength sizing override, and exit review.
    """
    st.caption("Relative Rotation Graph — sector ETFs vs SPY. Weekly data, trailing 8 weeks. Clockwise rotation is normal cycle progression.")

    # Sector filter for the chart
    chart_sectors = st.multiselect(
        "Sectors to display",
        options=ALL_SECTORS,
        default=ALL_SECTORS,
    )
    chart_data = [d for d in l15_data if d["sector"] in chart_sectors] if chart_sectors else l15_data

    with st.spinner("Building RRG chart..."):
        rrg_fig = build_rrg_chart(chart_data)
    if rrg_fig:
        st.plotly_chart(rrg_fig, use_container_width=True)
    else:
        st.warning("Insufficient data to build RRG chart.")

    st.divider()

    # Quadrant count summary
    n_improving = sum(1 for d in l15_data if d["quadrant"] == "Improving")
    n_leading   = sum(1 for d in l15_data if d["quadrant"] == "Leading")
    n_weakening = sum(1 for d in l15_data if d["quadrant"] == "Weakening")
    n_lagging   = sum(1 for d in l15_data if d["quadrant"] == "Lagging")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🔵 Improving", n_improving)
    mc2.metric("🟢 Leading",   n_leading)
    mc3.metric("🟡 Weakening", n_weakening)
    mc4.metric("🔴 Lagging",   n_lagging)

    # Flow strength inputs — manual entry from ETFdb.com, overrides quadrant-based sizing
    with st.expander("📊 Weekly Flow Strength — set to refine sizing"):
        st.caption(
            "Check [ETFdb Fund Flows](https://etfdb.com/etf-fund-flows/) for 1-week and 4-week "
            "directional inflows. Enter each sector's reading below — overrides RRG-based sizing."
        )
        flow_cols = st.columns(3)
        for i, (sector, etf) in enumerate(SECTOR_ETFS.items()):
            key = f"flow_{etf.lower()}"
            with flow_cols[i % 3]:
                st.session_state[key] = st.selectbox(
                    f"{etf} — {sector}",
                    FLOW_OPTS,
                    index=FLOW_OPTS.index(st.session_state.get(key, "Not set")),
                    key=f"flow_sel_{etf}",
                )

    st.divider()

    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        candidates = [d for d in l15_data if d["quadrant"] in ("Improving", "Leading")]
        with st.expander("ETF Entry Candidates", expanded=True):
            if candidates:
                rows = []
                for d in candidates:
                    flow_key      = f"flow_{d['etf'].lower()}"
                    flow_strength = st.session_state.get(flow_key, "Not set")
                    # Override sizing with user's flow reading if set
                    if flow_strength in FLOW_SIZE_MAP:
                        sizing, risk_pct, _ = FLOW_SIZE_MAP[flow_strength]
                    else:
                        sizing, risk_pct = d["sizing"], d["risk_pct"]

                    q_icon = "🔵" if d["quadrant"] == "Improving" else "🟢"
                    rows.append({
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
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("No Improving or Leading sectors currently.")

    with col_r:
        weakening = [d for d in l15_data if d["quadrant"] == "Weakening"]
        with st.expander("Weakening — Review Stops", expanded=True):
            if weakening:
                rows = []
                for d in weakening:
                    flow_key      = f"flow_{d['etf'].lower()}"
                    flow_strength = st.session_state.get(flow_key, "Not set")
                    rows.append({
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
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.success("No sectors in Weakening quadrant.")

    # Full sector table
    st.write("")
    with st.expander("All Sectors", expanded=True):
        q_icons = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}
        all_rows = []
        for d in l15_data:
            flow_key      = f"flow_{d['etf'].lower()}"
            flow_strength = st.session_state.get(flow_key, "Not set")
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
        st.dataframe(pd.DataFrame(all_rows), hide_index=True, use_container_width=True)


def _render_layer2_tab(results_df: pd.DataFrame, passes_df: pd.DataFrame,
                       half_df: pd.DataFrame, perm: str,
                       active_sectors: list, show_half: bool, show_all: bool) -> None:
    """Render the Layer 2 — Screener tab."""

    if perm == "Red":
        st.error("🔴 RED STATE — No new entries. Results shown for reference only.")

    st.caption(f"Sectors: {', '.join(active_sectors)}  ·  Universe: {len(results_df)} stocks")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Screened",        len(results_df))
    c2.metric("✅ Full Signal",  len(passes_df))
    c3.metric("⚠️ Half Signal",  len(half_df))
    c4.metric("❌ No Trade",     len(results_df[results_df["2-Speed"] == "NO TRADE"]))

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
            all_s = results_df.sort_values(["PASS", "2-Speed", "3M Ret"], ascending=[False, True, False])
            st.dataframe(fmt_df(all_s), hide_index=True, use_container_width=True)


def _render_charts_tab(passes_df: pd.DataFrame) -> None:
    """Render the Charts tab — individual stock charts for Full Signal candidates."""

    if passes_df.empty:
        st.info("No Full Signal candidates to chart.")
        return

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
.main, .main > div { background-color: #060C1C !important; }
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
[data-testid="stSidebar"] h3   { color: #FFFFFF !important; }

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

/* ── EXPANDER CARDS (#2550C8) ──────────────────────────────────────────── */
/* Expanders use secondaryBackgroundColor automatically — just style shape  */
[data-testid="stExpander"] {
    border: 1px solid rgba(147,182,250,0.40) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 0 1px rgba(147,182,250,0.10) !important;
}
/* Header label */
[data-testid="stExpander"] details summary p {
    color: #93B6FA !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
/* Toggle arrow */
[data-testid="stExpanderToggleIcon"] svg { color: #93B6FA !important; }
/* Separator line below header */
[data-testid="stExpander"] details summary {
    border-bottom: 1px solid rgba(147,182,250,0.25) !important;
    padding-bottom: 8px !important;
}

/* ── DATAFRAMES / TABLES ────────────────────────────────────────────────── */
[data-testid="stDataFrame"]       { border-radius: 9px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] > div { border-radius: 9px !important; }

/* ── ALERT BOXES ────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    border: 1px solid rgba(147,182,250,0.30) !important;
}

/* ── TABS ───────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"]           { background-color: transparent !important; border-bottom: 1px solid rgba(147,182,250,0.20) !important; }
[data-baseweb="tab"]                { color: #93B6FA !important; font-weight: 400 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid #93B6FA !important;
    font-weight: 500 !important;
}

/* ── MISC ───────────────────────────────────────────────────────────────── */
hr { border-color: rgba(147,182,250,0.25) !important; margin: 0.75rem 0 !important; }
p  { color: #FFFFFF !important; }

/* ── TEXT ON LIGHT PANELS (#D8E6FF) ────────────────────────────────────── */
[data-testid="stMetric"] p,
[data-testid="stMetric"] label,
[data-testid="stMetric"] span    { color: #3A5EAA !important; }
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p  { color: #3A5EAA !important; }
[data-testid="stMetricValue"] > div { color: #1D3B93 !important; }

/* ── SIDEBAR WIDGETS ────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div     { border-color: rgba(147,182,250,0.30) !important; }
[data-testid="stNumberInput"] > div > div > input { border-color: rgba(147,182,250,0.30) !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # ── SESSION STATE ─────────────────────────────────────────────────────────
    # Manual signals (set once per weekly review)
    defaults = {
        "eps_signal":    "Not set",
        "drawdown_state": "At or near peak — full risk",
        "lei_signal":    "Not set",
        "taylor_rule":   "Not set",
    }
    # Layer 1.5 flow strength per sector ETF (set in the L1.5 tab expander)
    for etf in SECTOR_ETFS.values():
        defaults[f"flow_{etf.lower()}"] = "Not set"

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
            index=["Not set", "↑ Rising 3+ weeks ✅", "Flat", "↓ Declining ⚠️"].index(st.session_state.eps_signal),
        )
        st.session_state.drawdown_state = st.selectbox(
            "Drawdown from Peak Equity",
            ["At or near peak — full risk", "5–10% drawdown — reduce risk",
             "10–15% drawdown — defensive", ">15% drawdown — cash"],
            index=["At or near peak — full risk", "5–10% drawdown — reduce risk",
                   "10–15% drawdown — defensive", ">15% drawdown — cash"].index(st.session_state.drawdown_state),
        )
        st.session_state.lei_signal = st.selectbox(
            "Conference Board LEI",
            ["Not set", "Rising ✅", "Flat", "6mo declining ⚠️"],
            index=["Not set", "Rising ✅", "Flat", "6mo declining ⚠️"].index(st.session_state.lei_signal),
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
        regime_ov = st.selectbox("Regime",           ["Auto", "Risk-on", "Reflation", "Deflation", "Stagflation", "Mixed"])
        perm_ov   = st.selectbox("Permission State", ["Auto", "Green", "Yellow", "Red"])

        st.divider()
        st.subheader("Screener Settings")
        min_vol   = st.number_input("Min Avg Dollar Volume ($M)", value=10, step=5, format="%d") * 1_000_000
        rs_lb     = st.slider("RS Lookback (days)", 21, 126, LB_3M)
        show_half = st.checkbox("Show Half Signal watchlist", value=True)
        show_all  = st.checkbox("Show full universe table",   value=False)

    # ── DATA LOADING ──────────────────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        macro_close = fetch_macro_data()
        l0          = calc_layer0(macro_close)

    if "error" in l0:
        st.error(l0["error"])
        return

    with st.spinner("Loading FRED indicators..."):
        fred_data = fetch_fred_data()

    with st.spinner("Computing sector rotation (RRG)..."):
        l15_data = calc_layer15(macro_close)

    # ── REGIME AND PERMISSION STATE ───────────────────────────────────────────
    regime = regime_ov if regime_ov != "Auto" else l0["regime"]

    rec_indicators = score_recession_composite(fred_data, st.session_state.lei_signal)
    rec_flags      = sum(1 for i in rec_indicators if not i["ok"])
    rec_total      = len(rec_indicators)

    perm, limits = calc_layer1(l0, rec_flags, st.session_state.eps_signal, perm_ov)

    # Sectors to screen — auto-set from regime, user can override in sidebar
    regime_sectors = {
        "Risk-on":     list(RISK_ON_SECTORS),
        "Reflation":   list(REFLATION_SECTORS),
        "Deflation":   list(DEFENSIVE_SECTORS),
        "Stagflation": list(REFLATION_SECTORS | {"Consumer Staples", "Utilities"}),
    }
    mixed_auto   = l0.get("leading_sectors", []) + l0.get("mixed_sectors", [])
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

    # ── SCREENER DATA ─────────────────────────────────────────────────────────
    universe_size = sum(len(SECTOR_TICKERS.get(s, [])) for s in active_sectors)
    with st.spinner(f"Loading {universe_size} stocks..."):
        try:
            sectors_key             = ",".join(sorted(active_sectors))
            close_sc, vol_sc, all_t = fetch_screener_data(sectors_key)
        except RuntimeError as e:
            st.error(str(e))
            return

    ticker_sector = {t: sec for sec in active_sectors for t in SECTOR_TICKERS.get(sec, [])}
    spy_sc        = close_sc["SPY"] if "SPY" in close_sc.columns else pd.Series(dtype=float)

    with st.spinner("Running screener..."):
        results_df = calc_layer2(close_sc, vol_sc, all_t, ticker_sector, spy_sc, min_vol, rs_lb)

    if results_df.empty:
        st.error("Screener returned no results. Check your internet connection and refresh.")
        return

    passes_df = results_df[results_df["PASS"]].sort_values(["Sector", "3M Ret"], ascending=[True, False])
    half_df   = results_df[(results_df["2-Speed"] == "HALF") & (~results_df["PASS"])].sort_values("3M Ret", ascending=False)

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    st.markdown("## Swing Trading Framework")
    st.caption(f"{datetime.now().strftime('%A, %B %d, %Y')}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime",        regime)
    c2.metric("Permission",    perm)
    c3.metric("SPY",           f"${l0['spy_price']:.2f}", f"{l0['spy_ret_1m']*100:+.1f}% 1M")
    c4.metric("Max Positions", limits["max_pos_label"])
    c5.metric("Risk / Trade",  f"{limits['risk_lo']}–{limits['risk_hi']}%" if limits["risk_hi"] > 0 else "No new trades")
    c6.metric("Max Heat",      f"{limits['heat']}%")

    if   perm == "Green":  st.success(f"🟢 **GREEN STATE** — Full deployment. {SETUP_STYLE['Green']}. {limits['max_pos_label']} positions.")
    elif perm == "Yellow": st.warning(f"🟡 **YELLOW STATE** — Selective. {SETUP_STYLE['Yellow']}. {limits['max_pos_label']} positions max.")
    else:                  st.error(  f"🔴 **RED STATE** — {SETUP_STYLE['Red']}. {limits['max_pos_label']} positions max.")

    if l0.get("liquidity_tighten") or l0.get("fnl_signal") == "OVERRIDE ACTIVE":
        reasons = []
        if l0.get("liquidity_tighten"):
            reasons.append("HYG/IEF ratio declining")
        if l0.get("fnl_signal") == "OVERRIDE ACTIVE":
            reasons.append(f"Fed Net Liquidity −${abs(l0.get('fnl_change_4w', 0)):.0f}B (4-week)")
        st.warning(f"⚠️ Liquidity override active — {'; '.join(reasons)}. Reduce exposure immediately.")
    if rec_flags >= 3:
        st.error(f"⚠️ Recession composite elevated: {rec_flags}/{rec_total} indicators flagging.")
    elif rec_flags >= 1:
        st.warning(f"⚠️ Recession composite: {rec_flags}/{rec_total} indicator(s) flagging — monitor.")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Layer 0 — Macro",
        "🔀 Layer 1.5 — Sector Rotation",
        "🔍 Layer 2 — Screener",
        "📈 Charts",
    ])

    with tab1:
        _render_layer0_tab(l0, fred_data, rec_indicators, rec_flags, rec_total, perm, limits, l15_data)

    with tab2:
        _render_layer15_tab(l15_data)

    with tab3:
        _render_layer2_tab(results_df, passes_df, half_df, perm, active_sectors, show_half, show_all)

    with tab4:
        _render_charts_tab(passes_df)


if __name__ == "__main__":
    main()
