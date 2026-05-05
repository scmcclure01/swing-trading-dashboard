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

ALL_SECTORS      = list(SECTOR_ETFS.keys())
RISK_ON_SECTORS  = {"Technology", "Financials", "Consumer Discretionary"}
REFLATION_SECTORS = {"Energy", "Materials", "Industrials"}
DEFENSIVE_SECTORS = {"Consumer Staples", "Utilities", "Health Care"}

PERM_LIMITS = {
    "Green":  {"max_pos": 20, "risk_lo": 0.75, "risk_hi": 1.00, "heat": 15},
    "Yellow": {"max_pos": 10, "risk_lo": 0.25, "risk_hi": 0.50, "heat": 8},
    "Red":    {"max_pos":  5, "risk_lo": 0.00, "risk_hi": 0.00, "heat": 3},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING  (cached 1 hour)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data():
    """SPY, sector ETFs, TLT, HYG, IEF for Layer 0."""
    tickers = ["SPY", "TLT", "HYG", "IEF"] + list(SECTOR_ETFS.values())
    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    close = raw["Close"]
    # VIX fetched separately — ticker has special chars that can cause issues
    try:
        vix = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
        close["^VIX"] = vix["Close"] if isinstance(vix["Close"], pd.Series) else vix["Close"].squeeze()
    except Exception:
        pass
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(sectors_key: str):
    """Download 1-year OHLCV for all tickers in the given sectors.
    sectors_key is a sorted comma-joined string so it's hashable for caching.
    """
    sectors = sectors_key.split(",")
    all_t   = [t for s in sectors for t in SECTOR_TICKERS.get(s, [])]
    raw = yf.download(all_t + ["SPY"], period="1y", auto_adjust=True, progress=False)
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
    if len(spy) < 63:
        return {"error": "Insufficient SPY data — try refreshing."}

    # SPY
    r["spy_price"]    = float(spy.iloc[-1])
    r["spy_ma20"]     = float(spy.rolling(20).mean().iloc[-1])
    r["spy_ma50"]     = float(spy.rolling(50).mean().iloc[-1])
    r["spy_above_20"] = bool(spy.iloc[-1] > spy.rolling(20).mean().iloc[-1])
    r["spy_above_50"] = bool(spy.iloc[-1] > spy.rolling(50).mean().iloc[-1])
    r["spy_ret_1m"]   = float(spy.iloc[-1] / spy.iloc[-21] - 1)
    r["spy_ret_3m"]   = float(spy.iloc[-1] / spy.iloc[-63] - 1)

    # TLT
    tlt = s("TLT")
    r["tlt_above_50"] = bool(tlt.iloc[-1] > tlt.rolling(50).mean().iloc[-1]) if len(tlt) >= 50 else None
    r["tlt_ret_1m"]   = float(tlt.iloc[-1] / tlt.iloc[-21] - 1) if len(tlt) >= 21 else None

    # Liquidity: HYG / IEF ratio trend
    hyg, ief = s("HYG"), s("IEF")
    if len(hyg) >= 21 and len(ief) >= 21:
        ief_a = ief.reindex(hyg.index).ffill()
        ratio = (hyg / ief_a).dropna()
        r["hyg_ief_ratio"]    = float(ratio.iloc[-1])
        r["liquidity_tighten"] = bool(ratio.iloc[-1] < ratio.iloc[-21])
    else:
        r["hyg_ief_ratio"]    = None
        r["liquidity_tighten"] = False

    # VIX
    vix = s("^VIX")
    r["vix"]         = float(vix.iloc[-1]) if len(vix) > 0 else None
    r["vix_elevated"] = bool(r["vix"] > 25) if r["vix"] else False

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

        rs_line    = ep_a / sa
        rs_1m      = float(ep_a.iloc[-1]/ep_a.iloc[-21] - 1) - float(sa.iloc[-1]/sa.iloc[-21] - 1)
        rs_3m      = float(ep_a.iloc[-1]/ep_a.iloc[-63] - 1) - float(sa.iloc[-1]/sa.iloc[-63] - 1)
        rs_new_hi  = bool(rs_line.iloc[-1] >= rs_line.iloc[-63:].max() * 0.98)
        ret_abs_1m = float(ep_a.iloc[-1]/ep_a.iloc[-21] - 1)
        ret_abs_3m = float(ep_a.iloc[-1]/ep_a.iloc[-63] - 1)

        if rs_1m > 0 and rs_3m > 0:
            trend = "Leading"
        elif rs_1m > 0 or rs_3m > 0:
            trend = "Mixed"
        else:
            trend = "Lagging"

        sector_rs[sector] = {
            "etf": etf, "price": round(float(ep_a.iloc[-1]), 2),
            "ret_1m": ret_abs_1m, "ret_3m": ret_abs_3m,
            "rs_1m": rs_1m, "rs_3m": rs_3m,
            "rs_new_hi": rs_new_hi, "trend": trend,
        }

    r["sector_rs"]       = sector_rs
    r["leading_sectors"] = [s for s, v in sector_rs.items() if v["trend"] == "Leading"]

    # Regime classification
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
# LAYER 1
# ─────────────────────────────────────────────────────────────────────────────
def calc_layer1(l0: dict, override: str = "Auto") -> tuple:
    if override != "Auto":
        perm = override
    elif l0.get("liquidity_tighten") and l0.get("vix_elevated"):
        perm = "Red"
    elif l0.get("spy_above_50") and l0.get("spy_ret_1m", 0) > 0:
        perm = "Green"
    elif l0.get("spy_above_50") or l0.get("spy_ret_1m", 0) > 0:
        perm = "Yellow"
    else:
        perm = "Red"
    return perm, PERM_LIMITS[perm]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 SCREENER
# ─────────────────────────────────────────────────────────────────────────────
def run_screener(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    tickers: list,
    ticker_sector: dict,
    spy: pd.Series,
    min_vol: int = 500_000,
    rs_lookback: int = 63,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        px  = close[t].dropna()
        vol = volume[t].dropna() if t in volume.columns else pd.Series(dtype=float)
        if len(px) < 63:
            continue

        price    = float(px.iloc[-1])
        ma20     = float(px.rolling(20).mean().iloc[-1])
        ma50     = float(px.rolling(50).mean().iloc[-1])
        avg_vol  = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0.0
        ret_1m   = float(px.iloc[-1] / px.iloc[-21] - 1)
        ret_3m   = float(px.iloc[-1] / px.iloc[-63] - 1)

        spy_a     = spy.reindex(px.index).ffill()
        rs_line   = px / spy_a
        rs_new_hi = bool(rs_line.iloc[-1] >= rs_line.iloc[-rs_lookback:].max() * 0.98)

        try:
            rsi_v  = float(RSIIndicator(close=px, window=14).rsi().iloc[-1])
            mo     = MACDIndicator(close=px, window_slow=26, window_fast=12, window_sign=9)
            m_hist = float(mo.macd_diff().iloc[-1])
            m_bull = bool(mo.macd().iloc[-1] > mo.macd_signal().iloc[-1])
        except Exception:
            rsi_v, m_hist, m_bull = 50.0, 0.0, False

        above_20 = price > ma20
        above_50 = price > ma50
        vol_ok   = avg_vol >= min_vol

        if   ret_1m > 0 and ret_3m > 0: two_spd = "FULL"
        elif ret_1m > 0 or  ret_3m > 0: two_spd = "HALF"
        else:                            two_spd = "NO TRADE"

        passes = above_20 and above_50 and vol_ok and rs_new_hi and two_spd == "FULL"

        rows.append({
            "Ticker":    t,
            "Sector":    ticker_sector.get(t, ""),
            "Price":     round(price, 2),
            "vs 20MA":   above_20,
            "vs 50MA":   above_50,
            "1M Ret":    ret_1m,
            "3M Ret":    ret_3m,
            "RS Hi":     rs_new_hi,
            "RSI":       round(rsi_v, 1),
            "MACD":      m_bull,
            "MACD Hist": round(m_hist, 4),
            "Avg Vol(M)": round(avg_vol / 1e6, 1),
            "2-Speed":   two_spd,
            "PASS":      passes,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER  (cached per ticker)
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

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.38, 0.12, 0.18, 0.16, 0.16],
        vertical_spacing=0.025,
        subplot_titles=[f"{ticker} — Price", "Volume", "RS vs SPY", "RSI (14)", "MACD"],
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

    fig.update_layout(
        height=750,
        paper_bgcolor="#111827", plot_bgcolor="#1f2937",
        font=dict(color="#d1d5db", size=11),
        margin=dict(l=50, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.02, bgcolor="rgba(0,0,0,0)"),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="#374151", row=i, col=1)
        fig.update_yaxes(gridcolor="#374151", row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pct(v):   return f"{v*100:+.1f}%"
def icon(v):  return "✅" if v else "❌"
def macd(v):  return "▲" if v else "▼"

def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ["1M Ret", "3M Ret"]:
        if col in d.columns:
            d[col] = d[col].apply(pct)
    for col in ["vs 20MA", "vs 50MA", "RS Hi"]:
        if col in d.columns:
            d[col] = d[col].apply(icon)
    if "MACD" in d.columns:
        d["MACD"] = d["MACD"].apply(macd)
    if "Avg Vol(M)" in d.columns:
        d["Avg Vol(M)"] = d["Avg Vol(M)"].apply(lambda x: f"{x:.1f}M")
    return d.drop(columns=["PASS", "MACD Hist"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── SIDEBAR (static controls) ─────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 Controls")
        st.caption(f"Loaded: {datetime.now().strftime('%b %d, %Y  %I:%M %p')}")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("Manual Overrides")
        regime_ov = st.selectbox(
            "Regime",
            ["Auto", "Risk-on", "Reflation", "Deflation", "Stagflation", "Mixed"],
        )
        perm_ov = st.selectbox("Permission State", ["Auto", "Green", "Yellow", "Red"])

        st.divider()
        st.subheader("Screener Settings")
        min_vol   = st.number_input("Min Avg Volume", value=500_000, step=100_000, format="%d")
        rs_lb     = st.slider("RS Lookback (days)", min_value=21, max_value=126, value=63)
        show_half = st.checkbox("Show Half Signal watchlist", value=True)
        show_all  = st.checkbox("Show full universe table",   value=False)

    # ── LOAD MACRO DATA ───────────────────────────────────────────────────────
    with st.spinner("Loading macro data..."):
        macro_close = fetch_macro_data()
        l0 = calc_layer0(macro_close)

    if "error" in l0:
        st.error(l0["error"])
        return

    # ── DERIVE REGIME / PERMISSION ────────────────────────────────────────────
    regime = regime_ov if regime_ov != "Auto" else l0["regime"]
    perm, limits = calc_layer1(l0, perm_ov)

    # Determine sectors to screen based on regime
    regime_sectors = {
        "Risk-on":     list(RISK_ON_SECTORS),
        "Reflation":   list(REFLATION_SECTORS),
        "Deflation":   list(DEFENSIVE_SECTORS),
        "Stagflation": list(REFLATION_SECTORS | {"Consumer Staples", "Utilities"}),
    }
    auto_sectors = regime_sectors.get(regime, l0.get("leading_sectors") or ALL_SECTORS)

    # Sector selector in sidebar (after we know auto_sectors)
    with st.sidebar:
        st.divider()
        st.subheader("Sectors to Screen")
        selected_sectors = st.multiselect(
            "Override if needed",
            ALL_SECTORS,
            default=auto_sectors,
            help="Auto-set from regime. Adjust here to add/remove sectors.",
        )
    active_sectors = selected_sectors if selected_sectors else auto_sectors

    # ── LOAD SCREENER DATA ────────────────────────────────────────────────────
    universe_size = sum(len(SECTOR_TICKERS.get(s, [])) for s in active_sectors)
    with st.spinner(f"Loading {universe_size} stocks..."):
        sectors_key            = ",".join(sorted(active_sectors))
        close_sc, vol_sc, all_t = fetch_screener_data(sectors_key)

    ticker_sector = {t: s for s in active_sectors for t in SECTOR_TICKERS.get(s, [])}
    spy_sc        = close_sc["SPY"] if "SPY" in close_sc.columns else pd.Series(dtype=float)

    with st.spinner("Running screener..."):
        results_df = run_screener(close_sc, vol_sc, all_t, ticker_sector, spy_sc, min_vol, rs_lb)

    if results_df.empty:
        st.error("Screener returned no results. Check your internet connection and refresh.")
        return

    passes_df = results_df[results_df["PASS"]].sort_values(["Sector", "3M Ret"], ascending=[True, False])
    half_df   = results_df[(results_df["2-Speed"] == "HALF") & (~results_df["PASS"])].sort_values("3M Ret", ascending=False)

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    st.markdown("## Swing Trading Framework")
    st.caption(f"{datetime.now().strftime('%A, %B %d, %Y')}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime",        regime)
    c2.metric("Permission",    perm)
    c3.metric("SPY",           f"${l0['spy_price']:.2f}", f"{l0['spy_ret_1m']*100:+.1f}% 1M")
    c4.metric("Max Positions", limits["max_pos"])
    c5.metric("Risk / Trade",  f"{limits['risk_lo']}–{limits['risk_hi']}%" if limits["risk_hi"] > 0 else "No new trades")
    c6.metric("Max Heat",      f"{limits['heat']}%")

    if   perm == "Green":  st.success(f"🟢 **GREEN** — Full deployment. Momentum bias. Up to {limits['max_pos']} positions.")
    elif perm == "Yellow": st.warning(f"🟡 **YELLOW** — Selective. {limits['max_pos']} positions max. Reduced risk.")
    else:                  st.error(  f"🔴 **RED** — No new entries. Protect capital.")

    if l0.get("liquidity_tighten"):
        st.warning("⚠️ Liquidity tightening (HYG/IEF ratio declining). Treat as override — reduce exposure.")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Layer 0 — Macro", "🔍 Layer 2 — Screener", "📈 Charts"])

    # ── TAB 1: MACRO DASHBOARD ────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("SPY")
            st.dataframe(pd.DataFrame([
                {"Signal": "Price",          "Value": f"${l0['spy_price']:.2f}"},
                {"Signal": "vs 20d MA",      "Value": f"{'✅' if l0['spy_above_20'] else '❌'}  ${l0['spy_ma20']:.2f}"},
                {"Signal": "vs 50d MA",      "Value": f"{'✅' if l0['spy_above_50'] else '❌'}  ${l0['spy_ma50']:.2f}"},
                {"Signal": "1-Month Return", "Value": pct(l0["spy_ret_1m"])},
                {"Signal": "3-Month Return", "Value": pct(l0["spy_ret_3m"])},
            ]), hide_index=True, use_container_width=True)

            st.subheader("Bond & Liquidity")
            tlt_above = l0.get("tlt_above_50")
            tlt_ret   = l0.get("tlt_ret_1m")
            hyg_r     = l0.get("hyg_ief_ratio")
            vix_v     = l0.get("vix")
            st.dataframe(pd.DataFrame([
                {"Signal": "TLT vs 50d MA",
                 "Value":  ("✅ Above" if tlt_above else "❌ Below") if tlt_above is not None else "N/A"},
                {"Signal": "TLT 1M Return",
                 "Value":  pct(tlt_ret) if tlt_ret is not None else "N/A"},
                {"Signal": "HYG/IEF Ratio",
                 "Value":  f"{hyg_r:.3f}  {'⬇️ Tightening' if l0.get('liquidity_tighten') else '➡️ Stable'}" if hyg_r else "N/A"},
                {"Signal": "VIX",
                 "Value":  f"{vix_v:.1f}  {'⚠️ Elevated' if l0.get('vix_elevated') else '✅ Normal'}" if vix_v else "N/A"},
            ]), hide_index=True, use_container_width=True)

        with col_b:
            st.subheader("Sector Relative Strength vs SPY")
            sr = l0.get("sector_rs", {})
            if sr:
                rows_sr = []
                for sec, v in sr.items():
                    emoji = "🟢" if v["trend"] == "Leading" else "🟡" if v["trend"] == "Mixed" else "🔴"
                    rows_sr.append({
                        "Sector": sec,
                        "ETF":    v["etf"],
                        "Price":  f"${v['price']:.2f}",
                        "1M RS":  pct(v["rs_1m"]),
                        "3M RS":  pct(v["rs_3m"]),
                        "RS Hi":  icon(v["rs_new_hi"]),
                        "Status": f"{emoji} {v['trend']}",
                        "_sort":  v["rs_3m"],
                    })
                sr_df = pd.DataFrame(rows_sr).sort_values("_sort", ascending=False).drop(columns=["_sort"])
                st.dataframe(sr_df, hide_index=True, use_container_width=True, height=380)

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
                all_s = results_df.sort_values(["PASS", "2-Speed", "3M Ret"], ascending=[False, True, False])
                st.dataframe(fmt_df(all_s), hide_index=True, use_container_width=True)

    # ── TAB 3: CHARTS ─────────────────────────────────────────────────────────
    with tab3:
        if passes_df.empty:
            st.info("No Full Signal candidates to chart. Run the screener first.")
        else:
            ticker_opts = passes_df["Ticker"].tolist()
            sel = st.selectbox(
                "Select ticker",
                ticker_opts,
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

            # Show all charts toggle
            if st.checkbox("Show all Full Signal charts"):
                for _, row in passes_df.iterrows():
                    t = row["Ticker"]
                    label = f"{t}  —  {row['Sector']}  |  1M: {pct(row['1M Ret'])}  |  3M: {pct(row['3M Ret'])}"
                    with st.expander(label, expanded=False):
                        with st.spinner(f"Loading {t}..."):
                            fig = build_chart(t)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
