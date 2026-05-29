"""
Swing Trading Framework — Regime-Aware Layer 2 Screener + Dashboard
--------------------------------------------------------------------
Requires:   pip install yfinance pandas ta plotly
Run:        python3 screener.py
Output:     Prints results to terminal + opens HTML dashboard in browser
"""

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser, os, warnings
warnings.filterwarnings('ignore')

# ── SECTOR & PERMISSION INPUT — asked at runtime ─────────────────────────────

ALL_SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Technology",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Utilities",
    "Health Care",
]

# Regime label is derived from sector selection — for display only
REGIME_PATTERNS = {
    frozenset(["Technology", "Financials", "Consumer Discretionary"]): "Risk-on",
    frozenset(["Technology", "Financials"]): "Risk-on",
    frozenset(["Energy", "Materials", "Industrials"]): "Reflation",
    frozenset(["Energy", "Materials"]): "Reflation",
    frozenset(["Energy", "Industrials"]): "Reflation",
    frozenset(["Consumer Staples", "Utilities", "Health Care"]): "Deflation",
    frozenset(["Consumer Staples", "Utilities"]): "Deflation",
    frozenset(["Utilities", "Health Care"]): "Deflation",
}

VALID_PERMISSIONS = ["Green", "Yellow", "Red"]

print("\n── Swing Trading Screener ──────────────────────────────────")
print("Based on your Layer 0 review, select the sectors showing")
print("RS leadership (outperforming SPY). Enter the numbers separated")
print("by commas, e.g.  1,2,3\n")

for i, s in enumerate(ALL_SECTORS, 1):
    print(f"  {i}. {s}")

while True:
    raw = input("\nLeading sectors: ").strip()
    try:
        picks = [int(x.strip()) for x in raw.split(",")]
        if all(1 <= p <= len(ALL_SECTORS) for p in picks) and len(picks) >= 1:
            active_sectors = [ALL_SECTORS[p - 1] for p in picks]
            break
    except ValueError:
        pass
    print(f"  ✗ Enter numbers between 1 and {len(ALL_SECTORS)}, separated by commas")

# Derive regime label for display
REGIME = REGIME_PATTERNS.get(frozenset(active_sectors), "Mixed")

while True:
    print("\nPermission state — Green | Yellow | Red")
    PERMISSION = input("Permission state: ").strip().capitalize()
    if PERMISSION in VALID_PERMISSIONS:
        break
    print(f"  ✗ Please enter: Green, Yellow, or Red")

print(f"\nRegime pattern: {REGIME}")
print(f"Sectors selected: {', '.join(active_sectors)}")
print("────────────────────────────────────────────────────────────\n")

SECTOR_TICKERS = {
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "DVN","HAL","BKR","FANG","MRO","APA","EQT","CTRA","TRGP","OKE",
        "KMI","WMB","LNG","CVI","MGY"
    ],
    "Materials": [
        "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","ALB",
        "DD","EMN","IFF","PPG","RPM","FMC","MOS","CF","BALL","IP",
        "PKG","SEE","CCK","AVY","SON","AMCR","CE","DOW","LYB","WLK"
    ],
    "Industrials": [
        "RTX","HON","UNP","UPS","BA","LMT","GE","CAT","DE","MMM",
        "ITW","EMR","ETN","PH","ROK","FDX","CSX","NSC","WM","RSG",
        "CTAS","CPRT","GWW","AME","TT","IR","CARR","OTIS","PWR","URI",
        "MAS","JCI","XYL","AXON","TDG","HWM","NOC","GD","LHX","LDOS",
        "HUBB","FTV","RRX","GNRC","SAIA","ODFL","JBHT","EXPD","TXT"
    ],
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","AMD","QCOM","TXN",
        "AMAT","LRCX","KLAC","MU","ADI","MCHP","CDNS","SNPS","FTNT",
        "PANW","CRWD","NOW","ZS","DDOG","NET"
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG",
        "ROST","ORLY","AZO","DHI","LEN","PHM","ULTA","YUM","DRI"
    ],
    "Financials": [
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW",
        "AXP","C","USB","PNC","TFC","COF","ICE","CME","SPGI","MCO"
    ],
    "Consumer Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","GIS","K",
        "SJM","HRL","CAG","CPB","MKC","CHD","CLX","KMB","MDLZ"
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","ETR","ED",
        "XEL","WEC","ES","AWK","DTE","FE","PPL","AEE","CMS","NI"
    ],
    "Health Care": [
        "LLY","UNH","JNJ","ABT","TMO","DHR","BMY","AMGN","ISRG","MDT",
        "SYK","BSX","EW","BDX","IDXX","DXCM"
    ],
}

MIN_AVG_VOLUME = 500_000
RS_LOOKBACK    = 63
SCREEN_PERIOD  = "1y"
CHART_PERIOD   = "6mo"

# ── BUILD UNIVERSE ────────────────────────────────────────────────────────────
all_tickers   = [t for s in active_sectors for t in SECTOR_TICKERS.get(s, [])]
ticker_sector = {t: s for s in active_sectors for t in SECTOR_TICKERS.get(s, [])}

print(f"\nRegime: {REGIME} | Permission: {PERMISSION}")
print(f"Sectors: {active_sectors} | Universe: {len(all_tickers)} stocks\n")

# ── DOWNLOAD SCREENING DATA ───────────────────────────────────────────────────
print("Downloading screening data (1y)...")
raw    = yf.download(all_tickers + ["SPY"], period=SCREEN_PERIOD, auto_adjust=True, progress=False)
close  = raw["Close"]
volume = raw["Volume"]
spy    = close["SPY"]
print(f"  {len(close)} trading days loaded\n")

# ── COMPUTE LAYER 2 SIGNALS ───────────────────────────────────────────────────
results = []

for ticker in all_tickers:
    if ticker not in close.columns:
        continue
    px  = close[ticker].dropna()
    vol = volume[ticker].dropna()
    if len(px) < 63:
        continue

    price   = px.iloc[-1]
    ma20    = px.rolling(20).mean().iloc[-1]
    ma50    = px.rolling(50).mean().iloc[-1]
    avg_vol = vol.rolling(20).mean().iloc[-1]
    ret_1m  = px.iloc[-1] / px.iloc[-21] - 1
    ret_3m  = px.iloc[-1] / px.iloc[-63] - 1

    spy_a       = spy.reindex(px.index).ffill()
    rs_line     = px / spy_a
    rs_current  = rs_line.iloc[-1]
    rs_high     = rs_line.iloc[-RS_LOOKBACK:].max()
    rs_new_high = rs_current >= rs_high * 0.98

    rsi_val   = RSIIndicator(close=px, window=14).rsi().iloc[-1]
    macd_obj  = MACD(close=px, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    macd_hist = macd_obj.macd_diff().iloc[-1]
    macd_bull = macd_line > macd_sig

    above_20 = price > ma20
    above_50 = price > ma50
    vol_ok   = avg_vol >= MIN_AVG_VOLUME

    if ret_1m > 0 and ret_3m > 0:
        two_speed = "FULL"
    elif ret_1m > 0 or ret_3m > 0:
        two_speed = "HALF"
    else:
        two_speed = "NO TRADE"

    passes = above_20 and above_50 and vol_ok and rs_new_high and two_speed == "FULL"

    results.append({
        "Ticker":    ticker,
        "Sector":    ticker_sector.get(ticker, ""),
        "Price":     round(price, 2),
        "vs 20MA":   above_20,
        "vs 50MA":   above_50,
        "1M Ret":    ret_1m,
        "3M Ret":    ret_3m,
        "RS Hi":     rs_new_high,
        "RSI":       round(rsi_val, 1),
        "MACD Bull": macd_bull,
        "MACD Hist": round(macd_hist, 3),
        "AvgVol(M)": round(avg_vol / 1e6, 1),
        "2-Speed":   two_speed,
        "PASS":      passes,
    })

df        = pd.DataFrame(results)
passes_df = df[df["PASS"]].sort_values(["Sector","3M Ret"], ascending=[True, False])
half_df   = df[(df["2-Speed"]=="HALF") & (~df["PASS"])].sort_values("3M Ret", ascending=False)

# ── TERMINAL PRINT ────────────────────────────────────────────────────────────
def fmt_ret(v): return f"{v*100:+.1f}%"
def fmt_bool(v): return "✅" if v else "❌"
def fmt_macd(v): return "▲" if v else "▼"

print("=" * 70)
print(f"  REGIME: {REGIME}  |  PERMISSION: {PERMISSION}  |  {pd.Timestamp.today().date()}")
print(f"  Screened: {len(results)} stocks")
print("=" * 70)
print(f"\n✅ FULL SIGNAL — {len(passes_df)} passing all Layer 2 filters\n")
for _, r in passes_df.iterrows():
    print(f"  {r['Ticker']:<6} {r['Sector']:<22} ${r['Price']:<8.2f} "
          f"1M:{fmt_ret(r['1M Ret'])} 3M:{fmt_ret(r['3M Ret'])} "
          f"RSI:{r['RSI']} MACD:{fmt_macd(r['MACD Bull'])}")

print(f"\n⚠️  HALF SIGNAL — top 10 watch list\n")
for _, r in half_df.head(10).iterrows():
    print(f"  {r['Ticker']:<6} {r['Sector']:<22} ${r['Price']:<8.2f} "
          f"1M:{fmt_ret(r['1M Ret'])} 3M:{fmt_ret(r['3M Ret'])} "
          f"RSI:{r['RSI']} MACD:{fmt_macd(r['MACD Bull'])}")

# ── BUILD HTML DASHBOARD ──────────────────────────────────────────────────────
print("\nBuilding HTML dashboard...")

PERMISSION_COLORS = {"Green": "#22c55e", "Yellow": "#f59e0b", "Red": "#ef4444"}
perm_color = PERMISSION_COLORS.get(PERMISSION, "#888")

def bool_cell(val, true_label="✅", false_label="❌"):
    color = "#16a34a" if val else "#dc2626"
    label = true_label if val else false_label
    return f'<span style="color:{color};font-weight:600">{label}</span>'

def ret_cell(val):
    color = "#16a34a" if val > 0 else "#dc2626"
    return f'<span style="color:{color};font-weight:600">{val*100:+.1f}%</span>'

def rsi_cell(val):
    if val >= 70:
        color = "#dc2626"
    elif val <= 30:
        color = "#2563eb"
    else:
        color = "#d1d5db"
    return f'<span style="color:{color};font-weight:600">{val:.1f}</span>'

def speed_cell(val):
    colors = {"FULL": "#16a34a", "HALF": "#f59e0b", "NO TRADE": "#dc2626"}
    return f'<span style="color:{colors.get(val,"#888")};font-weight:600">{val}</span>'

def build_table_rows(data):
    rows = ""
    for _, r in data.iterrows():
        pass_bg = "#1a2e1a" if r["PASS"] else ""
        rows += f"""<tr style="background:{pass_bg}">
            <td style="font-weight:700;color:#f9fafb">{r['Ticker']}</td>
            <td style="color:#9ca3af">{r['Sector']}</td>
            <td>${r['Price']:.2f}</td>
            <td>{bool_cell(r['vs 20MA'])}</td>
            <td>{bool_cell(r['vs 50MA'])}</td>
            <td>{ret_cell(r['1M Ret'])}</td>
            <td>{ret_cell(r['3M Ret'])}</td>
            <td>{bool_cell(r['RS Hi'])}</td>
            <td>{rsi_cell(r['RSI'])}</td>
            <td>{'<span style="color:#16a34a;font-weight:700">▲</span>' if r['MACD Bull'] else '<span style="color:#dc2626;font-weight:700">▼</span>'}</td>
            <td style="color:#9ca3af">{r['AvgVol(M)']:.1f}M</td>
            <td>{speed_cell(r['2-Speed'])}</td>
        </tr>"""
    return rows

# ── STOCK CHARTS FOR PASSING TICKERS ─────────────────────────────────────────
chart_html = ""
passing_tickers = passes_df["Ticker"].tolist()

if passing_tickers:
    print(f"  Downloading 6-month chart data for {len(passing_tickers)} passing stocks...")

    # Download SPY once for RS line calculation using Ticker API (more reliable than download())
    spy_hist  = yf.Ticker("SPY").history(period=CHART_PERIOD)
    spy_close = spy_hist["Close"]

    for ticker in passing_tickers:
        hist = yf.Ticker(ticker).history(period=CHART_PERIOD)
        if hist.empty:
            continue

        px  = hist["Close"].dropna()
        op  = hist["Open"].reindex(px.index)
        hi  = hist["High"].reindex(px.index)
        lo  = hist["Low"].reindex(px.index)
        vol = hist["Volume"].reindex(px.index)

        # Debug: confirm real prices (remove after first successful run)
        print(f"  {ticker} close sample: {px.iloc[-1]:.2f} (should be ~${passes_df[passes_df['Ticker']==ticker]['Price'].values[0]:.2f})")

        ma20_s = px.rolling(20).mean()
        ma50_s = px.rolling(50).mean()

        spy_a  = spy_close.reindex(px.index).ffill()
        rs_s   = px / spy_a

        rsi_s  = RSIIndicator(close=px, window=14).rsi()
        macd_o = MACD(close=px, window_slow=26, window_fast=12, window_sign=9)
        macd_l = macd_o.macd()
        macd_g = macd_o.macd_signal()
        macd_h = macd_o.macd_diff()

        # Convert everything to plain Python lists — eliminates numpy/pandas serialization issues
        def to_list(s):
            return [None if (v != v) else float(v) for v in s]  # NaN → None for Plotly

        dates    = [str(d)[:10] for d in px.index]  # 'YYYY-MM-DD' strings
        px_v     = to_list(px)
        op_v     = to_list(op)
        hi_v     = to_list(hi)
        lo_v     = to_list(lo)
        vol_v    = to_list(vol)
        ma20_v   = to_list(ma20_s)
        ma50_v   = to_list(ma50_s)
        rs_v     = to_list(rs_s)
        rsi_v    = to_list(rsi_s)
        macd_l_v = to_list(macd_l)
        macd_g_v = to_list(macd_g)
        macd_h_v = to_list(macd_h)

        sector = ticker_sector.get(ticker, "")
        row    = passes_df[passes_df["Ticker"] == ticker].iloc[0]

        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            row_heights=[0.38, 0.12, 0.18, 0.16, 0.16],
            vertical_spacing=0.025,
            subplot_titles=[
                f"{ticker} — Price + MAs (6mo)",
                "Volume",
                "RS Line vs SPY",
                "RSI (14)",
                "MACD (12/26/9)"
            ]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dates, open=op_v, high=hi_v, low=lo_v, close=px_v,
            name=ticker,
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=dates, y=ma20_v, name="20d MA",
            line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=ma50_v, name="50d MA",
            line=dict(color="#3b82f6", width=1.5)), row=1, col=1)

        # Volume — separate subplot
        vol_colors = ["#22c55e" if c >= o else "#ef4444"
                      for c, o in zip(px_v, op_v)]
        fig.add_trace(go.Bar(x=dates, y=vol_v, name="Volume",
            marker_color=vol_colors, opacity=0.7, showlegend=False), row=2, col=1)

        # RS line
        fig.add_trace(go.Scatter(x=dates, y=rs_v, name="RS Line",
            line=dict(color="#a78bfa", width=1.5), showlegend=False), row=3, col=1)

        # RSI — fixed 0-100 axis
        fig.add_trace(go.Scatter(x=dates, y=rsi_v, name="RSI",
            line=dict(color="#38bdf8", width=1.5), showlegend=False), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=4, col=1)

        # MACD
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444"
                       for v in macd_h.fillna(0).values]
        fig.add_trace(go.Bar(x=dates, y=macd_h_v, name="Hist",
            marker_color=hist_colors, showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_l_v, name="MACD",
            line=dict(color="#f59e0b", width=1.5), showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_g_v, name="Signal",
            line=dict(color="#a78bfa", width=1.5), showlegend=False), row=5, col=1)

        fig.update_layout(
            height=800,
            paper_bgcolor="#111827",
            plot_bgcolor="#1f2937",
            font=dict(color="#d1d5db", size=11),
            margin=dict(l=50, r=20, t=40, b=20),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", x=0, y=1.02,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            showlegend=True,
        )
        for i in range(1, 6):
            fig.update_xaxes(gridcolor="#374151", row=i, col=1)
            fig.update_yaxes(gridcolor="#374151", row=i, col=1)

        # Fixed RSI axis
        fig.update_yaxes(range=[0, 100], row=4, col=1)

        chart_div = fig.to_html(full_html=False, include_plotlyjs=False)
        chart_html += f"""
        <div class="chart-block">
            <div class="chart-header">
                <span class="chart-ticker">{ticker}</span>
                <span class="chart-sector">{sector}</span>
                <span class="chart-stat">1M: <b>{row['1M Ret']*100:+.1f}%</b></span>
                <span class="chart-stat">3M: <b>{row['3M Ret']*100:+.1f}%</b></span>
                <span class="chart-stat">RSI: <b>{row['RSI']}</b></span>
                <span class="chart-stat">Price: <b>${row['Price']:.2f}</b></span>
            </div>
            {chart_div}
        </div>"""

# ── FULL TABLE HTML ───────────────────────────────────────────────────────────
all_sorted = df.sort_values(["PASS","2-Speed","3M Ret"], ascending=[False, True, False])
table_rows = build_table_rows(all_sorted)

# ── ASSEMBLE HTML ─────────────────────────────────────────────────────────────
run_date = pd.Timestamp.today().strftime("%B %d, %Y")

# Pre-compute conditional blocks (required for Python 3.9 f-string compatibility)
no_trade_count  = len(df[df['2-Speed'] == 'NO TRADE'])
passes_table_html = (
    "<p class='no-pass'>No stocks passing all filters in current regime conditions.</p>"
    if len(passes_df) == 0
    else f"""<table>
    <thead><tr>
      <th>Ticker</th><th>Sector</th><th>Price</th>
      <th>vs 20MA</th><th>vs 50MA</th><th>1M Ret</th><th>3M Ret</th>
      <th>RS Hi</th><th>RSI</th><th>MACD</th><th>Avg Vol</th><th>Two-Speed</th>
    </tr></thead>
    <tbody>{build_table_rows(passes_df)}</tbody>
  </table>"""
)
charts_section_html = "<p class='no-pass'>No passing stocks to chart.</p>" if not chart_html else chart_html

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swing Trading Screener — {run_date}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  .header {{ background: #1e293b; border-bottom: 1px solid #334155; padding: 20px 32px; display: flex; align-items: center; gap: 24px; }}
  .header h1 {{ font-size: 1.4rem; font-weight: 700; color: #f8fafc; }}
  .badge {{ padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }}
  .regime-badge {{ background: #1e3a5f; color: #93c5fd; }}
  .perm-badge {{ background: #1c1a10; color: {perm_color}; border: 1px solid {perm_color}; }}
  .date-badge {{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }}
  .main {{ padding: 24px 32px; }}
  .summary-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px 22px; min-width: 140px; }}
  .stat-card .num {{ font-size: 2rem; font-weight: 800; }}
  .stat-card .lbl {{ font-size: 0.75rem; color: #94a3b8; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .pass-num {{ color: #22c55e; }}
  .half-num {{ color: #f59e0b; }}
  .total-num {{ color: #60a5fa; }}
  .section-title {{ font-size: 1rem; font-weight: 700; color: #f1f5f9; margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 1px solid #334155; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; background: #1e293b; border-radius: 10px; overflow: hidden; }}
  th {{ background: #0f172a; color: #94a3b8; font-weight: 600; padding: 10px 12px; text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; color: #cbd5e1; white-space: nowrap; }}
  tr:hover td {{ background: #263348; }}
  .chart-block {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; margin-bottom: 24px; overflow: hidden; }}
  .chart-header {{ padding: 14px 20px; display: flex; align-items: center; gap: 20px; border-bottom: 1px solid #334155; background: #0f172a; }}
  .chart-ticker {{ font-size: 1.2rem; font-weight: 800; color: #f8fafc; }}
  .chart-sector {{ color: #94a3b8; font-size: 0.85rem; }}
  .chart-stat {{ font-size: 0.85rem; color: #94a3b8; }}
  .chart-stat b {{ color: #e2e8f0; }}
  .no-pass {{ color: #94a3b8; font-style: italic; padding: 20px 0; }}
</style>
</head>
<body>

<div class="header">
  <h1>Swing Trading Screener</h1>
  <span class="badge regime-badge">⚙ {REGIME}</span>
  <span class="badge perm-badge">◉ {PERMISSION}</span>
  <span class="badge date-badge">{run_date}</span>
</div>

<div class="main">

  <div class="summary-row">
    <div class="stat-card"><div class="num total-num">{len(results)}</div><div class="lbl">Stocks Screened</div></div>
    <div class="stat-card"><div class="num pass-num">{len(passes_df)}</div><div class="lbl">Full Signal (Pass)</div></div>
    <div class="stat-card"><div class="num half-num">{len(half_df)}</div><div class="lbl">Half Signal (Watch)</div></div>
    <div class="stat-card"><div class="num" style="color:#94a3b8">{no_trade_count}</div><div class="lbl">No Trade</div></div>
  </div>

  <div class="section-title">✅ Full Signal — Layer 2 Candidates ({len(passes_df)} stocks)</div>
  {passes_table_html}

  <div class="section-title">⚠️ Half Signal — Watch List (top 15)</div>
  <table>
    <thead><tr>
      <th>Ticker</th><th>Sector</th><th>Price</th>
      <th>vs 20MA</th><th>vs 50MA</th><th>1M Ret</th><th>3M Ret</th>
      <th>RS Hi</th><th>RSI</th><th>MACD</th><th>Avg Vol</th><th>Two-Speed</th>
    </tr></thead>
    <tbody>{build_table_rows(half_df.head(15))}</tbody>
  </table>

  <div class="section-title">6-Month Charts — Passing Stocks</div>
  {charts_section_html}

  <div class="section-title">Full Universe — All {len(results)} Stocks</div>
  <table>
    <thead><tr>
      <th>Ticker</th><th>Sector</th><th>Price</th>
      <th>vs 20MA</th><th>vs 50MA</th><th>1M Ret</th><th>3M Ret</th>
      <th>RS Hi</th><th>RSI</th><th>MACD</th><th>Avg Vol</th><th>Two-Speed</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

</div>
</body>
</html>"""

# ── SAVE & OPEN ───────────────────────────────────────────────────────────────
out_dir   = os.path.dirname(os.path.abspath(__file__))
html_file = os.path.join(out_dir, f"screener_dashboard_{pd.Timestamp.today().date()}.html")
csv_file  = os.path.join(out_dir, f"screen_{REGIME.lower().replace('-','_')}_{PERMISSION.lower()}_{pd.Timestamp.today().date()}.csv")

with open(html_file, "w") as f:
    f.write(html)

df.to_csv(csv_file, index=False)

print(f"  Dashboard saved: {os.path.basename(html_file)}")
print(f"  CSV saved:       {os.path.basename(csv_file)}")
print(f"\nOpening dashboard in browser...\n")
webbrowser.open(f"file://{html_file}")
