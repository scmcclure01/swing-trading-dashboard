"""
Layer calculations — the analytical core of the framework.

Pure-ish computation over price/FRED data:
  L0  calc_layer0              — macro regime, SPY trend, sector RS, liquidity
      score_recession_composite — 5-model recession score
  L2  calc_layer2             — permission state (Green/Yellow/Red)
  L3  calc_layer3             — sector RRG (relative rotation)
  L4  calc_layer4             — stock selection filters
  L5  calc_layer5             — entry-trigger evaluation per candidate

calc_layer0 / calc_layer3 are @st.cache_data-decorated; the rest are pure
functions of their arguments (data is passed in, not fetched here). calc_layer0
is the only one that reaches out — to data.fred for Fed Net Liquidity.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from config import (
    LB_1M, LB_3M, LB_6M,
    RS_NEW_HI_THRESHOLD, VELOCITY_THRESHOLD,
    SECTOR_ETFS,
    RISK_ON_SECTORS, REFLATION_SECTORS, DEFENSIVE_SECTORS,
    PERM_LIMITS, PHASE_MAP,
)
from data.fred import fetch_fed_net_liquidity


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

        # Velocity Flag (v4): ROC 21 for the sector ETF
        roc_21 = float(ep_a.iloc[-1] / ep_a.iloc[-LB_1M] - 1)
        velocity_status = ("ACCELERATING" if roc_21 > VELOCITY_THRESHOLD
                           else "NORMAL" if roc_21 > 0.05
                           else "SLOW")

        sector_rs[sector] = {
            "etf":      etf,
            "price":    round(float(ep_a.iloc[-1]), 2),
            "ret_1m":   float(ep_a.iloc[-1] / ep_a.iloc[-LB_1M] - 1),
            "ret_3m":   float(ep_a.iloc[-1] / ep_a.iloc[-LB_3M] - 1),
            "rs_1m":    rs_1m,
            "rs_3m":    rs_3m,
            "rs_new_hi": rs_new_hi,
            "trend":    trend,
            "roc_21":   roc_21,
            "velocity": velocity_status,
        }

    r["sector_rs"]       = sector_rs
    r["leading_sectors"] = [sec for sec, v in sector_rs.items() if v["trend"] == "Leading"]
    r["mixed_sectors"]   = [sec for sec, v in sector_rs.items() if v["trend"] == "Mixed"]

    # Velocity Flag summary (v4)
    r["velocity_flags"]  = {sec: v["velocity"] for sec, v in sector_rs.items()}
    r["accelerating"]    = [sec for sec, v in sector_rs.items() if v["velocity"] == "ACCELERATING"]

    # Regime classification — use aggregate RS weight per category.
    # Sum the 3-month RS of all sectors in each group. Highest wins.
    # Stagflation = Reflation sectors leading AND Defensive sectors leading simultaneously.
    def _group_rs(group):
        return sum(sector_rs[s]["rs_3m"] for s in group if s in sector_rs)

    ro_rs = _group_rs(RISK_ON_SECTORS)
    re_rs = _group_rs(REFLATION_SECTORS)
    de_rs = _group_rs(DEFENSIVE_SECTORS)

    # Stagflation: reflation AND defensive both positive (growth down, inflation up)
    if re_rs > 0 and de_rs > 0 and ro_rs < re_rs and ro_rs < de_rs:
        r["regime"] = "Stagflation"
    elif ro_rs >= re_rs and ro_rs >= de_rs:
        r["regime"] = "Risk-on"
    elif re_rs >= ro_rs and re_rs >= de_rs:
        r["regime"] = "Reflation"
    else:
        r["regime"] = "Deflation"

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
# LAYER 1 — MONTHLY MACRO MISPRICING CHECK
# ─────────────────────────────────────────────────────────────────────────────

def score_layer1_mispricing(fwd_earnings_yield: float | None,
                            tips_10y: float | None,
                            nominal_10y: float | None,
                            gdpnow_signal: str = "Not set") -> dict:
    """
    Score the monthly macro-mispricing composite (framework Layer 1).

    Three checks, each scored Bullish (+1) / Neutral (0) / Bearish (-1):
      Check 1 — Earnings yield vs real rates: fwd EY minus 10y TIPS (DFII10)
                  > 4%  cheap (+1)  |  2-4% fair (0)  |  < 2% expensive (-1)
      Check 2 — GDPNow vs analyst EPS gap: manual (no clean FRED series).
                  Pass "Bullish"/"Bearish"/"Neutral"/"Not set".
      Check 3 — Equity risk premium: fwd EY minus 10y nominal (DGS10)
                  > 5% cheap (+1)  |  3-5% normal (0)  |  < 3% stretched (-1)

    Composite → sizing action (framework):
      +2..+3 → Full risk
      -1..+1 → Normal operations
      -2..-3 → Reduce all position sizes 30-50%, top-tier setups only

    Returns a dict with the three checks (name/value/score/label), the composite
    score, the sizing verdict, and a 'computable' flag (False if rate data missing).
    """
    checks = []
    computable = fwd_earnings_yield is not None and tips_10y is not None and nominal_10y is not None

    # ── Check 1: Earnings yield vs real rates (TIPS) ──────────────────────────
    if fwd_earnings_yield is not None and tips_10y is not None:
        spread1 = round(fwd_earnings_yield - tips_10y, 2)
        if spread1 > 4:
            s1, lbl1 = 1, "Cheap vs real rates"
        elif spread1 >= 2:
            s1, lbl1 = 0, "Fairly valued"
        else:
            s1, lbl1 = -1, "Expensive vs real rates"
        val1 = f"{spread1:+.2f}%"
    else:
        s1, lbl1, val1 = 0, "N/A", "N/A"
    checks.append({"name": "Earnings Yield − TIPS (real)", "value": val1,
                   "score": s1, "label": lbl1, "threshold": "> 4% cheap / < 2% rich"})

    # ── Check 2: GDPNow vs analyst EPS gap (manual) ───────────────────────────
    gdp_map = {"Bullish": 1, "Bearish": -1, "Neutral": 0, "Not set": 0}
    s2 = gdp_map.get(gdpnow_signal, 0)
    checks.append({"name": "GDPNow vs Analyst EPS", "value": gdpnow_signal,
                   "score": s2, "label": gdpnow_signal, "threshold": "manual (Atlanta Fed)"})

    # ── Check 3: Equity risk premium (vs nominal 10y) ─────────────────────────
    if fwd_earnings_yield is not None and nominal_10y is not None:
        erp = round(fwd_earnings_yield - nominal_10y, 2)
        if erp > 5:
            s3, lbl3 = 1, "Cheap"
        elif erp >= 3:
            s3, lbl3 = 0, "Normal"
        else:
            s3, lbl3 = -1, "Stretched"
        val3 = f"{erp:+.2f}%"
    else:
        s3, lbl3, val3 = 0, "N/A", "N/A"
    checks.append({"name": "Equity Risk Premium", "value": val3,
                   "score": s3, "label": lbl3, "threshold": "> 5% cheap / < 3% stretched"})

    composite = s1 + s2 + s3
    if composite >= 2:
        verdict, sizing = "Full risk", "No sizing adjustment — macro supports full risk."
    elif composite >= -1:
        verdict, sizing = "Normal operations", "Normal sizing."
    else:
        verdict, sizing = "Reduce sizing", "Reduce all position sizes 30-50%. Top-tier setups only. Tighten trailing stops."

    return {
        "checks":     checks,
        "composite":  composite,
        "verdict":    verdict,
        "sizing":     sizing,
        "computable": computable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9 — DRAWDOWN TIER (auto from peak equity)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical drawdown-state strings. These MUST match the sidebar dropdown options
# verbatim so the position-sizing logic (which matches on "Tier 2", "7–10%", etc.)
# reacts correctly whether the tier is auto-computed or manually selected.
DRAWDOWN_STATES = {
    "peak":      "At or near peak — full risk",
    "tier1":     "Tier 1: 0–7% drawdown — full operations",
    "tier2":     "Tier 2: 7–10% drawdown — reduce risk 50%",
    "tier3":     "Tier 3: 10–15% drawdown — defensive",
    "emergency": ">15% drawdown — 100% cash",
}


def drawdown_tier(current_equity: float, peak_equity: float) -> dict:
    """Map current vs peak equity to a Layer 9 drawdown tier.

    Returns {'pct': drawdown % (<= 0), 'state': canonical dropdown string,
             'label': short tier name, 'color': hex}. Thresholds per v4:
      0 to -7%   Tier 1 normal | -7 to -10% Tier 2 cut 50% |
      -10 to -15% Tier 3 defensive | < -15% emergency cash.
    """
    if peak_equity and peak_equity > 0:
        pct = (current_equity - peak_equity) / peak_equity * 100
    else:
        pct = 0.0

    if pct >= 0:
        key, label, color = "peak", "At peak", "#27500A"
    elif pct > -7:
        key, label, color = "tier1", "Tier 1: Normal", "#27500A"
    elif pct > -10:
        key, label, color = "tier2", "Tier 2: Reduce risk 50%", "#E07800"
    elif pct > -15:
        key, label, color = "tier3", "Tier 3: Defensive", "#CC1111"
    else:
        key, label, color = "emergency", "Emergency: 100% cash", "#CC1111"

    return {"pct": round(pct, 1), "state": DRAWDOWN_STATES[key],
            "label": label, "color": color}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — MARKET PERMISSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def calc_layer2(l0: dict, rec_flags: int, eps_signal: str, override: str = "Auto") -> tuple:
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
# LAYER 3 — SECTOR ROTATION (RRG)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def calc_layer3(close: pd.DataFrame) -> list:
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
# LAYER 4 — STOCKSEL
# ─────────────────────────────────────────────────────────────────────────────

def calc_layer4(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    tickers: list,
    ticker_sector: dict,
    spy: pd.Series,
    min_dollar_vol: float = 10_000_000,
    rs_lookback: int = LB_3M,
) -> pd.DataFrame:
    """
    Screen individual stocks against Layer 4 criteria:
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

        # % above 20d MA (v4 — for Accelerating Protocol screening)
        pct_above_20 = (price / ma20 - 1) if ma20 > 0 else 0.0

        rows.append({
            "Ticker":     t,
            "Sector":     ticker_sector.get(t, ""),
            "Price":      round(price, 2),
            "vs 20MA":    above_20,
            "vs 50MA":    above_50,
            "% > 20MA":   round(pct_above_20, 4),
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
# LAYER 5 — ENTRYTRIG
# ─────────────────────────────────────────────────────────────────────────────



def calc_layer5(
    passes_df: pd.DataFrame,
    half_df: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    spy: pd.Series,
    perm: str,
    l0: dict,
    earnings_dates: dict,
    ticker_sector: dict,
    rec_flags: int,
) -> tuple:
    """
    Evaluate Layer 5 entry trigger for each Layer 4 candidate.
    Returns (full_l5_df, half_l5_df).

    Per stock computes: vol ratio, MA distances, 10d EMA, 6-week base range,
    RS slope, earnings proximity, trigger type routing, entry/stop suggestion, verdict.

    Trigger type routing:
      Red state           → None (no entries)
      Sector accelerating → Accelerating Protocol (v4)
      rec_flags >= 4      → Pullback only
      Risk-on / Green     → Breakout preferred
      Mixed / Yellow      → Pullback preferred
    """
    today        = datetime.today().date()
    regime       = l0.get("regime", "Risk-on")
    accel_secs   = set(l0.get("accelerating", []))
    rec_override = rec_flags >= 4

    def _assess(df_in: pd.DataFrame, signal_tier: str) -> pd.DataFrame:
        rows = []
        for _, src in df_in.iterrows():
            t   = src["Ticker"]
            sec = src["Sector"]

            if t not in close.columns:
                continue

            px  = close[t].dropna()
            vol = volume[t].dropna() if t in volume.columns else pd.Series(dtype=float)

            if len(px) < 60:
                continue

            price  = float(px.iloc[-1])
            ma20   = float(px.rolling(20).mean().iloc[-1])
            ma50   = float(px.rolling(50).mean().iloc[-1])
            ema10  = float(px.ewm(span=10, adjust=False).mean().iloc[-1])

            avg_vol   = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0.0
            last_vol  = float(vol.iloc[-1]) if len(vol) > 0 else 0.0
            vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 0.0

            pct_vs_20 = (price / ma20 - 1) if ma20 > 0 else 0.0
            pct_vs_50 = (price / ma50 - 1) if ma50 > 0 else 0.0

            # 6-week base range for breakout stop / pivot reference
            base_px   = px.iloc[-30:]
            base_high = float(base_px.max())
            base_low  = float(base_px.min())

            # RS slope: compare today's RS vs 10 trading days ago
            spy_a = spy.reindex(px.index).ffill()
            rs    = (px / spy_a).dropna()
            rs_up = bool(len(rs) >= 10 and rs.iloc[-1] > rs.iloc[-10])

            # Earnings proximity (14 calendar days ≈ 10 business days)
            next_ed = earnings_dates.get(t)
            days_to_ed = None
            if next_ed is not None:
                try:
                    days_to_ed = (next_ed - today).days
                except Exception:
                    pass
            earnings_flag = (days_to_ed is not None and days_to_ed <= 14)

            # ── Trigger type routing ──────────────────────────────────────────
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

            # ── Verdict + suggested entry/stop ────────────────────────────────
            entry_px = None
            stop_px  = None
            verdict  = "⬜ NOT READY"
            notes    = []

            if earnings_flag:
                verdict = "❌ SKIP"
                notes.append(f"Earnings in {days_to_ed}d")

            elif trigger_type == "None":
                verdict = "❌ RED STATE"
                notes.append("No new entries")

            elif trigger_type == "Breakout":
                near_top = pct_vs_20 <= 0.08 and price <= base_high * 1.05
                vol_ok   = vol_ratio >= 1.4
                if near_top and vol_ok and rs_up:
                    verdict  = "🟢 ENTRY READY"
                    entry_px = round(price, 2)
                    stop_px  = round(base_low * 0.99, 2)
                elif near_top and vol_ratio >= 1.0:
                    verdict  = "🟡 WATCH"
                    entry_px = round(base_high * 1.001, 2)
                    stop_px  = round(base_low * 0.99, 2)
                    notes.append(f"Vol {vol_ratio:.1f}x — needs ≥1.4x on breakout day")
                else:
                    verdict  = "⬜ NOT READY"
                    entry_px = round(base_high * 1.001, 2)
                    stop_px  = round(base_low * 0.99, 2)
                    if not near_top: notes.append(f"{pct(pct_vs_20)} vs 20MA — not at pivot")
                    if not vol_ok:   notes.append(f"Vol {vol_ratio:.1f}x — needs ≥1.4x")
                    if not rs_up:    notes.append("RS not rising")

            elif trigger_type == "Pullback":
                near_20  = abs(pct_vs_20) <= 0.03
                near_50  = abs(pct_vs_50) <= 0.03
                vol_decl = vol_ratio <= 1.0
                if (near_20 or near_50) and vol_decl:
                    verdict = "🟢 ENTRY READY"
                    if near_20:
                        entry_px = round(price, 2)
                        stop_px  = round(ma20 * 0.98, 2)
                        notes.append("At 20d MA")
                    else:
                        entry_px = round(price, 2)
                        stop_px  = round(ma50 * 0.98, 2)
                        notes.append("At 50d MA")
                elif near_20 or near_50:
                    verdict = "🟡 WATCH"
                    entry_px = round(ma20 if near_20 else ma50, 2)
                    stop_px  = round((ma20 if near_20 else ma50) * 0.98, 2)
                    notes.append(f"Vol {vol_ratio:.1f}x — look for declining volume")
                else:
                    verdict  = "⬜ NOT READY"
                    entry_px = round(ma20, 2)
                    stop_px  = round(ma20 * 0.98, 2)
                    notes.append(f"{pct(pct_vs_20)} vs 20MA  |  {pct(pct_vs_50)} vs 50MA")

            elif trigger_type == "Accelerating":
                in_range = 0 <= pct_vs_20 <= 0.15
                vol_ok   = vol_ratio >= 1.0
                if in_range and vol_ok:
                    verdict  = "🟢 ENTRY READY"
                    entry_px = round(price, 2)
                    stop_px  = round(ema10 * 0.99, 2)
                    notes.append("Half size · 10d EMA stop · 6-week hold")
                elif in_range:
                    verdict  = "🟡 WATCH"
                    entry_px = round(price, 2)
                    stop_px  = round(ema10 * 0.99, 2)
                    notes.append(f"Vol {vol_ratio:.1f}x — needs ≥1.0x · Half size · 10d EMA stop")
                else:
                    verdict  = "⬜ NOT READY"
                    entry_px = round(price, 2)
                    stop_px  = round(ema10 * 0.99, 2)
                    notes.append(f"{pct(pct_vs_20)} above 20MA — outside 0–15% Accel window")

            # Append upcoming earnings note (if not already flagged)
            if next_ed is not None and not earnings_flag:
                try:
                    notes.append(f"Earnings: {next_ed.strftime('%b %-d')}")
                except Exception:
                    notes.append(f"Earnings: {next_ed}")

            rows.append({
                "Ticker":    t,
                "Sector":    sec,
                "Price":     round(price, 2),
                "Trigger":   trigger_type,
                "Vol Ratio": vol_ratio,
                "vs 20MA":   round(pct_vs_20, 4),
                "vs 50MA":   round(pct_vs_50, 4),
                "10d EMA":   round(ema10, 2),
                "Entry":     entry_px,
                "Stop":      stop_px,
                "RS ↑":      rs_up,
                "Verdict":   verdict,
                "Notes":     " · ".join(notes) if notes else "Verify in TradingView",
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    full_l5 = _assess(passes_df, "Full")
    half_l5 = _assess(half_df.head(15), "Half")
    return full_l5, half_l5

