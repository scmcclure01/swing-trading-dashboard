"""
Tests for the pure layer logic in layers.py.

Covers the rule-based decisions that don't need live data:
  - calc_layer2 permission-state transitions
  - score_recession_composite flag counting

calc_layer0/3/4/5 operate on price DataFrames and are exercised live in the app;
they're not unit-tested here (would require fixture price data).

Run:  python3 tests/test_layers.py   (or  python3 -m pytest tests/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit  # noqa: F401  (import guard — see note below)
from layers import (
    calc_layer2, score_recession_composite, score_layer1_mispricing,
    drawdown_tier, DRAWDOWN_STATES,
    is_etf_position, etf_exit_signals,
)


# ── calc_layer2 — permission state ───────────────────────────────────────────
_L0_GREEN = {"spy_ret_1m": 0.02, "spy_ret_6m": 0.05,
             "liquidity_tighten": False, "fnl_signal": "RISING"}


def test_green_when_all_clear():
    perm, _ = calc_layer2(_L0_GREEN, rec_flags=0, eps_signal="Flat")
    assert perm == "Green"


def test_liquidity_override_forces_red():
    l0 = dict(_L0_GREEN, fnl_signal="OVERRIDE ACTIVE")
    perm, _ = calc_layer2(l0, rec_flags=0, eps_signal="Flat")
    assert perm == "Red"


def test_spy_both_negative_is_red():
    l0 = dict(_L0_GREEN, spy_ret_1m=-0.02, spy_ret_6m=-0.05)
    perm, _ = calc_layer2(l0, rec_flags=0, eps_signal="Flat")
    assert perm == "Red"


def test_manual_override_passes_through():
    perm, _ = calc_layer2(_L0_GREEN, rec_flags=0, eps_signal="Flat", override="Yellow")
    assert perm == "Yellow"


def test_high_recession_flags_force_red():
    perm, _ = calc_layer2(_L0_GREEN, rec_flags=4, eps_signal="Flat")
    assert perm == "Red"


# ── score_recession_composite ────────────────────────────────────────────────
_FRED_CLEAN = {
    "recprob": 10.0, "recprob_date": "x",
    "sahm": 0.1, "sahm_date": "x",
    "cfnai": 0.2, "cfnai_date": "x",
    "t10y3m": 0.5, "t10y3m_date": "x",
}


def test_clean_composite_has_no_flags():
    inds = score_recession_composite(_FRED_CLEAN, "Rising ✅")
    assert len(inds) == 5
    assert sum(1 for i in inds if not i["ok"]) == 0


def test_stressed_composite_flags_curve_sahm_lei():
    bad = dict(_FRED_CLEAN, t10y3m=-0.3, sahm=0.8)
    inds = score_recession_composite(bad, "6mo declining ⚠️")
    assert sum(1 for i in inds if not i["ok"]) == 3


# ── score_layer1_mispricing ──────────────────────────────────────────────────
def test_l1_cheap_market_full_risk():
    # High earnings yield vs both rates → both checks bullish, GDP bullish → +3
    r = score_layer1_mispricing(fwd_earnings_yield=8.0, tips_10y=2.0,
                                nominal_10y=2.5, gdpnow_signal="Bullish")
    assert r["composite"] == 3
    assert r["verdict"] == "Full risk"
    assert r["computable"] is True


def test_l1_expensive_market_reduces_sizing():
    # Low earnings yield: EY-TIPS = 1.0 (<2 → -1), ERP = 0.5 (<3 → -1), GDP bearish (-1) → -3
    r = score_layer1_mispricing(fwd_earnings_yield=4.0, tips_10y=3.0,
                                nominal_10y=3.5, gdpnow_signal="Bearish")
    assert r["composite"] == -3
    assert r["verdict"] == "Reduce sizing"


def test_l1_normal_range():
    # EY-TIPS = 3.0 (neutral 0), ERP = 2.5 -> wait: 6-3.5=2.5 (<3 -> -1); use values for normal
    # fwd 6.5, tips 3.0 -> EY-TIPS=3.5 (neutral); nominal 3.0 -> ERP=3.5 (normal 0); GDP not set 0
    r = score_layer1_mispricing(fwd_earnings_yield=6.5, tips_10y=3.0,
                                nominal_10y=3.0, gdpnow_signal="Not set")
    assert r["composite"] == 0
    assert r["verdict"] == "Normal operations"


def test_l1_missing_rates_not_computable():
    r = score_layer1_mispricing(fwd_earnings_yield=None, tips_10y=2.0,
                                nominal_10y=2.5, gdpnow_signal="Not set")
    assert r["computable"] is False
    assert len(r["checks"]) == 3   # still returns structure


# ── drawdown_tier ────────────────────────────────────────────────────────────
def test_dd_at_peak():
    r = drawdown_tier(100000, 100000)
    assert r["pct"] == 0.0
    assert r["state"] == DRAWDOWN_STATES["peak"]


def test_dd_tier1_normal():
    r = drawdown_tier(95000, 100000)   # -5%
    assert r["pct"] == -5.0
    assert "Tier 1" in r["state"]


def test_dd_tier2_cut_risk():
    r = drawdown_tier(91000, 100000)   # -9% → Tier 2
    assert "Tier 2" in r["state"]
    assert "7–10%" in r["state"]       # token the sizing logic matches on


def test_dd_tier3_defensive():
    r = drawdown_tier(88000, 100000)   # -12% → Tier 3
    assert "Tier 3" in r["state"]


def test_dd_emergency():
    r = drawdown_tier(80000, 100000)   # -20% → emergency
    assert ">15%" in r["state"]


def test_dd_boundary_minus7_is_tier2():
    # exactly -7% is NOT > -7, so it falls to Tier 2 (matches app's original logic)
    r = drawdown_tier(93000, 100000)
    assert "Tier 2" in r["state"]


def test_dd_new_high_above_peak():
    r = drawdown_tier(110000, 100000)  # above peak → treated as at-peak
    assert r["state"] == DRAWDOWN_STATES["peak"]


# ── ETF exit monitoring ──────────────────────────────────────────────────────
def test_is_etf_position_by_layer():
    assert is_etf_position({"ticker": "AAPL", "layer": "Core"}) is True


def test_is_etf_position_by_ticker():
    assert is_etf_position({"ticker": "XLK", "layer": "Tactical"}) is True


def test_is_not_etf_position():
    assert is_etf_position({"ticker": "AAPL", "layer": "Tactical"}) is False


def test_exit_below_ma_triggers():
    sigs = etf_exit_signals("XLK", price=95.0, ma20=100.0,
                            flow_strength="Strong", rs_trend="Leading")
    assert any(s["trigger"] == "Below 20d MA" for s in sigs)
    assert sigs[0]["severity"] == "high"


def test_exit_flow_reversal_triggers():
    sigs = etf_exit_signals("XLE", price=105.0, ma20=100.0,
                            flow_strength="Outflows", rs_trend="Leading")
    assert any(s["trigger"] == "Flow reversal" for s in sigs)


def test_exit_rs_lagging_triggers_medium():
    sigs = etf_exit_signals("XLU", price=105.0, ma20=100.0,
                            flow_strength="Strong", rs_trend="Lagging")
    assert any(s["trigger"] == "RS declining" and s["severity"] == "medium" for s in sigs)


def test_exit_healthy_position_no_signals():
    sigs = etf_exit_signals("XLK", price=110.0, ma20=100.0,
                            flow_strength="Strong", rs_trend="Leading")
    assert sigs == []


def test_exit_handles_missing_data():
    # No price/MA, no flow, no RS → nothing fires, no crash
    sigs = etf_exit_signals("XLK", price=None, ma20=None,
                            flow_strength=None, rs_trend=None)
    assert sigs == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
