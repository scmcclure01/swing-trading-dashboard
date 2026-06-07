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
from layers import calc_layer2, score_recession_composite, score_layer1_mispricing


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
