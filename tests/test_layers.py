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
from layers import calc_layer2, score_recession_composite


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
