"""
Tests for trading_logic.py — the pure financial rules (entry/stop, period
mapping, position mutations).

Run from the framework root with:  python3 -m pytest tests/
(or just:  python3 tests/test_trading_logic.py)

These exist as a safety net: after any edit to trading_logic.py, run them to
confirm the math still behaves before the change reaches the live dashboard.
"""
import os
import sys
from datetime import date

# Make the framework root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_logic import (
    compute_entry_stop,
    period_to_range,
    add_open_position,
    close_position,
)

_LEVELS = {"price": 100, "ma20": 95, "ema10": 98, "base_high": 110, "base_low": 90}


# ── compute_entry_stop ───────────────────────────────────────────────────────
def test_breakout_entry_stop():
    entry, stop = compute_entry_stop("Breakout", _LEVELS)
    assert entry == round(110 * 1.001, 2)
    assert stop == round(90 * 0.99, 2)


def test_pullback_entry_stop():
    entry, stop = compute_entry_stop("Pullback", _LEVELS)
    assert entry == 95.0
    assert stop == round(95 * 0.98, 2)


def test_accelerating_entry_stop():
    entry, stop = compute_entry_stop("Accelerating", _LEVELS)
    assert entry == 100.0
    assert stop == round(98 * 0.99, 2)


# ── period_to_range ──────────────────────────────────────────────────────────
def test_ytd_range():
    today = date(2026, 6, 6)
    start, end, label = period_to_range("YTD", today)
    assert start == date(2026, 1, 1)
    assert end == today
    assert label == "YTD 2026"


def test_rolling_window_labels():
    today = date(2026, 6, 6)
    assert period_to_range("1M", today)[2] == "Last 30 Days"
    assert period_to_range("3M", today)[2] == "Last 3 Months"
    assert period_to_range("6M", today)[2] == "Last 6 Months"
    assert period_to_range("1Y", today)[2] == "Last 12 Months"


def test_custom_range():
    today = date(2026, 6, 6)
    start, end, _ = period_to_range("Custom", today, (date(2026, 3, 1), date(2026, 4, 1)))
    assert start == date(2026, 3, 1)
    assert end == date(2026, 4, 1)


# ── position mutations ───────────────────────────────────────────────────────
def _fresh_portfolio():
    return {"open_positions": [], "closed_positions": [], "cash_balance": 10000}


def test_add_open_position_debits_cash():
    d = _fresh_portfolio()
    add_open_position(d, ticker="AAPL", layer="Tactical", entry_date="2026-06-01",
                      entry_price=100, shares=10, stop_price=95, notes="")
    assert d["cash_balance"] == 9000          # 10000 - (100 * 10)
    assert len(d["open_positions"]) == 1


def test_close_position_credits_cash_and_moves_record():
    d = _fresh_portfolio()
    add_open_position(d, ticker="AAPL", layer="Tactical", entry_date="2026-06-01",
                      entry_price=100, shares=10, stop_price=95, notes="")
    close_position(d, ticker="AAPL", exit_date="2026-06-05", exit_price=110,
                   exit_reason="Target", notes="")
    assert d["cash_balance"] == 9000 + 1100   # proceeds = 110 * 10
    assert len(d["open_positions"]) == 0
    assert len(d["closed_positions"]) == 1


def test_close_unknown_ticker_is_noop():
    d = _fresh_portfolio()
    before = d["cash_balance"]
    close_position(d, ticker="ZZZZ", exit_date="x", exit_price=1,
                   exit_reason="Stop", notes="")
    assert d["cash_balance"] == before
    assert len(d["closed_positions"]) == 0


# Allow running the file directly without pytest installed.
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
