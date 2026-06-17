"""
Pure trading logic — no Streamlit, no I/O. Unit-testable.

Holds the financial rules that were previously embedded inside render callbacks:
  - entry/stop computation per L5 trigger type
  - performance-period → date-range mapping
  - open/close position mutations on the portfolio dict
"""
from datetime import date, timedelta


# ── Entry / stop per Layer 5 trigger ─────────────────────────────────────────
def compute_entry_stop(trigger: str, levels: dict) -> tuple[float, float]:
    """Return (entry, stop) for a trigger type given reference price levels.

    levels expects keys: price, ma20, ema10, base_high, base_low.
    Rules (framework L5):
      Breakout    — entry just above base high, stop just below base low
      Pullback    — entry at 20d MA, stop ~2% below it
      Accelerating— entry at current price, stop just below 10d EMA
    """
    if trigger == "Breakout":
        entry = round(levels["base_high"] * 1.001, 2)
        stop  = round(levels["base_low"] * 0.99, 2)
    elif trigger == "Pullback":
        entry = round(levels["ma20"], 2)
        stop  = round(levels["ma20"] * 0.98, 2)
    else:  # Accelerating
        entry = round(levels["price"], 2)
        stop  = round(levels["ema10"] * 0.99, 2)
    return entry, stop


# ── Performance period → (start, end, label) ─────────────────────────────────
def period_to_range(period: str, today: date,
                    custom: tuple[date, date] | None = None) -> tuple[date, date, str]:
    """Map a performance-period selection to a concrete date range and label."""
    if period == "YTD":
        return date(today.year, 1, 1), today, f"YTD {today.year}"
    if period == "1M":
        return today - timedelta(days=30), today, "Last 30 Days"
    if period == "3M":
        return today - timedelta(days=91), today, "Last 3 Months"
    if period == "6M":
        return today - timedelta(days=182), today, "Last 6 Months"
    if period == "1Y":
        return today - timedelta(days=365), today, "Last 12 Months"
    # Custom
    start = custom[0] if custom and len(custom) > 0 else date(today.year, 1, 1)
    end   = custom[1] if custom and len(custom) > 1 else today
    return start, end, f"{start} → {end}"


# ── Portfolio mutations ──────────────────────────────────────────────────────
def add_open_position(data: dict, *, ticker: str, layer: str, entry_date: str,
                      entry_price: float, shares: int, stop_price: float,
                      notes: str) -> dict:
    """Append an open position and debit cash by cost. Mutates and returns data."""
    cost = entry_price * int(shares)
    data.setdefault("open_positions", []).append({
        "ticker": ticker, "layer": layer,
        "entry_date": entry_date, "entry_price": entry_price,
        "shares": int(shares), "stop_price": stop_price,
        "current_price": None, "notes": notes,
    })
    data["cash_balance"] = data.get("cash_balance", 0) - cost
    return data


def close_position(data: dict, *, ticker: str, exit_date: str, exit_price: float,
                   exit_reason: str, notes: str) -> dict:
    """Move an open position to closed, credit cash by proceeds. Mutates/returns data.

    No-op if the ticker isn't open.
    """
    pos = next((p for p in data.get("open_positions", []) if p["ticker"] == ticker), None)
    if pos is None:
        return data
    proceeds = exit_price * pos["shares"]
    data.setdefault("closed_positions", []).insert(0, {
        "ticker": pos["ticker"], "layer": pos.get("layer", "Tactical"),
        "entry_date": pos["entry_date"], "exit_date": exit_date,
        "entry_price": pos["entry_price"], "exit_price": exit_price,
        "shares": pos["shares"], "exit_reason": exit_reason,
        "notes": notes,
    })
    data["open_positions"] = [p for p in data["open_positions"] if p["ticker"] != ticker]
    data["cash_balance"] = data.get("cash_balance", 0) + proceeds
    return data
