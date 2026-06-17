"""
Fed Net Liquidity — 4-Week Signal
Layer 0, Signal 6 of the Swing Trading Framework

Formula: WALCL - WTREGEN - RRPONTSYD (all from FRED, no API key required)
Signal:  4-week change > -$200B = Liquidity Override Active
Update:  Thursdays at 4:30 PM ET (H.4.1 release)
"""

import urllib.request
import csv
from io import StringIO
from datetime import datetime


def fetch_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url) as r:
        data = r.read().decode()
    rows = list(csv.reader(StringIO(data)))[1:]
    return {row[0]: float(row[1]) for row in rows if len(row) > 1 and row[1] not in ('.', '', 'NA')}


def get_fed_net_liquidity_signal(weeks_back=4, lookback_display=6):
    walcl = fetch_fred("WALCL")
    tga   = fetch_fred("WTREGEN")
    rrp   = fetch_fred("RRPONTSYD")

    dates = sorted(set(walcl) & set(tga) & set(rrp))
    net_liq = {d: walcl[d] - tga[d] - rrp[d] for d in dates}

    recent = sorted(net_liq.keys())[-(lookback_display):]
    latest_date = recent[-1]
    prior_date  = recent[-(weeks_back + 1)]

    current  = net_liq[latest_date]
    prior    = net_liq[prior_date]
    change_b = (current - prior) / 1000

    # Signal
    if change_b <= -200:
        signal = "OVERRIDE ACTIVE"
        detail = f"Declined ${abs(change_b):.0f}B — exceeds $200B threshold. Reduce exposure immediately."
    elif change_b < 0:
        signal = "DECLINING"
        detail = f"Declining but below override threshold (need -$200B, current: ${change_b:.1f}B). Monitor closely."
    else:
        signal = "RISING"
        detail = f"Rising ${change_b:+.1f}B — liquidity tailwind. No override."

    return {
        "as_of": latest_date,
        "current_b": round(current / 1000, 1),
        "prior_b": round(prior / 1000, 1),
        "prior_date": prior_date,
        "change_4w_b": round(change_b, 1),
        "signal": signal,
        "detail": detail,
        "recent": {d: round(net_liq[d] / 1000, 1) for d in recent},
    }


def print_report(result):
    print("=" * 52)
    print("  FED NET LIQUIDITY — 4-WEEK SIGNAL")
    print(f"  As of: {result['as_of']}  |  Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 52)

    print("\nRecent Weekly Readings ($B):")
    for date, val in result["recent"].items():
        marker = " <-- latest" if date == result["as_of"] else ""
        print(f"  {date}:  ${val:,.1f}B{marker}")

    print(f"\n4-Week Change:  ${result['change_4w_b']:+.1f}B")
    print(f"  {result['prior_date']}:  ${result['prior_b']:,.1f}B")
    print(f"  {result['as_of']}:  ${result['current_b']:,.1f}B")

    print(f"\nSIGNAL: {result['signal']}")
    print(f"  {result['detail']}")
    print("=" * 52)


if __name__ == "__main__":
    result = get_fed_net_liquidity_signal()
    print_report(result)
