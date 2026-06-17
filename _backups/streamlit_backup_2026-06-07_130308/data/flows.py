"""
ETF fund flows — scraped from etfdb.com.

NOTE: this depends on etfdb.com page markup (a `data-series` attribute embedding
the daily flow time series). It will break if that markup changes; failures are
returned per-ETF as {'flow_strength': 'N/A', 'error': ...} rather than raised.
"""
import json
import re
import urllib.request
from datetime import datetime

import streamlit as st

from config import SECTOR_ETFS


def _scrape_flows(ticker: str):
    """Return list of (timestamp_ms, flow_billions) for one ETF, or None."""
    url = f"https://etfdb.com/etf/{ticker}/"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    resp = urllib.request.urlopen(req, timeout=15)
    html = resp.read().decode()
    match = re.search(r"data-series='(\[\[[\d\.,\-\s\[\]e]+\]\])'", html)
    if not match:
        return None
    raw = json.loads(match.group(1))
    return [(ts, flow_b) for ts, flow_b in raw]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_fund_flows() -> dict:
    """
    Fetch actual daily fund flow data from etfdb.com for each sector ETF.

    Flow classification per ETF (framework Layer 3):
      Strong   : 1w net > +$150M  AND  4w net > +$300M
      Moderate : 1w net > +$50M   OR   4w net > +$100M (and not Strong)
      Outflows : 1w net < -$50M   OR   4w net < -$100M
      Weak     : everything else (small positive or flat)

    Returns dict keyed by ETF ticker → {
        'flow_strength', 'aum_1w_delta', 'aum_4w_delta',
        'direction_5d', 'direction_10d', 'as_of', 'error' (only on failure)
    }
    """
    results = {}
    for etf in SECTOR_ETFS.values():
        try:
            data = _scrape_flows(etf)
            if not data or len(data) < 20:
                results[etf] = {"flow_strength": "N/A", "error": "Insufficient flow data"}
                continue

            as_of = datetime.utcfromtimestamp(data[-1][0] / 1000).strftime('%Y-%m-%d')

            # 5-day (1 week) and 20-day (4 week) net flows — data is in $B
            flows_5d  = [v for _, v in data[-5:]]
            flows_10d = [v for _, v in data[-10:]]
            flows_20d = [v for _, v in data[-20:]]

            net_1w = sum(flows_5d) * 1000   # convert $B → $M
            net_4w = sum(flows_20d) * 1000

            # Directional consistency
            pos_5d  = sum(1 for v in flows_5d if v > 0)
            pos_10d = sum(1 for v in flows_10d if v > 0)

            # Classify per framework thresholds
            if net_1w > 150 and net_4w > 300:
                flow_strength = "Strong"
            elif net_1w > 50 or net_4w > 100:
                flow_strength = "Moderate"
            elif net_1w < -50 or net_4w < -100:
                flow_strength = "Outflows"
            else:
                flow_strength = "Weak"

            results[etf] = {
                "flow_strength":  flow_strength,
                "aum_1w_delta":   round(net_1w, 1),
                "aum_4w_delta":   round(net_4w, 1),
                "direction_5d":   f"{pos_5d}/5 positive",
                "direction_10d":  f"{pos_10d}/10 positive",
                "as_of":          as_of,
            }

        except Exception as e:
            results[etf] = {"flow_strength": "N/A", "error": str(e)}

    return results
