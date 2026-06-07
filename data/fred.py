"""
FRED data — recession composite, Fed Net Liquidity, T-bill rate.

Two access paths:
  - fredapi (needs FRED_API_KEY in Streamlit secrets) for the recession composite
  - public FRED CSV (no key) for Net Liquidity and as a T-bill fallback
"""
import csv
import urllib.request
from io import StringIO

import streamlit as st

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred_data() -> dict:
    """Fetch recession composite indicators from FRED (cached 24h — updates daily/monthly)."""
    if not FRED_AVAILABLE:
        return {"error": "fredapi package not installed"}
    try:
        api_key = st.secrets.get("FRED_API_KEY", "")
        if not api_key:
            return {"error": "Add FRED_API_KEY to Streamlit Secrets to enable recession composite"}
        fred = Fred(api_key=api_key)

        def latest(series_id, start="2023-01-01"):
            s = fred.get_series(series_id, observation_start=start).dropna()
            return float(s.iloc[-1]), str(s.index[-1].date())

        data = {}
        data["sahm"],    data["sahm_date"]    = latest("SAHMREALTIME")
        data["cfnai"],   data["cfnai_date"]   = latest("CFNAIMA3")
        data["t10y3m"],  data["t10y3m_date"]  = latest("T10Y3M")
        data["recprob"], data["recprob_date"] = latest("RECPROUSM156N")
        return data
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fed_net_liquidity() -> dict:
    """
    Fetch Fed Net Liquidity from FRED public CSV (no API key required).
    Formula: WALCL - WTREGEN - RRPONTSYD
    Signal: 4-week change <= -$200B → Override Active
    Updates Thursdays at 4:30 PM ET with the H.4.1 release.
    """
    def fetch_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read().decode()
        rows = list(csv.reader(StringIO(data)))[1:]
        return {row[0]: float(row[1]) for row in rows if len(row) > 1 and row[1] not in ('.', '', 'NA')}

    try:
        walcl = fetch_series("WALCL")
        tga   = fetch_series("WTREGEN")
        rrp   = fetch_series("RRPONTSYD")

        dates = sorted(set(walcl) & set(tga) & set(rrp))
        if len(dates) < 5:
            return {"error": "Insufficient FRED data"}

        net_liq = {d: walcl[d] - tga[d] - rrp[d] for d in dates}
        recent      = sorted(net_liq.keys())
        latest_date = recent[-1]
        prior_date  = recent[-5]  # 4 weeks back

        current  = net_liq[latest_date]
        prior    = net_liq[prior_date]
        change_b = (current - prior) / 1000  # convert $M → $B

        if change_b <= -200:
            signal = "OVERRIDE ACTIVE"
        elif change_b < 0:
            signal = "DECLINING"
        else:
            signal = "RISING"

        return {
            "as_of":        latest_date,
            "current_b":    round(current / 1000, 1),
            "prior_b":      round(prior / 1000, 1),
            "prior_date":   prior_date,
            "change_4w_b":  round(change_b, 1),
            "signal":       signal,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tbill_rate() -> float:
    """
    Fetch 3-Month T-Bill rate (DTB3) for earnings carry calculation.
    Tries: 1) fredapi (API key), 2) FRED public CSV, 3) yfinance ^IRX.
    Returns annualized rate as a percentage (e.g. 4.25 means 4.25%).
    Falls back to 4.25% if all sources fail.
    """
    import yfinance as yf

    # Method 1: fredapi (same as recession composite)
    if FRED_AVAILABLE:
        try:
            api_key = st.secrets.get("FRED_API_KEY", "")
            if api_key:
                fred = Fred(api_key=api_key)
                s = fred.get_series("DTB3", observation_start="2024-01-01").dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
        except Exception:
            pass

    # Method 2: FRED public CSV
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read().decode()
        rows = list(csv.reader(StringIO(data)))[1:]
        for row in reversed(rows):
            if len(row) > 1 and row[1] not in (".", "", "NA"):
                return float(row[1])
    except Exception:
        pass

    # Method 3: yfinance ^IRX (13-week T-bill index, quoted in basis points / 10)
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty:
            return float(irx["Close"].iloc[-1])
    except Exception:
        pass

    # Fallback
    return 4.25
