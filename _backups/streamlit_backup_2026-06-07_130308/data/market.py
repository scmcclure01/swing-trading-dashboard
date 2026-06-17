"""
Market data — price/volume downloads via yfinance.

Covers macro ETFs, the screener universe, live portfolio prices, and earnings
dates. All cached; cache TTLs match each series' update cadence.
"""
from datetime import datetime

import pandas as pd
import yfinance as yf
import streamlit as st

from config import SECTOR_ETFS, SECTOR_TICKERS


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data() -> pd.DataFrame:
    """Fetch 1-year daily close for SPY, sector ETFs, TLT, HYG, IEF, and VIX."""
    tickers = ["SPY", "TLT", "HYG", "IEF"] + list(SECTOR_ETFS.values())
    raw   = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
    close = raw["Close"]
    try:
        vix = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
        close["^VIX"] = vix["Close"].squeeze()
    except Exception:
        pass
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(sectors_key: str) -> tuple:
    """
    Fetch 1-year OHLCV for all tickers in the given sectors (comma-separated key).
    Returns (close_df, volume_df, tickers_list) or raises on network failure.
    """
    try:
        sectors = sectors_key.split(",")
        all_t   = [t for s in sectors for t in SECTOR_TICKERS.get(s, [])]
        raw     = yf.download(all_t + ["SPY"], period="1y", auto_adjust=True, progress=False)
        return raw["Close"], raw["Volume"], all_t
    except Exception as e:
        raise RuntimeError(f"Screener data fetch failed: {e}") from e


@st.cache_data(ttl=300, show_spinner=False)
def fetch_portfolio_prices(tickers_key: str) -> dict:
    """Fetch last close for open positions. Cached 5 min."""
    tickers = [t.strip() for t in tickers_key.split(",") if t.strip()]
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        closes = raw["Close"] if len(tickers) > 1 else raw["Close"].rename(tickers[0])
        if isinstance(closes, pd.Series):
            closes = closes.to_frame()
        return {t: float(closes[t].dropna().iloc[-1]) for t in tickers if t in closes.columns}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_earnings_dates(tickers_key: str) -> dict:
    """Fetch next earnings dates for a batch of tickers (cached 24h)."""
    tickers = tickers_key.split(",")
    today   = datetime.today().date()
    results = {}
    for t in tickers:
        try:
            cal     = yf.Ticker(t).calendar
            next_ed = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed is not None:
                    dates = ed if isinstance(ed, list) else [ed]
                    for d in dates:
                        try:
                            dt = pd.Timestamp(d).date()
                            if dt >= today:
                                next_ed = dt
                                break
                        except Exception:
                            pass
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for col in cal.columns:
                    try:
                        val = cal.at["Earnings Date", col]
                        dt  = pd.Timestamp(val).date()
                        if dt >= today:
                            next_ed = dt
                            break
                    except Exception:
                        pass
            results[t] = next_ed
        except Exception:
            results[t] = None
    return results
