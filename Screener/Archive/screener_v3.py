#!/usr/bin/env python3
"""
Autonomous Swing Trading Screener v3
=====================================
Regime-first approach: only scans stocks in regime-aligned sectors.
Dramatically reduces universe size → stays within yfinance rate limits.

Usage:
    python screener_v3.py --regime risk-on --permission green

How it works:
    1. Regime → aligned sector ETFs → pull ETF holdings (top constituents)
    2. Batch download 6-month OHLCV for those ~500-1200 names
    3. Apply L4 filters: price > 20d/50d MA, two-speed signal, RS vs SPY, volume
    4. Rank by RS strength, output CSV + summary
"""

import argparse
import datetime
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Regime → Sector ETFs ───
REGIME_SECTORS = {
    "risk-on": ["XLK", "XLY", "XLF", "XLI", "SMH"],
    "reflation": ["XLE", "XLB", "XLI"],
    "deflation": ["XLV", "XLP", "XLU"],
    "stagflation": ["XLE"],
}

# ─── Sector ETF → GICS sector name mapping ───
ETF_TO_SECTOR = {
    "XLK": "Technology",
    "XLY": "Consumer Cyclical",
    "XLF": "Financial Services",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Basic Materials",
    "XLV": "Healthcare",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "SMH": "Semiconductors",
}

# ─── Hardcoded sector universes (top liquid names per sector) ───
# These are the ~100-200 most liquid names per sector ETF.
# Much more reliable than trying to scrape ETF holdings.
# Update quarterly if needed.
SECTOR_UNIVERSES = {
    "XLK": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN", "ADBE",
        "IBM", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW", "ADI", "LRCX", "KLAC",
        "SNPS", "CDNS", "CRWD", "MSI", "APH", "MCHP", "FTNT", "ROP", "TEL", "NXPI",
        "ADSK", "MPWR", "ON", "ANSS", "KEYS", "CDW", "FSLR", "IT", "HPE", "HPQ",
        "ZBRA", "TYL", "EPAM", "AKAM", "SWKS", "TER", "NTAP", "JNPR", "PTC", "TRMB",
        "GEN", "WDC", "STX", "SMCI", "DELL", "PLTR", "NET", "DDOG", "ZS", "MDB",
        "SNOW", "TEAM", "HUBS", "WDAY", "VEEV", "SPLK", "OKTA", "ZM", "DOCN", "PATH",
        "S", "CFLT", "ESTC", "MNDY", "BILL", "GTLB", "IOT", "AI", "SAMSARA", "APP",
        "FICO", "ANET", "MRVL", "ARM", "UBER", "DASH", "SHOP", "TTD", "COIN",
    ],
    "SMH": [
        "NVDA", "AMD", "AVGO", "QCOM", "TXN", "AMAT", "ADI", "LRCX", "KLAC", "MCHP",
        "NXPI", "ON", "MPWR", "MRVL", "SWKS", "TER", "MKSI", "ENTG", "QRVO", "AMKR",
        "CRUS", "ONTO", "WOLF", "RMBS", "SMTC", "ACLS", "AOSL", "ALGM", "SITM", "POWI",
        "DIOD", "AMBA", "LSCC", "MTSI", "PI", "SLAB", "COHR", "IPGP", "IRTC", "MU",
        "INTC", "TSM", "ASML", "ARM", "SMCI", "GFS", "UMC", "ASX",
    ],
    "XLY": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ABNB",
        "CMG", "ORLY", "AZO", "ROST", "DHI", "LEN", "PHM", "NVR", "GPC", "GRMN",
        "POOL", "BBY", "DRI", "YUM", "DARDEN", "ULTA", "LULU", "DECK", "TPR", "RL",
        "HAS", "MAT", "WYNN", "LVS", "MGM", "CZR", "RCL", "CCL", "NCLH", "HLT",
        "MAR", "H", "EXPE", "LKQ", "KMX", "AN", "LAD", "CVNA", "CRVL", "DPZ",
        "WING", "TXRH", "EAT", "CAKE", "DINE", "JACK", "SHAK", "BROS", "CAVA",
        "BIRK", "ONON", "CROX", "SKX", "COLM", "VFC", "PVH", "CPRI", "ETSY",
        "W", "RH", "WSM", "FIVE", "OLLI", "DLTR", "DG", "COST", "WMT", "TGT",
    ],
    "XLF": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
        "SCHW", "C", "AXP", "CB", "MMC", "AON", "PGR", "ICE", "CME", "MCO",
        "MSCI", "AJG", "AFL", "TRV", "MET", "AIG", "PRU", "ALL", "FI", "COF",
        "DFS", "SYF", "USB", "PNC", "TFC", "FITB", "MTB", "HBAN", "CFG", "KEY",
        "RF", "ZION", "NDAQ", "CBOE", "FDS", "MKTX", "VIRT", "HOOD", "IBKR",
        "RJF", "LPLA", "EVR", "HLI", "PJT", "SF", "PIPR", "SEIC", "TROW",
        "BEN", "IVZ", "WBS", "FNB", "EWBC", "WAL", "PACW", "CMA", "SNV",
        "FHN", "ALLY", "OZK", "SSB", "BOKF", "GBCI", "PNFP", "WTFC",
    ],
    "XLI": [
        "GE", "CAT", "RTX", "HON", "UNP", "UPS", "DE", "LMT", "BA", "ADP",
        "ETN", "ITW", "GD", "NOC", "WM", "RSG", "CSX", "NSC", "PCAR", "EMR",
        "JCI", "TT", "CARR", "OTIS", "ROK", "FAST", "SWK", "MMM", "GWW", "IR",
        "PH", "DOV", "FTV", "AME", "XYL", "IEX", "RBC", "GNRC", "PWR", "HUBB",
        "BLDR", "TTC", "WAB", "AGCO", "AXON", "TDG", "HWM", "HEI", "TRMB",
        "VRSK", "CPRT", "PAYX", "CTAS", "ODFL", "SAIA", "XPO", "JBHT", "CHRW",
        "LSTR", "EXPD", "KEX", "WERN", "SNDR", "MATX", "GXO", "ALLE", "AOS",
        "WSO", "RRX", "MIDD", "MAS", "AAON", "SPXC", "RHI",
    ],
    "XLE": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY",
        "HES", "DVN", "FANG", "HAL", "BKR", "CTRA", "MRO", "APA", "CHRD", "OVV",
        "EQT", "RRC", "AR", "PR", "MTDR", "CRGY", "SM", "MGY", "NOG", "VTLE",
        "CRC", "CIVI", "TRGP", "WMB", "KMI", "OKE", "ET", "EPD", "MPLX", "PAA",
        "AM", "DTM", "DEN", "DINO", "HF", "PBF", "PARR", "CVI", "CLR",
    ],
    "XLB": [
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM", "DOW",
        "DD", "PPG", "CE", "EMN", "ALB", "IFF", "FMC", "CF", "MOS", "RPM",
        "BALL", "PKG", "IP", "WRK", "SEE", "SON", "AVY", "ATR", "AXTA",
        "HUN", "OLN", "TROX", "CC", "IOSP", "KWR", "CBT", "GEF", "SLVM",
    ],
    "XLV": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
        "BMY", "MDT", "ISRG", "SYK", "BSX", "GILD", "VRTX", "REGN", "ZTS", "BDX",
        "CI", "ELV", "HCA", "MCK", "COR", "HUM", "CNC", "MOH", "GEHC", "EW",
        "DXCM", "IDXX", "RMD", "HOLX", "BAX", "A", "IQV", "MTD", "WAT", "PKI",
        "TECH", "BIO", "ILMN", "ALGN", "PODD", "INSP", "NVCR", "HZNP", "VTRS",
    ],
    "XLP": [
        "PG", "PEP", "KO", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
        "KMB", "GIS", "SJM", "K", "HSY", "MKC", "HRL", "CAG", "CPB", "TSN",
        "BG", "ADM", "MNST", "KDP", "STZ", "TAP", "SAM", "BF-B", "DEO",
        "WBA", "KR", "SYY", "USFD", "PFGC", "CHD", "CLX", "SPB", "COTY",
    ],
    "XLU": [
        "NEE", "SO", "DUK", "D", "SRE", "AEP", "EXC", "XEL", "ED", "WEC",
        "ES", "EIX", "AWK", "ATO", "CMS", "DTE", "ETR", "FE", "PPL", "CEG",
        "AES", "LNT", "EVRG", "NI", "PNW", "OGE", "NRG", "VST", "CWEN",
    ],
}


def get_universe_for_regime(regime):
    """Get combined ticker list for regime-aligned sectors."""
    etfs = REGIME_SECTORS.get(regime, [])
    if not etfs:
        print(f"ERROR: Unknown regime '{regime}'")
        sys.exit(1)

    tickers = set()
    sector_labels = {}
    for etf in etfs:
        names = SECTOR_UNIVERSES.get(etf, [])
        for n in names:
            tickers.add(n)
            sector_labels[n] = ETF_TO_SECTOR.get(etf, etf)

    print(f"Regime '{regime}' → sectors: {', '.join(etfs)}")
    print(f"Universe: {len(tickers)} unique tickers across {len(etfs)} sectors")
    return sorted(tickers), sector_labels


def download_data(tickers, period="6mo"):
    """Batch download OHLCV data."""
    print(f"\nDownloading {len(tickers)} tickers + SPY...")
    all_tickers = tickers + ["SPY"]

    batch_size = 200
    frames = {}

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total = (len(all_tickers) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total} ({len(batch)} tickers)...", end=" ")
        t0 = time.time()

        try:
            data = yf.download(
                batch, period=period, group_by="ticker",
                threads=True, progress=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        df = data[t][["Close", "Volume"]].dropna()
                        if len(df) >= 50:
                            frames[t] = df
                    except (KeyError, TypeError):
                        pass
            elif len(batch) == 1 and len(data) >= 50:
                frames[batch[0]] = data[["Close", "Volume"]].dropna()

            print(f"{time.time()-t0:.1f}s")
        except Exception as e:
            print(f"error: {e}")

        if i + batch_size < len(all_tickers):
            time.sleep(2)

    print(f"  Got data for {len(frames)} tickers")
    return frames


def compute_and_filter(frames, sector_labels, min_price=5.0, min_advol=50000):
    """Compute L4 signals and filter."""
    print(f"\nComputing L4 signals...")

    spy = frames.get("SPY")
    if spy is None:
        print("ERROR: No SPY data")
        return pd.DataFrame()

    spy_close = spy["Close"].squeeze()
    results = []

    for ticker, df in frames.items():
        if ticker == "SPY":
            continue
        try:
            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            if len(close) < 50:
                continue

            price = float(close.iloc[-1])
            if price < min_price:
                continue

            # Moving averages
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            if price < ma20 or price < ma50:
                continue

            # Two-speed signal (1-month and 3-month returns)
            roc_1m = (price / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
            roc_3m = (price / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None

            if roc_1m is None:
                continue
            if roc_3m is not None:
                if roc_1m > 0 and roc_3m > 0:
                    two_speed = "Full"
                elif roc_1m > 0 or roc_3m > 0:
                    two_speed = "Half"
                else:
                    continue
            else:
                two_speed = "Half" if roc_1m > 0 else None
                if two_speed is None:
                    continue

            # Volume filter
            avg_vol = float(volume.tail(20).mean())
            avg_dollar_vol = avg_vol * price
            if avg_dollar_vol < min_advol:
                continue

            # RS vs SPY (21-day)
            common = close.index.intersection(spy_close.index)
            if len(common) < 21:
                continue
            rs = close.reindex(common) / spy_close.reindex(common)
            rs_chg = (float(rs.iloc[-1]) / float(rs.iloc[-21]) - 1) * 100
            if rs_chg <= 0:
                continue

            # Distance from 20d MA
            dist_ma20 = ((price - ma20) / ma20) * 100

            # MACD histogram direction (red→green crossover check)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            if len(histogram) >= 2:
                hist_prev = float(histogram.iloc[-2])
                hist_curr = float(histogram.iloc[-1])
                macd_crossover = hist_prev < 0 and hist_curr > 0
            else:
                macd_crossover = False

            # Volume trend (last 5 days vs 20-day avg)
            recent_vol = float(volume.tail(5).mean())
            vol_ratio = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 0

            results.append({
                "Ticker": ticker,
                "Sector": sector_labels.get(ticker, "Unknown"),
                "Price": round(price, 2),
                "MA20": round(ma20, 2),
                "MA50": round(ma50, 2),
                "Dist_MA20_pct": round(dist_ma20, 1),
                "ROC_1m": round(roc_1m, 1),
                "ROC_3m": round(roc_3m, 1) if roc_3m else None,
                "Two_Speed": two_speed,
                "RS_vs_SPY_21d": round(rs_chg, 2),
                "MACD_Crossover": macd_crossover,
                "Vol_Ratio_5d": vol_ratio,
                "Avg_Dollar_Vol": int(avg_dollar_vol),
                "Entry_Zone": (
                    "Near MA (pullback)" if dist_ma20 <= 3.0
                    else "Normal" if dist_ma20 <= 6.0
                    else "Extended (accel only)" if dist_ma20 <= 15.0
                    else "Too extended"
                ),
            })
        except Exception:
            continue

    df_out = pd.DataFrame(results)
    print(f"  Passing all L4 filters: {len(df_out)} stocks")
    return df_out


def output_results(df, regime, permission, output_dir):
    """Rank and output final watchlist."""
    today = datetime.date.today().strftime("%Y-%m-%d")

    if len(df) == 0:
        print("\nNo candidates found.")
        return None

    # Rank by RS strength
    df = df.sort_values("RS_vs_SPY_21d", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    # Save CSV
    csv_path = os.path.join(output_dir, f"screener_watchlist_{today}.csv")
    df.to_csv(csv_path)

    # Print summary
    print(f"\n{'='*72}")
    print(f"SCREENER RESULTS — {today}")
    print(f"Regime: {regime.upper()} | Permission: {permission.upper()}")
    print(f"{'='*72}")

    display_cols = ["Ticker", "Sector", "Price", "Two_Speed", "RS_vs_SPY_21d",
                    "Dist_MA20_pct", "Entry_Zone", "MACD_Crossover", "Vol_Ratio_5d"]

    print(f"\nTop 25 by relative strength:\n")
    print(df[display_cols].head(25).to_string())

    # Breakdowns
    near_ma = df[df["Dist_MA20_pct"] <= 3.0]
    normal = df[(df["Dist_MA20_pct"] > 3.0) & (df["Dist_MA20_pct"] <= 6.0)]
    extended = df[(df["Dist_MA20_pct"] > 6.0) & (df["Dist_MA20_pct"] <= 15.0)]
    too_ext = df[df["Dist_MA20_pct"] > 15.0]

    print(f"\n--- Entry proximity ---")
    print(f"Pullback zone (≤3% above MA20): {len(near_ma)}")
    print(f"Normal (3-6%): {len(normal)}")
    print(f"Extended / Accelerating only (6-15%): {len(extended)}")
    print(f"Too extended (>15%): {len(too_ext)}")

    macd_cross = df[df["MACD_Crossover"]]
    print(f"\n--- MACD histogram crossover (red→green today) ---")
    if len(macd_cross) > 0:
        print(macd_cross[["Ticker", "Sector", "Price", "Dist_MA20_pct", "RS_vs_SPY_21d"]].to_string())
    else:
        print("None today")

    full = df[df["Two_Speed"] == "Full"]
    half = df[df["Two_Speed"] == "Half"]
    print(f"\n--- Two-speed signal ---")
    print(f"Full: {len(full)} | Half: {len(half)}")

    print(f"\n--- Sector breakdown ---")
    print(df["Sector"].value_counts().to_string())

    # Highlight best setups
    best = df[(df["Dist_MA20_pct"] <= 6.0) & (df["Two_Speed"] == "Full")].head(10)
    if len(best) > 0:
        print(f"\n--- TOP SETUPS (Full signal + near entry zone) ---")
        print(best[display_cols].to_string())

    print(f"\n{'='*72}")
    print(f"Saved: {csv_path}")
    print(f"Total candidates: {len(df)}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Swing Trading Screener v3")
    parser.add_argument("--regime", default="risk-on",
                        choices=["risk-on", "reflation", "deflation", "stagflation"])
    parser.add_argument("--permission", default="green",
                        choices=["green", "yellow", "red"])
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-advol", type=float, default=50000)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    if args.permission == "red":
        print("RED state — no new entries per framework. Exiting.")
        return

    start = time.time()

    # Stage 1: Get regime-aligned universe
    tickers, sector_labels = get_universe_for_regime(args.regime)

    # Stage 2: Download data
    frames = download_data(tickers, period="6mo")

    # Stage 3: Compute signals and filter
    df = compute_and_filter(frames, sector_labels,
                            min_price=args.min_price, min_advol=args.min_advol)

    # Stage 4: Output
    result = output_results(df, args.regime, args.permission, args.output_dir)

    print(f"\nCompleted in {time.time()-start:.0f} seconds.")
    return result


if __name__ == "__main__":
    main()
