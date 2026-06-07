"""
Configuration — framework constants and universes.

Lookback windows, thresholds, sector/ETF maps, permission-state limits, and
sizing tables. No runtime logic, no st.* calls — safe to import anywhere.
"""

# ── Lookback periods (trading days) ──────────────────────────────────────────
LB_1M = 21    # ~1 calendar month
LB_3M = 63    # ~3 calendar months
LB_6M = 126   # ~6 calendar months

# ── Signal thresholds ────────────────────────────────────────────────────────
RS_NEW_HI_THRESHOLD = 0.98   # within 2% of recent peak = RS new high
VELOCITY_THRESHOLD  = 0.15   # v4 Velocity Flag: sector ETF ROC 21 > 15% = Accelerating

# ── Sector ETF universe ──────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Industrials":            "XLI",
    "Technology":             "XLK",
    "Semiconductors":         "SMH",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Utilities":              "XLU",
    "Health Care":            "XLV",
}

SECTOR_TICKERS = {
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES",
        "DVN","HAL","BKR","FANG","MRO","APA","EQT","CTRA","TRGP","OKE",
        "KMI","WMB","LNG","CVI","MGY",
    ],
    "Materials": [
        "LIN","APD","ECL","SHW","FCX","NEM","NUE","VMC","MLM","ALB",
        "DD","EMN","IFF","PPG","RPM","FMC","MOS","CF","BALL","IP",
        "PKG","SEE","CCK","AVY","SON","AMCR","CE","DOW","LYB","WLK",
    ],
    "Industrials": [
        "RTX","HON","UNP","UPS","BA","LMT","GE","CAT","DE","MMM",
        "ITW","EMR","ETN","PH","ROK","FDX","CSX","NSC","WM","RSG",
        "CTAS","CPRT","GWW","AME","TT","IR","CARR","OTIS","PWR","URI",
        "MAS","JCI","XYL","AXON","TDG","HWM","NOC","GD","LHX","LDOS",
        "HUBB","FTV","RRX","GNRC","SAIA","ODFL","JBHT","EXPD","TXT",
    ],
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ACN","AMD","QCOM","TXN",
        "AMAT","LRCX","KLAC","MU","ADI","MCHP","CDNS","SNPS","FTNT",
        "PANW","CRWD","NOW","ZS","DDOG","NET",
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG",
        "ROST","ORLY","AZO","DHI","LEN","PHM","ULTA","YUM","DRI",
    ],
    "Financials": [
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW",
        "AXP","C","USB","PNC","TFC","COF","ICE","CME","SPGI","MCO",
    ],
    "Consumer Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","GIS","K",
        "SJM","HRL","CAG","CPB","MKC","CHD","CLX","KMB","MDLZ",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","ETR","ED",
        "XEL","WEC","ES","AWK","DTE","FE","PPL","AEE","CMS","NI",
    ],
    "Health Care": [
        "LLY","UNH","JNJ","ABT","TMO","DHR","BMY","AMGN","ISRG","MDT",
        "SYK","BSX","EW","BDX","IDXX","DXCM",
    ],
}

ALL_SECTORS = list(SECTOR_ETFS.keys())

# ── Regime classification sets ───────────────────────────────────────────────
# Industrials appears in both Risk-on and Reflation per framework v3.
RISK_ON_SECTORS   = {"Technology", "Semiconductors", "Financials", "Consumer Discretionary", "Industrials"}
REFLATION_SECTORS = {"Energy", "Materials", "Industrials"}
DEFENSIVE_SECTORS = {"Consumer Staples", "Utilities", "Health Care"}

# ── Per-sector display colors for the RRG chart ──────────────────────────────
SECTOR_COLORS = {
    "Energy":                 "#F59E0B",   # amber
    "Materials":              "#10B981",   # emerald
    "Industrials":            "#3B82F6",   # blue
    "Technology":             "#A78BFA",   # violet
    "Semiconductors":         "#8B5CF6",   # purple
    "Financials":             "#EC4899",   # pink
    "Consumer Discretionary": "#F97316",   # orange
    "Consumer Staples":       "#06B6D4",   # cyan
    "Utilities":              "#EF4444",   # red
    "Health Care":            "#84CC16",   # lime
}
SECTOR_DASH = ["solid", "dash", "dot", "dashdot"]   # line-style overflow fallback

# ── Permission state limits and display labels ───────────────────────────────
PERM_LIMITS = {
    "Green":  {"max_pos": 20, "max_pos_label": "Up to 20", "risk_lo": 0.75, "risk_hi": 1.00, "heat": 15},
    "Yellow": {"max_pos": 10, "max_pos_label": "8–12",     "risk_lo": 0.25, "risk_hi": 0.50, "heat": 8},
    "Red":    {"max_pos":  5, "max_pos_label": "3–5",      "risk_lo": 0.00, "risk_hi": 0.00, "heat": 3},
}

SETUP_STYLE = {
    "Green":  "Momentum breakouts",
    "Yellow": "Pullbacks to 20d / 50d MA",
    "Red":    "No new entries — protect capital",
}

# ── Layer 3 flow strength options and sizing map (framework v3) ───────────────
FLOW_OPTS = ["Not set", "Weak", "Moderate", "Strong", "Outflows"]

FLOW_SIZE_MAP = {
    "Weak":     ("Quarter",  "0.19%", "1 week of modest inflows — watch closely"),
    "Moderate": ("Half",     "0.38%", "1–2 weeks consistent inflows — enter"),
    "Strong":   ("Full",     "0.75%", "2+ weeks accelerating inflows — full size"),
    "Outflows": ("Exit",     "—",     "Flow reversal — exit review immediately"),
}

# ── Layer 3 phase labels mapped from RRG quadrant ────────────────────────────
PHASE_MAP = {
    "Improving": "Phase 1 — Early",
    "Leading":   "Phase 2 — Confirmed",
    "Weakening": "Exiting",
    "Lagging":   "No Trade",
}
