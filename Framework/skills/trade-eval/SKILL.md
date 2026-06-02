---
name: trade-eval
description: Evaluate a specific trade candidate through the swing trading framework. Use this skill when the user shares a ticker and asks whether to trade it, asks for a framework evaluation, says "evaluate [ticker]", "run the framework on [ticker]", "is [ticker] a buy", or wants entry, stop, and size for a specific stock or ETF. Also use for Core ETF entry decisions and Layer 3 sector ETF evaluations.
---

# Trade Evaluation

The framework doc is at: `Framework/swing_trading_framework_v4.md`. Read relevant sections if you need rule detail.

Identify the trade type first, then run the appropriate path.

---

## Trade Type A — Individual Stock (Standard or Accelerating)

**Requires from user or recent weekly review:** current permission state, current regime, Velocity Flag status for this sector.

### Layer 4 — Stock Qualification

Pull via yfinance:
```python
import yfinance as yf
import pandas as pd

ticker = 'XXXX'
data = yf.download(ticker, period='1y', progress=False)

close = data['Close']
roc21 = (close.iloc[-1] / close.iloc[-22] - 1) * 100   # 1-month
roc63 = (close.iloc[-1] / close.iloc[-64] - 1) * 100   # 3-month
ma20 = close.rolling(20).mean().iloc[-1]
ma50 = close.rolling(50).mean().iloc[-1]
price = close.iloc[-1]
vol = data['Volume']
avg_vol = vol.rolling(50).mean().iloc[-1]
today_vol = vol.iloc[-1]

print(f"Price: {price:.2f} | MA20: {ma20:.2f} | MA50: {ma50:.2f}")
print(f"ROC21: {roc21:.1f}% | ROC63: {roc63:.1f}%")
print(f"Today vol: {today_vol:,.0f} | 50d avg: {avg_vol:,.0f} | Ratio: {today_vol/avg_vol:.2f}x")
```

Check:
- **Two-speed trend:** both ROC21 and ROC63 positive = full position; mixed = half; both negative = PASS
- **Price vs MAs:** above 20d and 50d? If not = PASS
- **RS line:** is the stock outperforming SPY over the last 3 months? (calculate: stock ROC63 vs SPY ROC63)
- **Earnings blackout:** any earnings within 10 business days? (web search) → if yes = PASS
- **Sector alignment:** does the sector match the current regime?
- **Earnings carry:** (forward earnings yield minus 3M T-bill). Positive = full; slightly negative = reduce 25% or skip
- **Energy stocks only:** run 3-layer check (CL1-CL2 term structure, COT percentile, XLE vs 50d SMA)

### Layer 5 — Entry Trigger

**Check which protocol applies:**

| Sector ROC 21 | Protocol |
|---|---|
| > +15% | Accelerating — see below |
| ≤ +15% | Standard |

**Standard entry:**
- Breakout: consolidation 3-8 weeks, breakout above base on 40%+ above-average volume, RS line at new high. Don't enter if >5% above pivot.
- Pullback: uptrend intact, price at 20d or 50d MA, reversal candle, declining volume. Stop = 1-2% below the MA.

**Accelerating Protocol (Velocity Flag active):**
- Entry allowed up to 12-15% above 20d MA
- Volume threshold: 1.0x average (not 1.4x)
- Stop: 10-day EMA (not base low)
- Max 3 Accelerating positions simultaneously
- Permission state YELLOW → quarter size (not half)

### Layer 6 — Position Sizing

```
Shares = (Account × Risk%) ÷ (Entry − Stop)
```

Risk% by permission state:
- GREEN: 0.75-1.0%
- YELLOW: 0.25-0.5%

Adjustments:
- Half two-speed signal → halve shares
- Accelerating Protocol → halve shares (after two-speed adjustment)
- Drawdown tier (7-10% below peak) → halve risk%
- Hard cap: no single position > 10% of account ($43K Tactical sleeve)

### Output Format — Individual Stock

```
TICKER — [TRADE / HALF SIZE / PASS]

Layer 4:
- Two-speed: ROC21 X% / ROC63 X% → [Full / Half / PASS]
- Price vs MAs: above 20d ($X) and 50d ($X)? [Yes/No]
- RS vs SPY: [outperforming / underperforming]
- Earnings blackout: [clear / XX days — PASS]
- Sector: [name] — regime-aligned? [Yes/No]
- Carry: [positive/negative]

Layer 5:
- Protocol: [Standard / Accelerating]
- Trigger: [Breakout above $X / Pullback to 20d at $X / Extended entry at $X]
- Volume: [X.Xx average]
- Initial stop: $X ([X]% risk)

Layer 6:
- Account risk: $X (X% of account)
- Shares: X
- Position value: $X (X% of Tactical sleeve)
- [Cap applied if triggered]

Targets:
- T1: $X (+X%) — sell 1/3
- T2: $X (+X%) — sell 1/3
- Trail: 20d MA after T2
- Max hold: [12 weeks / 6 weeks Accelerating]
```

---

## Trade Type B — Core ETF Entry

**Requires:** current permission state, current regime, Core positions already held (from playbook).

### Evaluation

1. Permission state GREEN or YELLOW? (RED = no Core entries)
2. Sector regime-aligned?
3. Layer 3 Phase 2 Confirmed? (3+ weeks sustained inflows, RS at new highs, individual stocks showing full two-speed signals)
4. Current Core ETF count < 3?
5. Adding this ETF stays within: 15% per ETF cap, 25% sector concentration cap, deployment floor target?

If all yes → enter at market, stop at 20d MA.

### Output Format — Core ETF

```
[ETF] Core Entry — [ENTER / PASS]

Phase status: [Phase 1 Early / Phase 2 Confirmed]
Regime alignment: [Yes/No]
Permission state: [GREEN/YELLOW/RED]
Current Core count: X/3

Entry: market (~$X)
Stop: 20d MA ($X)
Size: $X (X% of account) — [within 15% cap / capped at 15%]
Post-entry Core deployment: X% (target X%)
Deployment floor: [met / X% below floor]
```

---

## Trade Type C — Layer 3 Sector ETF

**Requires:** current permission state, current regime, ETF.com flow data (user provides or Claude fetches via web search).

### Evaluation

1. Permission state GREEN or YELLOW?
2. Sector regime-aligned?
3. Sector ETF above 20d MA?
4. Flow signal strength:
   - 1 week inflows, modest = Weak → quarter size
   - 1-2 weeks consistent = Moderate → half size
   - 2+ weeks accelerating = Strong → full size
5. RS line inflecting up vs SPY?
6. Any regime mismatch? (flag as anomaly, do not enter)

### Output Format — Layer 3 ETF

```
[ETF] Layer 3 — [ENTRY SIZE / WATCH / PASS]

Flow signal: [Weak/Moderate/Strong] — X weeks directional inflows
RS vs SPY: [inflecting up / at new highs / flat / declining]
ETF vs 20d MA: [X% above/below]
Regime alignment: [Yes/No — note mismatch if any]

Entry: market (~$X)
Stop: 20d MA ($X)
Risk%: X% (Green [Weak/Moderate/Strong] sizing)
Shares: X | Position value: $X

Phase outlook: [Phase 1 Early — watch for Phase 2 / already Phase 2 — evaluate Core]
Exit triggers: 2 consecutive weeks outflows / close below 20d MA on volume / RS declining 2+ weeks
```

---

Be direct. Give the verdict first, then the supporting data.
