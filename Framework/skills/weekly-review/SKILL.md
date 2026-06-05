---
name: weekly-review
description: Run the full swing trading weekly review. Use this skill whenever the user says "run weekly review", "weekly review", "run the review", "Sunday review", or asks for a macro/regime update, permission state, sector rotation check, or playbook update. This is the primary weekly workflow skill — trigger it proactively whenever the context suggests it's the start of a trading week.
---

# Swing Trading Weekly Review

The framework doc is at: `Framework/swing_trading_framework_v4.md`. Read relevant sections if you need rule detail. The playbook is the most recent `.xlsx` in `Playbooks/`. Read `screening_log.md` for recent context.

Execute every step in order. Never skip a layer.

---

## Step 1 — Layer 0: Pull Macro Data

Fetch all of the following via web search and yfinance. Flag anything unavailable.

**SPY trend (permission state input):**
- SPY 1-month return (21-day)
- SPY 6-month return (126-day)

**Sector RS vs SPY (4-week):** XLK, XLE, XLI, XLB, XLF, XLY, XLU, XLP, SMH — rank by relative return

**Velocity Flag — calculate ROC 21 for each sector ETF:**
```python
import yfinance as yf
tickers = ['XLK','XLE','XLI','XLB','XLF','XLY','XLU','XLP','SMH']
for t in tickers:
    data = yf.download(t, period='3mo', progress=False)
    roc21 = (data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) * 100
    print(f"{t}: ROC21 = {roc21:.1f}%")
```
Flag any sector with ROC 21 > +15% as 🔥 ACCELERATING.

**Macro signals (web search):**
- TLT 4-week direction
- HYG/IEF spread — widening or tightening?
- FRED CFNAIMA3 — latest reading (below -0.70 = recession signal)
- FRED SAHMREALTIME — latest reading (above 0.50 = triggered)
- FRED T10Y3M — yield curve spread
- FactSet Earnings Insight (insight.factset.com) — EPS revision direction, 3-week trend
- Crude oil CL1-CL2 term structure (backwardation = positive for energy)

**Fed Net Liquidity:** WALCL - WTREGEN - RRPONTSYD via FRED. Calculate 4-week change. Flag if decline > $200B.

---

## Step 2 — Score Recession Composite

Check each of the 5 models:
1. CFNAIMA3 below -0.70?
2. Sahm Rule ≥ 0.50?
3. T10Y3M inverted (negative)?
4. Chauvet-Piger (FRED: RECPROUSM156N) above 50%? (use last available)
5. Conference Board LEI declining 6+ months? (use last known if paywalled)

Score: X/5 models in recession territory.
- 0-1: normal
- 2-3: caution — reduce max positions 25%, tighten stops
- 4-5: treat as RED state regardless of SPY trend

---

## Step 3 — Set Permission State (Layer 2)

| SPY signals | Recession composite | State |
|---|---|---|
| Both positive | 0-3/5 | GREEN |
| Mixed | any | YELLOW |
| Both negative OR liquidity override | any | RED |
| Any | 4-5/5 | RED (override) |

State governs: max positions, risk per trade, Core allocation, entry style.

Check for liquidity override: HYG/IEF widening rapidly OR Fed Net Liquidity -$200B+ over 4 weeks → override to RED.

---

## Step 4 — Layer 3: Sector Rotation ETF Check

Pull from ETF.com fund flows tool and SSGA Sector Tracker (web search):
- Top 3 sectors by 1-week net inflows
- Top 3 sectors by 4-week net inflows
- Top 3 sectors by 1M and 3M price return (SSGA)

For each sector with consistent inflows:
- Regime-aligned? (cross-reference Step 3 regime)
- Sector ETF above 20d MA?
- Flow signal strength: 1 week = weak / 2 weeks consistent = moderate / 2+ weeks accelerating = strong
- Score: no action / watch / quarter / half / full entry

For any active Layer 3 ETF positions (from playbook):
- Flow reversal? (2 consecutive weeks outflows → exit)
- ETF above 20d MA? (below on volume → exit)
- RS line vs SPY declining 2+ weeks? (exit)
- Individual stocks from this sector entering? (Phase 3 transition — reduce ETF)

Flag regime mismatch anomalies (strong inflows into non-regime sector).

---

## Step 5 — Core Allocation Check

From the playbook, identify current Core ETF positions:
- How many Core ETFs held? (max 3)
- % of account deployed in Core vs target (GREEN=40%, YELLOW=20%, RED=0%)
- Total deployed capital (Core + Tactical) vs deployment floor (GREEN=40-60%, YELLOW=20-35%)

Evaluate:
- Any sectors newly Phase 2 Confirmed? → candidate for Core entry
- Any Core positions need exit? (20d MA violation, RS declining 2+ weeks, regime shift, permission state RED)
- If below deployment floor AND Phase 2 sectors exist → recommend Core ETF entries

---

## Step 6 — Open Position Management (Layer 8)

For each open Tactical position in the playbook, check:
- Hit +5%? → stop should be at breakeven
- Hit +8-12%? → 1/3 should be sold, stop at breakeven
- Hit +20-25%? → another 1/3 should be sold, trailing at 20d MA
- Close below 20d MA on volume? → flag for exit
- RS line declining? → flag for review
- Recession composite 4+/5? → move all stops to breakeven
- 12-week max hold reached? (6 weeks for Accelerating entries) → exit

For Accelerating Protocol positions: check 10d EMA stop specifically.

---

## Step 7 — Watchlist Rescore (Layer 4)

For each watchlist candidate, use yfinance to calculate:
- ROC 21 (1-month return)
- ROC 63 (3-month return)
- Two-speed signal: both positive = full / mixed = half / both negative = remove

Flag:
- Regime-aligned sector?
- Any new entry trigger visible? (breakout setup, pullback to MA)
- Velocity Flag active for this sector? → evaluate under Accelerating Protocol
- Active Core or Layer 3 ETF in this sector? → priority candidate

---

## Step 8 — Output

Produce a concise weekly summary covering:

**Layer 0 Snapshot:**
- SPY 1M / 6M returns
- Permission state
- Regime identification
- Liquidity override: yes/no
- Fed Net Liquidity 4-week change
- Recession composite score
- FactSet revision direction
- Velocity Flag: sectors flagged and ROC 21 readings

**Layer 3 Snapshot:**
- Top inflow sectors (1W and 4W)
- Any new ETF entry opportunities (with sizing)
- Active ETF position status

**Core Allocation Status:**
- Positions held, % deployed, vs floor
- Any entries or exits needed

**Position Actions:**
- Each open position: hold / take partial / exit / adjust stop
- Each watchlist name: two-speed score / entry trigger status

**Week's Rules:**
- Entry style bias (breakout vs pullback)
- Max positions for the week
- Risk per trade
- Any special conditions (Accelerating sectors, drawdown tier, recession caution)

Then update the playbook xlsx with new snapshot data and action items.

---

## Step 9 — Update Screening Log

Append a new dated entry to `screening_log.md` in the workspace folder. Use this format:

```
## YYYY-MM-DD Screening Pass (Automated Weekly Review)

**Regime:** [regime from Step 3]
**Permission state:** [state from Step 3]
**Sector bias:** Primary: [leading sectors] | Secondary: [mixed sectors] | Avoid: [lagging]
**Risk/trade:** [from permission state]
**Max positions:** [from permission state]

### Layer 3 — Sector ETF Flow Scan

| Sector | ETF | 1W Flow | 4W Flow | Signal | Phase | Action |
|--------|-----|---------|---------|--------|-------|--------|
[one row per sector with flow data from Step 4]

### Layer 4 — Individual Stock Screen

| Ticker | Sector | Two-Speed | Entry Zone | Verdict | Notes |
|--------|--------|-----------|------------|---------|-------|
[top candidates from Step 7 — ENTRY READY and WATCH only, max 15 rows]

### Open Position Actions

[one line per open position from Step 6: hold / partial / exit / adjust stop]
```

This creates the audit trail. Every weekly review must produce a screening log entry — no exceptions.

---

## Monthly Add-on (after CPI/PCE)

When running the monthly check, also pull:
- Forward earnings yield vs 10Y TIPS (DFII10) spread
- Equity risk premium vs 10Y nominal (DGS10)
- GDPNow direction (atlantafed.org) vs FactSet consensus EPS
- Taylor Rule gap (Atlanta Fed calculator)

Score mispricing composite (+1 bullish / 0 neutral / -1 bearish per check):
- +2 to +3: full risk
- -1 to +1: normal
- -2 to -3: reduce all position sizes 30-50%, tighten stops

---

Be direct. Readings, state, and what to do. No background.
