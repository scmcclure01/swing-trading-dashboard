# Swing Trading Weekly Workflow
**Framework:** v4.0
**Last updated:** 2026-06-01

---

## Before You Start
- This workflow runs every week, ideally Sunday morning
- Never skip a layer — each one gates the next
- The Streamlit dashboard handles Layers 0–5 automatically — open it first
- Claude reads the most recent playbook xlsx and portfolio.json directly — no attachments needed

---

## Layer 0 — Macro Regime Filter
*Goal: Identify the current Growth/Inflation/Liquidity environment*

**Auto-computed on the Streamlit dashboard (Macro & Permission tab):**
- SPY two-speed trend: 1-month (21d) and 6-month (126d) return
- Sector RS vs SPY (4-week): which ETFs are outperforming?
- TLT direction and TLT/SPY combined signal
- HYG/IEF credit spread (liquidity check)
- Fed Net Liquidity (WALCL - WTREGEN - RRPONTSYD from FRED) — 4-week change
- VIX level

**Auto-computed — Velocity Flag [v4]:**
- ROC 21 for each sector ETF (XLK, XLE, XLI, XLB, XLF, XLY, XLU, XLP, SMH)
- Any sector with ROC 21 > +15% = ACCELERATING — modified entry rules apply

**Auto-fetched via FRED API (recession composite):**
- CFNAIMA3, SAHMREALTIME, T10Y3M, Chauvet-Piger (RECPROUSM156N)

**Manual inputs (set in sidebar each week):**
- EPS Revisions direction (from FactSet Earnings Insight — insight.factset.com)
- Conference Board LEI trend (conference-board.org — paywalled)
- Taylor Rule deviation (monthly, after CPI/PCE — atlantafed.org)
- Drawdown from peak equity

---

## Layer 1 — Monthly Macro Mispricing Check
*Goal: Assess whether equities are rich or cheap vs real rates*

**Run monthly after CPI/PCE release, not weekly.** Set in the sidebar:
- Earnings yield vs 10Y TIPS spread (DFII10)
- Equity risk premium vs 10Y nominal (DGS10)
- GDPNow vs analyst EPS direction
- Taylor Rule gap

Score: +1 bullish / 0 neutral / -1 bearish per check. Composite -2 to -3 = reduce sizing 30-50%.

---

## Layer 2 — Market Permission State
*Goal: Determine how aggressively to trade*

Auto-computed on the dashboard from L0 inputs. Override in sidebar if needed.

| State | Condition | Max Positions | Risk/Trade | Core Allocation [v4] |
|-------|-----------|---------------|------------|---------------------|
| Green | SPY both positive + no overrides | Up to 20 | 0.75–1.0% | Full — up to 40% in 2-3 ETFs |
| Yellow | SPY mixed OR recession 2-3/5 OR EPS declining | 8–12 | 0.25–0.5% | Half — up to 20% in 1-2 ETFs |
| Red | SPY both negative OR liquidity override OR recession 4+/5 | 3–5 max | No new entries | ZERO — exit all Core |

---

## Layer 3 — Sector Rotation ETF
*Goal: Identify sector rotations early and enter via ETF before individual stocks set up*

**Run after Layer 2, before the screener. Use the Sector Rotation tab on the dashboard.**

The dashboard auto-computes the Relative Rotation Graph (RRG) and maps sectors to phases:
- Improving (RS < 100, Momentum > 100) → Phase 1 — Early
- Leading (RS > 100, Momentum > 100) → Phase 2 — Confirmed
- Weakening (RS > 100, Momentum < 100) → Exiting
- Lagging (RS < 100, Momentum < 100) → No Trade

**Set flow strength per sector in the L3 expander** (auto-seeded from implied flows, override with ETF.com data):

| Flow Strength | What You See | Size | Risk % (Green) |
|--------------|-------------|------|----------------|
| Weak | 1 week of inflows, RS just turning | Quarter | 0.1875% |
| Moderate | 1–2 weeks consistent inflows, RS inflecting | Half | 0.375% |
| Strong | 2+ weeks accelerating inflows, RS at new highs | Full | 0.75% |

Stop is always the 20d MA of the sector ETF.

**For existing ETF positions — check weekly:**
- Any 2 consecutive weeks of outflows (ETF.com)? → Exit
- 20d MA still holding? → If below on volume: exit
- RS line declining 2+ weeks? → Exit
- Individual names from this sector entering via Layer 5? → Reduce ETF proportionally (Phase 3)

**Regime mismatch:** If a sector has strong inflows but conflicts with the G/I/L regime, flag as anomaly. Stick to the regime unless overwhelming one-off evidence.

Sector cap: ETF + individual names in same sector ≤ 25% of account [v4].

---

## Core Allocation Check [v4]
*Goal: Ensure persistent regime-aligned exposure meets the deployment floor*

**Run after Layer 3. Use the Core Allocation tab on the dashboard.**

1. List current Core ETF positions (from sidebar inputs)
2. Check Core % deployed vs target (GREEN=40%, YELLOW=20%, RED=0%)
3. Check total deployed capital (Core + Tactical) vs deployment floor:
   - GREEN: 40–60% minimum
   - YELLOW: 20–35% minimum
   - RED: no floor — cash preservation
4. Any sectors newly Phase 2 Confirmed? → Candidate for Core entry (max 3 Core ETFs, 15% per ETF)
5. Any Core positions need exit? (20d MA violation on volume, RS declining 2+ weeks, regime shift, RED state, flow reversal)
6. If below deployment floor AND Phase 2 sectors exist → recommend Core entries to fill

Core entries: at market, stop at 20d MA. No breakout or volume threshold.

---

## Layer 4 — Run the Screener
*Goal: Generate individual stock candidates filtered to leading sectors*

**Use the Screener tab on the Streamlit dashboard.** Click "Run Screener" — it scans only regime-aligned sectors from Layer 0.

**Review the results:**
- **Actionable Setups** — Full signal + entry trigger confirmed (ENTRY READY or WATCH)
- **Watchlist** — Half signal candidates (half-size eligible, monitor)
- Check MACD direction — histogram red→green crossover is the signal, not above/below zero
- Stocks marked "Too extended" are for monitoring only — do not enter

**If an active Layer 3 ETF or Core position exists in a sector:** Flag individual names from that sector as priority candidates. Layer 3 ETF reduces proportionally as stocks enter (Phase 3). Core persists — no transition needed.

---

## Layer 5 — Entry Trigger
*Goal: Find the specific entry point for each individual stock candidate*

- Open each actionable candidate in **TradingView** to confirm the setup
- **Risk-on / Green regime**: breakout from base on 40%+ above-average volume
- **Mixed / Yellow regime**: pullback to 20d or 50d MA with reversal candle on declining volume
- **Velocity Flag active for this sector [v4]**: Accelerating Protocol — entries up to 12-15% above 20d MA, 1.0x volume, half size, 10d EMA stop, 6-week max hold, max 3 simultaneous
- No earnings within 10 business days
- Entry price defines your initial stop — do not enter without a clear stop level

Note: ETF entries (Layer 3) and Core entries do not use this trigger — they enter at market with 20d MA stop.

---

## Layer 6 — Position Sizing
*Goal: Size each position using the expected-loss method*

**Use the Position Sizer tab on the dashboard, or calculate manually:**

- **Formula**: Shares = (Account × Risk%) ÷ (Entry − Stop)
- Risk% set by permission state (Layer 2); for ETFs, by flow momentum (Layer 3)
- Max single Tactical position: 10% of account
- Max single Core ETF: 15% of account [v4]
- Accelerating Protocol: halve the share count [v4]
- In drawdown: reduce risk% per Layer 9 tiers

---

## Layer 7 — Portfolio Exposure Check
*Goal: Confirm total portfolio heat is within limits before entering*

**Use the Portfolio tab compliance section, or check manually:**

- Count all open positions (Core ETFs + Layer 3 ETFs + Tactical stocks)
- Total portfolio heat (sum of all open risk):
  - Green: max 15%
  - Yellow: max 8%
  - Red: max 3%
- Sector cap: all positions in same sector ≤ 25% of account [v4]
- Deployment floor check [v4]: total deployed capital above minimum for current state?
- If adding the new position would breach any limit → skip or reduce size

---

## Layer 8 — Trade Management (ongoing)
*Goal: Manage open positions systematically*

**Core ETF positions [v4]:**
- Trail stop at 20d MA — no partial profit rules
- **Exit if:** close below 20d MA on volume / RS declining 2+ weeks vs SPY / regime shifts against sector / permission state RED / 2 consecutive weeks outflows (ETF.com)
- No fixed max hold — Core persists with the regime

**Layer 3 ETF positions:**
- Trail stop at 20d MA — no partial profit rules
- **Exit if:** 2 consecutive weeks outflows / close below 20d MA on volume / RS declining 2+ weeks
- **Reduce proportionally** as individual stocks from the same sector are entered (Phase 3)
- 12-week max hold

**Tactical individual stocks — Standard:**
- Move stop to **breakeven** at +5%
- Sell **1/3 at +8–12%**, move stop to breakeven
- Sell **another 1/3 at +20–25%**, trail remainder at 20d MA
- **Exit if:** close below 20d MA on volume / RS declining / permission state Red / 12-week max hold reached

**Tactical individual stocks — Accelerating Protocol [v4]:**
- Move stop to **breakeven** at +5%
- Sell **1/3 at +8–10%**, move stop to breakeven
- Sell **another 1/3 at +15–20%**, trail remainder at 20d MA
- Stop: 10d EMA (pre-T1), 20d MA (post-T1)
- **6-week max hold** — exit if insufficient progress
- If Velocity Flag deactivates: manage as Standard from that point

---

## Layer 9 — Hard Risk Caps (check weekly)
*Goal: Enforce portfolio-level drawdown limits*

| Drawdown from Peak | Action |
|--------------------|--------|
| 0–7% | Normal operations. Core holds if stops intact. [v4] |
| 7–10% | Reduce risk/trade by 50%. No new Tactical. Core holds, no new Core. Tighten all stops. [v4] |
| 10–15% | Max 3-5 positions (including Core). Exit all Tactical. Core may hold 1 strongest. [v4] |
| >15% | 100% cash including Core. Wait for confirmed Green state. |

**Hedges during tightening liquidity:** cash, SH/PSQ inverse ETF (10–20%), TLT (deflationary regime)

---

## End of Week Checklist

- [ ] Layer 0 macro regime confirmed — SPY trend, sector RS, Velocity Flag, TLT, liquidity, Fed Net Liquidity
- [ ] Permission state set (Layer 2)
- [ ] Layer 3 ETF scan complete — RRG reviewed, flow strengths set, any new rotation opportunities?
- [ ] Core allocation checked [v4] — positions, % deployed, deployment floor met?
- [ ] Screener run with correct regime-aligned sectors (Layer 4)
- [ ] All actionable candidates reviewed in TradingView (Layer 5)
- [ ] Open Core positions checked — 20d MA, RS, regime, flows
- [ ] Open Layer 3 ETF positions checked — flow reversal, 20d MA, Phase 3 transition
- [ ] Open Tactical positions checked against Layer 8 rules
- [ ] Accelerating positions checked — 10d EMA stop, 6-week max hold [v4]
- [ ] Portfolio heat within Layer 7 limits (all positions combined)
- [ ] No hard risk cap breaches (Layer 9)
- [ ] Playbook xlsx updated with new snapshot data
- [ ] Trade journal updated with any closed trades
- [ ] Screening log entry added

---

*Framework v4.0 — Prometheus Research methodology (prometheus-research.com)*
