# Layer 3 — Sector Rotation ETF
*Proposal v2 — April 30, 2026*

---

## Position in the framework

Layer 3 sits between Permission State (Layer 2) and Stock Selection (Layer 4). The logic: sector ETF positions are entered before individual stocks set up cleanly, so the decision to enter the ETF must precede the individual stock screening pass. If Layer 2 is green or yellow, Layer 3 evaluates whether a sector rotation is underway and whether to put early capital to work via the sector ETF.

Sequence:
- Layer 0 — Macro regime
- Layer 2 — Permission state
- **Layer 3 — Sector rotation ETF entry**
- Layer 4 — Individual stock selection
- Layer 5 — Entry triggers

---

## The thesis

Money rotation into a sector precedes individual stock setups by 2–8 weeks. Institutional capital moves into sector ETFs first — the easiest, fastest way to get broad exposure. Individual stocks form bases and trigger clean entries only after the rotation is underway. Layer 3 catches the rotation early, puts capital to work, and transitions to individual names as they set up through Layer 4.

This is momentum trading. The signal is the flow of money. The exit is when the money leaves.

---

## Data sources

**ETF.com Fund Flows Tool** — [etf.com/etfanalytics/etf-fund-flows-tool](https://www.etf.com/etfanalytics/etf-fund-flows-tool)
Primary flow signal. Daily net inflows/outflows by ETF, filterable by sector. Free, updated daily. Check the 1-week and 4-week columns for the sector ETFs (XLK, XLE, XLI, XLF, XLB, XLY, etc.).

**SSGA Sector Tracker** — [ssga.com/us/en/intermediary/resources/sector-tracker](https://www.ssga.com/us/en/intermediary/resources/sector-tracker)
Price performance tracker (not flows). Use the 1M and 3M tabs to rank sectors by return — confirms which sectors are leading in price. Check weekly.

**State Street Monthly Flash Flows PDF** — published ~month-end at ssga.com
Backward-looking confirmation of where institutional money went during the prior month.

**TradingView** — sector ETF RS line vs SPY, 20d MA position, volume.

---

## Signal framework

### Phase 1 — Early indication (Week 1–2)

Conditions:
- Sector ETF shows 1–2 weeks of directionally consistent positive net inflows on ETF.com
- Sector ETF price at or above its 20d MA (not extended — close to it)
- Regime alignment: current G/I/L regime favors this sector
- RS line of the sector ETF vs SPY is inflecting upward

The signal is momentum, not a threshold. We are not looking for a specific dollar amount — we are looking for consistent directional flow. One-day spikes are noise (hedges, rebalancing). What matters is sustained directionality over 1–2 weeks.

**Action:** Enter sector ETF. Position size scales with flow momentum — see sizing below.

### Phase 2 — Confirmation (Weeks 3–8)

Confirmation signals:
- 3+ consecutive weeks of net inflows sustained
- Sector ETF RS line making new highs vs SPY
- Individual stocks in sector: two-speed signals turning Full (ROC 21 and ROC 63 both positive)
- Volume expanding on up weeks vs down weeks

**Action:** Scale up ETF position if not already at full size, OR begin entering individual names via Layer 4/3 and reduce ETF proportionally.

### Phase 3 — Transition to individual names

As individual stocks clear Layer 4 and trigger Layer 5 entries, the ETF position becomes redundant. Transition rules:
- Each individual stock entered in the sector reduces the ETF by one slot-equivalent
- Complete transition by Week 8–12
- If individual setups never materialize despite ETF strength, keep the ETF and trail it — the money is there, just not yet concentrated in individual names

---

## Position sizing — flow momentum based

Position size is not fixed. The volume and consistency of inflow momentum determines how aggressively to size the initial ETF entry.

| Flow signal strength | Position size | Risk % |
|---------------------|---------------|--------|
| Weak — 1 week, modest inflows, RS just turning | Quarter size | 0.1875% |
| Moderate — 1–2 weeks, consistent inflows, RS inflecting | Half size | 0.375% |
| Strong — 2+ weeks, accelerating inflows, RS making new highs | Full size | 0.75% |

Stop is always the 20d MA of the sector ETF. If the ETF closes below its 20d MA on volume, the rotation thesis is invalidated — exit regardless of size.

Entry defines the stop. Sizing is determined before entry, not adjusted after.

---

## Portfolio heat

ETF positions count toward portfolio heat calculations on equal footing with individual stock positions.

- Each ETF position occupies one slot toward the 20-position maximum
- ETF dollar risk counts toward the total portfolio heat percentage (Green: 15% max, Yellow: 8% max, Red: 3% max)
- Sector cap: ETF position + individual stock positions in the same sector ≤ 20% of account at any time

---

## Exit signals

Exits track the flow of money, not price targets. This is momentum — ride it until the momentum stops.

**Primary exits (any one triggers):**
1. **Flow reversal** — 2 consecutive weeks of net outflows after a sustained inflow period. Money leaving = rotation ending.
2. **Price** — close below 20d MA on above-average volume.
3. **RS line** — sector ETF RS line vs SPY declines for 2+ consecutive weeks.

**Secondary exits (flag for review, tighten stop to breakeven):**
4. **Regime shift** — G/I/L regime changes and no longer favors the sector. See regime mismatch protocol below.
5. **12-week max hold** — same rule as individual positions. No meaningful progress in 12 weeks → exit and redeploy.

---

## Regime mismatch protocol

When a sector is receiving strong, sustained inflows but it conflicts with the current G/I/L regime (e.g., Energy getting inflows in a Risk-On regime, or Utilities getting inflows when liquidity is loose):

1. **Flag it as an anomaly** — do not ignore it, do not chase it automatically.
2. **Evaluate the regime first** — ask whether the current regime classification has a flaw. Is there a cross-asset signal being misread? Is the regime on the edge of transitioning? If the regime appears correct, stay the course.
3. **Stick to the regime** — unless there is overwhelming, multi-source evidence that a one-off structural event is driving the flow (e.g., geopolitical supply shock into Energy during a Risk-On backdrop), do not override the regime to chase a mismatched sector.
4. **Log the anomaly** — note it in the weekly review. If the mismatch persists and the regime framework continues to explain it away while the sector keeps running, that is a signal to re-examine the regime methodology, not to capitulate into the trade.

The purpose: protect the framework's integrity. Chasing every flow signal regardless of regime turns this into noise-following. The regime is the filter that keeps us in high-probability setups.

---

## Weekly checklist addition (Sunday review)

Add to the Layer 0 data pull:

| Check | Source | Signal |
|-------|--------|--------|
| Top 3 sectors by 1-week net inflows | ETF.com fund flows | Directional momentum, not size |
| Top 3 sectors by 4-week net inflows | ETF.com fund flows | Sustained vs. spike |
| Top 3 sectors by 1M and 3M price return | SSGA Sector Tracker | Price leadership confirming flow |
| Sector ETF RS line vs SPY | TradingView | Inflecting = early; new highs = confirmed |
| Sector ETF 20d MA position | TradingView | Above = valid; below = wait |
| Sectors with 2+ weeks of outflows | ETF.com | Flag existing ETF positions for exit review |
| Any regime mismatch anomalies | Cross-reference | Log and evaluate |

---

## Current application — semis

As of late April/early May 2026, the semi rotation has already occurred. SMH moved ~40% in 3 weeks following the tariff selloff in early April. Layer 3 would have triggered in Week 1 (April 7–10) as flows turned positive and SMH reclaimed its 20d MA.

Current state: SMH is extended (28% above 20d MA), ROC 126 still negative, MACD rolling. The ETF entry window has passed. Individual name entries (NVDA, AVGO, AMAT, ADI) via Layers 4–3 are the right vehicle from here, pending May earnings clearance.

The lesson that shaped this layer: the flow signal fires early. Waiting for the ETF to be fully confirmed and extended means missing it. Layer 3 is designed to catch the signal at Week 1–2, not Week 6.

---

## Pending: full framework rebuild

All framework documents (swing_trading_framework_v2.md, positions_tracker.md, weekly_review_prompt.md, weekly_playbook template, screening_log.md) are to be updated to incorporate Layer 3 as a permanent part of the 8-layer system (which becomes a 9-layer system). This work is deferred — do not begin until directed.

---

*Sources: ETF.com fund flows methodology; SSGA Sector Tracker; State Street Monthly Flash Flows PDF (March 2026); Financial Modeling Prep — ETF Flows as Leading Indicators for Sector Rotation; CFRA Research — Analyzing ETF Flow Trends*
