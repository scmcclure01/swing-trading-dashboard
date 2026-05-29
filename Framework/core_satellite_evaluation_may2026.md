# Program Evaluation & Core-Satellite Restructuring Spec
## May 23, 2026

---

## Executive Summary

The swing trading framework has been operational since April 2026. It has not produced a single losing trade — all 8 losses in the trade journal pre-date the framework. The system's risk management, position sizing, and trade management rules are functioning correctly.

**The problem is capital utilization.** In a confirmed GREEN state with 18 open position slots and 12.1% remaining heat capacity, the framework kept ~85% of the account in cash while the leading sector (XLK/Tech) ran +40%. YTD realized P&L of +$768 on a $71K account translates to ~9% annualized — well below the 25-35% target.

**Root cause:** The entry trigger (Layer 3) is binary and speed-blind. It requires either a breakout on 40%+ volume or a pullback to 20d MA with reversal candle. In a normal rotation, this works. In a white-hot sector move, the pullback never comes and the framework says "wait" indefinitely.

**Solution:** Restructure from pure swing to Core-Satellite, add a Velocity Flag to Layer 0, and recalibrate drawdown tiers to match expanded tolerance.

---

## Current Performance (as of May 23, 2026)

| Metric | Value |
|--------|-------|
| Account size | ~$71,111 |
| YTD realized P&L | +$767.93 |
| Open positions | 2 (VRT runner, NVDA) |
| Total closed trades | 17 |
| Win rate | 53% (9W / 8L) |
| Avg winner | +$427 / +20.3% |
| Avg loser | -$194 / -5.1% |
| Reward:Risk ratio | 2.2:1 |
| Expectancy per trade | +$45 |
| Annualized return (projected) | ~9% |
| Capital utilization (GREEN state) | ~15% |
| Framework-only trades (post Apr 2026) | 0 losses |

**Key finding:** Strip out VRT's $1,485 partial and the remaining 16 trades are roughly flat. The system produces outsized winners when it catches them, but catches too few.

---

## Approved Changes

### 1. Core-Satellite Structure

**Core Allocation: 40% (~$28K)**
- 2-3 regime-aligned sector ETFs
- Entry: at market when Layer 0 confirms regime + Layer 1.5 confirms Phase 2
- No breakout or volume threshold required for Core entries
- Stop: 20d MA (same as current ETF positions)
- Hold: as long as regime supports — no fixed max hold
- Exit: close below 20d MA on volume, regime goes RED, or RS declining for 2 consecutive weeks

**Tactical Allocation: 60% (~$43K)**
- Current swing framework (Layers 2-6) for individual stocks
- Entry criteria unchanged in normal conditions
- Modified entry criteria when Velocity Flag is active (see below)

**Rules:**
- Core positions count toward portfolio heat on equal footing
- In RED state, Core exits to cash with everything else
- Core ETFs are selected from Layer 1.5 Phase 2 Confirmed sectors only
- Maximum 3 Core ETF positions simultaneously
- Each Core ETF max 15% of account (~$10.7K)

### 2. Velocity Flag (Layer 0 Addition)

**Trigger:** Sector ETF ROC 21 > +15% = "Accelerating" flag for that sector.

**When Accelerating flag is active for a sector:**
- Individual stock entries in that sector allowed up to 12-15% above 20d MA (vs normal ~6% effective ceiling)
- Stop moves to 10-day EMA (tighter, faster-responding) instead of base low
- Half position size (risk dollars stay constant despite wider entry)
- 6-week max hold instead of 12 weeks
- Volume threshold reduced to 1.0x average (sustained elevated volume in hot sectors)

**When flag deactivates (ROC 21 drops below +15%):**
- No new Accelerating entries
- Existing positions managed normally under Layer 6 rules
- Stop reverts to 20d MA after first partial taken

### 3. Drawdown Tiers Recalibrated

| Tier | Old Trigger | New Trigger | Protocol |
|------|-------------|-------------|----------|
| Tier 1 | -5% ($3,556) | -7% ($4,978) | Cut risk/trade 50%. No new tactical entries. Core holds if stops intact. |
| Tier 2 | -10% ($7,111) | -10% ($7,111) | Max 3 positions including Core. Exit all tactical. |
| Tier 3 | -15% ($10,667) | -15% ($10,667) | 100% cash. Exit everything. Wait for confirmed GREEN. |

**Rationale:** Emotional drawdown tolerance expanded from $3-5K to $7-10K to support higher capital deployment. Old Tier 1 at -5% would trigger too early with 40% Core deployment during normal market pullbacks.

### 4. Minimum Deployment Floor

- In GREEN state: target 40-60% deployed (Core + Tactical combined)
- If no tactical entries meet criteria, Core fills the floor via sector ETFs
- In YELLOW state: target 20-35% deployed
- In RED state: no floor — cash preservation overrides

---

## Expected Impact

**Conservative scenario (Core captures 60% of sector move, 1-2 tactical swings/month):**
- Core: ~$28K × 15% annual sector return × 60% capture = ~$2,520
- Tactical: current trajectory ~$6,700 annualized
- Combined: ~$9,200 = ~13% return

**Base scenario (Core captures 75%, velocity flag catches 1 hot sector/year):**
- Core: ~$28K × 18% × 75% = ~$3,780
- Tactical with velocity entries: ~$10,000
- Combined: ~$13,780 = ~19% return

**Optimistic scenario (Core + velocity + clean VRT-style runner):**
- Core: ~$4,500
- Tactical + velocity + runner: ~$15,000+
- Combined: ~$19,500+ = ~27%+ return

35% requires catching a major sector rotation early AND a VRT-caliber individual stock move in the same year.

---

## Documents to Update

1. `swing_trading_framework_v3.md` → v4
2. `weekly_review_prompt.md` — add Core checks, Velocity Flag, deployment floor
3. `playbook_template.py` — add Core position tab
4. `trade_journal.md` — add Core entry/exit template
5. Layer 0 dashboard artifact — add Velocity Flag and Core status

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| 40% Core allocation | Balances deployment with drawdown tolerance. At 40%, a 10% sector correction = ~$2,800 loss on Core — within expanded $7K Tier 1. |
| 2-3 ETFs max for Core | Concentrated enough to capture sector moves, diversified enough to avoid single-sector blowup. |
| Velocity threshold at ROC 21 > 15% | Historical analysis: sectors that move >15% in 21 days are in the top 5% of momentum. Below this, normal entry rules suffice. |
| Half position size on velocity entries | Compensates for extended entry risk. Keeps dollar risk per trade constant. |
| Tier 1 moved to -7% | Old -5% threshold would trigger during normal Core position fluctuations. -7% gives Core room to breathe while still protecting capital. |
| Weekly cadence maintained | Swing trading doesn't require daily monitoring. GTC orders handle execution between reviews. |
