# Swing Trading Weekly Workflow
**Framework:** v3.0
**Last updated:** 2026-04-30

---

## Before You Start
- This workflow runs every week, ideally Sunday morning
- Never skip a layer — each one gates the next
- The screener dashboard auto-runs at 7 AM Sunday (cron job) or run manually via **Run Screener.command**
- Claude reads `positions_tracker.md` directly — no attachments needed for the weekly review

---

## Layer 0 — Macro Regime Filter
*Goal: Identify the current Growth/Inflation/Liquidity environment and sector flow momentum*

**Claude auto-fetches:**
- **SPY trend**: 1-month and 6-month return both positive / mixed / both negative
- **Sector RS** — which ETFs are outperforming SPY this week?
  - XLE / XLB / XLI leading → Reflation (inflation rising)
  - XLK / XLY / XLF leading → Risk-on (growth rising, inflation falling)
  - XLU / XLP / XLV leading → Defensive / Deflation (growth falling)
  - Everything falling together → Stagflation / liquidity crisis
- **Sector flows** — ETF.com 1-week and 4-week net inflows by sector (feeds Layer 1.5)
- **TLT direction**: TLT falling + SPY rising = Reflation confirmed
- **Recession probability composite** (FRED: CFNAIMA3, SAHMREALTIME)
- **FactSet earnings revision direction** (weekly free report)

**Manual checks (you do these in TradingView/FRED):**
- HYG/IEF spread widening? Fed hiking or signaling hikes? → Liquidity override
- Fed Net Liquidity (jlb05013 indicator) — rising or falling over last 4 weeks?
- Chauvet-Piger (FRED: RECPROUSM156N) and Conference Board LEI — complete recession composite

---

## Layer 1 — Market Permission State
*Goal: Determine how aggressively to trade*

- **Green** (SPY 1M and 6M both positive): up to 20 positions, 0.75–1.0% risk/trade, momentum bias
- **Yellow** (mixed SPY signals): 8–12 positions, 0.25–0.5% risk/trade, selective entries only
- **Red** (both negative, or liquidity override): 3–5 positions max, no new entries

---

## Layer 1.5 — Sector Rotation ETF
*Goal: Identify sector rotations early and enter via ETF before individual stocks set up*

**Run after Layer 1, before running the screener.**

Check ETF.com and SSGA Sector Tracker for each sector:

| Question | Source | Signal |
|----------|--------|--------|
| Which sectors have 1-week inflows? | ETF.com | Directional momentum |
| Which sectors have 4-week inflows? | ETF.com | Sustained rotation |
| Which sectors lead on 1M and 3M price return? | SSGA Sector Tracker | Price confirming flow |
| Is the sector ETF above its 20d MA? | TradingView | Entry validity |
| Is the RS line inflecting up vs SPY? | TradingView | Trend confirmation |

**Sizing decision:**

| Flow Strength | What You See | Size | Risk % (Green) |
|--------------|-------------|------|----------------|
| Weak | 1 week of inflows, RS just turning | Quarter | 0.1875% |
| Moderate | 1–2 weeks consistent inflows, RS inflecting | Half | 0.375% |
| Strong | 2+ weeks accelerating inflows, RS at new highs | Full | 0.75% |

Stop is always the 20d MA of the sector ETF. If the ETF closes below it on volume, exit.

**For existing ETF positions — check weekly:**
- Any 2 consecutive weeks of outflows? → Exit
- 20d MA still holding? → If not, on volume: exit
- RS line declining 2+ weeks? → Exit
- Individual names from this sector entering via Layer 3? → Reduce ETF proportionally (Phase 3)

**Regime mismatch:** If a sector has strong inflows but conflicts with the G/I/L regime, flag it as an anomaly. Evaluate whether the regime classification is correct before acting. Stick to the regime unless there is overwhelming one-off evidence.

ETF positions count toward total position count and portfolio heat on equal footing with individual stocks. Sector cap: ETF + individual names in same sector ≤ 20% of account.

---

## Layer 2 — Run the Screener
*Goal: Generate individual stock candidates filtered to leading sectors*

- Double-click **Run Screener.command** in your Screener folder
- Enter the sectors showing RS leadership (from Layer 0 review) — e.g. `1,2,3`
- Enter the permission state (Green / Yellow / Red)
- Wait ~60–90 seconds for data to download
- Dashboard opens automatically in your browser

**Review the dashboard:**
- **Full Signal candidates** — stocks passing all Layer 2 filters (prioritize these)
- **Half Signal watch list** — mixed two-speed trend (monitor, not yet actionable)
- Check RSI on full signal candidates — avoid entries where RSI > 75 (extended)
- Check MACD direction — look for MACD crossing above signal line (histogram turning green). Absolute level (above/below zero) is not the signal.

**If an active Layer 1.5 ETF position exists in a sector:** Flag individual names from that sector as Phase 3 transition priorities. As they set up and enter, reduce the ETF proportionally.

---

## Layer 3 — Entry Trigger
*Goal: Find the specific entry point for each individual stock candidate*

- Open each full signal candidate in **TradingView**
- **Momentum/Green regime**: look for breakout from base on 40%+ above-average volume
- **Mean-reversion/Yellow regime**: look for pullback to 20d or 50d MA with reversal candle on declining volume
- No earnings within 10 business days
- Entry price defines your initial stop — do not enter without a clear stop level

Note: ETF entries (Layer 1.5) do not use this trigger. ETF entries are determined by flow momentum strength and 20d MA position.

---

## Layer 4 — Position Sizing
*Goal: Size each position using the expected-loss method*

- **Formula**: Shares = (Account × Risk%) ÷ (Entry − Stop)
- Risk% is set by permission state (Layer 1) and, for ETFs, by flow momentum strength (Layer 1.5)
- Max single position: 10% of account
- In drawdown: reduce risk% proportionally per Layer 7 tiers

---

## Layer 5 — Portfolio Exposure Check
*Goal: Confirm total portfolio heat is within limits before entering*

- Count all open positions (individual stocks AND ETF positions from Layer 1.5)
- Check total portfolio heat (sum of all open risk — stocks + ETFs)
  - Green: max 15% total heat
  - Yellow: max 8% total heat
  - Red: max 3% total heat
- Sector cap: ETF + individual names in same sector ≤ 20% of account
- If adding the new position would breach any limit → skip or reduce size

---

## Layer 6 — Trade Management (ongoing)
*Goal: Manage open positions systematically*

**Individual stocks:**
- Move stop to **breakeven** at +5%
- Sell **1/3 at +8–12%**, move stop to breakeven
- Sell **another 1/3 at +20–25%**, trail remainder at 20d MA
- **Exit immediately if:**
  - Close below 20d MA on volume
  - RS line declining
  - Permission state goes Red
  - 12-week max hold reached with insufficient progress

**ETF positions (Layer 1.5):**
- Trail stop at 20d MA — no partial profit rules
- **Exit if:** 2 consecutive weeks outflows / close below 20d MA on volume / RS line declining 2+ weeks
- **Reduce proportionally** as individual stocks from the same sector are entered

---

## Layer 7 — Hard Risk Caps (check weekly)
*Goal: Enforce portfolio-level drawdown limits*

- **5–10% portfolio drawdown**: cut risk/trade by 50%, no new positions (stocks or ETFs)
- **10–15% drawdown**: 3–5 positions max, no new entries of any type
- **15%+ drawdown**: 100% cash — wait for confirmed Green state before re-engaging
- **Hedges during tightening liquidity**: cash, SH/PSQ inverse ETF (10–20%), TLT (deflationary regime)

---

## End of Week Checklist

- [ ] Layer 0 macro regime confirmed — SPY trend, sector RS, sector flows, TLT, liquidity
- [ ] Permission state set (Layer 1)
- [ ] Layer 1.5 ETF scan complete — any new rotation opportunities? Any existing positions to exit?
- [ ] Screener run with correct sectors (Layer 2)
- [ ] All full signal candidates reviewed in TradingView (Layer 3)
- [ ] Open individual stock positions checked against Layer 6 rules
- [ ] Open ETF positions checked for flow reversal / 20d MA breach / Phase 3 transition
- [ ] Portfolio heat within Layer 5 limits (stocks + ETFs combined)
- [ ] No hard risk cap breaches (Layer 7)
- [ ] Tracker file updated with new Layer 0 and Layer 1.5 snapshots
- [ ] Trade journal updated with any closed trades
- [ ] Screening log entry added

---

*Framework v3.0 — Prometheus Research methodology (prometheus-research.com)*
