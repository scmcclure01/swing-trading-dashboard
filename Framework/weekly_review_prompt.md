# Weekly Review Prompt
*Framework v4.0 — Updated to include Core-Satellite Structure, Velocity Flag, and Deployment Floor*

Copy and paste this into any chat in the project to trigger a standardized weekly review.

---

## The prompt

```
Run weekly review. Read the most recent playbook xlsx from the Playbooks/ folder and screening_log.md from the workspace folder.

Execute in order:

1. LAYER 0 — pull current data via web search:
   - SPY 1-month and 6-month return
   - Top 3 sectors by 4-week RS vs SPY
   - TLT direction (4wk)
   - HYG/IEF spread direction
   - Fed Net Liquidity 4-week change (WALCL - WTREGEN - RRPONTSYD from FRED)
   - CFNAIMA3 latest reading
   - Sahm Rule latest reading
   - T10Y3M spread
   - FactSet forward EPS revision direction (latest Earnings Insight)
   - Crude oil CL1-CL2 term structure (for energy signal)

2. VELOCITY FLAG [v4] — calculate ROC 21 for each primary sector ETF:
   - XLK, XLE, XLI, XLB, XLF, XLY, XLU, XLP, SMH
   - Flag any sector with ROC 21 > +15% as ACCELERATING
   - Note which sectors are in the Accelerating zone

3. Score the recession composite (__/5 models in recession territory).

4. Identify the current macro regime and permission state (Layer 1).

5. Flag any liquidity override conditions.

6. CORE ALLOCATION CHECK [v4]:
   a. List current Core ETF positions (from playbook)
   b. Calculate Core % deployed vs target (GREEN=40%, YELLOW=20%, RED=0%)
   c. Are any sectors newly Phase 2 Confirmed? → Candidate for Core entry
   d. Do any existing Core positions need exit? (20d MA violation, RS declining, regime shift)
   e. Is total deployed capital (Core + Tactical) above the deployment floor?
      - GREEN floor: 40-60%
      - YELLOW floor: 20-35%
      - If below floor: recommend Core ETF entries to fill

7. LAYER 1.5 — Sector Rotation ETF review:
   a. Pull top 3 sectors by 1-week and 4-week net inflows from ETF.com fund flows tool
   b. Cross-reference with SSGA Sector Tracker for 1M and 3M price performance rank
   c. For each sector with consistent inflows:
      - Is it regime-aligned?
      - Is the sector ETF above its 20d MA?
      - How strong is the flow signal? (1 week = weak / 2 weeks consistent = moderate / 2+ weeks accelerating = strong)
      - Score: no action / watch / quarter-size entry / half-size entry / full-size entry
   d. For any active ETF positions in the tracker:
      - Check ETF.com for flow reversal (2 consecutive weeks outflows?)
      - Check 20d MA status
      - Check RS line vs SPY (declining 2+ weeks?)
      - Any individual stocks from that sector now entering (Phase 3 transition)?
   e. Flag any regime mismatch anomalies (inflows into non-regime-aligned sector)

8. LAYER 2 — For each OPEN POSITION in the playbook:
   - Check if first target (+8-12%), second target (+20-25%), or stop has been hit
   - Check if close below 20d MA on volume
   - Check if RS line is declining
   - Flag any action needed
   - For Accelerating Protocol positions [v4]: check 10d EMA stop and 6-week max hold

9. For each WATCHLIST candidate:
   - Re-score 2-speed trend signal (ROC 21 and ROC 63)
   - Confirm still aligned with current regime
   - Note any new entry trigger
   - If Velocity Flag active for this sector [v4]: evaluate under Accelerating Protocol
     (entries up to 12-15% above 20d MA, half size, 10d EMA stop, 6-week hold)
   - For stocks in sectors with active Core or Layer 1.5 ETF positions: flag as priority

10. Output an UPDATED playbook with:
    - New Last Updated date
    - New Layer 0 Weekly Snapshot filled in (including Velocity Flag status)
    - New Layer 1.5 Snapshot filled in (flow readings, ETF position status, any new opportunities)
    - Core Allocation status [v4] (positions, % deployed, vs floor)
    - Updated permission state and composite scores
    - Updated Week's Rules
    - Deployment floor check [v4]
    - Action items for the week

Be direct. No background. Just the readings, the state, and what to do.
```

---

## How to use it

**Weekly cadence (Sunday or Monday morning):**

1. Open a chat in this project
2. Paste the prompt above — no attachment needed, Claude reads the playbook directly
3. Claude runs the full Layer 0 + Velocity Flag + Core Check + Layer 1.5 checklist via web search, updates the playbook, and flags all position and ETF actions
4. Save the output as the new weekly playbook xlsx in the Playbooks/ folder

**Monthly cadence (after CPI or PCE release):**

Add this to the end of the prompt:

```
Also run LAYER 0.5 monthly mispricing check:
- Forward earnings yield vs 10Y TIPS (DFII10) spread
- Equity risk premium vs 10Y nominal (DGS10)
- GDPNow direction vs FactSet consensus EPS direction
- Taylor Rule gap from Atlanta Fed calculator
- Score the mispricing composite and flag sizing adjustments
```

**Per-trade cadence — individual stock (any time a setup appears):**

```
Evaluate [TICKER] through the framework.

Current permission state: [from last weekly review]
Current regime: [from last weekly review]
Velocity Flag active for this sector? [yes/no — check ROC 21]
Any active Core or Layer 1.5 ETF in this sector? [yes/no — from playbook]

Run Layers 2, 3, 4 and tell me: trade, half-size, or pass.
If Velocity Flag active: evaluate under Accelerating Protocol (half size, 10d EMA stop, 6-week hold).
If trade: entry trigger, stop, shares for $X account risk, targets.
If Core active in this sector: note — Core persists, no transition needed.
If Layer 1.5 ETF active in this sector: note Phase 3 transition plan.
```

**Per-trade cadence — Core ETF entry [v4]:**

```
Core ETF evaluation for [SECTOR / ETF ticker].

Current permission state: [from last weekly review]
Current regime: [from last weekly review]
Phase status: [Phase 1 Early / Phase 2 Confirmed]
Current Core deployment: [__% of target]

Is this sector Phase 2 Confirmed and regime-aligned?
If yes: entry at market, stop at 20d MA, size up to 15% of account.
Check deployment floor status after adding this position.
```

**Per-trade cadence — sector ETF (Layer 1.5):**

```
Layer 1.5 evaluation for [SECTOR / ETF ticker].

Current permission state: [from last weekly review]
Current regime: [from last weekly review]
Flow signal: [1-week inflow direction from ETF.com] / [4-week inflow direction]
ETF price vs 20d MA: [above / at / below]
RS line vs SPY: [inflecting up / making new highs / flat / declining]

Score the flow momentum: weak / moderate / strong.
Recommend: no action / watch / quarter-size / half-size / full-size entry.
If entry: ETF ticker, approximate entry, stop (20d MA level), position size for $X account.
Note: if sector reaches Phase 2, evaluate for Core allocation upgrade.
```

---

## Data Claude can auto-fetch via web search

- SPY 1-month and 6-month returns
- FRED: Sahm Rule (SAHMREALTIME), CFNAI 3-mo MA (CFNAIMA3), T10Y3M yield curve
- TLT 4-week direction
- FactSet EPS revision direction (insight.factset.com)
- ETF.com sector fund flows — 1-week and 4-week net inflows/outflows
- SSGA Sector Tracker — 1M and 3M sector price performance
- Sector ETF RS vs SPY (approximate, from recent price data)
- Sector ETF ROC 21 for Velocity Flag calculation [v4] (via yfinance in bash)

## Data requiring manual check (Claude cannot access)

- Fed Net Liquidity (TradingView jlb05013 indicator)
- HYG/IEF ratio 4-week direction (TradingView)
- Chauvet-Piger recession probability (FRED: RECPROUSM156N — sometimes accessible)
- Conference Board LEI trend (conference-board.org — paywalled)
- Sector ETF 20d MA exact level for Layer 1.5 and Core entry (TradingView — can approximate via yfinance)

---

## Notes

- Takes 30-60 seconds from your end. Claude handles the rest.
- If any data source is down or paywalled, Claude will flag it and use the most recent available reading.
- Layer 1.5 flow data from ETF.com may have a 1-day lag — use directional signal, not precise numbers.
- Position data comes from your playbook file — Claude cannot access your brokerage account.
- Velocity Flag ROC 21 is calculated automatically via yfinance — no manual input needed [v4].
- Core allocation check runs automatically as part of every weekly review [v4].
