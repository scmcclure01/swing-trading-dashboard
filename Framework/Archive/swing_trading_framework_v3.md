**PROMETHEUS-BASED SWING TRADING FRAMEWORK**

A Complete Systematic Rules-Based Methodology — Version 3.0

Hold Period: 1–12 Weeks  |  Platform: TradingView  |  Universe: All US Equities

*Incorporating: Prometheus Research + Zero-Cost Public Data Enhancements + Sector Rotation ETF Layer*

| **Version 3.0 — What's New** |
| --- |
| This document supersedes v2.0. All v2.0 content is preserved; v3.0 adds one new layer. |
| • Layer 1.5: NEW — Sector Rotation ETF (momentum-based early entry into rotating sectors) |
| The framework now consists of 9 layers plus the monthly mispricing check. |
| Items marked [v3] throughout this document indicate Version 3.0 additions. |
| Items marked [v2] indicate Version 2.0 additions still in effect. |

# **Framework Overview**

This framework is a systematic, rules-based swing trading methodology built on Prometheus Research (prometheus-research.com), incorporating their Basic Trend Program, G/I/L macro regime framework, expected-loss risk management, and sector rotation research — augmented with zero-cost public data signals from FRED, CFTC, FactSet, and the Atlanta Fed.

The framework consists of 9 layers plus one monthly layer, applied in sequence. Every trade decision flows through each layer in order. No layer can be skipped.

| **The 9 Layers + Monthly Check** |
| --- |
| Layer 0   — Macro Regime Filter (G/I/L + recession probability + liquidity + earnings revisions) |
| Layer 0.5 — Monthly Macro Mispricing Check |
| Layer 1   — Market Permission State (green / yellow / red) |
| Layer 1.5 — Sector Rotation ETF [v3] |
| Layer 2   — Stock Selection (sector bias, RS, two-speed trend, carry signal) |
| Layer 3   — Entry Trigger (breakout or MA pullback with volume) |
| Layer 4   — Position Sizing (expected-loss method, inverse-vol weighting) |
| Layer 5   — Dynamic Portfolio Exposure (breadth-scaled positions) |
| Layer 6   — Trade Management (stops, partials, trailing) |
| Layer 7   — Hard Risk Caps (drawdown thresholds, emergency protocols) |

# **Layer 0 — Macro Regime Filter**

## **Purpose**

The macro regime filter is the top-level permission system. Before looking at any stock or sector, you must understand the macroeconomic environment. This determines: (1) whether to trade at all, (2) which sectors to favor, and (3) whether momentum or mean-reversion will dominate.

## **The Three Pillars: Growth, Inflation, Liquidity**

| **Force** | **What It Measures** |
| --- | --- |
| Growth (G) | Economy expanding or contracting? Track via PMI, earnings trends, retail sales, industrial production. Rising = risk-on. |
| Inflation (I) | Inflation rising or falling? Track via CPI trends, commodity prices, TIPS breakevens. Rising = sector rotation. |
| Liquidity (L) | Credit expanding or contracting? Primarily Fed policy. Tightening = primary driver of large drawdowns across ALL asset classes. |

## **The Four Regimes + The Liquidity Override**

| **Regime** | **Growth** | **Inflation** | **Stocks / Sectors** |
| --- | --- | --- | --- |
| Risk-On | Rising | Falling | Strong uptrends. Full exposure. Tech, Discretionary, Financials lead. |
| Reflation | Rising | Rising | Stocks up, rotation underway. Energy, Materials, Industrials lead. |
| Deflation | Falling | Falling | Stocks weak. Reduce exposure. Staples, Utilities defensive. |
| Stagflation | Falling | Rising | Worst regime. Cash and commodities only. |

| **LIQUIDITY OVERRIDE — The Fifth State** |
| --- |
| When the Fed is actively hiking rates OR credit spreads are widening rapidly, ALL four regimes become irrelevant. |
| Tightening liquidity = single state: REDUCE EXPOSURE. Overrides every other signal. |
| Prometheus: 'The primary driver of large drawdowns comes from shocks in liquidity conditions.' |

## **Signal 1: SPY Two-Speed Trend (Market Permission)**

| **Condition** | **State** |
| --- | --- |
| 1-month AND 6-month return both positive | GREEN — full engagement |
| 1-month and 6-month mixed (one positive, one negative) | YELLOW — reduced engagement |
| 1-month AND 6-month both negative | RED — minimal exposure / cash |

## **Signal 2: Sector Relative Strength (Regime Flavor)**

| **What is outperforming SPY** | **Regime signal** |
| --- | --- |
| XLK (Tech), XLY (Discretionary), XLF (Financials) | Risk-on — growth rising, inflation falling |
| XLE (Energy), XLB (Materials), XLI (Industrials) | Reflation — inflation rising |
| XLU (Utilities), XLP (Consumer Staples) | Defensive — growth falling, reduce exposure |
| Everything falling together | Stagflation / tightening liquidity override |

## **Signal 3: Bond Market (Regime Confirmation)**

| **TLT vs SPY behavior** | **Regime implication** |
| --- | --- |
| TLT rising + SPY rising | Risk-on confirmed |
| TLT falling + SPY rising | Reflation regime |
| TLT rising + SPY falling | Deflationary recession — reduce significantly |
| TLT falling + SPY falling | Stagflation / liquidity crisis — emergency reduce |

## **Signal 4: Liquidity Check (The Override)**

- Is the Fed hiking or signaling hikes? If yes: defensive posture regardless of other signals.
- Is the yield curve inverted and steepening from inversion? Late-cycle warning.
- Are credit spreads (HYG vs IEF ratio) widening rapidly? Cut exposure immediately.
- Are recession probability indicators above 30%? Mean-reversion traps active; tighten stops.

## **Signal 5: Recession Probability Composite [v2]**

Check these five free models monthly. Count how many are in recession territory.

| **Model / Source** | **How to Read It** |
| --- | --- |
| Chauvet-Piger Smoothed Probability — FRED: RECPROUSM156N | Continuous 0-100%. Above 50% = recession signal. |
| Sahm Rule — FRED: SAHMREALTIME | Triggered when reading >= 0.50. Fast labor-market signal. |
| CFNAI 3-Month MA — FRED: CFNAIMA3 | Below -0.70 = recession signal. 85 underlying indicators. |
| 10Y-3M Yield Curve — FRED: T10Y3M (TradingView) | Inversion = 12-18 month forward warning. Watch for un-inversion. |
| Conference Board LEI — conference-board.org | Six months declining with diffusion below 50 = warning. |

| **Recession Composite Scoring** |
| --- |
| 0-1 models in recession territory   → Normal operations, full risk |
| 2-3 models in recession territory   → Caution — reduce max positions 25%, tighten stops |
| 4-5 models in recession territory   → Defensive — treat like yellow/red permission state regardless of SPY trend |
| Rule: When composite is 4+ models, do NOT buy breakouts. Pullback entries only — bounces often fail in recessions. |

## **Signal 6: Fed Net Liquidity Trend [v2]**

Track weekly. Formula: Fed Balance Sheet minus Treasury General Account minus Reverse Repo Facility.

| **Fed Net Liquidity — Setup** |
| --- |
| TradingView: Search 'Fed Net Liquidity' by jlb05013 or vbarink — free community indicator |
| Formula: WALCL - WTREGEN - RRPONTSYD (all available on FRED) |
| Update cadence: Every Thursday at 4:30 PM ET (H.4.1 release) |
| Signal: Rising trend = liquidity tailwind (risk-on). Declining trend = tightening = activate liquidity override. |
| Threshold: If Net Liquidity has declined >$200B over 4 weeks, treat as liquidity override active. |

## **Signal 7: FactSet Earnings Revision Direction [v2]**

Check every Friday. FactSet publishes 'Earnings Insight' free at insight.factset.com.

| **Revision Direction** | **Action** |
| --- | --- |
| Estimates revised UP 3+ weeks in a row | Confirms risk-on bias. Full exposure appropriate. |
| Estimates flat (within +/- 0.5%) | Neutral — no adjustment to framework needed. |
| Estimates revised DOWN 2+ weeks in a row | Reduce conviction on new longs. Tighten stops on existing positions. |
| Estimates revised DOWN sharply (>2% in one week) | Treat as yellow flag — reduce max positions immediately. |

## **Signal 8: Taylor Rule Deviation [v2]**

Check monthly after CPI and PCE releases. Use the Atlanta Fed's free Taylor Rule calculator at atlantafed.org/cqer/research/taylor-rule.

| **Taylor Rule Gap (Implied Rate vs Actual Rate)** | **Interpretation** |
| --- | --- |
| Positive gap (implied > actual by >1%) | Fed is too loose — pressure to hike. Hawkish surprise risk. Reduce rate-sensitive longs. |
| Near zero (-1% to +1%) | Fed approximately on target. Neutral. |
| Negative gap (implied < actual by >1%) | Fed is too tight — pressure to cut. Dovish surprise potential. Favors risk-on positioning. |

## **Signal 9: Sector Flow Momentum [v3]**

Check weekly alongside sector RS. This feeds directly into Layer 1.5 — if sustained inflows are detected, Layer 1.5 evaluates the ETF entry opportunity.

| **Source** | **What to Check** |
| --- | --- |
| ETF.com Fund Flows Tool (etf.com/etfanalytics/etf-fund-flows-tool) | 1-week and 4-week net inflows by sector ETF. Look for directional consistency, not size. |
| SSGA Sector Tracker (ssga.com/us/en/intermediary/resources/sector-tracker) | 1M and 3M price performance rank. Confirms which sectors are leading in price. |

What to look for: 1–2 weeks of directional inflow consistency into a sector ETF that is regime-aligned. Single-day spikes are noise. Sustained directional flow triggers Layer 1.5 evaluation.

## **How Regime Changes Your Trading Style**

| **Regime** | **Trading approach** |
| --- | --- |
| Risk-on | Buy breakouts aggressively. Hold winners longer. Full exposure. Momentum dominates. |
| Reflation | Momentum works in energy/commodities, may reverse in tech. Tighten tech stops. Rotate to commodity/industrial. |
| Deflationary / recession risk | Mean-reversion traps everywhere. Bounces fail. Smaller positions, tighter stops. Do not chase. |
| Stagflation / tightening liquidity | Almost nothing works on the long side. Cash is a position. 3-5 strongest names only or move to cash. |

## **Practical Weekly Layer 0 Checklist**

Run every Sunday or Monday morning.

| **Step** | **Action** |
| --- | --- |
| 1. SPY trend check | Calculate 21-day and 126-day price return. Both positive = green. Mixed = yellow. Both negative = red. |
| 2. Sector RS check | Which sector ETFs outperformed SPY last 4 weeks? Sets regime flavor and sector tilt. |
| 3. Sector flow check [v3] | ETF.com: which sectors have 1-week and 4-week net inflows? Note directional consistency. |
| 4. Bond confirmation | TLT direction vs SPY. Confirms or challenges regime reading. |
| 5. HYG/IEF spread | Falling ratio = credit spreads widening = liquidity tightening override. |
| 6. Fed Net Liquidity trend [v2] | TradingView indicator. Rising or falling over last 4 weeks? Declining >$200B = override active. |
| 7. Recession composite [v2] | Check CFNAIMA3 and SAHMREALTIME on FRED. How many of 5 models are in recession territory? |
| 8. FactSet earnings revisions [v2] | insight.factset.com — Earnings Insight PDF. Up or down trend in forward EPS estimates? |
| 9. Set week's rules | Based on 1-8: set max positions, sector tilt, ETF opportunity (Layer 1.5), entry style bias, stop tightness. |

Monthly additions (after CPI/PCE): Taylor Rule gap check + Macro Mispricing Check (Layer 0.5).

## **TradingView Macro Watchlist**

| **Symbol** | **Purpose** |
| --- | --- |
| SPY | Primary market trend — two-speed signal |
| QQQ | Growth/tech regime indicator |
| TLT | Bond market regime confirmation |
| HYG | Credit spreads / liquidity indicator |
| IEF | Use HYG/IEF ratio for spread tracking |
| GLD | Inflation hedge / risk-off |
| XLE | Energy — inflation regime leader |
| XLK | Technology — risk-on leader |
| XLB | Materials — reflation |
| XLI | Industrials — growth |
| XLU | Utilities — defensive |
| XLP | Consumer Staples — defensive |
| XLY | Consumer Discretionary — risk-on |
| XLF | Financials — risk-on / yield curve |
| SMH | Semiconductors — risk-on sub-sector [v3] |
| FRED:CFNAIMA3 | Business cycle gauge — below -0.70 = recession signal |
| FRED:SAHMREALTIME | Sahm Rule — above 0.50 = triggered |
| FRED:T10Y3M | Yield curve spread — inversion = warning |
| CL1!-CL2! | Crude oil term structure — backwardation = risk premium |

# **Layer 0.5 — Monthly Macro Mispricing Check**

## **Purpose**

Once a month (after CPI and PCE releases), run a valuation check to assess whether equities are rich or cheap relative to real interest rates and macro fundamentals. This check directly modifies position sizing across the entire portfolio when equities are materially overpriced vs. macro conditions.

## **The Three Mispricing Checks**

### **Check 1: Earnings Yield vs. Real Rates Spread**

Measure: Forward S&P 500 earnings yield minus 10-Year TIPS yield (FRED: DFII10).

| **Spread Reading** | **Action** |
| --- | --- |
| Above 4% (equities cheap vs real rates) | No sizing adjustment. Macro environment supports full risk. |
| 2-4% (equities fairly valued) | Normal operations. |
| Below 2% (equities expensive vs real rates) | Reduce max risk per trade by 25%. |
| Below 1% or negative (equities very expensive) | Reduce max risk per trade by 50%. Only highest-conviction setups. |

### **Check 2: Earnings Revision Gap vs. Macro-Implied EPS**

- GDP growth rate: Atlanta Fed GDPNow (atlantafed.org)
- If GDPNow is declining sharply AND analysts have not yet revised down EPS: bearish mispricing — tighten stops
- If GDPNow is accelerating AND analysts are still at depressed estimates: bullish mispricing — hold winners longer

### **Check 3: Equity Risk Premium**

Forward earnings yield minus 10-year nominal Treasury yield (FRED: DGS10).

| **ERP Level** | **Sizing Rule** |
| --- | --- |
| Above 5% | Equities cheap. Take full positions. |
| 3-5% | Normal range. No adjustment. |
| 2-3% | Elevated. Reduce max position size 20%. |
| Below 2% | Stretched. Reduce max position size 40%. Only the strongest setups. |

| **Monthly Mispricing Composite** |
| --- |
| Score each of the three checks: Bullish (+1), Neutral (0), Bearish (-1) |
| Composite +2 to +3: Full risk. |
| Composite -1 to +1: Normal operations. |
| Composite -2 to -3: Reduce all position sizes by 30-50%. Only top-tier setups. Tighten all trailing stops. |

# **Layer 1 — Market Permission State**

## **Purpose**

Layer 1 translates macro regime analysis from Layer 0 into a single actionable permission state: green, yellow, or red. This state governs the maximum number of positions, how aggressively you can size, and whether momentum or mean-reversion setups are preferred.

The permission state is a hard constraint. If red, you do not go looking for stock or ETF setups regardless of how attractive an individual chart looks.

After setting the permission state, proceed to Layer 1.5 to evaluate sector rotation ETF opportunity before individual stock screening.

| **GREEN — Full Engagement** |
| --- |
| Condition: SPY both signals positive + no liquidity override + recession composite below 2/5 |
| Maximum positions: up to 20 (including any ETF positions from Layer 1.5) |
| Position sizing: 0.75-1.0% account risk per trade |
| Preferred setups: Momentum breakouts, trend continuation |
| Trailing stops: Loose — give trends room to run |

| **YELLOW — Reduced Engagement** |
| --- |
| Condition: SPY mixed signals OR recession composite 2-3/5 OR earnings revisions declining |
| Maximum positions: 8-12 (including any ETF positions from Layer 1.5) |
| Position sizing: 0.25-0.5% account risk per trade |
| Preferred setups: Both momentum and mean-reversion, be selective |
| Trailing stops: Tighter — less room for error |

| **RED — Minimal Exposure / Cash** |
| --- |
| Condition: SPY both negative OR liquidity override OR recession composite 4-5/5 |
| Maximum positions: 3-5 (strongest only) or 100% cash |
| Position sizing: Minimum or no new positions — no new ETF entries via Layer 1.5 |
| Action: Manage existing positions toward exits, do not add |

## **Transition Rules**

- Green to Yellow: SPY 1-month turns negative while 6-month remains positive, OR recession composite moves to 2-3/5.
- Yellow to Red: SPY 6-month turns negative, OR liquidity override triggered, OR recession composite reaches 4+/5.
- Red to Yellow: SPY 1-month turns positive AND recession composite falls below 3/5. Light re-engagement with half-size positions.
- Yellow to Green: SPY 6-month turns positive AND recession composite below 2/5. Resume full engagement.

The liquidity override can jump you from green directly to red in a single week.

# **Layer 1.5 — Sector Rotation ETF [v3]**

## **Purpose**

Sector money rotation precedes individual stock setups by 2–8 weeks. Institutional capital moves into sector ETFs first — the easiest, fastest way to get broad exposure. Individual stocks form bases and trigger clean entries only after the rotation is underway.

Layer 1.5 catches the rotation early via the sector ETF, puts capital to work, and transitions to individual names via Layer 2 as setups mature. This is momentum trading. The signal is the flow of money. The exit is when the money leaves.

## **Data Sources**

| **Source** | **What It Provides** | **URL** |
| --- | --- | --- |
| ETF.com Fund Flows Tool | Daily net inflows/outflows by ETF — primary flow signal | etf.com/etfanalytics/etf-fund-flows-tool |
| SSGA Sector Tracker | Price performance of sector ETFs (1M, 3M tabs) — confirms price leadership | ssga.com/us/en/intermediary/resources/sector-tracker |
| State Street Monthly Flash Flows | Monthly institutional flow summary — backward-looking confirmation | ssga.com (published ~month-end) |
| TradingView | Sector ETF RS line vs SPY, 20d MA position, volume | — |

Note: The SSGA Sector Tracker shows price performance, not flows. ETF.com is the primary flow signal.

## **The Signal Framework**

### **Phase 1 — Early Indication (Week 1–2)**

Conditions required:
- Sector ETF shows 1–2 weeks of directionally consistent positive net inflows on ETF.com
- Sector ETF price at or above its 20d MA (not extended — close to it)
- Regime alignment: current G/I/L regime favors this sector
- RS line of the sector ETF vs SPY is inflecting upward

The signal is momentum, not a specific dollar threshold. One-day spikes are noise (hedges, rebalancing). Sustained directional flow over 1–2 weeks is the signal.

**Action:** Enter sector ETF. Position size scales with flow momentum strength — see sizing below.

### **Phase 2 — Confirmation (Weeks 3–8)**

Confirmation signals:
- 3+ consecutive weeks of net inflows sustained
- Sector ETF RS line making new highs vs SPY
- Individual stocks in the sector: two-speed signals turning Full (ROC 21 and ROC 63 both positive)
- Volume expanding on up weeks vs down weeks

**Action:** Scale up ETF position if not already at full size, OR begin entering individual names via Layer 2/3 and reduce ETF proportionally.

### **Phase 3 — Transition to Individual Names**

As individual stocks clear Layer 2 and trigger Layer 3 entries, the ETF position becomes redundant. Each individual stock entered in the sector reduces the ETF by one slot-equivalent. Complete transition by Week 8–12 of the rotation. If individual setups never materialize despite ETF strength, keep the ETF and trail it.

## **Position Sizing — Flow Momentum Based**

Position size is determined by the strength and consistency of the flow signal at entry. Stop is always the 20d MA of the sector ETF.

| **Flow Signal Strength** | **Size** | **Risk %** |
| --- | --- | --- |
| Weak — 1 week of modest inflows, RS just turning | Quarter size | 0.1875% (Green state) |
| Moderate — 1–2 weeks consistent inflows, RS inflecting | Half size | 0.375% (Green state) |
| Strong — 2+ weeks accelerating inflows, RS at new highs | Full size | 0.75% (Green state) |

Scale risk % to permission state (Yellow state: use half of the above figures).

## **Portfolio Heat and Slot Rules**

- ETF positions count toward portfolio heat on equal footing with individual stock positions
- Each ETF position occupies one slot toward the maximum position count (20 in Green, 8-12 in Yellow)
- Sector cap: ETF position + individual stock positions in the same sector ≤ 20% of account at any time
- ETF dollar risk counts toward total portfolio heat percentage ceiling

## **Exit Rules**

Exits track the flow of money, not price targets.

**Primary exits (any one triggers):**
1. Flow reversal — 2 consecutive weeks of net outflows on ETF.com after a sustained inflow period
2. Price — close below 20d MA on above-average volume
3. RS line — sector ETF RS line vs SPY declines for 2+ consecutive weeks

**Secondary exits (tighten stop to breakeven, prepare to exit):**
4. Regime shift — G/I/L regime changes and no longer favors the sector
5. 12-week max hold — same rule as individual positions

## **Regime Mismatch Protocol**

When a sector receives strong sustained inflows but conflicts with the current G/I/L regime:

1. Flag it as an anomaly — do not ignore, do not chase automatically
2. Evaluate the regime first — is there a flaw in the current regime classification? Is the regime at a transition point?
3. Stick to the regime — do not override the regime to chase a mismatched sector unless there is overwhelming multi-source evidence of a structural one-off event (e.g., geopolitical supply shock)
4. Log the anomaly in the weekly review — if the mismatch persists and the regime framework continues to explain it away while the sector keeps running, that is a signal to re-examine the regime methodology, not to capitulate into the trade

The regime is the filter that keeps entries in high-probability setups. Chasing every flow signal regardless of regime turns this into noise-following.

## **Weekly Layer 1.5 Checklist**

| **Check** | **Source** | **Signal** |
| --- | --- | --- |
| Top 3 sectors by 1-week net inflows | ETF.com | Directional momentum — not size |
| Top 3 sectors by 4-week net inflows | ETF.com | Sustained vs. spike |
| Top 3 sectors by 1M and 3M price return | SSGA Sector Tracker | Price leadership confirming flow |
| Sector ETF RS line vs SPY | TradingView | Inflecting = early; new highs = confirmed |
| Sector ETF 20d MA position | TradingView | Above = valid; below = wait |
| Any sectors with 2+ weeks of outflows | ETF.com | Flag existing ETF positions for exit review |
| Regime mismatch anomalies | Cross-reference | Log and evaluate |

# **Layer 2 — Stock Selection**

## **Purpose**

Layer 2 identifies individual stock candidates from the sectors identified in Layer 0/1/1.5. If an ETF position was entered in Layer 1.5, stocks from that sector become the priority for Layer 2 screening — as they set up and are entered, the ETF position transitions out (Phase 3).

Only runs when the permission state is green or yellow.

## **Step 1: Apply Sector Bias from Macro Regime**

| **Macro regime** | **Priority sectors** |
| --- | --- |
| Risk-on (growth up, inflation down) | Technology (XLK), Consumer Discretionary (XLY), Financials (XLF), Industrials (XLI) |
| Reflation (growth up, inflation up) | Energy (XLE), Materials (XLB), Industrials (XLI), Homebuilders (XHB) |
| Deflationary (growth down, inflation down) | Healthcare (XLV), Consumer Staples (XLP), Utilities (XLU) — very selective |
| Stagflation (growth down, inflation up) | Energy (XLE), commodities — minimal long exposure overall |

If a Layer 1.5 ETF position is active, prioritize individual names within that sector for screening.

## **Step 2: Relative Strength Screening**

- RS line (stock / SPY) should be in an uptrend, ideally making 52-week highs alongside price
- Avoid stocks where RS line is declining even if price is rising
- Price above both the 20-day and 50-day moving averages
- Average dollar volume at least 5x your intended position size
- No earnings announcement within 10 business days (unless intentional)
- No recent gap-downs or heavy distribution days in last 3 weeks

## **Step 3: Two-Speed Trend Signal**

| **Signal condition** | **Action** |
| --- | --- |
| 1-month AND 3-month return both positive | Eligible for full position |
| One positive, one negative (mixed) | Eligible for half position only |
| Both negative | Do not trade — no entry regardless of chart |

## **Step 4: Earnings Carry Signal [v2]**

| **Carry = Forward Earnings Yield minus 3-Month T-Bill Rate** | **Action** |
| --- | --- |
| Above +3% | Strong carry — full position eligible |
| 0% to +3% | Positive but modest — normal sizing |
| Negative | Reduce position size 25% or skip |

## **Step 5: Energy Three-Layer Signal [v2]**

For energy stocks and commodity-sensitive equities, run the additional three-layer check.

### **Layer A: Crude Oil Term Structure**
- TradingView: CL1!-CL2! (front-month minus second-month spread)
- Positive (backwardation): bullish for energy longs
- Negative (contango): headwind for energy longs

### **Layer B: COT Positioning**
- Source: CFTC Commitments of Traders (cftc.gov, published Friday 3:30 PM ET)
- Free visualization: tradingster.com/cot or barchart.com/futures/commitment-of-traders
- Below 20th percentile: specs extremely short — contrarian bullish
- Above 80th percentile: specs extremely long — fragile, higher reversal risk

### **Layer C: Short-Term Trend**
- XLE (or individual energy stock) above 50-day SMA: bullish
- Below 50-day SMA: bearish

| **Energy Signal Composite** |
| --- |
| Score each layer: +1 bullish, 0 neutral, -1 bearish |
| Composite +2 or +3: Full energy position eligible |
| Composite +1: Half position only |
| Composite 0 or below: Flat or skip energy exposure |
| Weight: Term structure 50%, COT 25%, trend 25% |

## **Building Your Watchlist**

- Output of Layer 2 is a ranked watchlist — not a trade list
- Maintain 20-40 names at all times
- Rank by RS line strength — strongest at the top
- Tag each with: two-speed signal (full/half/flat), carry score, energy signal if applicable
- Names in a sector with an active Layer 1.5 ETF position are priority candidates for Phase 3 transition

# **Layer 3 — Entry Trigger**

## **Purpose**

Layer 3 defines the specific price action event that initiates a trade. A stock can be on your watchlist for weeks before producing a valid entry. The watchlist alone is not a reason to enter — only the trigger is. The trigger also defines your initial stop loss.

Note: Layer 1.5 ETF entries use the sector ETF's 20d MA as the reference level, not this layer's individual stock triggers. This layer governs individual stock entries only.

## **Regime-Based Entry Selection**

| **Regime type** | **Preferred entry trigger** |
| --- | --- |
| Risk-on / momentum dominant | Breakout from base on 40%+ above-average volume |
| Mixed / transitional | Either setup, prefer pullbacks to reduce risk |
| Deflationary / recession risk | Pullback to moving average only |
| Stagflation / red state | No new entries |

## **Entry Trigger Type 1: Breakout from Base**

- Stock has been in tight sideways consolidation for 3-8 weeks
- Declining volume during the base — sellers drying up
- Breakout day: price closes above the top of the base on 40-50%+ above average volume
- RS line simultaneously making a new high
- Entry: buy on breakout day as price clears the pivot
- Do not chase: if stock is already 5%+ above pivot, pass
- Initial stop: just below the bottom of the base

## **Entry Trigger Type 2: Pullback to Moving Average**

- Stock is in a clear uptrend (higher highs, higher lows), RS line is positive
- Price pulls back to the 20-day or 50-day moving average on declining volume
- Entry trigger: bullish reversal candle at the MA — closes near the high of the day, above the MA
- Avoid: stocks pulling back on heavy volume — may be institutional selling
- Initial stop: 1-2% below the moving average that triggered the entry

## **Entry Confirmation Checklist**

- Permission state is green or yellow (Layer 1)
- Stock passed two-speed trend signal (Layer 2) — full or half
- Earnings carry is positive, or carry is only slightly negative with exceptional RS
- Valid entry trigger today (breakout or MA pullback with volume confirmation)
- Stop loss level clearly identified
- Position size calculated (Layer 4) and within portfolio limits (Layer 5)
- No earnings announcement within 10 business days
- If a Layer 1.5 ETF position is active in this sector: adding this stock begins Phase 3 transition — plan to reduce ETF proportionally

# **Layer 4 — Position Sizing (Expected-Loss Method)**

## **Purpose**

Layer 4 determines how large each position should be. Applies to both individual stock entries (Layer 3) and ETF entries (Layer 1.5). For ETF entries, the flow momentum strength determines the risk %, and the stop is the ETF's 20d MA.

## **The Core Formula**

| **Position Sizing Formula** |
| --- |
| Shares = (Account Value x Risk Per Trade %) / (Entry Price - Stop Price) |
|  |
| Example: |
| Account Value: $74,000 |
| Risk Per Trade: 0.75% = $555 |
| Entry Price: $197.00  │  Stop Price: $190.00  │  Risk Per Share: $7.00 |
| Shares to Buy: $555 / $7.00 = 79 shares |
| Position Value: 79 x $197 = $15,563 (21% of account — triggers 10% cap) |
| Capped shares: $7,400 / $197 = 37 shares |

Hard cap: No single position exceeds 10% of total account value regardless of formula output.

## **Risk Per Trade Guidelines**

| **Permission state** | **Individual stock** | **ETF (Layer 1.5)** |
| --- | --- | --- |
| Green | 0.75-1.0% | Quarter/Half/Full based on flow momentum |
| Yellow | 0.25-0.5% | Half the Green rate for equivalent signal strength |
| Red | No new positions | No new ETF entries |

## **Capital-Base-Dependent Sizing**

| **Account equity status** | **Adjustment** |
| --- | --- |
| At or near all-time high | Full risk allocation |
| 5-10% below peak | Reduce risk per trade by 25-50% |
| 10-15% below peak | Reduce to minimum — capital preservation mode |
| Beyond 15% drawdown | Emergency protocol — see Layer 7 |

# **Layer 5 — Dynamic Portfolio Exposure**

## **Purpose**

Layer 5 governs total simultaneous positions and overall portfolio heat. ETF positions from Layer 1.5 count on equal footing with individual stock positions toward all limits.

## **Signal Breadth Determines Total Exposure**

| **Condition** | **Max simultaneous positions** |
| --- | --- |
| Green state + strong breadth (15+ stocks in full trend) | Up to 20 (includes ETF positions) |
| Green state + moderate breadth (8-15 stocks in full trend) | 12-15 (includes ETF positions) |
| Yellow state, or green with limited breadth (<8 stocks) | 8-12 (includes ETF positions) |
| Red state | 3-5 maximum (strongest existing only) |
| Liquidity override active | 0-3 positions, hedged if possible |

## **Portfolio Heat Management**

| **Portfolio Heat Calculation** |
| --- |
| Portfolio Heat = Sum of (Risk Per Trade %) across all open positions (stocks + ETFs) |
|  |
| Maximum portfolio heat targets: |
| Green state: up to 15% total portfolio heat |
| Yellow state: up to 8% total portfolio heat |
| Red state: under 3% total portfolio heat |

## **Sector Concentration Limits**

- No more than 40% of total positions in any one sector
- Sector cap for combined ETF + individual names: ETF position + individual stock positions in the same sector ≤ 20% of account
- Always maintain at least 2 sectors represented in the portfolio

# **Layer 6 — Trade Management**

## **Purpose**

Layer 6 covers everything after a position is entered. Rules apply to both individual stock positions and Layer 1.5 ETF positions, with differences noted.

## **Stop Loss Framework**

### **Individual stock positions**
- Breakout entries: stop just below the bottom of the base
- MA pullback entries: stop 1-2% below the moving average that triggered the entry
- At +5%: move stop to breakeven
- Stop can only be raised, never lowered

### **ETF positions (Layer 1.5)**
- Stop is always the 20d MA of the sector ETF
- No breakeven rule — ETF stop trails at the 20d MA throughout the hold
- If ETF closes below its 20d MA on above-average volume: exit

## **Partial Profit Taking (Individual Stocks)**

| **Profit level** | **Action** |
| --- | --- |
| +8-12% from entry | Sell 1/3 of position. Move stop on remainder to breakeven. |
| +20-25% from entry | Sell another 1/3. Begin trailing stop on remainder at 20-day MA. |
| Final 1/3 | Hold with 20-day MA trailing stop for maximum gain. |

ETF positions do not use partial profit rules — they are managed via flow signals and MA trail, and transitioned out as individual names are entered (Phase 3).

## **Exit Triggers**

### **Individual stocks — exit any of:**
- Close below 50-day MA on above-average volume
- Permission state downgrades to red
- Macro regime shifts against the sector
- Significant negative fundamental event
- RS line begins declining persistently
- Recession composite moves to 4+/5 (tighten all stops to breakeven)
- FactSet earnings revisions turn sharply down (tighten stops one level)
- 12-week max hold reached with insufficient progress

### **ETF positions (Layer 1.5) — exit any of:**
- 2 consecutive weeks of net outflows on ETF.com (flow reversal)
- Close below 20d MA on above-average volume
- RS line vs SPY declines for 2+ consecutive weeks
- Regime shift against the sector (tighten stop to breakeven, prepare to exit)
- 12-week max hold

## **Time-Based Exits**

- Consolidating 3-4 weeks with no progress: reassess — is thesis still intact?
- Maximum hold: 12 weeks regardless of whether stop has been hit
- Exception: strong trend with clear price progress — hold with trailing stop

# **Layer 7 — Hard Risk Caps and Emergency Protocols**

## **Purpose**

Layer 7 is the ultimate defense against catastrophic loss. Applies to the combined portfolio including all ETF and individual stock positions.

## **Drawdown Thresholds and Actions**

| **Portfolio drawdown from peak** | **Required action** |
| --- | --- |
| 0-5% drawdown | Normal operations. Review open positions. |
| 5-10% drawdown | Caution zone. Reduce risk per trade by 50%. No new positions (stocks or ETFs). Tighten all stops. |
| 10-15% drawdown | Emergency protocol. Max 3-5 positions. No new entries. Exit toward cash. |
| >15% drawdown | Full stop. Exit all positions. 100% cash. Wait for confirmed green state before re-engaging. |

## **Liquidity-Driven Hedge Protocol**

- Cash: Simplest and cleanest hedge.
- Inverse ETF: SH (S&P 500 inverse) or PSQ (Nasdaq inverse) — 10-20% of portfolio as macro hedge.
- Treasury allocation: TLT in deflationary downturn.

## **Recovery Protocol After Emergency**

- Wait for SPY two-speed trend to re-enter green state
- Confirm recession composite is below 3/5 before re-engagement
- Begin with 25% of normal maximum positions
- Use minimum position sizes — 0.25% risk per trade maximum
- No Layer 1.5 ETF entries until full green state confirmed
- Spend 2-4 weeks in cautious re-engagement mode
- Expand to full engagement only after portfolio recovered 50% of prior drawdown AND macro regime clearly positive

# **The Complete Decision Tree**

| **Weekly Process (Sunday/Monday)** |
| --- |
| 1. Run Layer 0: Macro regime? (SPY trend, sector RS, sector flows, TLT, HYG/IEF) |
| 2. Run Layer 0 additions: Fed Net Liquidity trend? Recession composite score? FactSet revision direction? |
| 3. Set Layer 1: Permission state (green / yellow / red) |
| 4. Run Layer 1.5: Any sector with 1-2 weeks of directional inflows + regime alignment + ETF above 20d MA? [v3] |
|    — If yes: determine ETF entry size based on flow momentum strength |
|    — If existing ETF position: check flow reversal signals, 20d MA status |
| 5. Run Layer 2: Update individual stock watchlist with qualified candidates |
|    — Prioritize stocks in sectors with active Layer 1.5 ETF positions (Phase 3 transition candidates) |
| 6. Note current drawdown from peak — confirm sizing tier (Layer 4 / Layer 7) |
| 7. Count open positions (stocks + ETFs) and portfolio heat — confirm within Layer 5 limits |

| **Monthly Process (after CPI/PCE)** |
| --- |
| 1. Run Layer 0.5: Macro Mispricing Check |
| 2. Check Taylor Rule deviation (Atlanta Fed calculator) |
| 3. Adjust overall position sizing if composite is -2 or worse |

| **Per-Trade Process (Any day a setup appears)** |
| --- |
| For ETF entries (Layer 1.5): |
| 1. Is permission state green or yellow? If red: STOP |
| 2. Is the sector regime-aligned? |
| 3. Is the sector ETF above its 20d MA? |
| 4. What is the flow momentum strength? (Determines size: quarter/half/full) |
| 5. Does adding this ETF position stay within portfolio heat and position count limits? |
| 6. If all yes: enter with calculated size, stop at 20d MA |
|  |
| For individual stock entries (Layer 3): |
| 1. Is permission state green or yellow? |
| 2. Is the stock on the qualified watchlist? (Layer 2 criteria met?) |
| 3. Is the two-speed trend signal full or half? |
| 4. Is the earnings carry positive? |
| 5. For energy names: what is the three-layer composite score? |
| 6. Is there a valid entry trigger today? (Breakout or MA pullback with volume) |
| 7. What is the initial stop level? |
| 8. Calculate position size: Account x Risk% / (Entry - Stop) |
| 9. Does adding this position stay within portfolio heat and position count limits? |
| 10. Is there an active Layer 1.5 ETF in this sector? If yes: plan ETF reduction (Phase 3 transition) |
| 11. If all yes: enter with calculated size and stop |
| 12. Record: entry price, stop price, 1/3 target, max hold date (entry + 12 weeks) |

| **Per-Trade Management (Weekly review of open positions)** |
| --- |
| Individual stocks: |
| 1. Has the +8-12% first target been reached? → Sell 1/3, move stop to breakeven |
| 2. Has the +20-25% second target been reached? → Sell another 1/3, trail at 20d MA |
| 3. Has the stock violated the trailing stop? → Exit |
| 4. Has the permission state changed? → Tighten stops, no new adds |
| 5. Is the RS line still positive? → If declining: reassess thesis |
| 6. Has recession composite moved to 4+/5? → Move all stops to breakeven |
| 7. Has the 12-week max hold been reached with insufficient progress? → Exit |
|  |
| ETF positions (Layer 1.5): |
| 1. Check ETF.com: any reversal in inflows (2 consecutive weeks outflows)? → Exit |
| 2. Check sector ETF 20d MA: still above it? → If not, on volume: exit |
| 3. Check RS line: declining 2+ weeks vs SPY? → Exit |
| 4. Individual names in this sector now entering (Phase 3)? → Reduce ETF proportionally |
| 5. 12-week max hold reached? → Exit if not already transitioned |

# **Complete Weekly Review Template**

| **Item** | **This week's reading** |
| --- | --- |
| SPY 1-month return | ____% |
| SPY 6-month return | ____% |
| Permission state | Green / Yellow / Red |
| Fed Net Liquidity trend | Rising / Flat / Falling  │  4-week change: $____B |
| CFNAIMA3 reading | ____  (below -0.70 = warning) |
| Sahm Rule reading | ____  (above 0.50 = triggered) |
| Recession composite score | ____ / 5 models in recession territory |
| FactSet revision direction | Up / Flat / Down  │  Trend: ____ |
| Top sectors by 4-week RS | ____ |
| Top sectors by 1-week inflows (ETF.com) | ____ |
| Top sectors by 4-week inflows (ETF.com) | ____ |
| TLT direction | Up / Down / Flat |
| Regime identification | Risk-on / Reflation / Deflation / Stagflation |
| Liquidity override active? | Yes / No |
| **Layer 1.5 status** | Active ETF positions: ____ │ New opportunities: ____ │ Anomalies: ____ |
| Energy signal composite | +__ / 3  (for energy positions) |
| Current drawdown from peak | ____% |
| Open positions (stocks + ETFs) | ____ of max ____ |
| Portfolio heat | ____% |
| Priority sectors this week | ____ |
| Watchlist candidates (names) | ____ |

| **Monthly item (after CPI/PCE)** | **This month's reading** |
| --- | --- |
| Earnings yield vs TIPS spread | ____%  (neutral 2-4%, below 2% = headwind) |
| Equity risk premium | ____%  (neutral 3-5%, below 2% = stretched) |
| GDPNow vs analyst EPS direction | Bullish / Neutral / Bearish gap |
| Macro mispricing composite | +__ to -3  (below -1 = reduce sizing) |
| Taylor Rule deviation | ____% positive/negative |

# **Zero-Cost Data Sources Reference**

## **FRED Economic Data (fred.stlouisfed.org)**

| **FRED Series Code** | **What It Tracks** |
| --- | --- |
| CFNAIMA3 | Chicago Fed National Activity Index 3-month MA |
| SAHMREALTIME | Sahm Rule real-time recession indicator |
| RECPROUSM156N | Chauvet-Piger smoothed recession probability |
| T10Y3M | 10-year minus 3-month yield curve spread |
| WALCL | Fed total assets (balance sheet) |
| WTREGEN | Treasury General Account balance |
| RRPONTSYD | Overnight reverse repo facility |
| DFII10 | 10-year TIPS yield (real rate) |
| DGS10 | 10-year nominal Treasury yield |
| DTB3 | 3-month T-bill rate |
| DTWEXBGS | Trade-weighted US dollar index |

## **Other Free Sources**

| **Source / URL** | **What It Provides** |
| --- | --- |
| etf.com/etfanalytics/etf-fund-flows-tool | ETF sector fund flows — daily inflows/outflows [v3] |
| ssga.com/us/en/intermediary/resources/sector-tracker | Sector ETF price performance tracker [v3] |
| atlantafed.org/research-and-data/data/gdpnow | GDPNow real-time GDP estimate |
| atlantafed.org/cqer/research/taylor-rule | Taylor Rule calculator — monthly |
| insight.factset.com/topic/earnings | Earnings Insight PDF — forward EPS, revisions — weekly, free |
| multpl.com | S&P 500 forward P/E and earnings yield |
| tradingster.com/cot | COT data visualization |
| conference-board.org/topics/us-leading-indicators | Conference Board LEI — monthly |
| cftc.gov/MarketReports/CommitmentsofTraders | Raw CFTC COT data — every Friday 3:30 PM ET |

# **Quick Reference: Complete Rules Summary**

## **Permission State Rules**

| **State** | **Max positions / Risk / Approach** |
| --- | --- |
| Green | Up to 20 / 0.75-1.0% risk / Momentum breakouts |
| Yellow | 8-12 / 0.25-0.5% risk / Both setups |
| Red | 3-5 or cash / No new entries (stocks or ETFs) |

## **Layer 1.5 ETF Sizing**

| **Flow Signal** | **Size** | **Risk % (Green)** |
| --- | --- | --- |
| Weak (1 week, modest) | Quarter | 0.1875% |
| Moderate (1-2 weeks, consistent) | Half | 0.375% |
| Strong (2+ weeks, accelerating) | Full | 0.75% |
| Stop: always 20d MA of sector ETF | Exit: 2 consecutive weeks outflows OR below 20d MA on volume |

## **Two-Speed Trend Signal**

| **Signal** | **Position size allowed** |
| --- | --- |
| Both lookbacks positive (1mo and 3mo) | Full position size |
| Mixed (one positive, one negative) | Half position size |
| Both negative | No trade |

## **Position Sizing Formula**

Shares = (Account x Risk%) / (Entry - Stop). Max single position = 10% of account.

## **Profit Taking Rules**

| **Profit level** | **Action** |
| --- | --- |
| +5% | Move stop to breakeven |
| +8-12% | Sell 1/3, stop to breakeven |
| +20-25% | Sell another 1/3, trail remainder at 20d MA |
| Remainder | Trail at 20d MA, hold up to 12 weeks |

## **Drawdown Emergency Levels**

| **Drawdown from peak** | **Action** |
| --- | --- |
| 5-10% | Cut risk/trade 50%, no new positions |
| 10-15% | 3-5 positions max, no new entries |
| >15% | 100% cash, wait for confirmed green state |

## **Layer 1.5 Exit Checklist**

| **Signal** | **Action** |
| --- | --- |
| 2 consecutive weeks of net outflows (ETF.com) | Exit ETF position |
| ETF closes below 20d MA on volume | Exit ETF position |
| Sector RS vs SPY declining 2+ weeks | Exit ETF position |
| Individual stocks entering (Phase 3) | Reduce ETF proportionally |

## **Recession Composite Scoring**

| **Models in recession territory** | **Action** |
| --- | --- |
| 0-1 / 5 | Normal operations |
| 2-3 / 5 | Reduce max positions 25%, tighten stops |
| 4-5 / 5 | Treat as red state regardless of SPY trend |

*Framework v3.0. Based on Prometheus Research methodology (prometheus-research.com) augmented with zero-cost public data signals from FRED, CFTC, Atlanta Fed, FactSet, and ETF.com sector flow data. For systematic swing trading with 1-12 week hold periods.*
