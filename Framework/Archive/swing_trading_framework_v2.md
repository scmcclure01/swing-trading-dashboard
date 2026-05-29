**PROMETHEUS-BASED SWING TRADING FRAMEWORK**

A Complete Systematic Rules-Based Methodology — Version 2.0

Hold Period: 1–12 Weeks  |  Platform: TradingView  |  Universe: All US Equities

*Incorporating: Prometheus Research + Zero-Cost Public Data Enhancements*

| **Version 2.0 — What's New** |
| --- |
| This document is an updated version of the original Prometheus-Based Swing Trading Framework. |
| It incorporates all original 8 layers plus the following zero-cost public data enhancements: |
| • Layer 0: Recession Probability Composite (5 free FRED models) |
| • Layer 0: Fed Net Liquidity signal (TradingView indicator) |
| • Layer 0: FactSet Earnings Revision direction (weekly free report) |
| • Layer 0: Taylor Rule Deviation (Atlanta Fed free calculator) |
| • Layer 0.5: NEW — Monthly Macro Mispricing Check |
| • Layer 2: Earnings carry signal for individual stocks |
| • Layer 2: Energy three-layer signal for commodity-sensitive names |
| • Weekly checklist: Updated with all new additions |
| Items marked [NEW] throughout this document indicate Version 2.0 additions. |

# **Framework Overview**

This framework is a systematic, rules-based swing trading methodology built on Prometheus Research (prometheus-research.com), incorporating their Basic Trend Program, G/I/L macro regime framework, expected-loss risk management, and sector rotation research — augmented with zero-cost public data signals from FRED, CFTC, FactSet, and the Atlanta Fed.

The framework consists of 8 layers plus one new monthly layer, applied in sequence. Every trade decision flows through each layer in order. No layer can be skipped.

| **The 8 Layers + Monthly Check** |
| --- |
| Layer 0   — Macro Regime Filter (G/I/L + recession probability + liquidity + earnings revisions) |
| Layer 0.5 — Monthly Macro Mispricing Check [NEW] |
| Layer 1   — Market Permission State (green / yellow / red) |
| Layer 2   — Stock Selection (sector bias, RS, two-speed trend, carry signal [NEW]) |
| Layer 3   — Entry Trigger (breakout or MA pullback with volume) |
| Layer 4   — Position Sizing (expected-loss method, inverse-vol weighting) |
| Layer 5   — Dynamic Portfolio Exposure (breadth-scaled positions) |
| Layer 6   — Trade Management (stops, partials, trailing) |
| Layer 7   — Hard Risk Caps (drawdown thresholds, emergency protocols) |

# **Layer 0 — Macro Regime Filter**

## **Purpose**

The macro regime filter is the top-level permission system. Before looking at any stock, you must understand the macroeconomic environment. This determines: (1) whether to trade at all, (2) which sectors to favor, and (3) whether momentum or mean-reversion will dominate.

Version 2.0 adds four zero-cost signals to Layer 0: a Recession Probability Composite, a Fed Net Liquidity Trend, a FactSet Earnings Revision direction, and a Taylor Rule Deviation check.

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

## **Signal 5: Recession Probability Composite [NEW]**

Check these five free models monthly. Count how many are in recession territory. Use the composite to gate position sizing and stop tightness.

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

## **Signal 6: Fed Net Liquidity Trend [NEW]**

Track weekly. Formula: Fed Balance Sheet minus Treasury General Account minus Reverse Repo Facility.

| **Fed Net Liquidity — Setup** |
| --- |
| TradingView: Search 'Fed Net Liquidity' by jlb05013 or vbarink — free community indicator |
| Formula: WALCL - WTREGEN - RRPONTSYD (all available on FRED) |
| Update cadence: Every Thursday at 4:30 PM ET (H.4.1 release) |
| Signal: Rising trend = liquidity tailwind (risk-on). Declining trend = tightening = activate liquidity override. |
| Threshold: If Net Liquidity has declined >$200B over 4 weeks, treat as liquidity override active. |

## **Signal 7: FactSet Earnings Revision Direction [NEW]**

Check every Friday. FactSet publishes 'Earnings Insight' free at insight.factset.com. Look for one thing: are forward S&P 500 EPS estimates being revised up or down?

| **Revision Direction** | **Action** |
| --- | --- |
| Estimates revised UP 3+ weeks in a row | Confirms risk-on bias. Full exposure appropriate. |
| Estimates flat (within +/- 0.5%) | Neutral — no adjustment to framework needed. |
| Estimates revised DOWN 2+ weeks in a row | Reduce conviction on new longs. Tighten stops on existing positions. |
| Estimates revised DOWN sharply (>2% in one week) | Treat as yellow flag — reduce max positions immediately. |

## **Signal 8: Taylor Rule Deviation [NEW]**

Check monthly after CPI and PCE releases. Use the Atlanta Fed's free Taylor Rule calculator at atlantafed.org/cqer/research/taylor-rule.

| **Taylor Rule Gap (Implied Rate vs Actual Rate)** | **Interpretation** |
| --- | --- |
| Positive gap (implied > actual by >1%) | Fed is too loose — pressure to hike. Hawkish surprise risk. Reduce rate-sensitive longs. |
| Near zero (-1% to +1%) | Fed approximately on target. Neutral. |
| Negative gap (implied < actual by >1%) | Fed is too tight — pressure to cut. Dovish surprise potential. Favors risk-on positioning. |

## **How Regime Changes Your Trading Style**

| **Regime** | **Trading approach** |
| --- | --- |
| Risk-on | Buy breakouts aggressively. Hold winners longer. Let trailing stops run. Full exposure. Momentum dominates. |
| Reflation | Momentum works in energy/commodities, may reverse in tech. Tighten tech stops. Rotate to commodity/industrial. |
| Deflationary / recession risk | Mean-reversion traps everywhere. Bounces fail. Smaller positions, tighter stops. Do not chase. |
| Stagflation / tightening liquidity | Almost nothing works on the long side. Cash is a position. 3-5 strongest names only or move to cash. |

## **Practical Weekly Layer 0 Checklist**

Run every Sunday or Monday morning. Should take 15-20 minutes total.

| **Step** | **Action** |
| --- | --- |
| 1. SPY trend check | Calculate 21-day and 126-day price return. Both positive = green. Mixed = yellow. Both negative = red. |
| 2. Sector RS check | Which sector ETFs outperformed SPY last 4 weeks? Sets regime flavor and sector tilt. |
| 3. Bond confirmation | TLT direction vs SPY. Confirms or challenges regime reading. |
| 4. HYG/IEF spread | Falling ratio = credit spreads widening = liquidity tightening override. |
| 5. Fed Net Liquidity trend [NEW] | TradingView indicator. Rising or falling over last 4 weeks? Declining >$200B = override active. |
| 6. Recession composite [NEW] | Check CFNAIMA3 and SAHMREALTIME on FRED. How many of 5 models are in recession territory? |
| 7. FactSet earnings revisions [NEW] | insight.factset.com — Earnings Insight PDF. Up or down trend in forward EPS estimates? |
| 8. Set week's rules | Based on 1-7: set max positions, sector tilt, entry style bias, stop tightness. |

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
| FRED:CFNAIMA3 | Business cycle gauge — below -0.70 = recession signal [NEW] |
| FRED:SAHMREALTIME | Sahm Rule — above 0.50 = triggered [NEW] |
| FRED:T10Y3M | Yield curve spread — inversion = warning [NEW] |
| CL1!-CL2! | Crude oil term structure — backwardation = risk premium [NEW] |

# **Layer 0.5 — Monthly Macro Mispricing Check [NEW]**

## **Purpose**

Once a month (after CPI and PCE releases), run a 5-minute valuation check to assess whether equities are rich or cheap relative to real interest rates and macro fundamentals. This check directly modifies position sizing across the entire portfolio when equities are materially overpriced vs. macro conditions.

This layer sits between the weekly regime filter and the permission state because it operates at a lower frequency and affects sizing rather than direction.

## **The Three Mispricing Checks**

### **Check 1: Earnings Yield vs. Real Rates Spread**

Measure: Forward S&P 500 earnings yield minus 10-Year TIPS yield.

- Forward earnings yield: From FactSet Earnings Insight (forward P/E inverted) or multpl.com

- 10-Year TIPS yield: FRED: DFII10 or TradingView FRED:DFII10

- Historical average spread: approximately 3-4%

| **Spread Reading** | **Action** |
| --- | --- |
| Above 4% (equities cheap vs real rates) | No sizing adjustment. Macro environment supports full risk. |
| 2-4% (equities fairly valued) | Normal operations. |
| Below 2% (equities expensive vs real rates) | Reduce max risk per trade by 25%. This is a valuation headwind. |
| Below 1% or negative (equities very expensive) | Reduce max risk per trade by 50%. Only highest-conviction setups. |

### **Check 2: Earnings Revision Gap vs. Macro-Implied EPS**

Compare the direction of FactSet's consensus S&P 500 forward EPS against what macro conditions imply.

Macro-implied EPS logic: When GDP growth is above trend, inflation is moderate, and the dollar is stable, earnings should be growing. When one or more of these deteriorate, macro-implied EPS falls — often before analysts revise down. A gap between what macro says and what analysts expect is a mispricing.

- GDP growth rate: Atlanta Fed GDPNow (atlantafed.org) — updated 6-7x per month

- Dollar trend: FRED: DTWEXBGS or TradingView DXY

- If GDPNow is declining sharply AND analysts have not yet revised down EPS: treat as bearish mispricing — tighten stops

- If GDPNow is accelerating AND analysts are still at depressed estimates: treat as bullish mispricing — hold winners longer

### **Check 3: Equity Risk Premium**

Forward earnings yield minus 10-year nominal Treasury yield (FRED: DGS10). When the ERP compresses below 2%, equities are pricing in nearly perfect conditions. When it expands above 5%, equities are compensating heavily for risk.

| **ERP Level** | **Sizing Rule** |
| --- | --- |
| Above 5% | Equities cheap. Take full positions. Market underestimates risk/reward. |
| 3-5% | Normal range. No adjustment. |
| 2-3% | Elevated. Reduce max position size 20%. Higher bar for new entries. |
| Below 2% | Stretched. Reduce max position size 40%. Only the strongest setups. |

| **Monthly Mispricing Composite** |
| --- |
| Score each of the three checks: Bullish (+1), Neutral (0), Bearish (-1) |
| Composite +2 to +3: Full risk. Macro pricing supports longs. |
| Composite -1 to +1: Normal operations. No adjustment. |
| Composite -2 to -3: Reduce all position sizes by 30-50%. Only top-tier setups. Tighten all trailing stops. |
| Time: 10 minutes once a month. Data: All free from FactSet, FRED, Atlanta Fed. |

# **Layer 1 — Market Permission State**

## **Purpose**

Layer 1 translates macro regime analysis from Layer 0 into a single actionable permission state: green, yellow, or red. This state governs the maximum number of positions, how aggressively you can size, and whether momentum or mean-reversion setups are preferred.

The permission state is a hard constraint. If red, you do not go looking for stock setups regardless of how attractive an individual chart looks.

| **GREEN — Full Engagement** |
| --- |
| Condition: SPY both signals positive + no liquidity override + recession composite below 2/5 |
| Maximum positions: up to 20 |
| Position sizing: 0.75-1.0% account risk per trade |
| Preferred setups: Momentum breakouts, trend continuation |
| Trailing stops: Loose — give trends room to run |

| **YELLOW — Reduced Engagement** |
| --- |
| Condition: SPY mixed signals OR recession composite 2-3/5 OR earnings revisions declining |
| Maximum positions: 8-12 |
| Position sizing: 0.25-0.5% account risk per trade |
| Preferred setups: Both momentum and mean-reversion, be selective |
| Trailing stops: Tighter — less room for error |

| **RED — Minimal Exposure / Cash** |
| --- |
| Condition: SPY both negative OR liquidity override OR recession composite 4-5/5 |
| Maximum positions: 3-5 (strongest only) or 100% cash |
| Position sizing: Minimum or no new positions |
| Preferred setups: None — do not look for new entries |
| Action: Manage existing positions toward exits, do not add |

## **Transition Rules**

- Green to Yellow: SPY 1-month turns negative while 6-month remains positive, OR recession composite moves to 2-3/5.

- Yellow to Red: SPY 6-month turns negative, OR liquidity override triggered, OR recession composite reaches 4+/5.

- Red to Yellow: SPY 1-month turns positive AND recession composite falls below 3/5. Light re-engagement with half-size positions.

- Yellow to Green: SPY 6-month turns positive AND recession composite below 2/5. Resume full engagement.

Important: The liquidity override can jump you from green directly to red in a single week.

# **Layer 2 — Stock Selection**

## **Purpose**

Layer 2 identifies individual stock candidates. Goal: stocks in the right macro sector, showing strong relative strength vs. the market, in a positive two-speed trend, with a positive carry signal. Only runs when the permission state is green or yellow.

## **Step 1: Apply Sector Bias from Macro Regime**

| **Macro regime** | **Priority sectors** |
| --- | --- |
| Risk-on (growth up, inflation down) | Technology (XLK), Consumer Discretionary (XLY), Financials (XLF), Industrials (XLI) |
| Reflation (growth up, inflation up) | Energy (XLE), Materials (XLB), Industrials (XLI), Homebuilders (XHB) |
| Deflationary (growth down, inflation down) | Healthcare (XLV), Consumer Staples (XLP), Utilities (XLU) — very selective |
| Stagflation (growth down, inflation up) | Energy (XLE), commodities — minimal long exposure overall |

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
| 1-month AND 3-6 month return both positive | Eligible for full position — both windows confirm |
| One positive, one negative (mixed) | Eligible for half position only — higher risk |
| Both negative | Do not trade — no entry regardless of chart |

## **Step 4: Earnings Carry Signal [NEW]**

For each stock candidate, calculate whether holding the stock generates positive carry relative to the risk-free rate. This filters out stocks that look good technically but offer no fundamental compensation for risk.

| **Earnings Carry Formula** |
| --- |
| Carry = Forward Earnings Yield - 3-Month T-Bill Rate |
|  |
| Forward Earnings Yield = 1 / Forward P/E (available on TradingView via request.financial()) |
| 3-Month T-Bill Rate = FRED: DTB3 (updated daily, ~4.3% as of early 2026) |
|  |
| Example: Stock with forward P/E of 20 = earnings yield of 5.0% |
| 5.0% - 4.3% = +0.7% carry — positive, stock compensates for risk |
|  |
| Interpretation: |
| Carry > +3%: Strong carry — full position eligible |
| Carry 0% to +3%: Positive but modest — normal sizing |
| Carry < 0%: Negative carry — reduce position size 25% or skip |

## **Step 5: Energy Three-Layer Signal [NEW]**

When trading energy stocks (XLE names, oil services, refiners) or commodity-sensitive equities, run this additional three-layer check before entry.

### **Layer A: Crude Oil Term Structure (Backwardation Check)**

- TradingView: Type 'CL1!-CL2!' in the symbol bar to see the front-month minus second-month spread

- Positive spread (backwardation): Supply tight, risk premium exists — bullish for energy longs

- Negative spread (contango): Oversupply, storage costs drag returns — headwind for energy longs

- Signal: Annualized roll yield above +5% = backwardation bullish. Below -5% = contango bearish.

### **Layer B: COT Positioning (Speculative vs Hedging Dominance)**

- Source: CFTC Commitments of Traders report — published every Friday at 3:30 PM ET at cftc.gov

- Free visualization: tradingster.com/cot or barchart.com/futures/commitment-of-traders

- Watch: Managed Money (speculative) net positioning as a percentile of its 1-year range

- Below 20th percentile: Specs extremely short — contrarian bullish signal

- Above 80th percentile: Specs extremely long — fragile positioning, higher reversal risk

- Speculative-dominant markets (very high managed money open interest) are more prone to violent reversals

### **Layer C: Short-Term Trend**

- Price of XLE (or individual energy stock) above its 50-day SMA: bullish

- Price below its 50-day SMA: bearish

| **Energy Signal Composite** |
| --- |
| Score each layer: +1 bullish, 0 neutral, -1 bearish |
| Composite +2 or +3: Full energy position eligible |
| Composite +1: Half position only |
| Composite 0 or below: Flat or skip energy exposure |
| Weight: Term structure 50%, COT 25%, trend 25% |
| Apply this check to: XLE, XOM, CVX, COP, SLB, PSX, VLO, and any oil services names |

## **Building Your Watchlist**

- Output of Layer 2 is a ranked watchlist — not a trade list

- Maintain 20-40 names at all times so you always have candidates when setups develop

- Rank by RS line strength — strongest at the top

- Tag each with: two-speed signal (full/half/flat), carry score (positive/negative), energy signal if applicable

- Remove any name where RS line begins declining or stock closes below 50-day MA on heavy volume

# **Layer 3 — Entry Trigger**

## **Purpose**

Layer 3 defines the specific price action event that initiates a trade. A stock can be on your watchlist for weeks before producing a valid entry. The watchlist alone is not a reason to enter — only the trigger is. The trigger also defines your initial stop loss.

## **Regime-Based Entry Selection**

| **Regime type** | **Preferred entry trigger** |
| --- | --- |
| Risk-on / momentum dominant | Breakout from base on 40%+ above-average volume — buy strength confirming trend |
| Mixed / transitional | Either setup, but prefer pullbacks to reduce risk |
| Deflationary / recession risk | Pullback to moving average only — breakout buys have higher failure rate |
| Stagflation / red state | No new entries regardless of setup quality |

## **Entry Trigger Type 1: Breakout from Base**

- Stock has been in tight sideways consolidation for 3-8 weeks

- Declining volume during the base — sellers drying up

- Breakout day: price closes above the top of the base (the pivot point)

- Breakout volume: at least 40-50% above the stock's average daily volume

- Ideal: RS line simultaneously making a new high

- Entry: buy on breakout day as price clears the pivot, or on the next open

- Do not chase: if stock is already 5%+ above pivot, pass — entry is extended

- Initial stop: just below the bottom of the base

## **Entry Trigger Type 2: Pullback to Moving Average**

- Stock is in a clear uptrend (higher highs, higher lows), RS line is positive

- Price pulls back to the 20-day or 50-day moving average on declining volume

- Entry trigger: bullish reversal day at the MA — closes near the high of the day, above the MA

- Avoid: stocks pulling back on heavy volume — may be institutional selling

- Entry: buy near the close of the reversal day or open of the following day

- Initial stop: 1-2% below the moving average that triggered the entry

## **Entry Confirmation Checklist**

- Permission state is green or yellow (Layer 1)

- Stock passed two-speed trend signal (Layer 2) — full or half

- Earnings carry is positive, or carry is only slightly negative with exceptional RS

- Entry trigger is valid (breakout or MA pullback with volume confirmation)

- Stop loss level clearly identified

- Position size calculated (Layer 4) and within portfolio limits (Layer 5)

- No earnings announcement within 10 business days

# **Layer 4 — Position Sizing (Expected-Loss Method)**

## **Purpose**

Layer 4 determines how large each position should be. Correct sizing keeps a string of losses from being catastrophic and ensures winners can compound meaningfully. This framework uses Prometheus's expected-loss approach rather than pure volatility targeting.

## **Why Not Pure Volatility Targeting**

Volatility targeting exposes portfolios to correlation risks — during market stress, all stocks fall together, causing losses far larger than any single-asset vol model predicts. Instead, size positions so that even in a worst-case correlation scenario (everything moves against you), total portfolio drawdown stays within a defined maximum.

## **The Core Formula**

| **Position Sizing Formula** |
| --- |
| Shares = (Account Value x Risk Per Trade %) / (Entry Price - Stop Price) |
|  |
| Example: |
| Account Value: $100,000 |
| Risk Per Trade: 1% = $1,000 |
| Entry Price: $50.00  │  Stop Price: $46.00  │  Risk Per Share: $4.00 |
| Shares to Buy: $1,000 / $4.00 = 250 shares |
| Position Value: 250 x $50 = $12,500 (12.5% of account) |
|  |
| Result: If stopped out, you lose exactly $1,000 (1% of account). No more. |

## **Risk Per Trade Guidelines**

| **Permission state** | **Max risk per trade** |
| --- | --- |
| Green — full engagement | 0.75% - 1.0% of account per trade |
| Yellow — reduced engagement | 0.25% - 0.5% of account per trade |
| Red — minimal / cash | No new positions, or 0.1-0.25% maximum |

## **Inverse-Volatility Weighting (Risk Parity)**

This happens automatically in the formula: a volatile stock with a wide stop requires fewer shares than a stable stock with a tight stop, even if the dollar risk is the same. This is risk parity at the individual stock level — each position contributes approximately equal risk to the portfolio.

## **Capital-Base-Dependent Sizing**

| **Account equity status** | **Adjustment** |
| --- | --- |
| At or near all-time high | Full risk allocation — compound aggressively |
| 5-10% below peak (mild drawdown) | Reduce risk per trade by 25-50% |
| 10-15% below peak (significant drawdown) | Reduce to minimum — capital preservation mode |
| Beyond 15% drawdown | Emergency protocol — see Layer 7 |

Hard cap: No single position exceeds 10% of total account value regardless of formula output.

# **Layer 5 — Dynamic Portfolio Exposure**

## **Purpose**

Layer 5 governs total simultaneous positions and overall portfolio heat. Take more risk when more signals are positive (more diversification = higher Sharpe = more risk-friendly). Scale back as the number of confirming signals decreases.

## **Signal Breadth Determines Total Exposure**

| **Condition** | **Max simultaneous positions** |
| --- | --- |
| Green state + strong breadth (15+ stocks in full trend) | Up to 20 positions |
| Green state + moderate breadth (8-15 stocks in full trend) | 12-15 positions |
| Yellow state, or green with limited breadth (<8 stocks) | 8-12 positions |
| Red state | 3-5 positions maximum (strongest existing only) |
| Liquidity override active | 0-3 positions, hedged if possible |

## **Portfolio Heat Management**

| **Portfolio Heat Calculation** |
| --- |
| Portfolio Heat = Sum of (Risk Per Trade %) across all open positions |
| Example: 10 positions each risking 0.75% = 7.5% total portfolio heat |
|  |
| Maximum portfolio heat targets: |
| Green state: up to 15% total portfolio heat |
| Yellow state: up to 8% total portfolio heat |
| Red state: under 3% total portfolio heat |

## **Sector Concentration Limits**

- No more than 40% of total positions in any one sector

- Risk-on regime: up to 40% tech, maintain 2-3 other sectors

- Reflation regime: up to 40% energy/materials combined, maintain some growth exposure

- Always maintain at least 2 sectors represented in the portfolio

# **Layer 6 — Trade Management**

## **Purpose**

Layer 6 covers everything after a position is entered: managing stops, taking partial profits, trailing the remaining position, and deciding when to exit. Good trade management separates a system that captures big winners from one that consistently gives back gains.

## **Stop Loss Framework**

### **Initial Stop**

- Breakout entries: stop just below the bottom of the base

- MA pullback entries: stop 1-2% below the moving average that triggered the entry

- Stop is set at entry and cannot be moved lower — only raised

- If the stop requires more than max risk per trade even at minimum shares, skip the trade

### **Moving to Breakeven**

- Once position is up ~5% from entry, move stop to breakeven

- Locks in a no-loss outcome. Removes psychological pressure.

- Do not move to breakeven too early — stock needs room to breathe

### **Trailing Stop**

- After +8-10% gain and first partial taken: begin trailing at 20-day MA

- Weekly: note 20-day MA level. Close below on above-average volume = exit signal.

- Strong trend: 20-day MA trail may hold for full 1-12 week period

- Weaker trend or yellow state: consider tighter 10-day MA trail

## **Partial Profit Taking**

| **Profit level** | **Action** |
| --- | --- |
| +8-12% from entry | Sell 1/3 of position. Move stop on remainder to breakeven. |
| +20-25% from entry | Sell another 1/3. Begin trailing stop on remainder at 20-day MA. |
| Final 1/3 | Hold with 20-day MA trailing stop for maximum gain. Can run weeks. |

## **Exit Triggers**

- Close below 50-day MA on above-average volume — distribution signal, exit remaining position

- Permission state downgrades to red — exit weaker positions first, tighten stops on stronger

- Macro regime shifts against the sector (e.g., inflation spike hits tech) — reassess thesis

- Significant negative fundamental event — exit immediately, do not wait for the stop

- RS line begins declining persistently — thesis weakening

- Recession composite moves to 4+/5 while in a long — tighten stops to breakeven immediately [NEW]

- FactSet earnings revisions turn sharply down while holding longs — tighten stops one level [NEW]

## **Time-Based Exits**

- Consolidating 3-4 weeks with no progress: reassess — is thesis still intact?

- Maximum hold: 12 weeks regardless of whether stop has been hit

- Exception: strong trend with clear price progress — hold with trailing stop

# **Layer 7 — Hard Risk Caps and Emergency Protocols**

## **Purpose**

Layer 7 is the ultimate defense against catastrophic loss. While Layers 0-6 manage risk in normal conditions, Layer 7 defines what happens when portfolio drawdowns exceed expectations and emergency action is required.

## **The Expected-Loss Philosophy**

Do not target a fixed volatility number. Instead: set a maximum acceptable drawdown (15%), and ensure total portfolio risk is sized so that even a worst-case correlation blowup stays within that limit. In high-correlation periods (crashes), this means you will be less exposed than a pure vol-targeting approach.

## **Drawdown Thresholds and Actions**

| **Portfolio drawdown from peak** | **Required action** |
| --- | --- |
| 0-5% drawdown | Normal operations. Review open positions, no forced action. |
| 5-10% drawdown | Caution zone. Reduce risk per trade by 50%. No new positions until recovery above 5%. Tighten all trailing stops. |
| 10-15% drawdown | Emergency protocol. Max 3-5 positions. No new entries. Only manage existing toward exits. |
| >15% drawdown | Full stop. Exit all positions. 100% cash. Wait for confirmed green state before re-engaging. |

## **Liquidity-Driven Hedge Protocol**

When the liquidity override is active (Fed hiking, credit spreads widening), going to cash is not the only option:

- Cash: Reduce position count and let cash be the hedge. Simplest and cleanest.

- Inverse ETF: Add SH (S&P 500 inverse) or PSQ (Nasdaq inverse) equal to 10-20% of portfolio as macro hedge.

- VIX exposure: Small position in VIXY during tightening environments. Prometheus uses risk-parity-weighted VIX exposure.

- Treasury allocation: In deflationary downturn, TLT tends to rise when stocks fall — hold as a hedge.

Hedges are regime-specific tools. Exit them when permission state returns to green.

## **Recovery Protocol After Emergency**

- Wait for SPY two-speed trend to re-enter green state — both 1-month and 6-month positive

- Confirm recession composite is below 3/5 before re-engagement [NEW]

- Begin with 25% of normal maximum positions (5 positions instead of 20)

- Use minimum position sizes — 0.25% risk per trade maximum

- Spend 2-4 weeks in cautious re-engagement mode

- Only expand to full engagement after portfolio recovered 50% of prior drawdown AND macro regime clearly positive

## **Record Keeping**

- Current drawdown from peak equity

- Total portfolio heat (sum of all position risk %)

- Number of open positions

- Current permission state and recession composite score [NEW]

- Any active hedges and their rationale

# **The Complete Decision Tree**

| **Weekly Process (Sunday/Monday) — 15-20 minutes** |
| --- |
| 1. Run Layer 0: Macro regime? (SPY trend, sector RS, TLT, HYG/IEF) |
| 2. Run Layer 0 additions: Fed Net Liquidity trend? Recession composite score? FactSet revision direction? |
| 3. Set Layer 1: Permission state (green / yellow / red) |
| 4. Run Layer 2: Update watchlist with qualified candidates |
| 5. Note current drawdown from peak — confirm sizing tier (Layer 4 / Layer 7) |
| 6. Count open positions and portfolio heat — confirm within Layer 5 limits |

| **Monthly Process (after CPI/PCE) — 10 minutes** |
| --- |
| 1. Run Layer 0.5: Macro Mispricing Check |
| - Forward earnings yield vs TIPS yield spread |
| - Equity risk premium vs. historical average |
| - GDPNow direction vs. analyst consensus EPS |
| 2. Check Taylor Rule deviation (Atlanta Fed calculator) |
| 3. Adjust overall position sizing if composite is -2 or worse |

| **Per-Trade Process (Any day a setup appears)** |
| --- |
| 1. Is permission state green or yellow? (If red: STOP — no new entries) |
| 2. Is the stock on the qualified watchlist? (Layer 2 criteria met?) |
| 3. Is the two-speed trend signal full or half? |
| 4. Is the earnings carry positive? |
| 5. For energy names: what is the three-layer composite score? |
| 6. Is there a valid entry trigger today? (Breakout or MA pullback with volume) |
| 7. What is the initial stop level? |
| 8. Calculate position size: Account x Risk% / (Entry - Stop) |
| 9. Does adding this position stay within portfolio heat and position count limits? |
| 10. If all yes: enter with calculated size and stop |
| 11. Record: entry price, stop price, 1/3 target, max hold date (entry + 12 weeks) |

| **Per-Trade Management (Weekly review of open positions)** |
| --- |
| 1. Has the +8-12% first target been reached? → Sell 1/3, move stop to breakeven |
| 2. Has the +20-25% second target been reached? → Sell another 1/3, trail at 20d MA |
| 3. Has the stock violated the trailing stop? → Exit |
| 4. Has the permission state changed? → Tighten stops, no new adds |
| 5. Is the RS line still positive? → If declining: reassess thesis |
| 6. Has recession composite moved to 4+/5? → Move all stops to breakeven [NEW] |
| 7. Has the 12-week max hold been reached with insufficient progress? → Exit and redeploy |

# **Complete Weekly Review Template**

| **Item** | **This week's reading** |
| --- | --- |
| SPY 1-month return | ____% |
| SPY 6-month return | ____% |
| Permission state | Green / Yellow / Red |
| Fed Net Liquidity trend [NEW] | Rising / Flat / Falling  │  4-week change: $____B |
| CFNAIMA3 reading [NEW] | ____  (below -0.70 = warning) |
| Sahm Rule reading [NEW] | ____  (above 0.50 = triggered) |
| Recession composite score [NEW] | ____ / 5 models in recession territory |
| FactSet revision direction [NEW] | Up / Flat / Down  │  Trend: ____ |
| Top performing sector (4 weeks) | ____ |
| TLT direction | Up / Down / Flat |
| Regime identification | Risk-on / Reflation / Deflation / Stagflation |
| Liquidity override active? | Yes / No |
| Energy signal composite [NEW] | +__ / 3  (for energy positions) |
| Current drawdown from peak | ____% |
| Open positions | ____ of max ____ |
| Portfolio heat | ____% |
| Priority sectors this week | ____ |
| Watchlist candidates (names) | ____ |

| **Monthly item (after CPI/PCE)** | **This month's reading** |
| --- | --- |
| Earnings yield vs TIPS spread [NEW] | ____%  (neutral 2-4%, below 2% = headwind) |
| Equity risk premium [NEW] | ____%  (neutral 3-5%, below 2% = stretched) |
| GDPNow vs analyst EPS direction [NEW] | Bullish / Neutral / Bearish gap |
| Macro mispricing composite [NEW] | +__ to -3  (below -1 = reduce sizing) |
| Taylor Rule deviation [NEW] | ____% positive/negative  (Atlanta Fed calc) |

# **Zero-Cost Data Sources Reference [NEW]**

All sources below are free. Bookmark these for your weekly and monthly workflow.

## **FRED Economic Data (fred.stlouisfed.org)**

| **FRED Series Code** | **What It Tracks** |
| --- | --- |
| CFNAIMA3 | Chicago Fed National Activity Index 3-month MA — business cycle gauge |
| SAHMREALTIME | Sahm Rule real-time recession indicator — labor market |
| RECPROUSM156N | Chauvet-Piger smoothed recession probability |
| T10Y3M | 10-year minus 3-month yield curve spread |
| WALCL | Fed total assets (balance sheet) |
| WTREGEN | Treasury General Account balance |
| RRPONTSYD | Overnight reverse repo facility |
| DFII10 | 10-year TIPS yield (real rate) |
| DGS10 | 10-year nominal Treasury yield |
| DTB3 | 3-month T-bill rate (for carry calculations) |
| DTWEXBGS | Trade-weighted US dollar index |
| BAMLH0A0HYM2 | High-yield credit spread (OAS) |
| NFCI | Chicago Fed National Financial Conditions Index |
| PCEPILFE | Core PCE inflation year-over-year |

## **Other Free Sources**

| **Source / URL** | **What It Provides** |
| --- | --- |
| atlantafed.org/research-and-data/data/gdpnow | GDPNow real-time GDP estimate — updated 6-7x/month |
| atlantafed.org/cqer/research/taylor-rule | Taylor Rule calculator — check monthly after CPI/PCE |
| insight.factset.com/topic/earnings | Earnings Insight PDF — forward EPS, revisions, sector breakdown — weekly, free |
| multpl.com | S&P 500 forward P/E and earnings yield — updated regularly |
| tradingster.com/cot | COT data visualization — managed money positioning charts — free |
| barchart.com/futures/commitment-of-traders | COT charts and tables — free |
| conference-board.org/topics/us-leading-indicators | Conference Board LEI — monthly, free headline |
| shillerdata.com | Robert Shiller CAPE data — monthly, free download |
| cftc.gov/MarketReports/CommitmentsofTraders | Raw CFTC COT data — every Friday 3:30 PM ET |
| newyorkfed.org/research/policy/rstar | Holston-Laubach-Williams r-star estimate — quarterly |

## **TradingView Signals (Free)**

| **TradingView Symbol / Indicator** | **What It Provides** |
| --- | --- |
| FRED:CFNAIMA3 | Business cycle gauge — add to macro watchlist |
| FRED:SAHMREALTIME | Sahm Rule — add to macro watchlist |
| FRED:T10Y3M | Yield curve — add to macro watchlist |
| FRED:DFII10 | Real rates (TIPS yield) — monthly mispricing check |
| CL1!-CL2! | Crude oil front-to-second month spread — backwardation/contango |
| 'Fed Net Liquidity' by jlb05013 | WALCL minus TGA minus RRP — community indicator, free |
| 'COT Managed Money Crude' | Search community scripts — COT data for energy |
| 'Earnings Yield vs Risk Free Rate' | Search community scripts — carry signal approximation |

# **Quick Reference: Complete Rules Summary**

## **Permission State Rules**

| **State** | **Max positions / Risk / Approach** |
| --- | --- |
| Green | Up to 20 / 0.75-1.0% risk / Momentum breakouts / Recession composite <2/5 |
| Yellow | 8-12 / 0.25-0.5% risk / Both setups / Recession composite 2-3/5 or mixed signals |
| Red | 3-5 or cash / No new entries / Recession composite 4+/5 or both SPY signals negative |

## **Two-Speed Trend Signal**

| **Signal** | **Position size allowed** |
| --- | --- |
| Both lookbacks positive (1mo and 3-6mo) | Full position size |
| Mixed (one positive, one negative) | Half position size |
| Both negative | No trade |

## **Position Sizing Formula**

Shares = (Account x Risk%) / (Entry - Stop)

Max single position = 10% of account

Capital at peak: full risk %. In drawdown: reduce proportionally.

## **Carry Signal (New)**

| **Carry = Earnings Yield minus T-Bill Rate** | **Action** |
| --- | --- |
| Above +3% | Full position eligible |
| 0% to +3% | Normal sizing |
| Negative | Reduce size 25% or skip |

## **Profit Taking Rules**

| **Profit level** | **Action** |
| --- | --- |
| +8-12% | Sell 1/3, move stop to breakeven |
| +20-25% | Sell another 1/3, trail remainder at 20d MA |
| Remainder | Trail at 20d MA, hold up to 12 weeks |

## **Drawdown Emergency Levels**

| **Drawdown from peak** | **Action** |
| --- | --- |
| 5-10% | Cut risk/trade 50%, no new positions |
| 10-15% | 3-5 positions max, no new entries |
| >15% | 100% cash, wait for confirmed green state |

## **Recession Composite Scoring**

| **Models in recession territory** | **Action** |
| --- | --- |
| 0-1 / 5 | Normal operations |
| 2-3 / 5 | Reduce max positions 25%, tighten stops |
| 4-5 / 5 | Treat as red state regardless of SPY trend |

## **Energy Three-Layer Signal**

| **Composite score** | **Action** |
| --- | --- |
| +2 or +3 | Full energy position eligible |
| +1 | Half position only |
| 0 or below | Flat or skip energy exposure |

*Framework v2.0. Based on Prometheus Research methodology (prometheus-research.com) augmented with zero-cost public data signals from FRED, CFTC, Atlanta Fed, and FactSet. For systematic swing trading with 1-12 week hold periods.*
