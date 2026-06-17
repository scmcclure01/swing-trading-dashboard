# Framework Audit — Action Items
**Audit date:** 2026-06-04
**Last updated:** 2026-06-07

---

## Summary

| Criticality | Total | Completed | Remaining |
|-------------|-------|-----------|-----------|
| Critical | 4 | 3 | 1 |
| High | 9 | 8 | 1 |
| Medium | 10 | 10 | 0 |
| Low | 11 | 7 | 4 |
| **Total** | **34** | **28** | **6** |

---

## Remaining Items

### Critical

- [ ] **C3: Fix SPY lookback conflation in project instructions** — Project settings change (user must update manually). SPY permission state uses 1M/6M. Stock-level two-speed uses 1M/3M. Current text says "3-6 month" which conflates them.

### High

- [ ] **H9: Review L3 rotation trigger philosophy and flow strength thresholds** — Now that we have real fund flow data from etfdb.com, revisit the entire Layer 3 signal framework. Are the $150M/$300M/$50M/$100M thresholds correct? Should thresholds be relative to ETF AUM? Is directional consistency a better signal than net dollar amount? Backtest using historical flow data (goes back to 2015).

### Medium

*(all medium items complete)*

### Low

- [ ] **L8: Rebuild ETF exit monitoring on true 2-consecutive-week signals** — The ETF exit-alert card (Portfolio tab) currently uses single-snapshot proxies: flow "Outflows" classification and RS "Lagging" trend. The framework actually requires *2 consecutive weeks* of net outflows OR 2+ weeks of RS decline. The current proxy does not accomplish this and is not trusted. Rebuild: persist weekly flow + RS history (e.g. a small rolling store in portfolio.json or a separate JSON), and fire the flow/RS exit triggers only on a genuine 2-week streak. The 20d-MA trigger is exact and can stay. Until rebuilt, treat the flow/RS alerts as informational only.

- [ ] **L9: Build Energy three-layer signal (Layer 4 Step 5)** — Not yet in the app. Framework specifies a 3-part composite for energy/commodity names: (A) crude term structure CL1!−CL2! (backwardation bullish), (B) CFTC COT positioning percentile (below 20th = contrarian bullish, above 80th = fragile), (C) XLE vs 50d SMA trend. Score +1/0/−1 each, weight 50/25/25, composite sizes energy entries. Main effort is sourcing/parsing CFTC COT data (cftc.gov, published Fri 3:30 PM ET; or tradingster.com/barchart). Narrow applicability (energy names only) — hence low priority.

- [ ] **L10: Make Taylor Rule feed sizing instead of informational-only** — The Taylor Rule deviation is a manual monthly sidebar dropdown but the Gate Summary marks it "Informational" and nothing consumes it. Framework treats a >1% deviation as a positioning/sizing input (Fed too loose → reduce rate-sensitive longs; too tight → favors risk-on). Decide the concrete sizing/tilt effect and wire it in, or remove the input if it's not going to drive anything.

- [ ] **L11: Add volume confirmation to exit triggers** — Framework repeatedly specifies "close below MA *on above-average volume*" for exits (Tactical 50d/20d and ETF 20d). The Portfolio tab's trade-management status and the new ETF exit card key off price vs MA but do not gate on the volume condition. Add the above-average-volume check so a low-volume dip below the MA doesn't read as a hard exit.

*(L1–L7 resolved — see Completed Items)*

---

## Completed Items

### Critical
- [x] **C1: Standardize layer count across all docs** — Renumbered all layers L0–L9 (eliminated L0.5/L1.5). Updated 15 files, 284 line changes. App.py function names updated.
- [x] **C2: Rewrite weekly_workflow.md for v4** — Full rewrite. Added Core allocation check, Velocity Flag, deployment floor, corrected drawdown tiers, updated sector cap to 25%, references Streamlit instead of CLI.
- [x] **C4: Resolve account value mismatch ($71K vs $100K)** — Dynamic account value from portfolio.json (cash_balance + open positions at live prices). Removed manual sidebar inputs. Peak equity tracked for drawdown. Actual cash balance set to $56,764.25 from Fidelity.

### High
- [x] **H1: Replace implied fund flows with actual etfdb.com data** — Scraped real daily fund flows from etfdb.com. No API key needed. Returns 5-day and 4-week net flows in $M plus directional consistency.
- [x] **H2: Filter "Too extended" stocks from screener** — PASS filter now enforces entry zone. "Too extended" never passes. Added separate Monitoring table for stocks with strong fundamentals but untradeable distance from MA.
- [x] **H3: Automate screening log updates** — Added Step 9 to weekly review skill. Every review auto-appends a dated entry to screening_log.md.
- [x] **H4: Fix missing _render_position_sizer function** — Removed dead code call from L5 tab.
- [x] **H5: Track peak equity for drawdown** — peak_equity tracked in portfolio.json as high-water mark. Drawdown computed dynamically.
- [x] **H6: Deprecate CLI screener** — Archived screener.py, screener_v3.py, Run Screener.command, run_screener.sh to Screener/Archive/.
- [x] **H7: Add SMH to Velocity Flag** — Added Semiconductors/SMH to SECTOR_ETFS, SECTOR_COLORS, RISK_ON_SECTORS. Now appears in RRG, Velocity Flag, fund flows, and regime classification.
- [x] **H8: Fix duplicate trade #4** — Renumbered trade journal 1–21 sequentially.

### Medium
- [x] **M1: Migrate native Streamlit components** — L3 tab migrated to _tile()/_card()/_gate_bar_html(). Design standard updated with exceptions table.
- [x] **M2: Remove "Mixed" regime** — Replaced with aggregate RS weight classification. Always produces one of 4 framework regimes.
- [x] **M3: Document RRG-to-Phase mapping** — Added RRG quadrant → Phase mapping table to framework doc.
- [x] **M4: Yellow state risk default** — Confirmed 0.5% is intentional (top of 0.25-0.5% range). No change.
- [x] **M5: Sector concentration check** — Position sizer now checks 25% sector cap against existing open positions.
- [x] **M6: Verify QCOM sizing** — Confirmed oversized (13.7% vs 10% cap, sized against old $100K). Flagged in portfolio.json. Live position, no adjustment.
- [x] **M7: Document stop buffer rule** — "Just below base" → "1% below base" in framework doc.
- [x] **M8: Pending order fields** — Added stop_price_plan, risk_dollars_est, max_hold to XLK pending order.
- [x] **M9: Implement Earnings Carry signal** — Forward EY (1/forwardPE from yfinance) minus DTB3 (FRED CSV). Screener displays carry spread and label (Strong >+3%, Positive 0–3%, Negative <0%). Negative carry auto-flags in notes and reduces position size 25% in the sizer.
- [x] **M10: Reconcile 50d MA exit vs 20d MA trailing stop** — No conflict: pre-T2 hard exit = 50d MA on volume, post-T2 trailing = 20d MA (tighter, replaces 50d rule). Updated exit triggers and quick reference checklist in framework doc.

### Low — resolved 2026-06-06
- [x] **L1: Archive or delete screener.py (v1)** — Confirmed no live references. Only matches in app.py are the `run_screener_v3` function name, not the archived file. Already in Screener/Archive/.
- [x] **L2: Delete app_archive_20260508.py** — Deleted from project root. Git retains history.
- [x] **L3: Delete duplicate layer0_dashboard_backup.html files** — Deleted both copies (root and Dashboards/Archive/). Superseded by Streamlit.
- [x] **L4: Update .gitignore for lock files and __pycache__** — Already covered: `.gitignore` contains `__pycache__/` and `.~lock.*`. No change needed.
- [x] **L5: Document intentional gate bar dot color difference** — Added explanatory note to streamlit_design_standard.md (dot `#1D7A2A` is intentionally lighter than text `#27500A` for legibility on the `#D6F0D6` gate-bar background).
- [x] **L6: Update weekly_review_prompt.md manual check section** — Moved Fed Net Liquidity, HYG/IEF, and Chauvet-Piger from "manual check" to "auto-fetch" with their automation sources noted.
- [x] **L7: Generate current-week playbook, schedule recurring review** — Closed with no action per user. Will be caught on next weekly cadence.
