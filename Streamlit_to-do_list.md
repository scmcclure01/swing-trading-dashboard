# Streamlit / Automation To-Do List

Running list of deferred items and automation improvements. Updated as new items are identified.

---

## Automation — Data Inputs

- [ ] **Gmail connector for FactSet automation** — Install Gmail MCP connector so Claude can receive the weekly FactSet Earnings Insight PDF directly from email, extract key metrics (revision direction, guidance ratio, market reaction asymmetry), and feed into the weekly Layer 0 output automatically. No manual download required.

- [x] **Fed Net Liquidity — FRED automation** — Hardwired into `calc_layer0()` in `app.py`. Fetches WALCL - WTREGEN - RRPONTSYD from FRED public CSV (no API key), computes 4-week change, and renders in the Bond & Liquidity card on the L0/L2 dashboard tab. ✅ Complete.

- [ ] **ETF Fund Flows — database automation** — Currently manual (ETFdb.com weekly check). When database work begins, automate Layer 3 flow signal using implied flows: Δ shares outstanding × price via yfinance. Requires persistent DB to store weekly snapshots. FMP API (~$15/mo) is an alternative data source — evaluate `/stable/etf/` endpoints. *(Do not begin until directed.)*

- [x] **HYG/IEF ratio** — Hardwired into `calc_layer0()` via `fetch_macro_data()` yfinance download. Computes ratio, compares to 1M ago, sets `liquidity_tighten` flag, and renders in the Bond & Liquidity card. ✅ Complete.

- [x] **Chauvet-Piger recession probability** — Hardwired into `fetch_fred_data()` via FRED API (RECPROUSM156N). Feeds into the recession composite on the dashboard. ✅ Complete.

- [ ] **Conference Board LEI** — Currently manual. Evaluate automation options.

---

## Framework — Document Rebuild

- [x] **Full framework rebuild — Layer 3 integration** — Completed 2026-05-23. Framework rebuilt as v4 with Core-Satellite structure, Velocity Flag, recalibrated drawdown tiers, and deployment floor. All docs updated. ✅ Complete.

---

## Streamlit Dashboard

- [x] **Wire fed_net_liquidity.py into dashboard** — Duplicate of Fed Net Liquidity item above. Already live in the L0/L2 tab. ✅ Complete.

- [ ] **Add FactSet earnings revision card** — Once Gmail connector is live, auto-populate the FactSet signal (revision direction, guidance ratio) on the dashboard each Friday.

- [ ] **(Low priority) Screener tab — blue header on top two tables** — The Actionable Setups and Watchlist tables use `st.data_editor` (glide-data-grid canvas) for the Size checkboxes. CSS `th`/`td` selectors and `--gdg-*` theme variables both failed to color the header blue to match the bottom Monitoring table. Keep checkboxes as-is. Revisit later — likely requires converting to `cb_table` HTML with a separate selection control, or a Streamlit version where the canvas honors injected theme vars.

---

## Portfolio Management

- [ ] **Portfolio tracker & trade journal reconciliation** — Download CSV from Fidelity and reconcile against `portfolio.json` and `trade_journal.md`. Clean up any discrepancies.

---

## UI — Revisit After Road Time

- [ ] **(Low priority) Position Sizer — cards vs. table** — Currently renders each queued order as a per-order card (more detail per order, but more vertical scrolling with many orders). Considered a true table (all orders visible at once, management details via hover/second line) but kept cards for the detail. Revisit after a few weeks of real use to decide if a compact table reads better with larger queues. Decided 2026-06-07.

---

*Last updated: 2026-06-07*
