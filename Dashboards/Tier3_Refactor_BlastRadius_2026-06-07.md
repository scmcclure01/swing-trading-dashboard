# Tier 3 Refactor — Blast Radius Analysis

**Date:** June 7, 2026
**Question:** Does extracting the screener / charts / portfolio out of `app.py` break any other tab?
**Answer:** It's safe to do, but there are **3 specific coupling points** that must be preserved. Miss any one and a *different* tab breaks silently (no error, just wrong/empty data).

---

## The cross-tab coupling map

All cross-tab data flows through `st.session_state`. The tabs are not independent — three of them form a **producer → consumer chain**:

```
Screener tab (_render_layer4_tab)   ── PRODUCES ──▶   session_state:
   run_screener_v3(...)                                 screener_results
                                                        screener_passes
                                                        screener_regime
                                                        screener_half
                                                        screener_last_run
                                                        sizer_queue
        │                                                     │
        │                                                     ▼
        │                          Position Sizer tab  ── CONSUMES ──▶ screener_results, sizer_queue
        │                          Charts tab          ── CONSUMES ──▶ screener_passes
        ▼
   (main() reads screener_results/passes via .get() and routes them into tabs 5 & 6)
```

**Plain-English version:** The Screener tab is the *only writer* of these keys. The Position Sizer and Charts tabs are *readers only*. If you run the Sizer before the Screener, it correctly shows "Run the screener first." That ordering dependency is real and intentional — any extraction must keep these exact key names and the write-before-read flow intact.

---

## The 3 coupling points that must survive the refactor

### 1. Shared `st.session_state` keys (the data bus)
The screener writes 6 keys that two other tabs read. **Extraction rule:** these keys are the public contract. If the screener moves to `screener.py`, it must still write the *same string keys* to the *same* `st.session_state`, and the Sizer/Charts modules must read those same keys. Don't rename them, don't namespace them, don't move them to a local object.

| Key | Written by | Read by |
|---|---|---|
| `screener_results` | Screener tab | Position Sizer tab, main() |
| `screener_passes` | Screener tab | Charts tab |
| `screener_regime`, `screener_half`, `screener_last_run` | Screener tab | Screener tab (cache check) |
| `sizer_queue` | Screener tab (+ self) | Position Sizer tab |

### 2. Shared module-level functions (not just render code)
Two heavy functions live at module scope and are called from inside tabs:
- `run_screener_v3()` (line 231, ~140 lines) — called only by the Screener tab.
- `build_chart()` (line 626) and `build_rrg_chart()` (line 718) — `build_chart` called by Charts tab, `build_rrg_chart` by Sector Rotation tab.

**Extraction rule:** if `build_chart`/`build_rrg_chart` move into a `charts.py`, the **Sector Rotation tab (`_render_layer3_tab`) also imports `build_rrg_chart`** — so charting can't be coupled 1:1 to the Charts tab. It's shared by two tabs. Put it in a standalone `charts.py` both import from.

### 3. ⚠️ Duplicate `cb_table` — pre-existing landmine
`cb_table` is defined **twice**: `app.py:841` and `ui_components.py:41`. The in-`app.py` copy currently shadows the import. If you extract tabs into separate modules, each module will `from ui_components import cb_table` and silently get the *other* implementation — which has different styling/preset behavior. **This must be resolved first** (delete the `app.py` copy, standardize on `ui_components.cb_table`) or the refactor will subtly change table rendering across every tab. This was already flagged as Tier 3 item #10 in the audit; it's now confirmed as a blocker, not a nicety.

---

## Tabs that are SAFE to extract independently (low coupling)

- **Portfolio tab** (`_render_portfolio_tab`) — reads `account_value` and `core_positions` but doesn't depend on screener/sizer output. Self-contained except for the shared portfolio.json helpers. **Safest to extract first.**
- **Core Allocation tab** (`_render_core_tab`) — reads `account_value`, `core_pct_deployed`, `core_positions`; all set in `main()` / sidebar. No screener dependency.
- **Macro & Permission tab** (`_render_layer0_2_tab`) — your flagship. Reads `eps_signal`, `taylor_rule`, `drawdown_state` (sidebar-set). **No screener coupling** — which means the Tier 1 visual redesign of this tab is independent of the Tier 3 refactor. They don't block each other.

---

## Recommended safe extraction order

1. **Resolve the duplicate `cb_table` first** (blocker — point #3 above).
2. Extract `charts.py` (`build_chart`, `build_rrg_chart`) — shared by 2 tabs, pure functions, easy to test.
3. Extract `screener.py` (`run_screener_v3` + the Screener tab render) — keep all 6 session_state keys identical.
4. Extract `portfolio.py` (Portfolio tab + portfolio.json helpers) — lowest coupling.
5. Leave per-tab render functions in a `tabs/` package last.
6. `app.py` becomes a thin orchestrator (~300 lines): page config, sidebar, session_state init, tab dispatch.

**Safety net:** run `tests/test_layers.py` and `tests/test_trading_logic.py` after each extraction step. The session_state contract (the 6 keys) is what to manually verify between steps — those tests won't catch a renamed key.

---

## Verdict

The Tier 3 refactor **will not break other tabs IF** the 6 session_state keys keep their exact names, `build_rrg_chart` is treated as shared by two tabs (not owned by the Charts tab), and the duplicate `cb_table` is resolved first. None of these is a deep problem — they're a checklist. Extract one module at a time, run tests between each, verify the screener→sizer→charts flow manually after the screener move.

Critically: **the Tier 1 visual redesign of the Macro tab has zero screener coupling**, so we can do that redesign now without waiting on (or risking) the refactor.
