# MT5 Straddle Strategies — README

Two standalone, **stateless** trading strategies. Neither keeps trade/order/result
state in memory — every method re-derives what it needs from MT5 directly
(`positions_get()`, `orders_get()`, `history_deals_get()`). A bot restart loses
nothing, and neither an in-memory circuit breaker nor an in-memory streak
counter can get permanently stuck.

---

## 1. `straddle_strategy.py` — Session-Open Straddle

**Edge:** Direction at session-open is close to a coin flip, but *when* a
breakout is likely is predictable (AUC 0.75, 4 walk-forward folds). Rather
than guess direction, place a buy-stop + sell-stop straddle at each symbol's
own best entry hour and let the market pick.

**Mechanics**
| Stage | Rule |
|---|---|
| Entry | Buy-stop + sell-stop at each symbol's fixed trigger hour (OCO — first fill cancels the other) |
| SL | Fixed distance: 25 pips (FX) / $20 (XAU) / $0.12 (XAG) / $600 (BTC) |
| Exit | Breakeven at trigger, then trail from 1R — no fixed TP |
| Max hold | Adaptive: force-close 1h before the symbol's *own next* trigger, capped so it never bridges a weekend (except BTC, which trades 7 days/week) |
| Breaker | **Cross-pair**: pauses all new entries if 2+ symbols each show 2+ consecutive losses. Hard fallback — never pauses longer than 10h, so it can't freeze permanently. A per-pair breaker was tested and made results worse. |

**Symbols & validated trigger hours (2024–2026 backtest, Exness M15, $90 start, 1% risk)**

| Symbol | Hour (UTC) | Trades | Win% | Result |
|---|---|---|---|---|
| EURUSDm | 08:00 | 503 | 51.1% | 567 pips |
| USDJPYm | 08:00 | 513 | 54.0% | 2,823 pips |
| GBPUSDm | 04:00 | 504 | 51.0% | 1,173 pips (nearly doubled vs. flat-24h-cap version's 625p) |
| XAUUSDm | 01:00 | 473 | 60.0% | $2,460.45 (best of a 22-hour sweep) |
| BTCUSDm | 15:00 | 512 | 56.6% | $789.25 (no cross-asset confirmation, held up on train/test split + year-normalized checks) |

GBPUSD deliberately does **not** use 08:00 UTC — UK data releases 06:00–07:00 UTC,
so 04:00 sits ahead of the catalyst while 08:00 only catches the retest.

**Gating flags (enabled with caveats)**
- `GOLD_ENABLED = True` — min balance $2,000 (SL=$20 → ~22% of a $90 account at the 0.01 lot floor)
- `BTC_ENABLED = True` — min balance $600. **Contract size unverified** — assumes 1.0 lot = 1 BTC. Check `mt5.symbol_info("BTCUSDm").trade_contract_size` on the real account.
- `SILVER_ENABLED = True` — min balance $300. **Contract size unverified** — assumes 1.0 lot = 5,000 oz. Same check needed for XAGUSDm.

**Recent fixes baked into this revision**
- XAGUSDm config was missing `pip` (silently fell back to FX default 0.0001, would've placed both legs on top of price) and `cancel_hour` (KeyError crash).
- `_get_tick()` now always calls `symbol_select()` first — root cause of a live GBPUSDm 04:00 miss: `symbol_info_tick()` returns `None` for symbols not in Market Watch, it doesn't auto-add them.
- `_safe_order_send()` wraps every `order_send()` call — a `None` return (AutoTrading off, bad filling mode, dropped connection) used to crash on `.retcode`.
- Floating-point rounding fix for the breakeven/trail SL comparison (was silently re-attempting the same BE move forever).

**Open TODOs**
- [ ] Verify `trade_contract_size` for BTCUSDm and XAGUSDm before trusting position sizes.
- [ ] Re-check gold sizing math once account balance actually supports it.

---

## 2. `news_spike_strategy.py` — Scheduled-Release Spike

**Edge:** Straddle at NFP/CPI/FOMC release, but simplified exit: **no TP, no
reload chain** — a hard 1-minute force-close, win or lose, then done for that
event (no re-entry even if time remains).

**Mechanics**
| Stage | Rule |
|---|---|
| Entry | Buy-stop + sell-stop, placed **only** in a 5-second pre-release window (stored schedule times are 5s early; window opens at the stored time and closes hard at the real release — no post-release retry) |
| SL | Real per-symbol stop, can trigger before the 1-min mark |
| Exit | Hard 60s force-close, whichever comes first (SL or timer) |
| Sizing | Per-symbol `risk_pct`, computed live from `trade_tick_value`/`trade_tick_size`, clamped to broker min/max/step |
| Flatten | Any position/pending order on the symbol (any magic) is flattened 10 minutes before release — handles the fact that shared symbols also carry `straddle_strategy.py`'s (different-magic) positions |

**Validation status**

| Symbol | Status |
|---|---|
| **XAUUSDm** | **Validated** — 2024-01-02 to 2025-12-05, minute-level, control-tested: NFP 59.1%, CPI 54.5%, FOMC 66.7% win rate, all beat random-time controls |
| XAGUSDm | Unvalidated — offset/SL guessed by scaling off gold |
| XCUUSDm | Single anecdote only — one NFP event (2026-08-07): +$29.89/lot, no SL touch. FOMC leg untestable (data gap) |

As of 2026-09-04, the four FX pairs (EURUSDm, GBPUSDm, USDJPYm, USDCADm) were
**removed entirely** — strategy now trades only XAUUSDm / XAGUSDm / XCUUSDm.
Risk budget was redistributed to preserve the original 33.2143% total across
just these three (XAU 14.04% / XAG 14.04% / XCU 5.26%) rather than shrinking it.

**Why the entry window changed to "pre-release only, 5s early" (2026-09-04)**
Live rejections at the exact release moment (`"Invalid price"`, retcode 10006)
traced to two compounding effects that only exist *at/after* release:
1. the anchor price can move between being read and the order reaching the broker during the spike, and
2. brokers widen `trade_stops_level` dynamically the instant high-impact volatility hits.

Placing the straddle as a **resting** order 5s before release sidesteps both,
since stop-distance validation happens at placement time, not continuously.
This requires ~1s main-loop polling — slower polling risks stepping over the
5-second window entirely.

**Other fixes baked into this revision**
- `has_open_position()` / `has_own_open_trade()` — magic-filtered, so the main
  loop's routing no longer mistakes another strategy's position on a shared
  symbol for "already open" and skips `check_and_place()` (which also skipped
  the flatten logic).
- Unconditional final flatten check immediately before order placement, independent of the window-based flatten timing.
- Lot sizing pulls live tick value/size instead of a static guess table (previously off by orders of magnitude, e.g. USDJPYm computing 42 lots).

**Status:** demo/paper only across all symbols — do not run on real money yet.

---

## Shared conventions
- Each strategy owns a unique `MAGIC` number (straddle: `20260716`, news spike: `20260807`) so they never mistake each other's positions for their own.
- Both wrap every `mt5.order_send()` in a helper that checks for `None` before touching `.retcode`.
- Both re-derive filling mode (`ORDER_FILLING_IOC`/`FOK`/`RETURN`) per call from `symbol_info().filling_mode` rather than hardcoding one mode.