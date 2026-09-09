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
- `SILVER_ENABLED = True` — min balance $300. **Contract size unverified** — assumes 1.0 lot = 5,000 oz. Same check needed for XAGUSDm. **Still open as of 2026-09-08** — the news spike strategy's own Sept 4 P&L walkthrough (see below) leaned on this same assumed 5,000oz figure and it's still never been confirmed against the live account.

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
| SL | Real per-symbol stop, can trigger before the 1-min mark. **Confirmed 2026-09-08 on real Sept 4 tick data: this is a price level, not an enforced payment cap** — a fast post-release cascade can jump straight through it (measured ~$40 worst-case range on gold in the 300ms after one real trigger, against a nominal $7 SL). See Sizing below for how this is currently managed. |
| Exit | Hard 60s force-close, whichever comes first (SL or timer). Exit deviation is effectively uncapped (`EXIT_DEVIATION = 500`, was 10) as of 2026-09-08 — see "Deviation fix" below. |
| Sizing | **Flat 7% per symbol** (changed 2026-09-08, was differentiated 14.04%/14.04%/5.26%), computed live from `trade_tick_value`/`trade_tick_size`, clamped to broker min/max/step |
| Flatten | Any position/pending order on the symbol (any magic) is flattened 10 minutes before release — handles the fact that shared symbols also carry `straddle_strategy.py`'s (different-magic) positions |

**Symbols — copper removed 2026-09-08**

Only **XAUUSDm** and **XAGUSDm** are traded now. XCUUSDm (copper) was removed
after reviewing Sept 4 2026 NFP on real tick data alongside gold/silver:
copper's confirmed contract spec (contract_size=1.0, tick_value=0.01,
tick_size=0.01 → only $1 of P&L per $1 price move per 1.0 lot) is drastically
lower leverage than gold/silver, which forced the sizing formula toward an
unusually large lot count (~0.67 lots at 7% risk) just to hit the same
target risk in dollars — combined with copper's `volume_min/max/step` still
being unconfirmed against the broker (flagged since 2026-08-30, never
resolved), this was the least-trusted of the three symbols and was dropped
rather than carried further unvalidated.

**Validation status**

| Symbol | Status |
|---|---|
| **XAUUSDm** | **Validated** — 2024-01-02 to 2025-12-05, minute-level, control-tested: NFP 59.1%, CPI 54.5%, FOMC 66.7% win rate, all beat random-time controls. One real tick-level walkthrough (Sept 4 2026 NFP) also confirmed the mechanic can survive a violent, discontinuous cascade **when direction cooperates** — it does NOT establish the SL reliably caps loss when direction doesn't cooperate. Still a single event. |
| XAGUSDm | Unvalidated — offset/SL guessed by scaling off gold. One real tick-level walkthrough exists (Sept 4 2026 NFP, clean win) but the dollar figure leaned on an **assumed 5,000oz contract size**, never confirmed against `symbol_info("XAGUSDm").trade_contract_size` — treat as directionally right, not precisely right. |
| ~~XCUUSDm~~ | Removed 2026-09-08 (see above) |

As of 2026-09-04, the four FX pairs (EURUSDm, GBPUSDm, USDJPYm, USDCADm) were
**removed entirely** — as of 2026-09-08, copper followed. Strategy now trades
only XAUUSDm / XAGUSDm. Total risk budget: ~33.2% (7 symbols, original) →
~14% (flat 7% × 2, current).

**Why the entry window is "pre-release only, 5s early"**
Live rejections at the exact release moment (`"Invalid price"`, retcode 10006)
traced to two compounding effects that only exist *at/after* release:
1. the anchor price can move between being read and the order reaching the broker during the spike, and
2. brokers widen `trade_stops_level` dynamically the instant high-impact volatility hits.

Placing the straddle as a **resting** order 5s before release sidesteps both,
since stop-distance validation happens at placement time, not continuously.
This requires ~1s main-loop polling — slower polling risks stepping over the
5-second window entirely. **A narrowing to 3s was considered and explicitly
rejected on 2026-09-08** — the slippage/liquidity-gap mechanism happens in the
seconds *after* the real release triggers the order, not in how many seconds
early the resting order was placed. Moving from 5s to 2-3s early doesn't
touch that mechanism at all, it only shrinks the window's margin for the
main loop's poll cycle to reliably land inside it. Left at 5.0.

**Deviation fix (2026-09-08)**
`_close_position_at_market()` and `_flatten_symbol()` previously used
`deviation=10` on their market-order requests. At real event volatility
(measured spread of 340pts on a live release bar; separately, a real ~$40
tick-level price cascade on Sept 4), a close attempted at that tight a
tolerance could get **rejected outright** rather than filled-with-slippage —
`manage_open_trade()` would log "will retry next cycle," meaning the "hard
60-second" exit could silently drift to 65s, 90s, or longer while it kept
retrying through the worst of the move. Both methods now use a shared
`EXIT_DEVIATION = 500` constant instead — deviation is a maximum tolerance,
not a target, so this costs nothing on calm exits and only changes behavior
on volatile ones, which is exactly when a guaranteed exit matters most.

**Other fixes baked into this revision**
- `has_open_position()` / `has_own_open_trade()` — magic-filtered, so the main
  loop's routing no longer mistakes another strategy's position on a shared
  symbol for "already open" and skips `check_and_place()` (which also skipped
  the flatten logic).
- Unconditional final flatten check immediately before order placement, independent of the window-based flatten timing.
- Lot sizing pulls live tick value/size instead of a static guess table (previously off by orders of magnitude, e.g. USDJPYm computing 42 lots).

**Known operational quirk — daily gold/silver break**
XAUUSDm/XAGUSDm both pause for a daily settlement break, roughly **20:58–22:02
UTC** (confirmed via Exness's published schedule) — **21:58–23:02 WAT** in
Joseph's local time. The strategy's own tick-staleness check (>10 min old =
"market likely closed") correctly detects and logs this — confirmed 2026-09-08
against a live test run, not a bug. Worth remembering when scheduling manual
test entries: anything landing inside this window will show stale-tick
"market closed" messages that are expected, not a fault.

**Open TODO — unresolved as of 2026-09-08**
- [ ] **`main_news_spike.py` has `LIVE = False` hardcoded**, meaning the bot
      always connects via `MT5_USERNAME_TRIAL`/`MT5_SERVER_TRIAL` regardless
      of intent. During a 2026-09-08 test, the bot logged "Pre-NFP flatten
      window — already flat" for XAUUSDm/XAGUSDm while the real MT5 terminal
      (checked directly on Joseph's phone) showed both symbols genuinely
      open with real losses. Leading theory: the bot's demo-account session
      is stale/disconnected and looking at a different account entirely than
      the one being checked manually — not a bug in `news_spike_strategy.py`
      or `main_news_spike.py`'s logic itself, which both read back correctly
      on review. **Not yet confirmed** — needs `mt5.account_info()` run
      directly on the VPS against `MT5_USERNAME_TRIAL`/`MT5_SERVER_TRIAL` to
      compare login/server against the account shown on the phone, and a
      live tick check (`symbol_info_tick("XAUUSDm")`) to see if that session
      is actually receiving data. Re-test planned for a time outside the
      daily break window (see above).

**Status:** demo/paper only across all symbols — do not run on real money yet.

---

## Shared conventions
- Each strategy owns a unique `MAGIC` number (straddle: `20260716`, news spike: `20260807`) so they never mistake each other's positions for their own.
- Both wrap every `mt5.order_send()` in a helper that checks for `None` before touching `.retcode`.
- Both re-derive filling mode (`ORDER_FILLING_IOC`/`FOK`/`RETURN`) per call from `symbol_info().filling_mode` rather than hardcoding one mode.