# Monthly Trend Strategy

A position trading strategy that uses the **monthly timeframe for directional bias**
and **M15 bars for entry timing**.

Derived from 25 months of backtested M15 data (Jun 2024 – Jun 2026) across EURUSDm
and USDJPYm on Exness. Designed to run fully automated in **live MT5** with no manual
intervention.

---

## ⚠️ Concurrent Operation — Read Before Running

This strategy does **not** run alone on the account. As of 2026-08-07, four other
strategies are live or in active demo testing, each as its own standalone process:

| Strategy | File | MAGIC | Symbols |
|----------|------|-------|---------|
| Straddle (daily) | `straddle_strategy.py` | `20260716` | EURUSDm, USDJPYm, GBPUSDm, XAUUSDm, BTCUSDm |
| News Confirm | `news_confirm_strategy.py` | `20260801` | XAUUSDm |
| News Reload | `news_reload_strategy.py` | `20260810` | XAUUSDm |
| News Spike | `news_spike_strategy.py` | `20260807` | XAUUSDm (NFP/CPI/FOMC) |
| **Monthly Trend** | *(this strategy)* | **`??? — not yet documented, see below`** | EURUSDm, USDJPYm |

### 🔴 Direct symbol overlap with the straddle bot — the one that actually matters

The three news strategies only ever touch XAUUSDm, so they have zero overlap with
this strategy. **`straddle_strategy.py` is different — it trades EURUSDm and
USDJPYm too, the exact same two pairs this strategy trades.** That means on any
given day, it's entirely possible for the straddle bot and this strategy to both
have an open position on EURUSDm (or USDJPYm) **at the same time**, from two
completely independent decision processes.

This needs three things to be true before it's safe to run both together:

**1. This strategy MUST have its own unique MAGIC number**, different from
`20260716`, `20260801`, `20260807`, and `20260810`. The current version of this
README doesn't state one — if the actual code doesn't set a unique `MAGIC` and
filter every position/order read by it (the same pattern every other strategy in
this project uses — see `straddle_strategy.py`'s `_get_position()` for the
reference implementation), fix that in code before running this alongside the
straddle bot, not just in this document.

**2. The MT5 account MUST be in hedging mode, not netting mode.** In netting mode,
MT5 allows only ONE position per symbol per account, full stop — regardless of
magic number. If both this strategy and the straddle bot tried to hold independent
positions on EURUSDm at the same time under netting, MT5 would merge them into a
single net position, and BOTH strategies' own SL/trail/sizing logic would silently
operate on the wrong thing. This is not a reporting quirk — it would actively
corrupt both strategies' actual risk management. Confirm the account mode before
enabling this strategy live or on demo alongside the straddle bot.

**3. Combined risk exposure is not automatically managed.** Each strategy sizes
its own positions independently off the current account balance — this strategy
targets 1% risk per trade (see Position Sizing below), and the straddle bot
targets its own 1% risk per trade on the same pairs. Neither strategy is aware of
the other's open risk. In the worst case (both fire on the same pair on the same
day), realized risk on that one pair could be closer to ~2% of the account, not
the 1% either strategy's own math assumes in isolation. Worth being deliberate
about this rather than assuming it's handled — it isn't, by either strategy.

### What's genuinely fine, no action needed
- No overlap at all with the three news strategies (XAUUSDm only) — this
  strategy never touches gold.
- Each strategy is fully stateless and reads MT5 directly (positions, orders,
  deal history) — none of them share any in-memory state, so a crash or restart
  in one doesn't corrupt another, regardless of the symbol-overlap question above.

---

## 📌 Strategy Overview

- Prior month candle direction sets the trade bias for the entire month
- Entry only in Week 1–2 of the month — weeks 3 and 4 trend exhausts
- No M15 candle filters — monthly bias alone drives direction
- Structural stop loss based on prior day high/low (min 35 pips)
- Trailing stop triggered at 1R moves SL to breakeven immediately
- After breakeven trail by 20 pips per bar — captures full monthly moves
- Trailing also reduces losses — once 1R is reached the trade cannot lose
- One trade per symbol at a time

No fixed take profit. The trail handles all exits.

---

## 📊 Backtest Results (Exness spreads, Jun 2024 – Jun 2026)

| Pair | Trades | Win rate | Net pips | Net $ (0.01 lots) | Entry weeks |
|------|--------|----------|----------|-------------------|-------------|
| EURUSDm | 124 | 44% | +1,182p | +$118 | Week 1 + 2 |
| USDJPYm | 78 | 50% | +3,541p | +$273 | Week 1 only |
| **Combined** | **202** | | **+4,723p** | **+$391** | |

Trail settings: trigger at 1R, trail 20 pips after BE.

---

## 🕒 Entry Windows

### EURUSDm — Week 1 and Week 2 of month, Mon–Thu

| Window | UTC | WAT (UTC+1) |
|--------|-----|-------------|
| London open | 07:00–08:00 | 08:00–09:00 |
| NY overlap | 13:00–14:00 | 14:00–15:00 |

### USDJPYm — Week 1 of month only, Mon–Thu

| Window | UTC | WAT (UTC+1) |
|--------|-----|-------------|
| Tokyo open | 00:00 | 01:00 |
| London open | 07:00–08:00 | 08:00–09:00 |
| NY overlap | 13:00–14:00 | 14:00–15:00 |

**Week of month:** Day 1–7 = Week 1, Day 8–14 = Week 2, Day 15+ = no trade.

**Note on the straddle bot's own EURUSDm/USDJPYm trigger hours (08:00 UTC for
both):** this strategy's London-open window (07:00–08:00 UTC) sits directly
adjacent to that. Worth being aware that on any Mon–Thu in Week 1–2 of the
month, both strategies could be evaluating entries on the same pair within
roughly the same hour — not a bug in either one, just a real timing proximity
worth knowing about given the overlap already flagged above.

---

## 🚫 No-Trade Rules

- November and December — volatility too low
- Week 3 and Week 4 of every month — trend exhausts, reversals increase
- Friday — no new entries, hard close any open trade at 14:00 UTC (15:00 WAT)
- 21:00–23:00 UTC — spread too wide
- Weekends — never

---

## ⚙️ Entry Logic (all must pass in order)

1. Symbol is configured (EURUSDm or USDJPYm)
2. Current month is not November or December
3. Current day is Mon–Thu (Friday blocked)
4. Current hour is in the allowed window for this symbol
5. Current date is in Week 1 or 2 (EURUSDm) or Week 1 only (USDJPYm)
6. Prior month closed above its open → **BUY** | below → **SELL**
7. Structural SL (prior day low for buys, prior day high for sells) ≥ 35 pips
8. Strategy not paused (monthly failure or regime monitor)

No M15 candle condition. No EMA. No body ratio. No weekly confirmation.
The monthly bias alone determines direction.

---

## 🛡️ Trade Management

| Event | Action |
|-------|--------|
| Price reaches 1R | Move SL to breakeven — trade now risk-free |
| After breakeven | Trail SL by 20 pips tracking best price |
| Friday 14:00 UTC (15:00 WAT) | Hard close — no weekend holds |
| SL hit by MT5 | Clear trade state, look for next signal |

**SL is never widened. Never add to a position. No fixed TP.**

---

## 🔒 Protection Systems

### Monthly failure tracker
If 3 losses occur during active weeks in the same month, the strategy skips
all remaining entries for that symbol for the rest of the month.
Resets automatically on the 1st of each new month.

### Rolling regime monitor
Tracks last 6 trades across all symbols. If fewer than 2 wins in the last 6,
strategy pauses all entries until the win rate recovers.

**Note:** both of these protections are scoped to THIS strategy's own trades
only (filtered by its own MAGIC number, once that's added — see the
Concurrent Operation section above). They cannot see or react to the straddle
bot's win/loss streaks on the same pairs, and vice versa.

---

## 💰 Position Sizing

```
lot = (balance × risk_pct%) / (sl_pips × pip_value_per_lot)
minimum lot = 0.01
```

Default risk: **1% per trade**. One position per symbol at a time **for this
strategy** — this does not account for a simultaneous straddle-bot position on
the same symbol; see Concurrent Operation above.

At $9 account: expect $3–10 loss / $6–30 gain per trade at 0.01 lots.
At $100 account: lot sizing formula kicks in properly.

---

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_pct` | 1.0 | % of balance risked per trade |
| `min_sl_pips` | 35.0 | Minimum structural SL distance |
| `trail_trigger_r` | 1.0 | Move to breakeven at this R level |
| `trail_pips` | 20.0 | Trail distance after breakeven (pips) |

---

## ⚙️ Symbol Configuration

```python
SYMBOL_CONFIG = {
    "EURUSDm": {
        "pip":         0.0001,
        "spread":      0.00008,              # 8 points (Exness)
        "sl_buffer":   0.0003,
        "entry_weeks": [1, 2],               # Week 1 and 2 of month
        "entry_hours": [7, 8, 13, 14],       # London open + NY overlap
    },
    "USDJPYm": {
        "pip":         0.01,
        "spread":      0.018,                # 18 points (Exness)
        "sl_buffer":   0.03,
        "entry_weeks": [1],                  # Week 1 only
        "entry_hours": [0, 7, 8, 13, 14],   # Tokyo + London + NY overlap
    },
}
```

---

## 📁 Usage

```python
strategy = MonthlyTrendStrategy(
    risk_pct=1.0,
    min_sl_pips=35.0,
    trail_trigger_r=1.0,
    trail_pips=20.0,
    backtest_mode=False,
    initial_balance=100.0,
)

# Entry — call on every M15 bar close
signal = strategy.generate_signal("EURUSDm")
if signal["signal"]:
    # place order: signal["entry_price"], ["stop_loss"], ["lot_size"]
    # take_profit is None — trail manages the exit

# Trade management — call on every M15 bar close while trade is open
status = strategy.manage_open_trade("EURUSDm")

# When trade closes (SL hit or trail exit)
strategy.record_result("EURUSDm", was_win=True, current_month=6)

# New month reset
strategy.reset_month("EURUSDm", new_month=7)

# Check status
print(strategy.get_performance_summary("EURUSDm"))
```

---

## 📦 Data Fetched Internally per Symbol

| Timeframe | Bars | Purpose |
|-----------|------|---------|
| M15 | 300 | Entry timing |
| D1 | 5 | Structural SL (prior day high/low) |
| MN1 | 3 | Monthly bias direction |

No external data needed. Pass only the symbol.

---

## ⚠️ Key Findings from Backtest

- M15 entry conditions (EMA, body ratio, RSI) add no value — removed entirely.
- Monthly bias follow-through: 58.3% WR on EURUSD over 25 months — the real edge.
- Week of month is critical: all edge in Weeks 1–2. Weeks 3–4 consistently negative.
- Trailing at 1R beats fixed 3R by 10x in net pips — captures the large monthly moves.
- Trailing also eliminates losses on trades that reach 1R then reverse (exit at BE).
- Exness spreads: EURUSDm 0.8p, USDJPYm 0.18p — negligible cost.
- GBPUSD, EURJPY, CADJPY all failed — monthly bias WR too low on those pairs.
- Nov–Dec dead months. Mar, Apr, Jun strongest (+420p, +505p, +439p on EURUSD).
- USDJPY Week 1 strongest single window across all tests (+3,541p net).

---

## 📋 Before Enabling Alongside the Straddle Bot — Checklist

- [ ] Confirm this strategy's actual `.py` code has a unique `MAGIC` number
      (not `20260716`, `20260801`, `20260807`, or `20260810`), and every
      position/order read is filtered by it
- [ ] Confirm the MT5 account is in **hedging mode**, not netting mode
- [ ] Be aware combined risk on EURUSDm/USDJPYm can reach ~2% on days both
      strategies fire, not the 1% either one assumes alone
- [ ] Each strategy still runs as its own separate process/terminal — no
      shared loop, matching every other strategy in this project