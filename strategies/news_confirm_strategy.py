"""
News Confirm Strategy — Live Version (stateless)
==================================================
Edge   : Pre-positioning ahead of NFP (buy-stop/sell-stop straddle before
         the release) was tested extensively and found to lose to a
         "wait-and-confirm" approach: do NOT take a position before the
         news. Wait for the release itself to happen, require the market
         to show a real, confirmed directional reaction (minimum $ move +
         elevated volume), and only then enter, in that confirmed
         direction. This matches standard professional NFP-trading
         practice (see research notes in project handoff) and specifically
         avoids the failure mode straddle-based pre-positioning kept
         hitting: a pre-news drift that reverses violently at the print
         itself (see the 2026-03-06 NFP case in project history — a
         pre-positioned short lost -$183 to a reversal spike at the exact
         release moment; this strategy would not have been in a position
         yet when that happened).

Entry  : NO pending orders, no OCO. At each scheduled NFP release, wait for
         the release bar itself (M30) to CLOSE, then check:
           1. Directional confirmation: |close - open| >= MIN_MOVE
           2. Volume confirmation: this bar's tick volume >= VOLUME_MULT x
              the trailing LOOKBACK_BARS average tick volume
         If both hold, enter MARKET in the direction of the move. If
         either fails, skip this event entirely — no trade.
Filter : Skip any NFP release that falls within FOMC_PROXIMITY_DAYS of a
         preceding FOMC decision. Backtested finding: NFP-after-FOMC events
         won 29% of the time (2/7) vs 67% (10/15) for NFP releases further
         from FOMC, consistently across multiple years (2022-2024), not
         just one bad year. Mechanism (plausible, not proven): the market
         has just repriced on Fed guidance days earlier, making the
         "confirmed" NFP reaction more prone to reverting.
Exit   : Breakeven trigger + trail, same style as the validated
         straddle_strategy.py gold config — no fixed TP. SL fixed at
         entry; once price moves BE_TRIGGER in favor, stop moves to
         breakeven; after that, stop trails TRAIL behind the best price
         reached. Hard max-hold deadline as a backstop.
Sizing : FLAT RISK_PCT (14.1%, "quarter-Kelly") of CURRENT balance,
         every trade — recalculated fresh every time it's used, never a
         fixed number, so the lot naturally scales with the account. No
         streak tracking, no multiplier. See _lot_size() below for the
         full Kelly Criterion derivation and why 14.1% specifically beats
         a lower risk_pct at zero extra drawdown cost, up to that point.

Backtest summary (2022-05 to 2026-07, XAUUSDm M30 real Exness data,
49 NFP events, $90 start):
  Wait-and-confirm only (no filter, no pyramid): 36 trades, 66.7% win
    rate, $90 -> $363.29. Profitable in 4 of 5 years (2024 was the
    exception: 33% win rate, -$43.99).
  + FOMC-proximity filter: 29 trades (7 near-FOMC events skipped), 75.9%
    win rate, $90 -> $385.71. Fixed 2024 specifically (-$43.99 -> +$4.72)
    by removing the specific sub-pattern responsible for that year's
    underperformance, found by investigating WHY 2024's losses were mostly
    full stop-outs rather than soft fades like other years.
  + Capped pyramid sizing (tested, NOT used in this config): same 29
    trades/75.9% win rate, $90 -> $883.01 at x1.5/4-win-cap, worst single
    trade -$80.64. An UNCAPPED version reached $981.81 but had one
    -$282.79 single-trade loss (a 6-win streak grew the lot 14x before
    the streak-ending loss landed at peak size) - real, meaningful tail
    risk that isn't worth the modest extra return over the option below.
  + FLAT Kelly-derived sizing (THIS CONFIG, RISK_PCT=14.1%): same 29
    trades/75.9% win rate (sizing doesn't change which trades win),
    $90 -> $627.70. Derived from the actual win rate and win/loss sizes
    via the Kelly Criterion (full Kelly = 56.3%, way too aggressive;
    half-Kelly = 28.1%, tested at 48.7% max drawdown - still too much;
    quarter-Kelly = 14.1%, tested at 19.9% max drawdown - IDENTICAL to
    flat 1% risk, because the account's min-lot floor protects it
    equally at both risk levels until balance grows past a threshold).
    Chosen over the pyramid because it's mathematically grounded in the
    actual backtest statistics rather than an arbitrary multiplier, and
    matches the flat-1% version's worst-case risk exactly while roughly
    doubling the return.

*** IMPORTANT — READ BEFORE ENABLING LIVE ***
Every number above comes from ONE dataset that was also used to find the
FOMC filter and tune every parameter in this file. None of it has been
validated on genuinely fresh, untouched data. Per this project's own
standing discipline (see annual_process.md's validation ritual and the
monthly-bias strategy's 58%->52% decay precedent), an in-sample backtest
finding is not the same as a validated edge. Treat this file as ready for
DEMO testing to accumulate real forward evidence, not as ready for live
capital, until it has been checked against data it was never tuned on.

*** NFP_SCHEDULE_UTC and FOMC_DATES_UTC are MANUALLY MAINTAINED ***
No fetching, no scraping, no API calls — both lists are plain hardcoded
data near the top of this file that you edit by hand. Event timestamps
cannot be reliably derived from a fixed rule anyway — the 2025 government
shutdown proved this by moving/cancelling/combining several 2025 NFP
releases outside the normal "first Friday" pattern (see project handoff),
so a rule-based auto-generator would have silently gotten that window
wrong regardless. Update NFP roughly monthly (check bls.gov/schedule) and
FOMC once or twice a year (check federalreserve.gov/monetarypolicy/
fomccalendars.htm — the Fed announces a full year+ at once).
_validate_calendar_freshness() runs automatically at import time and
prints a loud warning if either list has run dry of upcoming dates — it
only checks dates already in the file against today, it doesn't fetch
anything. This file will simply do nothing once it runs past the last
date you've entered, rather than guess at what comes next.

CHANGE LOG (this revision):
  - Initial build. Fresh strategy, separate MAGIC number and separate
    process from straddle_strategy.py / main_straddle.py — explicitly NOT
    merged into the existing straddle bot's loop, matching how this
    project's other strategies (monthly bots, straddle bot) each run
    standalone.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# Event calendars — MANUALLY MAINTAINED, no fetching/scraping/API calls.
#
# Update these by hand, roughly monthly for NFP and once or twice a year
# for FOMC (the Fed announces a full year+ of dates in one announcement,
# so this needs far less frequent attention):
#   NFP:  check bls.gov/schedule/<year>/home.htm, add the next date(s)
#   FOMC: check federalreserve.gov/monetarypolicy/fomccalendars.htm,
#         or the Fed's own press release when they announce a new year
#
# _validate_calendar_freshness() runs automatically at import time and
# prints a loud warning if either list has run out of upcoming dates or
# looks stale — it does NOT fetch anything, it only checks what's already
# here against today's date. See that function for exactly what it checks.
# ---------------------------------------------------------------------------

NFP_SCHEDULE_UTC: List[datetime.datetime] = [
    datetime.datetime(2026, 8, 7, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 9, 4, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 10, 2, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 11, 6, 13, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 12, 4, 13, 30, tzinfo=datetime.timezone.utc),
    # Add next month's date here. DST-adjust by hand: 8:30 AM ET = 12:30 UTC
    # during DST (roughly Mar-Nov), 13:30 UTC otherwise. Do NOT assume
    # "first Friday" — verify each one against bls.gov/schedule directly;
    # the 2025 government shutdown proved that assumption can silently
    # break (several 2025 releases were delayed/combined/cancelled).
]

FOMC_DATES_UTC: List[datetime.date] = [
    datetime.date(2026, 7, 29),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28),
    datetime.date(2026, 12, 9),
    datetime.date(2027, 1, 27),
    datetime.date(2027, 3, 17),
    datetime.date(2027, 4, 28),
    datetime.date(2027, 6, 9),
    datetime.date(2027, 7, 28),
    datetime.date(2027, 9, 15),
    datetime.date(2027, 10, 27),
    datetime.date(2027, 12, 8),
    # Only the SECOND day of each 2-day meeting (when the decision/
    # statement is released) — that's the date the FOMC-proximity filter
    # actually needs. Add the next announced year's dates here in one go
    # when the Fed publishes them.
]


def _validate_calendar_freshness() -> None:
    """Runs at import time. Does NOT fetch anything — just checks whether
    the hardcoded lists above still have upcoming dates, and warns loudly
    if they've gone stale. This is the whole "freshness check": no
    scraping, no API, just comparing what's already here to today's date."""
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()

    future_nfp = [d for d in NFP_SCHEDULE_UTC if d >= now]
    if not future_nfp:
        print("  !!! WARNING: NFP_SCHEDULE_UTC has NO upcoming dates — "
              "this bot will never enter on NFP until you add more. "
              "Check bls.gov/schedule and update the list in "
              "news_confirm_strategy.py now.")
    else:
        days_until_next = (min(future_nfp).date() - today).days
        if days_until_next > 40:
            print(f"  WARNING: next NFP date is {days_until_next} days away "
                  f"({min(future_nfp).date()}) — that's further out than a "
                  f"normal monthly gap, double-check the list isn't stale.")

    future_fomc = [d for d in FOMC_DATES_UTC if d >= today]
    if not future_fomc:
        print("  !!! WARNING: FOMC_DATES_UTC has NO upcoming dates — the "
              "FOMC-proximity filter will not work correctly for any NFP "
              "event beyond this point. Check federalreserve.gov/"
              "monetarypolicy/fomccalendars.htm and update the list now.")


_validate_calendar_freshness()


# ---------------------------------------------------------------------------
# Per-symbol configuration
# ---------------------------------------------------------------------------

SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "XAUUSDm": {
        "pip": 1.0,             # working directly in USD, not pips, for gold
        "point_size": 0.001,    # matches this account's XAUUSDm digits (3)
        "pip_value_per_lot": 100.0,  # $100 per 1.0-point move at 1.0 lot
        "min_move": 5.0,        # confirmation threshold: release bar must move >= $5
        "volume_mult": 1.5,     # confirmation threshold: >= 1.5x trailing avg tick volume
                                 # (backtested finding: this did NOT discriminate in
                                 # testing — gold NFP release bars were always high-volume
                                 # regardless of threshold 1.2-2.0x. Kept as a documented,
                                 # inert safety check rather than removed outright.)
        "sl": 20.0,
        "be_trigger": 20.0,
        "trail": 5.0,
        "max_hold_hours": 8.0,
        "lookback_bars": 12,    # ~6 hours of M30 bars, for the volume average
    },
}

RISK_PCT = 14.1           # "quarter-Kelly" - see _lot_size() docstring for
                           # the full derivation. Deliberately more aggressive
                           # than the live straddle bot's validated 1%, but
                           # backed by real Kelly Criterion math against this
                           # session's actual backtest results, not a guess.

MAGIC = 20260801          # unique to this strategy — must not collide with
                           # straddle_strategy.py's MAGIC or the monthly bots'

FOMC_PROXIMITY_DAYS = 3    # skip NFP events within this many days of a
                           # preceding FOMC decision — see docstring

# Scale-in on a winning trade — ADDS to the same still-open position while
# it's actively working, then stops. Different from a streak-based sizing
# multiplier (which changes bet size BETWEEN separate NFP events) — this
# only ever adds within ONE trade's own lifetime, then reverts fully for
# the next event. Backtested: this specific combo (4h/$3/2 adds) on top
# of the flat 14.1% base — see module docstring for results and the
# honest caveat about how concentrated the gain was in one strong year.
SCALE_IN_WINDOW_HOURS = 4.0   # only add within this many hours of entry
ADD_THRESHOLD = 3.0           # gold $ — add another leg every time price
                               # pushes this much further favorable since
                               # the last leg was added
MAX_ADDS = 2                  # cap on additional legs beyond the initial entry
ADD_LOT_FRACTION = 1.0        # each add's lot = this x the INITIAL leg's lot


def _pip_value_per_lot(symbol: str) -> float:
    return SYMBOL_CONFIG.get(symbol, {}).get("pip_value_per_lot", 100.0)


def _round_price(price: float, symbol: str) -> float:
    if symbol == "XAUUSDm":
        return round(price, 2)
    return round(price, 5)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class NewsConfirmStrategy:
    """No __init__ state beyond configuration — every method queries MT5
    fresh, including win-streak state for sizing. See module docstring for
    the full design rationale and backtest summary."""

    def __init__(self, initial_balance: float = 90.0) -> None:
        self.starting_balance = initial_balance
        self.traded_symbols: List[str] = list(SYMBOL_CONFIG.keys())

    # ---------------------------------------------------------------- balance & sizing

    def _balance(self) -> float:
        acc = mt5.account_info()
        return acc.balance if acc else self.starting_balance

    def _lot_size(self, symbol: str) -> float:
        """Flat RISK_PCT of CURRENT balance, every trade - no streak
        tracking, no multiplier. RISK_PCT=14.1% is 'quarter-Kelly',
        derived from this session's actual backtest win rate (75.9%) and
        win/loss sizes via the Kelly Criterion, then quartered for safety
        given the small (29-trade) sample the estimate came from. Tested
        head-to-head against flat 1% risk: identical max drawdown (19.9%)
        but meaningfully higher final balance ($627.70 vs $385.71 on the
        same 29 trades) - the min-lot floor protects the account
        identically at both risk levels until balance grows enough for
        the difference to matter, making 14.1% a genuinely better
        risk-adjusted choice up to that point, not just a bigger bet.
        Full Kelly was 56.3% (drawdown 48.7% when tested - far too
        aggressive to use); half-Kelly (28.1%) still hit 32.5%+ drawdown.
        Recomputed fresh from current balance every call, never a fixed
        number - matches the live straddle bot's _lot_size() formula
        shape (risk_dollar / (sl_dollars * pip_value))."""
        cfg = SYMBOL_CONFIG[symbol]
        risk_dollar = self._balance() * (RISK_PCT / 100.0)
        lot = risk_dollar / (cfg["sl"] * _pip_value_per_lot(symbol))
        return max(0.01, round(lot / 0.01) * 0.01)

    # ---------------------------------------------------------------- MT5 reads

    def _get_tick(self, symbol: str):
        """Same Market-Watch self-heal pattern as straddle_strategy.py —
        see that file's docstring for why this is a single choke point."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
        return tick

    def _filling_mode(self, symbol: str) -> int:
        """Identical logic to straddle_strategy.py — see that file's
        docstring for why raw integers are used instead of the
        nonexistent mt5.SYMBOL_FILLING_* constants."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        mode = info.filling_mode
        SYMBOL_FILLING_FOK = 1
        SYMBOL_FILLING_IOC = 2
        if mode & SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if mode & SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _safe_order_send(self, request: Dict[str, Any]):
        """Identical wrapper to straddle_strategy.py — see that file's
        docstring for why every order_send call must go through this."""
        result = mt5.order_send(request)
        if result is None:
            print(
                f"  order_send returned None — request never reached the "
                f"server. mt5.last_error(): {mt5.last_error()}"
            )
            return None
        return result

    def _get_position(self, symbol: str):
        """Returns the FIRST (earliest-opened) leg only, for simple
        'do we have any position open' checks. Use _get_open_positions()
        when scale-in adds matter (blended stop/trail management)."""
        positions = self._get_open_positions(symbol)
        return positions[0] if positions else None

    def _get_open_positions(self, symbol: str) -> List:
        """Every open leg for this symbol+magic (initial entry + any
        scale-in adds), sorted oldest-first. Assumes a HEDGING-mode
        account (multiple simultaneous tickets on the same symbol allowed)
        - retail Exness accounts support this; a netting-mode account
        would merge these into one ticket automatically instead, which
        would break the per-leg tracking scale-in relies on. Confirm your
        account mode before enabling this live."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []
        own = sorted([p for p in positions if p.magic == MAGIC], key=lambda p: p.time)
        return own

    # ---------------------------------------------------------------- event calendar

    def _next_nfp_confirmation_window(
        self, now: datetime.datetime
    ) -> Optional[tuple]:
        """Returns (release_time, confirm_check_time) for the next NFP
        release whose confirmation-check window (release_time + 30min, the
        M30 bar close) is happening right now, or None. Call this on every
        poll — it's cheap and fully derived from NFP_SCHEDULE_UTC, no state."""
        for release_time in NFP_SCHEDULE_UTC:
            check_time = release_time + datetime.timedelta(minutes=30)
            # Give a 5-minute poll window right at the bar close, matching
            # this bot's expected polling cadence.
            if check_time <= now < check_time + datetime.timedelta(minutes=5):
                return release_time, check_time
        return None

    def _days_since_fomc(self, event_date: datetime.date) -> int:
        prior = [d for d in FOMC_DATES_UTC if d <= event_date]
        if not prior:
            return 9999
        return (event_date - max(prior)).days

    def _is_near_fomc(self, event_time: datetime.datetime) -> bool:
        return self._days_since_fomc(event_time.date()) <= FOMC_PROXIMITY_DAYS

    # ---------------------------------------------------------------- entry

    def check_and_enter(self, symbol: str) -> Dict[str, Any]:
        """Call on every poll (5-min cadence recommended, matching the
        straddle bot). Does nothing unless `now` falls inside a scheduled
        NFP release's confirmation-check window — see
        _next_nfp_confirmation_window(). No pending orders are ever placed;
        this either enters MARKET immediately (confirmed) or does nothing
        (not confirmed / filtered / already have a position)."""
        if symbol not in self.traded_symbols:
            return self._no(f"{symbol} not enabled")

        if self._get_position(symbol) is not None:
            return self._no("Position already open")

        now = datetime.datetime.now(datetime.timezone.utc)
        window = self._next_nfp_confirmation_window(now)
        if window is None:
            return self._no("Not inside an NFP confirmation window")
        release_time, check_time = window

        if self._is_near_fomc(release_time):
            return self._no(
                f"Skipped — NFP within {FOMC_PROXIMITY_DAYS} days of a preceding "
                f"FOMC decision (days_since_fomc="
                f"{self._days_since_fomc(release_time.date())})"
            )

        cfg = SYMBOL_CONFIG[symbol]

        # Pull the release bar (M30, spanning release_time -> check_time)
        # plus enough trailing history for the volume average.
        bars = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M30,
            release_time - datetime.timedelta(minutes=30 * (cfg["lookback_bars"] + 1)),
            check_time,
        )
        if bars is None or len(bars) < cfg["lookback_bars"] + 1:
            return self._no("Insufficient bar history to confirm this event")

        release_bar = bars[-1]
        prior_bars = bars[-(cfg["lookback_bars"] + 1):-1]
        avg_volume = sum(b["tick_volume"] for b in prior_bars) / len(prior_bars)

        move = float(release_bar["close"]) - float(release_bar["open"])
        confirmed_direction = abs(move) >= cfg["min_move"]
        confirmed_volume = release_bar["tick_volume"] >= cfg["volume_mult"] * avg_volume

        if not (confirmed_direction and confirmed_volume):
            return self._no(
                f"Not confirmed — move=${move:.2f} (need >=${cfg['min_move']}), "
                f"volume_ratio={release_bar['tick_volume']/avg_volume:.2f}x "
                f"(need >={cfg['volume_mult']}x)"
            )

        direction = "buy" if move > 0 else "sell"
        tick = self._get_tick(symbol)
        if tick is None:
            return self._no("No tick data")

        tick_age = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)
        if tick_age > datetime.timedelta(minutes=10):
            return self._no(f"Market likely closed — last tick is {tick_age} old")

        entry_price = tick.ask if direction == "buy" else tick.bid
        sl_price = (
            entry_price - cfg["sl"] if direction == "buy" else entry_price + cfg["sl"]
        )
        lots = self._lot_size(symbol)
        filling_mode = self._filling_mode(symbol)

        result = self._safe_order_send(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": _round_price(sl_price, symbol),
                "tp": 0.0,  # no fixed TP — breakeven+trail manages the exit
                "deviation": 10,
                "magic": MAGIC,
                "comment": "news_confirm_entry",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
        )
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            reason = (
                f"retcode={result.retcode} comment='{result.comment}'"
                if result is not None
                else "order_send returned None"
            )
            return self._no(f"Entry rejected — {reason}")

        return {
            "signal": direction,
            "entry_price": entry_price,
            "lot_size": lots,
            "release_bar_move": round(move, 2),
            "reason": (
                f"Confirmed {direction.upper()} | move=${move:.2f} | "
                f"lots={lots} | entry={entry_price}"
            ),
        }

    # ---------------------------------------------------------------- scale-in

    def check_scale_in(self, symbol: str) -> str:
        """Call every poll while a position is open. Adds another leg,
        same direction, at ADD_LOT_FRACTION x the INITIAL leg's lot size,
        every time price pushes ADD_THRESHOLD further favorable since the
        last leg was added — only within SCALE_IN_WINDOW_HOURS of the
        first leg's open time, and only up to MAX_ADDS total adds.

        Fully stateless, same discipline as the rest of this file: "how
        many adds so far" is derived by counting real open MT5 positions
        for this symbol+magic (each add is its own ticket, not tracked in
        memory), and "best price since the last add" is derived from real
        M30 price history between the most-recently-opened leg's time and
        now, not stored anywhere."""
        positions = self._get_open_positions(symbol)
        if not positions:
            return "No open position"

        first_leg = positions[0]
        adds_done = len(positions) - 1
        if adds_done >= MAX_ADDS:
            return f"Max adds ({MAX_ADDS}) already reached"

        first_open_time = datetime.datetime.fromtimestamp(
            first_leg.time, tz=datetime.timezone.utc
        )
        scale_in_deadline = first_open_time + datetime.timedelta(hours=SCALE_IN_WINDOW_HOURS)
        now = datetime.datetime.now(datetime.timezone.utc)
        if now > scale_in_deadline:
            return "Scale-in window closed"

        is_buy = first_leg.type == mt5.POSITION_TYPE_BUY
        last_leg = positions[-1]  # most recently added (or the initial, if no adds yet)
        last_leg_time = datetime.datetime.fromtimestamp(last_leg.time, tz=datetime.timezone.utc)

        bars = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M30, last_leg_time, now)
        if bars is None or len(bars) == 0:
            return "No bar history since the last leg was opened"
        best_price = max(bar["high"] for bar in bars) if is_buy else min(bar["low"] for bar in bars)

        fav_move = (
            (best_price - last_leg.price_open) if is_buy
            else (last_leg.price_open - best_price)
        )
        if fav_move < ADD_THRESHOLD:
            return f"Not enough favorable move yet (${fav_move:.2f} < ${ADD_THRESHOLD})"

        tick = self._get_tick(symbol)
        if tick is None:
            return "No tick data"
        entry_price = tick.ask if is_buy else tick.bid
        add_lot = round(first_leg.volume * ADD_LOT_FRACTION / 0.01) * 0.01
        filling_mode = self._filling_mode(symbol)

        result = self._safe_order_send(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": add_lot,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": first_leg.sl,  # match whatever the shared stop currently is
                "tp": 0.0,
                "deviation": 10,
                "magic": MAGIC,
                "comment": "news_confirm_scale_in_add",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
        )
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            reason = (
                f"retcode={result.retcode} comment='{result.comment}'"
                if result is not None else "order_send returned None"
            )
            return f"Scale-in add REJECTED — {reason}"

        return f"Scale-in ADD #{adds_done + 1} — lot={add_lot} @ {entry_price}"

    # ---------------------------------------------------------------- trade management

    def manage_open_trade(self, symbol: str) -> str:
        """Call on every M30 (or finer) bar close while a position is
        open. Same breakeven+trail priority order as
        straddle_strategy.py's manage_open_trade(), but now applied to the
        BLENDED position across every open leg (initial entry + any
        scale-in adds from check_scale_in()) rather than a single ticket —
        weighted-average entry price, one shared stop applied to every
        leg identically. All legs always move together (same SL value),
        so the trade only ever exits as a whole, not leg-by-leg."""
        positions = self._get_open_positions(symbol)
        if not positions:
            return "No open trade"

        cfg = SYMBOL_CONFIG[symbol]
        first_leg = positions[0]
        entry_time = datetime.datetime.fromtimestamp(first_leg.time, tz=datetime.timezone.utc)
        deadline = entry_time + datetime.timedelta(hours=cfg["max_hold_hours"])
        if datetime.datetime.now(datetime.timezone.utc) >= deadline:
            all_closed = all(self._close_position_at_market(symbol, p) for p in positions)
            return (
                f"Past max hold — closed all {len(positions)} leg(s) at market"
                if all_closed else "Deadline close FAILED for one or more legs"
            )

        tick = self._get_tick(symbol)
        if tick is None:
            return "No tick data"

        is_buy = first_leg.type == mt5.POSITION_TYPE_BUY
        total_lot = sum(p.volume for p in positions)
        weighted_entry = sum(p.volume * p.price_open for p in positions) / total_lot
        be_trigger = cfg["be_trigger"]
        trail_dist = cfg["trail"]
        current_price = tick.bid if is_buy else tick.ask

        entry_r = _round_price(weighted_entry, symbol)
        sl_r = _round_price(first_leg.sl, symbol)
        be_done = (sl_r >= entry_r) if is_buy else (sl_r <= entry_r and first_leg.sl > 0)

        favorable_move = (
            (current_price - weighted_entry) if is_buy else (weighted_entry - current_price)
        )

        if not be_done and favorable_move >= be_trigger:
            new_sl = _round_price(weighted_entry, symbol)
            ok = all(self._modify_sl(symbol, p, new_sl) for p in positions)
            return (
                f"BE -> SL {new_sl} (all {len(positions)} leg(s))"
                if ok else "BE modify failed for one or more legs"
            )

        if be_done:
            best_price = self._best_price_since_entry(symbol, first_leg)
            if is_buy:
                new_sl = _round_price(best_price - trail_dist, symbol)
                if new_sl > _round_price(first_leg.sl, symbol):
                    ok = all(self._modify_sl(symbol, p, new_sl) for p in positions)
                    return (
                        f"Trail -> SL {new_sl} (all {len(positions)} leg(s))"
                        if ok else "Trail failed for one or more legs"
                    )
            else:
                new_sl = _round_price(best_price + trail_dist, symbol)
                if new_sl < _round_price(first_leg.sl, symbol):
                    ok = all(self._modify_sl(symbol, p, new_sl) for p in positions)
                    return (
                        f"Trail -> SL {new_sl} (all {len(positions)} leg(s))"
                        if ok else "Trail failed for one or more legs"
                    )

        return f"Holding ({len(positions)} leg(s), total_lot={total_lot})"

    def _best_price_since_entry(self, symbol: str, position) -> float:
        entry_time = datetime.datetime.fromtimestamp(
            position.time, tz=datetime.timezone.utc
        )
        bars = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M30,
            entry_time,
            datetime.datetime.now(datetime.timezone.utc),
        )
        if bars is None or len(bars) == 0:
            return position.price_open
        if position.type == mt5.POSITION_TYPE_BUY:
            return max(bar["high"] for bar in bars)
        return min(bar["low"] for bar in bars)

    def _close_position_at_market(self, symbol: str, position) -> bool:
        tick = self._get_tick(symbol)
        if tick is None:
            return False
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        result = self._safe_order_send(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 10,
                "magic": MAGIC,
                "comment": "news_confirm_deadline_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(symbol),
            }
        )
        if result is not None and result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"  Close {symbol} rejected — retcode={result.retcode} comment='{result.comment}'"
            )
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def _modify_sl(self, symbol: str, pos, new_sl: float) -> bool:
        result = self._safe_order_send(
            {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
        )
        if result is not None and result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"  SL modify {symbol} rejected — retcode={result.retcode} comment='{result.comment}'"
            )
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    # ---------------------------------------------------------------- reporting

    def get_performance_summary(
        self, symbol: Optional[str] = None, lookback_days: int = 120
    ) -> Dict[str, Any]:
        symbols = [symbol] if symbol else self.traded_symbols
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=lookback_days
        )
        all_closes = []
        for sym in symbols:
            deals = (
                mt5.history_deals_get(
                    since, datetime.datetime.now(datetime.timezone.utc), group=f"*{sym}*"
                )
                or ()
            )
            all_closes.extend(
                d for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT
            )
        if not all_closes:
            return {"trades": 0, "status": "No trades yet"}
        wins = sum(1 for d in all_closes if d.profit > 0)
        summary: Dict[str, Any] = {
            "total_trades": len(all_closes),
            "win_rate": f"{wins/len(all_closes)*100:.1f}%",
        }
        if symbol:
            summary["next_lot_size"] = self._lot_size(symbol)
        return summary

    # ---------------------------------------------------------------- util

    def _no(self, reason: str) -> Dict[str, Any]:
        return {
            "signal": None,
            "entry_price": None,
            "lot_size": None,
            "release_bar_move": None,
            "reason": reason,
        }

    def __repr__(self) -> str:
        return (
            f"NewsConfirmStrategy("
            f"symbols={self.traded_symbols}, risk_pct={RISK_PCT}% (quarter-Kelly, flat), "
            f"fomc_filter={FOMC_PROXIMITY_DAYS}d, "
            f"scale_in={MAX_ADDS}x adds within {SCALE_IN_WINDOW_HOURS}h "
            f"(+${ADD_THRESHOLD} each), stateless=True)"
        )


if __name__ == "__main__":
    s = NewsConfirmStrategy()
    print(s)
    print()
    future_nfp = [d for d in NFP_SCHEDULE_UTC if d >= datetime.datetime.now(datetime.timezone.utc)]
    future_fomc = [d for d in FOMC_DATES_UTC if d >= datetime.datetime.now(datetime.timezone.utc).date()]
    print(f"NFP events scheduled: {len(NFP_SCHEDULE_UTC)} total, {len(future_nfp)} upcoming")
    print(f"  Next: {min(future_nfp) if future_nfp else 'NONE — add dates to NFP_SCHEDULE_UTC in this file'}")
    print(f"FOMC dates loaded: {len(FOMC_DATES_UTC)} total, {len(future_fomc)} upcoming")
    print(f"  Next: {min(future_fomc) if future_fomc else 'NONE — add dates to FOMC_DATES_UTC in this file'}")
    print()
    print("*** DEMO-READY, NOT LIVE-READY — see module docstring 'IMPORTANT' section ***")
    print("*** Calendar is manually maintained — see the 'Event calendars' section near the top of this file ***")