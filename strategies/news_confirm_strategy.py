"""
News Spike Strategy — Live Version (stateless)
================================================
Edge   : Straddle-at-release entry (same OCO pattern as
         news_reload_strategy.py), but radically simplified on the exit
         side: NO take-profit, NO reload chain. Once a position opens, it
         gets a HARD 1-MINUTE force-close — win or lose, whatever it's
         worth at that moment, it closes. After that single close (or an
         SL hit, if that happens first), this specific event is DONE —
         no re-entry, no matter how much time is left in the trigger
         window.

         The idea: capture only the very first, immediate reaction to a
         scheduled release, then get out before a possible reversal —
         rather than staying in the market and riding it (which hurt
         other mechanics tested this session: the 2026-03-06 NFP
         reversal, and the 2024-12-06 NFP bar that spiked $12+ and then
         round-tripped almost all the way back to flat within 30 min).
         One shot per event, then done.

Entry  : Buy-stop + sell-stop straddle at the release, offset $3 from the
         anchor price. Whichever side fills is the trade. Now covers
         THREE event types — NFP, CPI, and FOMC — each on its own
         schedule (NFP/CPI at 8:30 AM ET, FOMC at 2:00 PM ET).
SL     : Real $5 broker-side stop — can still trigger before the
         1-minute mark if price moves against it fast enough.
Exit   : Hard 1-minute force-close, no TP, no reload. Whichever comes
         first — the SL hitting, or 60 seconds elapsing — ends the
         trade.
Sizing : Kelly-flat 14.1% base lot formula (same as the other two news
         strategies), computed once at entry — no reload here to grow it
         further.
Filter : NONE. Earlier versions of this project used a "skip NFP within
         3 days of a preceding FOMC" filter — that was found (and
         genuinely helped) for the RELOAD mechanic specifically, because
         near-FOMC NFP events tended to whipsaw over that mechanic's
         longer holding period. This mechanic only holds for 60 seconds,
         well before that kind of reversal has time to develop. Tested
         directly: removing the filter for THIS mechanic on real 1-min
         data improved the NFP result (50.0%->59.1% win rate, $278.52->
         $784.06 final) rather than hurting it — all 4 previously-
         filtered events turned out to be wins. Lesson: a filter that
         helps one mechanic isn't automatically right for a different
         one built on the same event calendar.

*** BACKTEST EVIDENCE — REAL, MINUTE-LEVEL, CONTROL-TESTED ***
Unlike the first version of this file (built with zero evidence, purely
from live observation on 2026-08-07), this has now been properly tested
against genuine 1-minute XAUUSD data (2024-01-02 to 2025-12-05) for all
three event types, each checked against a random-time control (200
bootstrap draws of the same number of random, non-event timestamps) to
confirm the edge isn't just riding the broader 2024-2025 gold uptrend:
  NFP  (22 events, no filter): 59.1% win rate, $90 -> $784.06.
  CPI  (22 events): 54.5% win rate, $90 -> $321.90. Beat ALL 200 random
       draws (100th percentile) - directly contradicts an earlier,
       different mechanic (tight-TP-reload) which tested badly on CPI;
       the lesson there was that CPI doesn't suit a sustained reload
       chain, not that CPI itself lacks a real reaction edge.
  FOMC (15 events): 66.7% win rate (best of the three), $90 -> $111.56
       (smallest $ total - gold reacts less dramatically to FOMC than to
       NFP/CPI surprises). Beat 99% of 200 random draws. Notably the
       LEAST likely of the three to be a trend-bias artifact, since wins
       split nearly evenly between buy (8) and sell (7).
A consistent finding across all three: roughly 20-23% of losses hit the
SL within the SAME MINUTE as entry - a real, fast whipsaw-reversal risk,
present but not fatal to any of the three.

*** STILL DEMO ONLY ***
Real evidence now, but still one historical window (2024-2025), not
tested against a second independent one, and no spread cost was modeled
(the 1-min data source had no spread column) - real results would be
modestly lower. Not live-capital-ready.

CHANGE LOG (this revision):
  - Expanded from NFP-only to three event types: NFP, CPI, FOMC - each
    on its own real, source-verified schedule.
  - Removed the FOMC-proximity filter entirely - see "Filter" above for
    why it was mechanic-specific, not a universal rule, and actively
    hurt this mechanic's NFP results when tested directly.
  - Docstring rewritten to reflect real backtest evidence (see above) -
    this is no longer a zero-evidence experiment.

CHANGE LOG (initial):
  - Initial build. Fourth standalone strategy, own MAGIC number, own
    process — same pattern as straddle_strategy.py, news_confirm_
    strategy.py, and news_reload_strategy.py never sharing a loop.
  - Includes the "already traded this event" guard from the start,
    given a live bug found in news_reload_strategy.py on 2026-08-07 (a
    fast-resolving chain let a second straddle open within the same
    5-min trigger window) — critical here too, arguably more so, since
    this mechanic resolves in under a minute almost every time.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# Event calendars — MANUALLY MAINTAINED. Three separate schedules, one per
# event type, each source-verified (BLS for NFP/CPI, Federal Reserve for
# FOMC) — do not assume a fixed-rule pattern for any of them; the 2025
# government shutdown proved that assumption can silently break.
# ---------------------------------------------------------------------------

NFP_SCHEDULE_UTC: List[datetime.datetime] = [
    datetime.datetime(2026, 8, 7, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 9, 4, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 10, 2, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 11, 6, 13, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 12, 4, 13, 30, tzinfo=datetime.timezone.utc),
    # Add next month's date here. DST-adjust by hand: 8:30 AM ET = 12:30 UTC
    # during DST (roughly Mar-Nov), 13:30 UTC otherwise.
]

CPI_SCHEDULE_UTC: List[datetime.datetime] = [
    datetime.datetime(2026, 8, 12, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 9, 11, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 10, 14, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 11, 10, 13, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 12, 10, 13, 30, tzinfo=datetime.timezone.utc),
    # Same 8:30 AM ET / DST rule as NFP. Check bls.gov/schedule/news_release/cpi.htm
]

FOMC_SCHEDULE_UTC: List[datetime.datetime] = [
    datetime.datetime(2026, 9, 16, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 10, 28, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 12, 9, 19, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 1, 27, 19, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 3, 17, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 4, 28, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 6, 9, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 7, 28, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 9, 15, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 10, 27, 18, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2027, 12, 8, 19, 0, tzinfo=datetime.timezone.utc),
    # Decision time is 2:00 PM ET = 18:00 UTC during DST, 19:00 UTC otherwise.
    # Add each new year's dates in one go once the Fed publishes them.
]


def _validate_calendar_freshness() -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    for name, schedule in (
        ("NFP", NFP_SCHEDULE_UTC),
        ("CPI", CPI_SCHEDULE_UTC),
        ("FOMC", FOMC_SCHEDULE_UTC),
    ):
        future = [d for d in schedule if d >= now]
        if not future:
            print(
                f"  !!! WARNING: {name}_SCHEDULE_UTC has NO upcoming dates — "
                f"update the list in news_spike_strategy.py now."
            )
        else:
            days_until_next = (min(future) - now).days
            if days_until_next > 40:
                print(
                    f"  WARNING: next {name} date is {days_until_next} days away "
                    f"({min(future)}) — double-check the list isn't stale."
                )


_validate_calendar_freshness()

# ---------------------------------------------------------------------------
# Per-symbol configuration
# ---------------------------------------------------------------------------

SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "XAUUSDm": {
        "pip": 1.0,
        "point_size": 0.001,
        "pip_value_per_lot": 100.0,
        "offset": 3.0,  # straddle distance from anchor, in $
        "sl": 5.0,  # stop-loss, in $
        "max_hold_seconds": 60.0,  # hard force-close — the whole point of this strategy
    },
}

RISK_PCT = 14.1  # same quarter-Kelly base as the other two news strategies —
# UNCHANGED from the NFP-only version, see _base_lot() below.
MAGIC = 20260807  # unique to this strategy — must not collide with
# straddle_strategy.py (20260716), news_confirm_strategy.py
# (20260801), or news_reload_strategy.py (20260810)


def _pip_value_per_lot(symbol: str) -> float:
    return SYMBOL_CONFIG.get(symbol, {}).get("pip_value_per_lot", 100.0)


def _round_price(price: float, symbol: str) -> float:
    if symbol == "XAUUSDm":
        return round(price, 2)
    return round(price, 5)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class NewsSpikeStrategy:
    """No __init__ state beyond configuration — every method queries MT5
    fresh. See module docstring for full design rationale."""

    def __init__(self, initial_balance: float = 90.0) -> None:
        self.starting_balance = initial_balance
        self.traded_symbols: List[str] = list(SYMBOL_CONFIG.keys())

    # ---------------------------------------------------------------- balance & sizing

    def _balance(self) -> float:
        acc = mt5.account_info()
        return acc.balance if acc else self.starting_balance

    def _base_lot(self, symbol: str) -> float:
        cfg = SYMBOL_CONFIG[symbol]
        risk_dollar = self._balance() * (RISK_PCT / 100.0)
        lot = risk_dollar / (cfg["sl"] * _pip_value_per_lot(symbol))
        return max(0.01, round(lot / 0.01) * 0.01)

    # ---------------------------------------------------------------- MT5 reads

    def _get_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
        return tick

    def _filling_mode(self, symbol: str) -> int:
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
        result = mt5.order_send(request)
        if result is None:
            print(f"  order_send returned None — mt5.last_error(): {mt5.last_error()}")
            return None
        return result

    def _get_position(self, symbol: str):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None
        own = [p for p in positions if p.magic == MAGIC]
        return own[0] if own else None

    def _get_pending_orders(self, symbol: str) -> Dict[str, Any]:
        orders = mt5.orders_get(symbol=symbol) or ()
        own = [o for o in orders if o.magic == MAGIC]
        buy = next((o for o in own if o.type == mt5.ORDER_TYPE_BUY_STOP), None)
        sell = next((o for o in own if o.type == mt5.ORDER_TYPE_SELL_STOP), None)
        return {"buy": buy, "sell": sell}

    # ---------------------------------------------------------------- event calendar

    def _next_event_trigger_window(
        self, now: datetime.datetime
    ) -> Optional[Tuple[datetime.datetime, str]]:
        """Checks all three schedules (NFP, CPI, FOMC) and returns
        (release_time, event_type) if `now` falls inside any of their
        5-min placement windows — or None. No proximity filter between
        event types; see module docstring 'Filter' section for why."""
        for event_type, schedule in (
            ("NFP", NFP_SCHEDULE_UTC),
            ("CPI", CPI_SCHEDULE_UTC),
            ("FOMC", FOMC_SCHEDULE_UTC),
        ):
            for release_time in schedule:
                if release_time <= now < release_time + datetime.timedelta(minutes=5):
                    return release_time, event_type
        return None

    def _event_already_traded(
        self, symbol: str, release_time: datetime.datetime
    ) -> bool:
        """Has this SPECIFIC event already produced a completed trade
        today? Critical here — this mechanic typically resolves in under a
        minute, well inside the 5-min trigger window, so without this
        check the bot would happily re-enter multiple times for the same
        event (exactly the bug found live in news_reload_strategy.py on
        2026-08-07). Derived from real MT5 deal history, not stored."""
        deals = (
            mt5.history_deals_get(
                release_time, datetime.datetime.now(datetime.timezone.utc)
            )
            or []
        )
        own_closes = [
            d
            for d in deals
            if d.symbol == symbol and d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT
        ]
        return len(own_closes) > 0

    # ---------------------------------------------------------------- entry

    def check_and_place(self, symbol: str) -> Dict[str, Any]:
        """Call on every poll (1-min cadence recommended — see
        main_news_spike.py). Places the straddle AT the release moment
        for whichever event type (NFP/CPI/FOMC) is currently inside its
        trigger window. manage_open_trade() then handles the hard
        1-minute force-close — there is no TP, no reload here."""
        if symbol not in self.traded_symbols:
            return self._no(f"{symbol} not enabled")

        if self._get_position(symbol) is not None:
            return self._no("Position already open")

        pending = self._get_pending_orders(symbol)
        if pending["buy"] is not None or pending["sell"] is not None:
            return self._no("Straddle already pending")

        now = datetime.datetime.now(datetime.timezone.utc)
        window = self._next_event_trigger_window(now)
        if window is None:
            return self._no("Not inside any event trigger window")
        release_time, event_type = window

        if self._event_already_traded(symbol, release_time):
            return self._no(
                f"This {event_type} event already produced a completed trade — "
                "no re-entry."
            )

        cfg = SYMBOL_CONFIG[symbol]
        tick = self._get_tick(symbol)
        if tick is None:
            return self._no("No tick data")

        tick_age = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.datetime.fromtimestamp(tick.time, tz=datetime.timezone.utc)
        if tick_age > datetime.timedelta(minutes=10):
            return self._no(f"Market likely closed — last tick is {tick_age} old")

        anchor = (tick.bid + tick.ask) / 2.0
        offset = cfg["offset"]
        sl = cfg["sl"]

        buy_stop = _round_price(anchor + offset, symbol)
        sell_stop = _round_price(anchor - offset, symbol)
        buy_sl = _round_price(buy_stop - sl, symbol)
        sell_sl = _round_price(sell_stop + sl, symbol)

        lots = self._base_lot(symbol)
        # Pending orders expire quickly here — no point leaving them
        # armed past the trigger window itself, since this strategy fires
        # once and is done.
        expiration = int((release_time + datetime.timedelta(minutes=5)).timestamp())
        filling_mode = self._filling_mode(symbol)

        tickets: Dict[str, Optional[int]] = {"buy": None, "sell": None}
        for side, order_type, price, stop in (
            ("buy", mt5.ORDER_TYPE_BUY_STOP, buy_stop, buy_sl),
            ("sell", mt5.ORDER_TYPE_SELL_STOP, sell_stop, sell_sl),
        ):
            result = self._safe_order_send(
                {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": symbol,
                    "volume": lots,
                    "type": order_type,
                    "price": price,
                    "sl": stop,
                    "tp": 0.0,
                    "magic": MAGIC,
                    "comment": "news_spike_entry",
                    "type_time": mt5.ORDER_TIME_SPECIFIED,
                    "expiration": expiration,
                    "type_filling": filling_mode,
                }
            )
            if result is None:
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                tickets[side] = result.order
            else:
                print(
                    f"  {side.upper()} {symbol} rejected — retcode={result.retcode} "
                    f"comment='{result.comment}'"
                )

        if tickets["buy"] is None or tickets["sell"] is None:
            for t in tickets.values():
                if t is not None:
                    self._safe_order_send(
                        {"action": mt5.TRADE_ACTION_REMOVE, "order": t}
                    )
            return self._no("Order send failed — rolled back")

        return {
            "signal": "straddle",
            "buy_stop": buy_stop,
            "sell_stop": sell_stop,
            "lot_size": lots,
            "reason": f"[{event_type}] Straddle placed | buy={buy_stop} sell={sell_stop} | lots={lots}",
        }

    # ---------------------------------------------------------------- OCO / cleanup

    def manage_pending_orders(self, symbol: str) -> str:
        pending = self._get_pending_orders(symbol)
        if pending["buy"] is None and pending["sell"] is None:
            return "No pending straddle"

        pos = self._get_position(symbol)
        if pos is not None:
            leftover = (
                pending["sell"] if pos.type == mt5.ORDER_TYPE_BUY else pending["buy"]
            )
            if leftover is not None:
                self._safe_order_send(
                    {"action": mt5.TRADE_ACTION_REMOVE, "order": leftover.ticket}
                )
            bias = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"
            return f"{bias.upper()} filled — opposite order cancelled"

        now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        for order in (pending["buy"], pending["sell"]):
            if (
                order is not None
                and order.time_expiration
                and now_ts >= order.time_expiration
            ):
                self._safe_order_send(
                    {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
                )
        remaining = self._get_pending_orders(symbol)
        if remaining["buy"] is None and remaining["sell"] is None:
            return "Neither side filled — straddle cancelled"
        return "Pending"

    # ---------------------------------------------------------------- trade management

    def manage_open_trade(self, symbol: str) -> str:
        """Call on every poll while a position is open. The ENTIRE job of
        this method: has more than max_hold_seconds (60) elapsed since
        this position opened? If yes, close it at market immediately,
        regardless of profit or loss. No TP check, no reload — this is
        deliberately the simplest possible exit."""
        pos = self._get_position(symbol)
        if pos is None:
            return "No open trade"

        cfg = SYMBOL_CONFIG[symbol]
        open_time = datetime.datetime.fromtimestamp(pos.time, tz=datetime.timezone.utc)
        elapsed = (
            datetime.datetime.now(datetime.timezone.utc) - open_time
        ).total_seconds()

        if elapsed < cfg["max_hold_seconds"]:
            return f"Holding — {elapsed:.0f}s elapsed, force-close at {cfg['max_hold_seconds']:.0f}s"

        closed = self._close_position_at_market(symbol, pos)
        return (
            f"{cfg['max_hold_seconds']:.0f}s elapsed — force-closed at market"
            if closed
            else "Force-close FAILED — will retry next cycle"
        )

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
                "comment": "news_spike_force_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(symbol),
            }
        )
        if result is not None and result.retcode != mt5.TRADE_RETCODE_DONE:
            print(
                f"  Close {symbol} rejected — retcode={result.retcode} comment='{result.comment}'"
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
                    since,
                    datetime.datetime.now(datetime.timezone.utc),
                    group=f"*{sym}*",
                )
                or ()
            )
            all_closes.extend(
                d for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT
            )
        if not all_closes:
            return {"trades": 0, "status": "No trades yet"}
        wins = sum(1 for d in all_closes if d.profit > 0)
        return {
            "total_trades": len(all_closes),
            "win_rate": f"{wins/len(all_closes)*100:.1f}%",
        }

    # ---------------------------------------------------------------- util

    def _no(self, reason: str) -> Dict[str, Any]:
        return {
            "signal": None,
            "buy_stop": None,
            "sell_stop": None,
            "lot_size": None,
            "reason": reason,
        }

    def __repr__(self) -> str:
        return (
            f"NewsSpikeStrategy(symbols={self.traded_symbols}, "
            f"events=[NFP,CPI,FOMC], risk_pct={RISK_PCT}% (quarter-Kelly, single-shot), "
            f"max_hold={SYMBOL_CONFIG['XAUUSDm']['max_hold_seconds']:.0f}s, "
            f"filter=None, one_shot_per_event=True, stateless=True)"
        )


if __name__ == "__main__":
    s = NewsSpikeStrategy()
    print(s)
    print()
    now = datetime.datetime.now(datetime.timezone.utc)
    for name, schedule in (
        ("NFP", NFP_SCHEDULE_UTC),
        ("CPI", CPI_SCHEDULE_UTC),
        ("FOMC", FOMC_SCHEDULE_UTC),
    ):
        future = [d for d in schedule if d >= now]
        print(f"{name} events scheduled: {len(schedule)} total, {len(future)} upcoming")
        print(f"  Next: {min(future) if future else 'NONE — add dates'}")
    print()
    print("*** Backed by real, control-tested 1-min data — see module docstring ***")
    print(
        "*** Still DEMO ONLY — one historical window, not yet out-of-sample tested ***"
    )
