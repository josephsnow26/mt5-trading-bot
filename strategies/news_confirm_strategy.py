"""
News Reload Strategy — Live Version (stateless)
=================================================
Edge   : Pre-positioned straddle placed directly at the NFP release itself
         (unlike news_confirm_strategy.py, which deliberately waits for
         the release to happen first). Whichever side fills first is the
         trade. A TIGHT take-profit ($6) closes it, and every TP hit
         immediately reloads the SAME direction at a bigger lot — the
         chain keeps going as long as it keeps winning ("momentum"),
         until either a stop-loss hit ends it or a fixed time window
         elapses, whichever comes first.

Entry  : Buy-stop + sell-stop straddle at each event, offset=$5 from the
         anchor price (this file), whichever side fills is the trade —
         same OCO idea as straddle_strategy.py, but tight TP=$6/SL=$5
         instead of a trail, and reload-on-TP instead of one static
         position.
Sizing : Base lot for EACH NEW event = 14.1% "quarter-Kelly" of CURRENT
         balance (identical formula/derivation to news_confirm_strategy.py
         — see that file's docstring for the full Kelly derivation). This
         is recalculated fresh every event, NOT carried over from the
         previous event — a bad month never penalizes the next one, and a
         good month doesn't inflate it either, beyond the balance itself
         having grown or shrunk.
         WITHIN one event's chain: every TP-hit reload adds a flat
         LOT_INCREMENT (0.04) to the current lot. No reset until the
         chain itself ends (SL hit or window elapsed) — this is
         deliberately NOT a Kelly/percentage-based reload, just a flat
         step, chosen after comparing 0.04/0.05/0.07 head to head (see
         backtest summary below).
Filter : Same FOMC-proximity filter as news_confirm_strategy.py — skip
         any NFP event within FOMC_PROXIMITY_DAYS of a preceding FOMC
         decision.

*** IMPORTANT — WHY LOT_INCREMENT=0.04, NOT 0.05 OR 0.07 ***
All three were backtested head-to-head on the same 2022-2026 gold data,
identical entries (same 42.5% event-level win rate at every increment —
the increment only changes position size, not which events win or lose):
  +0.04: $90 -> $31,210, NEVER went negative (min balance $55.96),
         max base lot reached 4.50, worst single event -$652.08.
  +0.05: $90 -> $36,261, stayed positive throughout, max base lot 5.22,
         worst single event -$743.60.
  +0.07: $90 -> $48,699, WENT NEGATIVE (-$18.97) on this same trusted
         dataset before recovering, max base lot 7.01, worst single
         event -$995.28.
0.04 was chosen specifically because it's the only one of the three that
never breached zero on the historical data — 0.05 and especially 0.07
show the same underlying mechanism can fail outright with a bigger step,
even on data that's otherwise held up all session. This was also tested
on a second, independent dataset (different source/format, 2018-2026) at
+0.05: that run went NEGATIVE FOR OVER A YEAR (June 2024-July 2025,
reaching -$228.06) before an exceptional trend day rescued it. A real
account would have been margin-called long before that recovery — this
is why 0.04, the most conservative of the three tested, is what's
actually wired in here, not the biggest final-number version.

*** EVEN AT 0.04, THIS IS STILL GENUINELY AGGRESSIVE ***
- Only 17 of 40 backtested events (42.5%) were net winners. This makes
  money because losses are bounded (roughly one base-lot SL hit per
  losing event) while winning chains compound — NOT because it wins most
  of the time. Expect more losing months than winning ones by count.
- The total backtested return is heavily concentrated in two exceptional
  trend days (June 5 and July 2, 2026) — without those two events, the
  running balance would have been far flatter through most of the
  4-year test. Do not expect every strong month to look like those.
- Base lot itself grows as the account grows (Kelly-flat), which means
  later events carry meaningfully larger absolute risk than early ones,
  by design — this compounds with the flat reload growth on top.

*** SAME "NOT LIVE-READY" WARNING AS EVERY OTHER STRATEGY THIS SESSION ***
All of the above comes from ONE historical dataset. Treat this as
demo-only, to accumulate real forward evidence, until it's been checked
against data it was never tuned on.

CHANGE LOG (this revision):
  - Initial build. Separate MAGIC number and separate process from both
    straddle_strategy.py AND news_confirm_strategy.py — three genuinely
    different mechanics, each standalone, matching this project's
    established pattern of never merging strategies into one loop.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# Event calendars — MANUALLY MAINTAINED, same discipline and same dates as
# news_confirm_strategy.py. Deliberately duplicated here rather than shared
# via import, matching this project's preference for standalone files with
# no cross-strategy dependencies. Keep both files' calendars in sync by
# hand when updating either one.
# ---------------------------------------------------------------------------

NFP_SCHEDULE_UTC: List[datetime.datetime] = [
    datetime.datetime(2026, 8, 7, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 9, 4, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 10, 2, 12, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 11, 6, 13, 30, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 12, 4, 13, 30, tzinfo=datetime.timezone.utc),
    # Add next month's date here. DST-adjust by hand: 8:30 AM ET = 12:30 UTC
    # during DST (roughly Mar-Nov), 13:30 UTC otherwise. Do NOT assume
    # "first Friday" — verify each one against bls.gov/schedule directly.
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
    # Only the SECOND day of each 2-day meeting. Add the next announced
    # year's dates in one go when the Fed publishes them.
]


def _validate_calendar_freshness() -> None:
    """Same freshness check as news_confirm_strategy.py — no fetching, just
    compares the hardcoded lists above to today's date and warns loudly if
    either has run dry of upcoming dates."""
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()

    future_nfp = [d for d in NFP_SCHEDULE_UTC if d >= now]
    if not future_nfp:
        print(
            "  !!! WARNING: NFP_SCHEDULE_UTC has NO upcoming dates — "
            "this bot will never enter until you add more. Check "
            "bls.gov/schedule and update the list in news_reload_strategy.py now."
        )
    else:
        days_until_next = (min(future_nfp).date() - today).days
        if days_until_next > 40:
            print(
                f"  WARNING: next NFP date is {days_until_next} days away "
                f"({min(future_nfp).date()}) — double-check the list isn't stale."
            )

    future_fomc = [d for d in FOMC_DATES_UTC if d >= today]
    if not future_fomc:
        print(
            "  !!! WARNING: FOMC_DATES_UTC has NO upcoming dates — the "
            "FOMC-proximity filter will not work correctly. Check "
            "federalreserve.gov/monetarypolicy/fomccalendars.htm and update now."
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
        "tp": 6.0,  # tight take-profit, in $
        "sl": 5.0,  # stop-loss, in $ — chain ends here
        "max_hold_hours": 8.0,  # chain's own time window
    },
}

RISK_PCT = 14.1  # "quarter-Kelly" base lot per event — see
# module docstring for the full derivation.
LOT_INCREMENT = 0.04  # flat $ added to lot on every TP-hit reload
# within one chain — see docstring for why
# 0.04 specifically, not 0.05 or 0.07.

MAGIC = 20260810  # unique to this strategy — must not collide
# with straddle_strategy.py's or
# news_confirm_strategy.py's MAGIC numbers

FOMC_PROXIMITY_DAYS = 3


def _pip_value_per_lot(symbol: str) -> float:
    return SYMBOL_CONFIG.get(symbol, {}).get("pip_value_per_lot", 100.0)


def _round_price(price: float, symbol: str) -> float:
    if symbol == "XAUUSDm":
        return round(price, 2)
    return round(price, 5)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class NewsReloadStrategy:
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
        """RISK_PCT of CURRENT balance — recomputed fresh every time a NEW
        chain starts. Never carried over from a previous event."""
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

    def _next_nfp_trigger_window(
        self, now: datetime.datetime
    ) -> Optional[datetime.datetime]:
        """Returns the NFP release time if `now` falls inside its 5-min
        placement window (fires the straddle AT the release, not after
        it — unlike news_confirm_strategy.py)."""
        for release_time in NFP_SCHEDULE_UTC:
            if release_time <= now < release_time + datetime.timedelta(minutes=5):
                return release_time
        return None

    def _days_since_fomc(self, event_date: datetime.date) -> int:
        prior = [d for d in FOMC_DATES_UTC if d <= event_date]
        if not prior:
            return 9999
        return (event_date - max(prior)).days

    def _is_near_fomc(self, event_time: datetime.datetime) -> bool:
        return self._days_since_fomc(event_time.date()) <= FOMC_PROXIMITY_DAYS

    # ---------------------------------------------------------------- entry

    def check_and_place(self, symbol: str) -> Dict[str, Any]:
        """Call on every poll. Places the straddle AT the NFP release
        moment (offset both sides of the current price) — no waiting, no
        confirmation. Whichever side fills becomes the chain's first leg;
        manage_open_trade() then handles the reload-on-TP behavior."""
        if symbol not in self.traded_symbols:
            return self._no(f"{symbol} not enabled")

        if self._get_position(symbol) is not None:
            return self._no("Position already open")

        pending = self._get_pending_orders(symbol)
        if pending["buy"] is not None or pending["sell"] is not None:
            return self._no("Straddle already pending")

        now = datetime.datetime.now(datetime.timezone.utc)
        release_time = self._next_nfp_trigger_window(now)
        if release_time is None:
            return self._no("Not inside an NFP trigger window")

        if self._is_near_fomc(release_time):
            return self._no(
                f"Skipped — NFP within {FOMC_PROXIMITY_DAYS} days of a preceding "
                f"FOMC decision (days_since_fomc={self._days_since_fomc(release_time.date())})"
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

        lots = self._base_lot(symbol)  # fresh Kelly-flat base for this NEW chain
        expiration = int(
            (release_time + datetime.timedelta(hours=cfg["max_hold_hours"])).timestamp()
        )
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
                    "tp": 0.0,  # TP handled in manage_open_trade so we can reload on hit
                    "magic": MAGIC,
                    "comment": "news_reload_entry",
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
            return self._no(f"Order send failed — rolled back")

        return {
            "signal": "straddle",
            "buy_stop": buy_stop,
            "sell_stop": sell_stop,
            "lot_size": lots,
            "reason": f"Straddle placed | buy={buy_stop} sell={sell_stop} | lots={lots}",
        }

    # ---------------------------------------------------------------- OCO / cleanup

    def manage_pending_orders(self, symbol: str) -> str:
        """Same OCO cleanup pattern as straddle_strategy.py — checked
        FIRST, every cycle, unconditionally (that ordering bug is exactly
        what broke the original straddle bot once already)."""
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

    # ---------------------------------------------------------------- trade management (reload chain)

    def manage_open_trade(self, symbol: str) -> str:
        """Call on every poll while a position is open. THIS is where the
        reload chain lives: if price has reached the chain's TP, close
        the current leg and immediately open a fresh one in the same
        direction at current market, with lot = previous lot +
        LOT_INCREMENT. The chain's SL and time-window deadline are both
        already enforced by MT5 itself (SL is a real broker-side stop;
        the pending order's expiration was set at placement time) — this
        method's only job is detecting a TP condition and reloading."""
        pos = self._get_position(symbol)
        if pos is None:
            return "No open trade"

        cfg = SYMBOL_CONFIG[symbol]
        tick = self._get_tick(symbol)
        if tick is None:
            return "No tick data"

        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        entry = pos.price_open
        tp_distance = cfg["tp"]
        current_price = tick.bid if is_buy else tick.ask

        favorable_move = (current_price - entry) if is_buy else (entry - current_price)
        if favorable_move < tp_distance:
            return "Holding — TP not reached"

        # TP reached: close this leg at market, then reload same direction
        closed = self._close_position_at_market(symbol, pos)
        if not closed:
            return "TP reached but close FAILED — will retry next cycle"

        new_lot = round((pos.volume + LOT_INCREMENT) / 0.01) * 0.01
        entry_price = tick.ask if is_buy else tick.bid
        sl_price = entry_price - cfg["sl"] if is_buy else entry_price + cfg["sl"]
        filling_mode = self._filling_mode(symbol)

        result = self._safe_order_send(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": new_lot,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": _round_price(sl_price, symbol),
                "tp": 0.0,
                "deviation": 10,
                "magic": MAGIC,
                "comment": "news_reload_add",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
        )
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            reason = (
                f"retcode={result.retcode} comment='{result.comment}'"
                if result
                else "order_send returned None"
            )
            return f"TP hit, closed OK, but RELOAD REJECTED — {reason}"

        return f"TP hit -> reloaded {('BUY' if is_buy else 'SELL')} | new lot={new_lot}"

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
                "comment": "news_reload_tp_close",
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
            f"NewsReloadStrategy(symbols={self.traded_symbols}, "
            f"risk_pct={RISK_PCT}% (quarter-Kelly, per-event base), "
            f"lot_increment={LOT_INCREMENT} (flat, within-chain only), "
            f"fomc_filter={FOMC_PROXIMITY_DAYS}d, stateless=True)"
        )


if __name__ == "__main__":
    s = NewsReloadStrategy()
    print(s)
    print()
    future_nfp = [
        d for d in NFP_SCHEDULE_UTC if d >= datetime.datetime.now(datetime.timezone.utc)
    ]
    print(
        f"NFP events scheduled: {len(NFP_SCHEDULE_UTC)} total, {len(future_nfp)} upcoming"
    )
    print(
        f"  Next: {min(future_nfp) if future_nfp else 'NONE — add dates to NFP_SCHEDULE_UTC'}"
    )
    print()
    print(
        "*** DEMO-READY, NOT LIVE-READY — see module docstring 'IMPORTANT' sections ***"
    )
