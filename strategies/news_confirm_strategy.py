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

Entry  : Buy-stop + sell-stop straddle at the release, offset from the
         anchor price (per-symbol distance below). Whichever side fills
         is the trade. Covers NFP, CPI, and FOMC — each on its own
         schedule (NFP/CPI at 8:30 AM ET, FOMC at 2:00 PM ET).
SL     : Real per-symbol stop — can still trigger before the 1-minute
         mark if price moves against it fast enough.
Exit   : Hard 1-minute force-close, no TP, no reload. Whichever comes
         first — the SL hitting, or 60 seconds elapsing — ends the
         trade.
Sizing : Kelly-flat 14.1% base lot formula (same as the other two news
         strategies), computed once at entry — no reload here to grow it
         further.
Filter : NONE. See prior revision's docstring — the FOMC-proximity
         filter was tested and removed for THIS mechanic specifically
         (helped the reload chain, hurt the spike mechanic). Not
         re-litigated here.

*** VALIDATED EVIDENCE — XAUUSDm ONLY ***
XAUUSDm is backed by real, minute-level, control-tested backtest data
(2024-01-02 to 2025-12-05, see prior revision's docstring for the full
NFP/CPI/FOMC breakdown — 59.1%/54.5%/66.7% win rates, all beat random-
time controls).

*** *** EVERYTHING ELSE IN THIS FILE IS STRUCTURAL, NOT VALIDATED *** ***
XAGUSDm and 4 major USD pairs (EURUSDm, GBPUSDm, USDJPYm, USDCADm) were
added on 2026-08-11 at Joseph's explicit instruction, with explicit
sign-off to skip backtesting: same mechanic, same 60s force-close, same
Kelly-flat 14.1% sizing formula. Trimmed from an earlier 8-symbol pass
(USDCHFm/AUDUSDm/NZDUSDm dropped 2026-08-11 at Joseph's request).
Silver's offset/SL are new, uncalibrated guesses scaled roughly off
gold's $3/$5 by relative price level.

XCUUSDm (copper) was added on 2026-08-30, same basis as XAGUSDm/FX —
Joseph's explicit instruction, explicit sign-off to skip full
validation. Unlike the others, it has a single-event data point behind
it: NFP 2026-08-07 12:30 UTC, tested on real 1-min XCUUSDm data. Buy
leg filled and rode a clean spike to +$29.89/lot (12/20pt offset/SL)
with no SL touch before the 60s force-close. One event is an anecdote,
not a backtest — treat this exactly like XAGUSDm/the FX pairs: zero
statistical validation, no random-time control, no year-over-year
check. The FOMC leg of that same session (2026-07-29 18:00 UTC)
couldn't be tested at all — the supplied CSV had a hard data gap
covering the entire decision window.

None of the 6 non-gold symbols have any event-reaction backtest behind
them. Unknown and unverified for all 6:
  - whether gold's spike edge (fast post-release reaction, fade before
    60s) generalizes to FX majors, silver, or copper at all
  - correct offset/SL distance per symbol (reused/extended, not fitted)
  - decimal/point precision matching Exness's real quoting for each
    symbol (assumed from typical broker convention, not confirmed
    against this account) — XCUUSDm confirmed live 2026-08-30:
    digits=2, contract_size=1.0, tick_value=0.01, tick_size=0.01
  - real volume_min/volume_max/volume_step for XCUUSDm — not yet
    confirmed, _base_lot() will clamp to whatever the broker actually
    returns at runtime
Treat every non-XAUUSDm symbol here as an unvalidated experiment until
it has its own backtested evidence — same standing as GOLD_ENABLED was
treated before its own backtest existed.

*** STILL DEMO ONLY (all symbols) ***

CHANGE LOG (2026-08-30):
  - Added XCUUSDm (copper), risk_pct=3.2143% (same share as each FX
    pair). Offset=12.0 / SL=20.0 (price units, digits=2) — carried over
    from the single-event NFP backtest, not independently fitted to
    copper's own volatility. Total budget across all 7 symbols is now
    33.2143% (was 30%) — unchanged unless/until Joseph asks for a
    rebalance.

CHANGE LOG (2026-08-12):
  - Re-added the unconditional final flatten check immediately before
    order placement in check_and_place() — this had been dropped during
    a revert and is the actual fix for the EURUSDm-not-closing-before-
    NFP bug reported live. The window-based flatten (_next_flatten_window,
    5 min before release) depends on the main loop calling
    check_and_place() every cycle in that window; if the main loop's
    per-symbol routing sends a symbol to manage_open_trade() instead
    (observed live 2026-08-12 on EURUSDm, one minute before release, no
    position ever placed by this bot so manage_open_trade() correctly
    reported "No open trade" every time — meaning check_and_place(),
    and therefore the flatten logic, was never even called for it), the
    window-based flatten silently never runs for that symbol. The final
    check at the last possible moment before order placement is a second,
    independent layer that doesn't depend on the window timing lining up
    with when the main loop happens to call this method — it runs
    unconditionally every time check_and_place() reaches that point,
    which happens across many cycles inside the whole 5-min trigger
    window, not just the exact cycle the window check catches.
  - Trimmed to 6 symbols (XAUUSDm, XAGUSDm, EURUSDm, GBPUSDm, USDJPYm,
    USDCADm) per Joseph's request — USDCHFm/AUDUSDm/NZDUSDm removed.
  - FX offset/SL settled at 12/20 pips (reverted from an intermediate
    8/15p tighten that was catching fakeouts on slower releases).
  - Flatten lead time set to 5 minutes (down from an initial 10).
  - Added _next_flatten_window()/_flatten_symbol(): closes ANY position
    and cancels ANY pending order on a symbol, regardless of magic
    number, in the lead-up to a scheduled event — deliberate, simpler
    alternative to hedging-mode detection/blocking. This strategy shares
    EURUSDm/GBPUSDm/USDJPYm/XAUUSDm with straddle_strategy.py
    (MAGIC=20260716); flattening the symbol before entry means there's
    nothing left for a netting-mode account to merge with.
  - Lot sizing rewritten in a prior pass to pull live from
    mt5.symbol_info(symbol).trade_tick_value/trade_tick_size instead of
    a static pip-value guess table, which had been wrong by orders of
    magnitude for several symbols (2026-08-11: USDJPYm computed 42
    lots, XAGUSDm 17.62 lots, off hand-guessed constants).

CHANGE LOG (2026-08-11, earlier same day):
  - Added XAGUSDm + 7 major USD pairs (later trimmed to 4, see above).
  - Added `decimals` to SYMBOL_CONFIG per symbol and generalized
    `_round_price()` off it instead of a single XAUUSDm special-case.
  - Added automatic hedging-mode check at import time (informational;
    superseded operationally by the flatten-before-entry approach above,
    but left in place as a diagnostic).

CHANGE LOG (prior revision):
  - Expanded from NFP-only to three event types: NFP, CPI, FOMC - each
    on its own real, source-verified schedule.
  - Removed the FOMC-proximity filter entirely - helped the reload
    chain, hurt this mechanic when tested directly.

CHANGE LOG (initial):
  - Initial build. Fourth standalone strategy, own MAGIC number, own
    process — same pattern as straddle_strategy.py, news_confirm_
    strategy.py, and news_reload_strategy.py never sharing a loop.
  - Includes the "already traded this event" guard from the start,
    given a live bug found in news_reload_strategy.py on 2026-08-07.
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


def _validate_hedging_mode() -> None:
    """Informational only now — the flatten-before-entry mechanism
    (_next_flatten_window / _flatten_symbol) is the operative protection
    against cross-strategy interference, not this check. Left in place
    as a diagnostic since it's cheap and still useful context."""
    acc = mt5.account_info()
    if acc is None:
        print("  WARNING: mt5.account_info() unavailable — cannot verify "
              "hedging mode.")
        return
    margin_mode = getattr(acc, "margin_mode", None)
    HEDGING = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
    if margin_mode != HEDGING:
        print(
            f"  NOTE: account is NOT in hedging mode (margin_mode={margin_mode}). "
            "Flatten-before-entry logic handles the overlap risk with "
            "straddle_strategy.py regardless, but worth knowing."
        )
    else:
        print("  Hedging mode confirmed.")


_validate_calendar_freshness()
_validate_hedging_mode()

# ---------------------------------------------------------------------------
# Per-symbol configuration
# ---------------------------------------------------------------------------
# offset/sl are in PRICE units (not pips) — already converted below so the
# entry code never has to know per-symbol pip size.
#
# Lot sizing pulls LIVE from mt5.symbol_info(symbol).trade_tick_value/
# trade_tick_size, clamped to the broker's real volume_min/max/step —
# see _base_lot(). No static pip-value guess table.
#
# decimals controls price rounding in _round_price().

SYMBOL_CONFIG: Dict[str, Dict[str, Any]] = {
    "XAUUSDm": {
        "pip": 1.0,
        "point_size": 0.001,
        "offset": 3.0,       # $
        "sl": 5.0,           # $
        "risk_pct": 8.5714,  # of the 30% total budget — highest single share, only validated symbol
        "decimals": 2,
        "max_hold_seconds": 60.0,
    },
    "XAGUSDm": {
        "pip": 0.01,
        "point_size": 0.001,
        "offset": 0.04,      # $ — UNCALIBRATED
        "sl": 0.06,          # $ — UNCALIBRATED
        "risk_pct": 8.5714,  # of the 30% total budget — matches gold, despite zero evidence behind this symbol
        "decimals": 3,
        "max_hold_seconds": 60.0,
    },
    "XCUUSDm": {
        "pip": 0.01,
        "point_size": 0.01,
        "offset": 12.0,      # price units — carried over from the single NFP
                              # 2026-08-07 backtest ("scaled 4x gold" config),
                              # NOT independently fitted to copper's own volatility
        "sl": 20.0,           # price units — same single-event basis as offset
        "risk_pct": 3.2143,  # of the total budget — matched to FX-pair share,
                              # conservative given single-event evidence only
        "decimals": 2,        # confirmed live 2026-08-30: digits=2, contract_size=1.0,
                              # tick_value=0.01, tick_size=0.01
        "max_hold_seconds": 60.0,
    },
    "EURUSDm": {
        "pip": 0.0001,
        "point_size": 0.00001,
        "offset": 0.0012,    # 12 pips
        "sl": 0.0020,        # 20 pips
        "risk_pct": 3.2143,  # of the 30% total budget — 12.86% split evenly across 4 FX pairs
        "decimals": 5,
        "max_hold_seconds": 60.0,
    },
    "GBPUSDm": {
        "pip": 0.0001,
        "point_size": 0.00001,
        "offset": 0.0012,    # 12 pips
        "sl": 0.0020,        # 20 pips
        "risk_pct": 3.2143,  # of the 30% total budget — 12.86% split evenly across 4 FX pairs
        "decimals": 5,
        "max_hold_seconds": 60.0,
    },
    "USDJPYm": {
        "pip": 0.01,
        "point_size": 0.001,
        "offset": 0.12,      # 12 pips
        "sl": 0.20,          # 20 pips
        "risk_pct": 3.2143,  # of the 30% total budget — 12.86% split evenly across 4 FX pairs
        "decimals": 3,
        "max_hold_seconds": 60.0,
    },
    "USDCADm": {
        "pip": 0.0001,
        "point_size": 0.00001,
        "offset": 0.0012,    # 12 pips
        "sl": 0.0020,        # 20 pips
        "risk_pct": 3.2143,  # of the 30% total budget — 12.86% split evenly across 4 FX pairs
        "decimals": 5,
        "max_hold_seconds": 60.0,
    },
}

RISK_PCT = 14.1  # fallback default only if a symbol's config is missing
# risk_pct — every symbol in SYMBOL_CONFIG now sets its own (2026-08-12
# allocation, scaled 2026-08-12 from an initial 14% total to 30%:
# 8.57% XAU + 8.57% XAG + 3.21%×4 FX = 30% total, replacing the old
# flat 14.1% applied independently to each symbol). 2026-08-30: +3.2143%
# for XCUUSDm brings the total to 33.2143% — not rebalanced down, by
# choice, pending Joseph's call on whether to trim elsewhere.
MAGIC = 20260807  # unique to this strategy — must not collide with
# straddle_strategy.py (20260716), news_confirm_strategy.py
# (20260801), or news_reload_strategy.py (20260810)


def _round_price(price: float, symbol: str) -> float:
    decimals = SYMBOL_CONFIG.get(symbol, {}).get("decimals", 5)
    return round(price, decimals)


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
        """Sizing pulled LIVE from the broker, not guessed. trade_tick_value
        / trade_tick_size gives $-per-1.0-price-unit-move per lot, already
        converted to account currency by MT5. Clamps to the symbol's real
        volume_min/volume_max/volume_step. Uses each symbol's own risk_pct
        (see SYMBOL_CONFIG) rather than a flat global RISK_PCT — split
        2026-08-12 per Joseph's allocation: 4% XAU, 4% XAG, 1.5% each for
        the 4 FX pairs, scaled 2026-08-12 to a 30% total budget (up from an
        initial 14%) instead of each symbol independently risking the
        full 14.1% — well above this project's own quarter-Kelly ceiling,
        Joseph's explicit choice."""
        cfg = SYMBOL_CONFIG[symbol]
        info = mt5.symbol_info(symbol)
        if info is None or not info.trade_tick_size:
            print(f"  {symbol}: symbol_info unavailable — defaulting to 0.01 floor")
            return 0.01

        value_per_unit_per_lot = info.trade_tick_value / info.trade_tick_size
        risk_pct = cfg.get("risk_pct", RISK_PCT)
        risk_dollar = self._balance() * (risk_pct / 100.0)
        raw_lot = risk_dollar / (cfg["sl"] * value_per_unit_per_lot)

        vol_min = info.volume_min or 0.01
        vol_max = info.volume_max or raw_lot
        vol_step = info.volume_step or 0.01

        if raw_lot > vol_max:
            print(
                f"  {symbol}: target risk implies {raw_lot:.2f} lots, "
                f"clamped to broker max {vol_max:.2f} — actual $ risk on this "
                f"trade will be LESS than the {risk_pct}% target."
            )

        lot = max(vol_min, min(raw_lot, vol_max))
        lot = round(lot / vol_step) * vol_step
        return round(max(lot, vol_min), 2)

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

    def has_open_position(self, symbol: str) -> bool:
        """Public, magic-filtered check for the main loop to use instead
        of a broker-wide open-trade count. The bug this fixes: main_
        news_spike.py was calling mt5_config.get_open_trades_count(symbol)
        — which counts ANY position on that symbol, any magic — to decide
        whether to route to manage_open_trade() or check_and_place(). On
        shared symbols (EURUSDm/GBPUSDm/USDJPYm/XAUUSDm, also traded by
        straddle_strategy.py MAGIC=20260716), that count was almost always
        >0 from the OTHER bot's position, so this strategy got routed to
        manage_open_trade() every cycle (correctly reporting "No open
        trade" for its own magic) and never reached check_and_place() at
        all — meaning the flatten-before-entry logic never ran either.
        This method wraps the same magic-filtered _get_position() already
        used everywhere else in this file, so the main loop's routing
        decision uses the same definition of "open" as the rest of the
        strategy."""
        return self._get_position(symbol) is not None

    def has_own_open_trade(self, symbol: str) -> bool:
        """Public, magic-filtered check for the main loop to use instead
        of a broker-wide open-trades count. main_news_spike.py was
        calling mt5_config.get_open_trades_count(symbol=symbol), which
        counts ANY open position on that symbol regardless of magic —
        so with straddle_strategy.py (MAGIC=20260716) holding EURUSD/
        GBPUSD/USDJPY/XAUUSD open nearly 24/7, the main loop routed
        those symbols to manage_open_trade() every cycle (which
        correctly reported 'not mine') and never reached
        check_and_place() at all — the flatten-before-entry logic never
        ran as a result. This method filters by MAGIC before answering,
        same as everything else in this file."""
        return self._get_position(symbol) is not None

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
        5-min placement windows — or None."""
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
        today? Derived from real MT5 deal history, not stored."""
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

    def _next_flatten_window(
        self, now: datetime.datetime, lead_minutes: float = 5.0
    ) -> Optional[Tuple[datetime.datetime, str]]:
        """Returns (release_time, event_type) if `now` is inside the
        pre-event flatten window — lead_minutes before release, up to
        release itself."""
        for event_type, schedule in (
            ("NFP", NFP_SCHEDULE_UTC),
            ("CPI", CPI_SCHEDULE_UTC),
            ("FOMC", FOMC_SCHEDULE_UTC),
        ):
            for release_time in schedule:
                flatten_start = release_time - datetime.timedelta(minutes=lead_minutes)
                if flatten_start <= now < release_time:
                    return release_time, event_type
        return None

    def _flatten_symbol(self, symbol: str) -> Optional[str]:
        """Closes ANY open position and cancels ANY pending order on this
        symbol, regardless of magic number — deliberate design choice:
        simpler than detecting/blocking on hedging mode. Called from two
        places in check_and_place() — the window-based check AND an
        unconditional final check right before order placement — because
        the window-based one alone depends on the main loop calling
        check_and_place() for this symbol during that specific 5-min
        stretch, which isn't guaranteed by this file's own logic (that's
        the main loop's job, outside this file)."""
        actions: List[str] = []

        for pos in mt5.positions_get(symbol=symbol) or ():
            tick = self._get_tick(symbol)
            if tick is None:
                continue
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            result = self._safe_order_send(
                {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": tick.bid if is_buy else tick.ask,
                    "deviation": 10,
                    "magic": MAGIC,
                    "comment": "news_spike_pre_event_flatten",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": self._filling_mode(symbol),
                }
            )
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                actions.append(f"closed position magic={pos.magic} ticket={pos.ticket}")

        for order in mt5.orders_get(symbol=symbol) or ():
            result = self._safe_order_send(
                {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
            )
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                actions.append(f"cancelled pending magic={order.magic} ticket={order.ticket}")

        return "; ".join(actions) if actions else None

    # ---------------------------------------------------------------- entry

    def check_and_place(self, symbol: str) -> Dict[str, Any]:
        """Call on every poll (1-min cadence recommended — see
        main_news_spike.py). Places the straddle AT the release moment
        for whichever event type (NFP/CPI/FOMC) is currently inside its
        trigger window. manage_open_trade() then handles the hard
        1-minute force-close — there is no TP, no reload here."""
        if symbol not in self.traded_symbols:
            return self._no(f"{symbol} not enabled")

        now = datetime.datetime.now(datetime.timezone.utc)
        flatten_window = self._next_flatten_window(now)
        if flatten_window is not None:
            flatten_release, flatten_event = flatten_window
            result = self._flatten_symbol(symbol)
            if result:
                return self._no(
                    f"Pre-{flatten_event} flatten ({flatten_release}): {result}"
                )
            return self._no(f"Pre-{flatten_event} flatten window — already flat")

        if self._get_position(symbol) is not None:
            return self._no("Position already open")

        pending = self._get_pending_orders(symbol)
        if pending["buy"] is not None or pending["sell"] is not None:
            return self._no("Straddle already pending")

        window = self._next_event_trigger_window(now)
        if window is None:
            return self._no("Not inside any event trigger window")
        release_time, event_type = window

        if self._event_already_traded(symbol, release_time):
            return self._no(
                f"This {event_type} event already produced a completed trade — "
                "no re-entry."
            )

        # Unconditional final safety net — runs every single time
        # check_and_place() reaches this point, regardless of whether the
        # window-based flatten above fired earlier in a different cycle.
        # This is the actual fix for the EURUSDm-not-closing-before-NFP
        # bug: the window check only helps on cycles where the main loop
        # happens to route this symbol through check_and_place() during
        # the 5-min flatten window specifically. This check has no such
        # dependency — it runs on every cycle that gets this far, across
        # the whole 5-min trigger window, so a missed window-cycle for
        # one symbol doesn't leave it permanently unguarded.
        flatten_result = self._flatten_symbol(symbol)
        if flatten_result:
            return self._no(f"Flattened at entry-time (final check): {flatten_result}")

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
        regardless of profit or loss."""
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

    def get_performance_by_symbol(self, lookback_days: int = 120) -> Dict[str, Any]:
        """Per-symbol breakdown — worth checking regularly since 5 of 6
        symbols are unvalidated."""
        return {sym: self.get_performance_summary(sym, lookback_days) for sym in self.traded_symbols}

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
            f"events=[NFP,CPI,FOMC], risk_pct=per-symbol (XAU/XAG 8.57% each, FX/copper 3.21% each, 33.2143% total), "
            f"max_hold=60s, filter=None, one_shot_per_event=True, stateless=True, "
            f"validated=['XAUUSDm'], unvalidated={[s for s in SYMBOL_CONFIG if s != 'XAUUSDm']})"
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
    print("*** XAUUSDm backed by real, control-tested 1-min data ***")
    print("*** All other symbols: structural extension only, minimal or zero backtest ***")
    print("*** Still DEMO ONLY — do not run any of this on real money ***")