import time
from decouple import config, AutoConfig
from mt5.meter_trader_config import MetaTraderConfig
from strategies.news_confirm_strategy import (
    NewsSpikeStrategy,
    NFP_SCHEDULE_UTC,
    CPI_SCHEDULE_UTC,
    FOMC_SCHEDULE_UTC,
    EARLY_ENTRY_SECONDS,
)
from datetime import datetime, timezone, timedelta
import os


def reload_decouple():
    KEYS = [
        "MT5_USERNAME",
        "MT5_PASSWORD",
        "MT5_SERVER",
        "MT5_USERNAME_TRIAL",
        "MT5_PASSWORD_TRIAL",
        "MT5_SERVER_TRIAL",
        "MT5_PATHWAY",
    ]
    for k in KEYS:
        os.environ.pop(k, None)
    AutoConfig._instances = {}


# ---------------------------------------------------------------------------
# Dynamic polling (added 2026-09-04)
# ---------------------------------------------------------------------------
# news_spike_strategy.py's entry window is only EARLY_ENTRY_SECONDS (5s)
# wide — see that file's own module docstring CHANGE LOG. At the old flat
# 30s cadence, a poll cycle can step clean over that 5-second window
# without ever checking inside it, silently losing the event (this is what
# happened live on CPI: gold traded, silver's check landed just late
# enough to miss the window entirely).
#
# Fix: poll every 1 second whenever a scheduled event is within
# TIGHT_BAND_MINUTES, poll every 30 seconds (as before) the rest of the
# time. This gets reliable window coverage exactly when it matters without
# hammering MT5 with 1s IPC calls 24/7 for windows that only exist a few
# minutes a month.
TIGHT_POLL_SECONDS = 1
NORMAL_POLL_SECONDS = 30
TIGHT_BAND_MINUTES = 2  # safely brackets both the entry window and the
                         # tail end of the flatten window on either side


def _seconds_to_next_event(now: datetime) -> float:
    """Seconds until the NEXT UPCOMING scheduled release across all three
    calendars — forward-looking only, and corrected to the REAL release
    time (schedule constants are stored EARLY_ENTRY_SECONDS/5s early, per
    news_spike_strategy.py — same constant imported above so both files
    always agree on what "real release time" means, no duplicated/
    guessed offset here). Once `now` passes a release time, that event
    stops counting entirely (tight polling reverts to NORMAL_POLL_SECONDS
    immediately at the real release moment, not TIGHT_BAND_MINUTES after
    it). Used ONLY to choose polling cadence — no trading/entry logic
    depends on this function; that logic lives entirely in
    news_spike_strategy.py itself.

    NOTE: because this stops being "tight" right at real release, the 60s
    force-close (manage_open_trade()) is only checked on the NORMAL
    30s cadence from that point on, not 1s. In practice this means the
    force-close can fire anywhere from 60 to ~89 seconds after fill,
    same margin as the plain 30s-polling straddle bot, rather than the
    tight ~60-61s this tight-polling band would otherwise give it."""
    real_events = [
        e + timedelta(seconds=EARLY_ENTRY_SECONDS)
        for e in (NFP_SCHEDULE_UTC + CPI_SCHEDULE_UTC + FOMC_SCHEDULE_UTC)
    ]
    future = [e for e in real_events if e > now]
    if not future:
        return float("inf")
    return (min(future) - now).total_seconds()


def sleep_until_next_tick(now: datetime, interval: int):
    """Sleeps until the next interval-aligned second mark (:00/:01/:02...
    for interval=1, :00/:30 for interval=30), not just 'interval seconds
    from whenever this was called' — keeps cycles clock-aligned rather
    than drifting."""
    seconds_to_wait = interval - (now.second % interval)
    if seconds_to_wait <= 0:
        seconds_to_wait = interval
    time.sleep(seconds_to_wait)


reload_decouple()

LIVE = False  # NOT backtested at all for most symbols — see
# news_spike_strategy.py's module docstring "VALIDATED EVIDENCE" section.
# Demo only.

# Fourth standalone process, own magic number, no shared loop with
# straddle_strategy.py, news_confirm_strategy.py's own confirm mechanic,
# or news_reload_strategy.py.


def main():

    # ── MT5 ──────────────────────────────────────────────────────────────
    mt5_config = MetaTraderConfig()
    mt5_settings = {
        "username": config("MT5_USERNAME" if LIVE else "MT5_USERNAME_TRIAL"),
        "password": config("MT5_PASSWORD" if LIVE else "MT5_PASSWORD_TRIAL"),
        "server": config("MT5_SERVER" if LIVE else "MT5_SERVER_TRIAL"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }

    print(f"Mode: {'LIVE' if LIVE else 'DEMO'}{mt5_settings}")

    if not mt5_config.start_mt5(mt5_settings):
        print("MT5 failed to start")
        return

    # ── Strategy ─────────────────────────────────────────────────────────
    strategy = NewsSpikeStrategy(initial_balance=90.0)

    print(f"{strategy}\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        # ONE timestamp per cycle, passed into every symbol's check this
        # cycle. Fixes the gold-traded/silver-didn't gap: previously each
        # check_and_place(symbol) call fetched its own fresh `now`
        # internally, so a slower symbol earlier in the loop (real
        # order_send() round-trips, slower during a volatile print) could
        # push the clock far enough that a later symbol in the SAME cycle,
        # for the SAME event, saw its 5s window already closed. All
        # symbols checked this cycle are now judged against the exact same
        # instant, regardless of loop position or how long earlier symbols
        # took.
        now = datetime.now(timezone.utc)

        print("=" * 55)
        print(f"{now.strftime('%A %d %B %Y — %H:%M:%S UTC')}")
        print("=" * 55)

        for symbol in strategy.traded_symbols:
            print(f"\n{symbol}")

            pending_status = strategy.manage_pending_orders(symbol)
            if pending_status not in ("No pending straddle", "Pending"):
                print(f"   {pending_status}")

            # has_own_open_trade() filters by this strategy's own MAGIC
            # before answering — see news_spike_strategy.py docstring for
            # the bug this fixes on shared symbols (EURUSDm/GBPUSDm/
            # USDJPYm/XAUUSDm, also traded by straddle_strategy.py).
            if strategy.has_own_open_trade(symbol):
                status = strategy.manage_open_trade(symbol)
                print(f"   {status}")
                continue

            if pending_status == "Pending":
                print(f"   {pending_status}")
                continue

            signal = strategy.check_and_place(symbol, now)
            print(f"   {signal['reason']}")

        print("\nCycle done")

        gap_seconds = _seconds_to_next_event(now)
        interval = (
            TIGHT_POLL_SECONDS
            if gap_seconds <= TIGHT_BAND_MINUTES * 60
            else NORMAL_POLL_SECONDS
        )
        sleep_until_next_tick(datetime.now(timezone.utc), interval)


if __name__ == "__main__":
    main()