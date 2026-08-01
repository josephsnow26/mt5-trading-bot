import time
import MetaTrader5 as mt5
from decouple import config, AutoConfig
from mt5.meter_trader_config import MetaTraderConfig
from strategies.news_confirm_strategy import NewsConfirmStrategy
from datetime import datetime, timezone
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


def sleep_until_next_5min():
    now = datetime.now(timezone.utc)
    seconds_past = (now.minute % 5) * 60 + now.second
    seconds_to_wait = (5 * 60) - seconds_past
    time.sleep(seconds_to_wait)


reload_decouple()

LIVE = True  # this strategy has not been validated on fresh out-of-sample
# data yet — see news_confirm_strategy.py's module docstring
# "IMPORTANT" section. Leave this False until that changes.

# SYMBOLS is derived from the strategy itself (traded_symbols), same
# pattern as main_straddle.py — SYMBOL_CONFIG in news_confirm_strategy.py
# is the single source of truth for which symbols are actually live.

# Runs as its OWN process, entirely separate from main_straddle.py's loop —
# same "no monthly bot abeg just this" decision that kept the straddle bot
# standalone applies here: different mechanic, different magic number,
# different risk profile, no reason to share a loop.


def main():

    # ── MT5 ──────────────────────────────────────────────────────────────
    mt5_config = MetaTraderConfig()
    mt5_settings = {
        "username": config("MT5_USERNAME" if LIVE else "MT5_USERNAME_TRIAL"),
        "password": config("MT5_PASSWORD" if LIVE else "MT5_PASSWORD_TRIAL"),
        "server": config("MT5_SERVER" if LIVE else "MT5_SERVER_TRIAL"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }

    print(f"Mode: {'LIVE' if LIVE else 'DEMO'}")

    if not mt5_config.start_mt5(mt5_settings):
        print("MT5 failed to start")
        return

    # ── Strategy ─────────────────────────────────────────────────────────
    strategy = NewsConfirmStrategy(initial_balance=90.0)

    print(f"{strategy}\n")

    # No restore_open_trades() call needed — same stateless design as the
    # straddle bot. Every method re-derives from MT5 directly.

    # ── Main loop ─────────────────────────────────────────────────────────
    # Simpler than main_straddle.py's loop by design: this strategy never
    # places a pending order, so there is no OCO cleanup step and no
    # "still pending, neither side filled yet" branch — every cycle is
    # either "manage an open position" or "check whether right now is
    # inside a scheduled NFP confirmation window and act on it."
    while True:
        now = datetime.now(timezone.utc)

        print("=" * 55)
        print(f"{now.strftime('%A %d %B %Y — %H:%M UTC')}")
        print("=" * 55)

        for symbol in strategy.traded_symbols:
            print(f"\n{symbol}")

            open_trades = mt5_config.get_open_trades_count(symbol=symbol)

            # ── manage a filled position ──────────────────────────────
            if open_trades > 0:
                # check_scale_in() runs first, every cycle — adds another
                # leg if the trade is winning and still inside its window.
                # Safe to call even when no add is currently due; it just
                # returns a status string explaining why it didn't act.
                scale_status = strategy.check_scale_in(symbol)
                print(f"   [scale-in] {scale_status}")

                status = strategy.manage_open_trade(symbol)
                print(f"   {status}")

                # detect the trade closing this cycle (SL, trail, or the
                # max-hold deadline) — read the outcome straight from MT5
                # history rather than tracking it ourselves, matching the
                # straddle bot's own pattern.
                if mt5_config.get_open_trades_count(symbol=symbol) == 0:
                    deals = (
                        mt5.history_deals_get(
                            int(time.time()) - 86400, int(time.time())
                        )
                        or []
                    )
                    own_deals = [
                        d
                        for d in deals
                        if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT
                    ]
                    was_win = own_deals[-1].profit > 0 if own_deals else False
                    print(f"   {'WIN' if was_win else 'LOSS'}")
                    print(f"   {strategy.get_performance_summary(symbol)}")

                continue

            # ── look for a new entry ───────────────────────────────────
            # check_and_enter() internally handles ALL of: "is now inside
            # an NFP confirmation window", "is this event too close to
            # FOMC", "did the release bar actually confirm direction+
            # volume", and "what lot size does the capped pyramid say to
            # use right now" — nothing further to do here whether or not
            # it actually enters.
            signal = strategy.check_and_enter(symbol)
            print(f"   {signal['reason']}")

        print("\nCycle done")
        sleep_until_next_5min()


if __name__ == "__main__":
    main()
