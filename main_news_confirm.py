import time
from decouple import config, AutoConfig
from mt5.meter_trader_config import MetaTraderConfig
from strategies.news_confirm_strategy import NewsSpikeStrategy
from datetime import datetime, timezone
import os


def reload_decouple():
    KEYS = [
        "MT5_USERNAME", "MT5_PASSWORD", "MT5_SERVER",
        "MT5_USERNAME_TRIAL", "MT5_PASSWORD_TRIAL", "MT5_SERVER_TRIAL",
        "MT5_PATHWAY",
    ]
    for k in KEYS:
        os.environ.pop(k, None)
    AutoConfig._instances = {}


def sleep_until_next_30sec():
    """30-second polling — tighter than the earlier 1-minute cadence.
    With a hard 60-second force-close, 1-min polling meant the actual
    close could land anywhere from 60 to 119 seconds after entry
    (whatever the next poll happened to catch). 30-sec polling narrows
    that to 60-89 seconds — closer to the real intent, and catches the
    trigger window and same-minute SL risk with finer resolution too.
    Aligned to :00/:30 second marks, not just 'every 30s from whenever
    the loop started'."""
    now = datetime.now(timezone.utc)
    seconds_to_wait = 30 - (now.second % 30)
    time.sleep(seconds_to_wait)


reload_decouple()

LIVE = False   # NOT backtested at all — see news_spike_strategy.py's
               # module docstring "DATA LIMITATION" section. Demo only.

# Fourth standalone process, own magic number, no shared loop with
# straddle_strategy.py, news_confirm_strategy.py, or news_reload_strategy.py.


def main():

    # ── MT5 ──────────────────────────────────────────────────────────────
    mt5_config = MetaTraderConfig()
    mt5_settings = {
        "username":    config("MT5_USERNAME"      if LIVE else "MT5_USERNAME_TRIAL"),
        "password":    config("MT5_PASSWORD"      if LIVE else "MT5_PASSWORD_TRIAL"),
        "server":      config("MT5_SERVER"        if LIVE else "MT5_SERVER_TRIAL"),
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
        now = datetime.now(timezone.utc)

        print("=" * 55)
        print(f"{now.strftime('%A %d %B %Y — %H:%M:%S UTC')}")
        print("=" * 55)

        for symbol in strategy.traded_symbols:
            print(f"\n{symbol}")

            pending_status = strategy.manage_pending_orders(symbol)
            if pending_status not in ("No pending straddle", "Pending"):
                print(f"   {pending_status}")

            open_trades = mt5_config.get_open_trades_count(symbol=symbol)

            if open_trades > 0:
                status = strategy.manage_open_trade(symbol)
                print(f"   {status}")
                continue

            if pending_status == "Pending":
                print(f"   {pending_status}")
                continue

            signal = strategy.check_and_place(symbol)
            print(f"   {signal['reason']}")

        print("\nCycle done")
        sleep_until_next_30sec()


if __name__ == "__main__":
    main()