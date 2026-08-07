import time
import MetaTrader5 as mt5
from decouple import config, AutoConfig
from mt5.meter_trader_config import MetaTraderConfig
from strategies.news_confirm_strategy import NewsReloadStrategy
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


def sleep_until_next_1min():
    """1-minute polling, not the usual 5 — deliberate for THIS strategy
    specifically. A reload chain can move $6+ between polls; at 5-min
    polling, a fast move could blow past several $6 reload steps in one
    gap instead of triggering each one individually, diverging from what
    was backtested (which assumes each $6 increment gets caught close to
    when it happens). 1-min keeps live behavior much closer to that
    assumption. See news_reload_strategy.py's module docstring."""
    now = datetime.now(timezone.utc)
    seconds_past = now.second
    seconds_to_wait = 60 - seconds_past
    time.sleep(seconds_to_wait)


reload_decouple()

LIVE = True  # this strategy has NOT been validated on fresh out-of-sample
# data, and the reload mechanic is genuinely aggressive even
# at its most conservative tested setting — see
# news_reload_strategy.py's module docstring. Leave False.

# Runs as its OWN process, entirely separate from main_straddle.py AND
# main_news_confirm.py — three different mechanics, three different magic
# numbers, no shared loop. Safe to run all three side by side.


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
    strategy = NewsReloadStrategy(initial_balance=90.0)

    print(f"{strategy}\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    # Same shape as main_straddle.py: OCO cleanup runs first, every cycle,
    # unconditionally — that ordering bug is exactly what broke the
    # original straddle bot once already. manage_open_trade() here does
    # double duty: it's both the TP-detection AND the reload trigger.
    while True:
        now = datetime.now(timezone.utc)

        print("=" * 55)
        print(f"{now.strftime('%A %d %B %Y — %H:%M UTC')}")
        print("=" * 55)

        for symbol in strategy.traded_symbols:
            print(f"\n{symbol}")

            pending_status = strategy.manage_pending_orders(symbol)
            if pending_status not in ("No pending straddle", "Pending"):
                print(f"   {pending_status}")

            open_trades = mt5_config.get_open_trades_count(symbol=symbol)

            # ── manage a filled position (includes reload-on-TP) ───────
            if open_trades > 0:
                status = strategy.manage_open_trade(symbol)
                print(f"   {status}")

                # A chain fully ends either via SL hit (broker-side stop,
                # detected here as "no more open position") or the pending
                # order's own expiration — either way, once nothing is
                # open AND nothing is pending, the chain is over.
                if (
                    mt5_config.get_open_trades_count(symbol=symbol) == 0
                    and strategy._get_pending_orders(symbol)["buy"] is None
                    and strategy._get_pending_orders(symbol)["sell"] is None
                ):
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
                    print(
                        f"   CHAIN ENDED — last leg was a {'WIN' if was_win else 'LOSS'}"
                    )
                    print(f"   {strategy.get_performance_summary(symbol)}")

                continue

            if pending_status == "Pending":
                print(f"   {pending_status}")
                continue

            # ── look for a new entry (fires AT the NFP release) ────────
            signal = strategy.check_and_place(symbol)
            print(f"   {signal['reason']}")

        print("\nCycle done")
        sleep_until_next_1min()


if __name__ == "__main__":
    main()
