import MetaTrader5 as mt5
from decouple import config
from symbols import get_symbols
import pandas as pd
from trading_manager import TradeManager  # your TradeManager class
from meter_trader_config import MetaTraderConfig
from datetime import datetime, timedelta
import time
from datetime import datetime, timezone

"""
Example usage of MACDTrendStrategy with MetaTraderConfig
Shows clean integration and proper error handling.
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from strategies.macd_strategy import MACDTrendStrategy, RestrictiveVolumeStrategy


live = True
if live:
    mt5_config = {
        "username": config("MT5_USERNAME"),
        "password": config("MT5_PASSWORD"),
        "server": config("MT5_SERVER"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }
else:
    mt5_config = {
        "username": config("MT5_USERNAME_TRIAL"),
        "password": config("MT5_PASSWORD_TRIAL"),
        "server": config("MT5_SERVER_TRIAL"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }


project_settings = mt5_config


def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info not found for {symbol}")
    return round(price, info.digits)

def main():
    from decouple import config
    from datetime import datetime, timedelta

    # ============================================================
    # 1. INITIALIZE MT5
    # ============================================================
    mt5_config = MetaTraderConfig()
    LIVE_MODE = True

    def load_mt5_settings(live: bool) -> dict:
        return {
            "username": config("MT5_USERNAME" if live else "MT5_USERNAME_TRIAL"),
            "password": config("MT5_PASSWORD" if live else "MT5_PASSWORD_TRIAL"),
            "server": config("MT5_SERVER" if live else "MT5_SERVER_TRIAL"),
            "mt5_pathway": config("MT5_PATHWAY"),
        }

    mt5_settings = load_mt5_settings(LIVE_MODE)

    print(f"\n🎯 Trading Mode: {'LIVE' if LIVE_MODE else 'DEMO'}")

    if not mt5_config.start_mt5(mt5_settings):
        print("❌ Failed to start MT5. Exiting...")
        return

    # ============================================================
    # 2. INITIALIZE STRATEGY
    # ============================================================
    strategy = RestrictiveVolumeStrategy(
        favorite_hours=[8, 9, 10, 13, 14],
        favorite_weekdays=[1, 2, 3],
        volume_spike_multiplier=2.0,
        sl_method="atr",
        tp_method="atr",
        sl_atr_multiplier=2.0,
        tp_atr_multiplier=4.0,
    )

    print(f"\n🧠 Strategy Loaded: {strategy}")
    print(f"📊 Minimum bars required: {strategy.min_bars}\n")

    # ============================================================
    # 3. TRADING PARAMETERS
    # ============================================================
    SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm"]
    TIMEFRAME = mt5.TIMEFRAME_M15
    TIMEFRAME_MINUTES = 15
    MAX_OPEN_TRADES = 3

    last_run_minute = None

    print("=" * 60)
    print("🔍 SCANNING FOR SIGNALS")
    print("=" * 60)

    # ============================================================
    # 4. MAIN LOOP
    # ============================================================
    while True:
        now = datetime.now()

        if now.minute % TIMEFRAME_MINUTES != 0:
            continue

        if last_run_minute == now.minute:
            continue

        last_run_minute = now.minute
        print("\n" + "=" * 60)
        print(f"🕒 {now.strftime('%A, %d %B %Y — %I:%M:%S %p')}")
        print("=" * 60)

        for symbol in SYMBOLS:
            print(f"\n📊 {symbol}")

            data = mt5_config.get_market_data_date_range(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start_date=now - timedelta(days=30),
                end_date=now,
            )

            if data.empty:
                print("   ⚠️ No data available")
                continue

            signal = strategy.generate_signal(data)

            # ---------------- SAFE LOGGING ----------------
            print(f"   Trend: {signal.get('trend')}")
            print(f"   Reason: {signal.get('reason')}")
            print(f"   Position Size: {signal.get('position_size')}")

            if not signal["signal"]:
                print("   ⏸️ No signal")
                continue

            # ---------------- SIGNAL DETAILS ----------------
            entry = signal["entry_price"]
            sl = signal["stop_loss"]
            tp = signal["take_profit"]

            print(f"\n   🎯 SIGNAL: {signal['signal'].upper()}")
            print(f"   Entry: {entry:.5f}")
            print(f"   SL: {sl:.5f}")
            print(f"   TP: {tp:.5f}")

            risk = abs(entry - sl)
            reward = abs(tp - entry)

            print(f"   Risk: {risk:.5f} | Reward: {reward:.5f} | RR: {reward / risk:.2f}")

            open_trades = mt5_config.get_open_trades_count()

            if open_trades >= MAX_OPEN_TRADES:
                print(f"   🚫 Trade skipped — open trades ({open_trades}/{MAX_OPEN_TRADES})")
                continue

            # ---------------- EXECUTION ----------------
            EXECUTE_TRADE = True  # 🔴 SET TO FALSE TO DRY-RUN

            if EXECUTE_TRADE:
                sl = normalize_price(symbol, sl)
                tp = normalize_price(symbol, tp)

                success = mt5_config.execute_trade(
                    symbol=symbol,
                    signal=signal["signal"],
                    stop_loss=sl,
                    take_profit=tp,
                    lot_size=signal["position_size"],  # DO NOT normalize
                    strategy_name=strategy.__class__.__name__,
                )

                print("   ✅ Trade executed" if success else "   ❌ Trade failed")

        print("\n✅ Cycle complete")


if __name__ == "__main__":
    main()
