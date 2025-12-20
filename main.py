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
from strategies.macd_strategy import MACDTrendStrategy


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
    # ============================================================
    # 1. INITIALIZE MT5
    # ============================================================

    from decouple import config

    mt5_config = MetaTraderConfig()

    # Toggle between live and demo accounts
    live = True

    if live:
        mt5_settings = {
            "username": config("MT5_USERNAME"),
            "password": config("MT5_PASSWORD"),
            "server": config("MT5_SERVER"),
            "mt5_pathway": config("MT5_PATHWAY"),
        }
    else:
        mt5_settings = {
            "username": config("MT5_USERNAME_TRIAL"),
            "password": config("MT5_PASSWORD_TRIAL"),
            "server": config("MT5_SERVER_TRIAL"),
            "mt5_pathway": config("MT5_PATHWAY"),
        }

    print(f"🎯 Trading Mode: {'LIVE' if live else 'DEMO'}")
    print(mt5_config.start_mt5(mt5_settings))

    if not mt5_config.start_mt5(mt5_settings):
        print("Failed to start MT5. Exiting...")
        return

    # ============================================================
    # 2. INITIALIZE STRATEGY
    # ============================================================

    strategy = MACDTrendStrategy(
        ema_period=200, macd_fast=12, macd_slow=26, macd_signal=9, risk_reward_ratio=2.5
    )

    print(f"\n{strategy}")
    print(f"Min bars required: {strategy.min_bars}\n")

    # ============================================================
    # 3. DEFINE TRADING PARAMETERS
    # ============================================================

    symbols = ["EURUSDm", "GBPUSDm", "USDJPYm"]  # Adjust to your broker's symbols
    timeframe = mt5.TIMEFRAME_M15

    # ============================================================
    # 4. SCAN SYMBOLS FOR SIGNALS
    # ============================================================

    TIMEFRAME_MINUTES = 15
    last_run_minute = None
    print("=" * 60)
    print("SCANNING FOR SIGNALS")
    print("=" * 60)
    while True:
        now = datetime.now()

        # Check alignment (e.g. 15m candle close)
        if now.minute % TIMEFRAME_MINUTES == 0:
            # Prevent running multiple times in same minute
            if last_run_minute != now.minute:
                last_run_minute = now.minute

                readable_time = now.strftime("%A, %d %B %Y — %I:%M:%S %p")
                print("\n" + "=" * 60)
                print(f"🕒 {readable_time}")
                print("=" * 60)

                for symbol in symbols:
                    print(f"\n📊 Analyzing {symbol}...")

                    # Fetch historical data
                    data = mt5_config.get_market_data_date_range(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        # Enough for 200 EMA calculation
                    )

                    # Check if data was fetched
                    if data.empty:
                        print(f"   ⚠️ No data available for {symbol}")
                        continue

                    print(f"   ✅ Loaded {len(data)} bars")

                    # Generate signal
                    signal_result = strategy.generate_signal(data)

                    # Display results
                    print(f"\n   Trend: {signal_result['trend']}")
                    print(f"   Current Price: {signal_result['entry_price']:.5f}")
                    print(f"   EMA 200: {signal_result['ema_200']:.5f}")
                    print(f"   MACD: {signal_result['macd_now']:.5f}")
                    print(f"   Signal Line: {signal_result['macd_signal_now']:.5f}")
                    print(f"   Reason: {signal_result['reason']}")
                    print(f"   Position Size: {signal_result['position_size']}")

                    # Handle trading signal
                    if signal_result["signal"]:
                        print(f"\n   🎯 SIGNAL: {signal_result['signal'].upper()}")
                        print(f"   Entry: {signal_result['entry_price']:.5f}")
                        print(f"   Stop Loss: {signal_result['stop_loss']:.5f}")
                        print(f"   Take Profit: {signal_result['take_profit']:.5f}")

                        # Calculate risk
                        risk = abs(
                            signal_result["entry_price"] - signal_result["stop_loss"]
                        )
                        reward = abs(
                            signal_result["take_profit"] - signal_result["entry_price"]
                        )
                        print(
                            f"   Risk: {risk:.5f} | Reward: {reward:.5f} | RR: {reward/risk:.1f}"
                        )

                        MAX_OPEN_TRADES = 3

                        open_trades = mt5_config.get_open_trades_count()

                        if open_trades >= MAX_OPEN_TRADES:
                            print(
                                f"   🚫 Trade skipped — open trades limit reached "
                                f"({open_trades}/{MAX_OPEN_TRADES})"
                            )
                        else:
                            print(
                                f"   🚫 Trade count — open trades  " f"({open_trades})"
                            )

                        # Execute trade (optional - uncomment to enable live trading)
                        execute = "y"
                        # Get symbol info once
                       
                        sl = normalize_price(symbol, signal_result["stop_loss"])
                        tp = normalize_price(symbol, signal_result["take_profit"])
                        if execute.lower() == "y":
                            success = mt5_config.execute_trade(
                                symbol=symbol,
                                signal=signal_result["signal"],
                                stop_loss=sl,
                                take_profit=tp,
                                lot_size=signal_result[
                                    "position_size"
                                ],  # ❌ DO NOT normalize
                                strategy_name="MACD_Trend",
                            )
                            if success:
                                print("   ✅ Trade executed successfully!")
                            else:
                                print("   ❌ Trade execution failed")
                    else:
                        print(f"   ⏸️ No signal")

                print("✅ Cycle complete")

        # Check every second (cheap + accurate)
        # time.sleep(1)

    # print("=" * 60)
    # print("SCANNING FOR SIGNALS")
    # print("=" * 60)
    # while True:
    #     now = datetime.now()
    #     readable_time = now.strftime("%A, %d %B %Y — %I:%M:%S %p")

    #     print("\n" + "=" * 60)
    #     print(f"🕒 {readable_time}")
    #     print("=" * 60)
    #     for symbol in symbols:
    #         print(f"\n📊 Analyzing {symbol}...")

    #         # Fetch historical data
    #         data = mt5_config.get_market_data_date_range(
    #             symbol=symbol,
    #             timeframe=timeframe,
    #             start_date=datetime.now() - timedelta(days=30),
    #             end_date=datetime.now(),
    #             # Enough for 200 EMA calculation
    #         )

    #         # Check if data was fetched
    #         if data.empty:
    #             print(f"   ⚠️ No data available for {symbol}")
    #             continue

    #         print(f"   ✅ Loaded {len(data)} bars")

    #         # Generate signal
    #         signal_result = strategy.generate_signal(data)

    #         # Display results
    #         print(f"\n   Trend: {signal_result['trend']}")
    #         print(f"   Current Price: {signal_result['entry_price']:.5f}")
    #         print(f"   EMA 200: {signal_result['ema_200']:.5f}")
    #         print(f"   MACD: {signal_result['macd_now']:.5f}")
    #         print(f"   Signal Line: {signal_result['macd_signal_now']:.5f}")
    #         print(f"   Reason: {signal_result['reason']}")
    #         print(f"   Position Size: {signal_result['position_size']}")

    #         # Handle trading signal
    #         if signal_result["signal"]:
    #             print(f"\n   🎯 SIGNAL: {signal_result['signal'].upper()}")
    #             print(f"   Entry: {signal_result['entry_price']:.5f}")
    #             print(f"   Stop Loss: {signal_result['stop_loss']:.5f}")
    #             print(f"   Take Profit: {signal_result['take_profit']:.5f}")

    #             # Calculate risk
    #             risk = abs(signal_result["entry_price"] - signal_result["stop_loss"])
    #             reward = abs(
    #                 signal_result["take_profit"] - signal_result["entry_price"]
    #             )
    #             print(
    #                 f"   Risk: {risk:.5f} | Reward: {reward:.5f} | RR: {reward/risk:.1f}"
    #             )

    #             MAX_OPEN_TRADES = 3

    #             open_trades = mt5_config.get_open_trades_count()

    #             if open_trades >= MAX_OPEN_TRADES:
    #                 print(
    #                     f"   🚫 Trade skipped — open trades limit reached "
    #                     f"({open_trades}/{MAX_OPEN_TRADES})"
    #                 )
    #             else:
    #                 print(f"   🚫 Trade count — open trades  " f"({open_trades})")

    #             # Execute trade (optional - uncomment to enable live trading)
    #             execute = "y"
    #             if execute.lower() == "y":
    #                 success = mt5_config.execute_trade(
    #                     symbol=symbol,
    #                     signal=signal_result["signal"],
    #                     stop_loss=signal_result["stop_loss"],
    #                     take_profit=signal_result["take_profit"],
    #                     lot_size=signal_result[
    #                         "position_size"
    #                     ],  # Auto-calculate based on 1% risk
    #                     strategy_name="MACD_Trend",
    #                 )
    #                 if success:
    #                     print("   ✅ Trade executed successfully!")
    #                 else:
    #                     print("   ❌ Trade execution failed")
    #         else:
    #             print(f"   ⏸️ No signal")

    # # ============================================================
    # # 5. RETRIEVE TRADE HISTORY (OPTIONAL)
    # # ============================================================

    # print("\n" + "=" * 60)
    # print("TRADE HISTORY")
    # print("=" * 60)

    # history = mt5_config.get_trade_history(
    #     start_time=datetime.now() - timedelta(days=7),
    #     end_time=datetime.now(),
    # )

    # if not history.empty:
    #     print(f"\nLast 5 trades:")
    #     print(history[['symbol', 'type', 'volume', 'price', 'profit', 'exit_reason']].tail())
    #     print(f"\nTotal P/L: ${history['profit'].sum():.2f}")
    # else:
    #     print("No trade history found")

    # # ============================================================
    # # 6. CLEANUP
    # # ============================================================

    # mt5_config.shutdown()
    # print("\n✅ Program completed")


if __name__ == "__main__":
    main()
