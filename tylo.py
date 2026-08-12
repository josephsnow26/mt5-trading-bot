from datetime import datetime, timezone
import MetaTrader5 as mt5
from decouple import config
import pandas as pd
import os
import time


if not mt5.initialize(
    path=config("MT5_PATHWAY"),
    login=int(config("MT5_USERNAME")),
    password=config("MT5_PASSWORD"),
    server=config("MT5_SERVER"),
):
    print("MT5 init failed:", mt5.last_error())
    exit()

print("MT5 connected:", mt5.account_info().server)
configs = {
    "XAUUSDm": 0.14, "XAGUSDm": 0.23, "EURUSDm": 0.28,
    "GBPUSDm": 0.35, "USDJPYm": 0.56, "USDCADm": 0.47,
}
total = 0
for symbol, lot in configs.items():
    mt5.symbol_select(symbol, True)
    time.sleep(0.5)  # let the terminal finish syncing specs after select

    info = mt5.symbol_info(symbol)
    print(f"{symbol}: contract_size={info.trade_contract_size} "
          f"margin_initial={info.margin_initial} "
          f"currency_margin={info.currency_margin} "
          f"currency_profit={info.currency_profit}")

    tick = mt5.symbol_info_tick(symbol)
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot, tick.ask)
    print(f"   → {lot} lot = ${margin} margin | last_error={mt5.last_error()}")
    total += margin or 0

print(f"TOTAL: ${total:.2f}")

acc = mt5.account_info()
print(f"Account leverage: 1:{acc.leverage}")

for symbol in ["XAUUSDm", "XAGUSDm", "EURUSDm", "GBPUSDm", "USDJPYm", "USDCADm"]:
    info = mt5.symbol_info(symbol)
    print(f"{symbol}: trade_calc_mode={info.trade_calc_mode} "
          f"trade_mode={info.trade_mode} "
          f"volume_min={info.volume_min} volume_max={info.volume_max}")