from mt5_lib import MetaTraderConfig
import MetaTrader5
from decouple import config
from symbols import get_symbols
from datetime import datetime, timedelta


project_settings = {
    "mt5": {
        "username": config("MT5_USERNAME"),
        "password": config("MT5_PASSWORD"),
        "server": config("MT5_SERVER"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }
}
symbols = get_symbols().get('symbols')

meta_trader_config = MetaTraderConfig()

# initialize and login mt5
meta_trader_config.start_mt5(project_settings=project_settings)
end = datetime.now()
start = end - timedelta(days=1)

# df_trades  = meta_trader_config.get_trade_history(start, end)
# print(df_trades)
meta_trader_config.run_trading_loop(symbols=symbols, timeframe=MetaTrader5.TIMEFRAME_M1)
