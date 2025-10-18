from mt5_lib import MetaTraderConfig
import MetaTrader5
from decouple import config
from datetime import datetime
project_settings = {
    "mt5": {
        "username": config("MT5_USERNAME"),
        "password": config("MT5_PASSWORD"),
        "server": config("MT5_SERVER"),
        "mt5_pathway": config("MT5_PATHWAY"),
    }
}


meta_trader_config = MetaTraderConfig(project_settings=project_settings)

# initialize and login mt5
meta_trader_config.start_mt5()



meta_trader_config.run_trading_loop(symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"], timeframe=MetaTrader5.TIMEFRAME_M1, delay=60)



# Get data by date range
meta_trader_config.get_market_data_date_range(
    timeframe=MetaTrader5.TIMEFRAME_M15,
    start_date=datetime(2025, 4, 1),
    end_date=datetime(2025, 10, 1),
    symbol="EURUSDm",
    download = "csv"
)

