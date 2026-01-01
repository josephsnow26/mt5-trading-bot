from mt5.mt5_lib import MetaTraderConfig
from datetime import datetime, timedelta
from main import project_settings

meta_trader_config = MetaTraderConfig()
meta_trader_config.start_mt5(project_settings=project_settings)

start = datetime.now() - timedelta(days=7)
end = datetime.now()
df_trades  = meta_trader_config.get_trade_history(start, end)
print(df_trades)