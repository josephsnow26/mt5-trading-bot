from datetime import datetime
import MetaTrader5

# Import your modules
from meter_trader_config import MetaTraderConfig
from reset import TradingSystem
from strategies.macd_strategy import MACDTrendStrategy
from strategies.ma_strategy import MATrendStrategy
from mt5_data_provider import MT5DataProvider
from backtester import Backtester
from main import project_settings


# 1. Setup MT5
mt5_config = MetaTraderConfig()
print(project_settings)
mt5_config.start_mt5(project_settings)


# MT5 timeframe mapping
MT5_TIMEFRAME_MAP = {
    MetaTrader5.TIMEFRAME_M1: "1 Min",
    MetaTrader5.TIMEFRAME_M5: "5 Min",
    MetaTrader5.TIMEFRAME_M10: "10 Min",
    MetaTrader5.TIMEFRAME_M15: "15 Min",
    MetaTrader5.TIMEFRAME_M30: "30 Min",
    MetaTrader5.TIMEFRAME_H1: "1 Hour",
    MetaTrader5.TIMEFRAME_H4: "4 Hour",
    MetaTrader5.TIMEFRAME_D1: "1 Day",
    MetaTrader5.TIMEFRAME_W1: "1 Week",
    MetaTrader5.TIMEFRAME_MN1: "1 Month",
}

timeframe = MetaTrader5.TIMEFRAME_M15

# 2. Create data provider
data_provider = MT5DataProvider(mt5_config)

# 3. Fetch data
symbols = ["USDJPYm","EURUSDm","GBPUSDm"]  # Example symbols
data_dict = data_provider.fetch_multiple_symbols(
    symbols=symbols,
    timeframe=timeframe,
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 20),
)



strategy = MACDTrendStrategy(risk_reward_ratio=3.0)
# strategy = MATrendStrategy(risk_reward_ratio=3.0)


# 5. Create trading system (optional - for risk management)
correlation_groups = {"EURUSD": "USD", "GBPUSD": "USD", "USDJPY": "USD"}

trading_system = TradingSystem(
    balance=100,
    risk_per_trade=0.02,
    max_positions=5,
    max_daily_loss=0.05,
    max_drawdown=0.15,
    correlation_groups=correlation_groups,
)

# 6. Create backtester
backtester = Backtester(
    strategy=strategy,
    trading_system=trading_system,
    timeframe=MT5_TIMEFRAME_MAP[timeframe],
)

# 7. Run backtest
results = backtester.run(symbols=symbols, df_dict=data_dict)

# 8. Export results
if results:
    results["trades_df"].to_csv("trades.csv", index=False)
    results["equity_curve_df"].to_csv("equity.csv", index=False)
