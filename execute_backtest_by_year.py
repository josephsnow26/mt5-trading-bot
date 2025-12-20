from datetime import datetime
import MetaTrader5

# Import your modules
from meter_trader_config import MetaTraderConfig
from reset import TradingSystem
from strategies.macd_strategy import MACDTrendStrategy
from strategies.ma_strategy import MATrendStrategy
from strategies.ma_pull_back_strategy import MATrendPullbackStrategy
from strategies.bollinger_pro import BollingerReversionProStrategy
from strategies.bolinger import BollingerReversionStrategy
from strategies.macd_trend_pull_back import TrendPullbackVolatilityStrategy
from strategies.new_macd_strategy import (
    EMARibbonRSIStrategy,
    ImprovedMACDTrendStrategy,
    AggressiveScalpStrategy,
    SupportResistanceBouncer,
    MicroAccountScalper,
    RSI2MeanReversion,
    StochasticBounceScalper,
    StructureBasedStrategy,
    AdaptiveTrendMomentumStrategy
)
from mt5_data_provider import MT5DataProvider
from backtester import Backtester
from main import project_settings
from datetime import datetime
import pandas as pd

# 1. Setup MT5
mt5_config = MetaTraderConfig()
mt5_config.start_mt5(project_settings)


def backtest_by_year_df(
    strategies_df, symbols, start_year, end_year, initial_balance=30
):
    """
    Run backtests for multiple strategies year by year using a DataFrame to hold strategies.

    Parameters
    ----------
    strategies_df : pd.DataFrame
        DataFrame with columns ["strategy_name", "strategy_instance"]
    symbols : list
        List of symbols to trade
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)
    initial_balance : float
        Starting balance for each year

    Returns
    -------
    pd.DataFrame
        Aggregated yearly results for all strategies
    """
    results_list = []

    for year in range(start_year, end_year + 1):
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31, 23, 59, 59)

        # Fetch data for the year
        df_dict = data_provider.fetch_multiple_symbols(
            symbols=symbols,
            timeframe=timeframe,
            start_date=year_start,
            end_date=year_end,
        )

        print(f"\n{'='*50}\n📅 BACKTESTING YEAR: {year}\n{'='*50}\n")

        for idx, row in strategies_df.iterrows():
            strategy_name = row["strategy_name"]
            strategy_instance = row["strategy_instance"]

            # Reset trading system for each strategy/year
            trading_system = TradingSystem(
                balance=initial_balance,
                risk_per_trade=0.02,
                max_positions=5,
                max_daily_loss=0.05,
                max_drawdown=0.15,
                correlation_groups={"EURUSD": "USD", "GBPUSD": "USD", "USDJPY": "USD"},
            )

            backtester = Backtester(
                strategy=strategy_instance,
                trading_system=trading_system,
                timeframe="1H",  # adjust as needed
            )

            # Filter out symbols with no data
            filtered_df_dict = {
                sym: df
                for sym, df in df_dict.items()
                if df is not None and not df.empty
            }
            if not filtered_df_dict:
                print(
                    f"⚠️ No data available for any symbol in {year}, skipping strategy {strategy_name}"
                )
                continue

            # Run backtest
            results = backtester.run(
                symbols=list(filtered_df_dict.keys()),
                df_dict=filtered_df_dict,
                start_date=year_start,
                end_date=year_end,
            )

            if results:
                results_list.append(
                    {
                        "year": year,
                        "strategy": strategy_name,
                        "total_trades": results["total_trades"],
                        "winning_trades": results["winning_trades"],
                        "losing_trades": results["losing_trades"],
                        "win_rate": results["win_rate"],
                        "net_pnl": results["net_pnl"],
                        "return_pct": results["return_pct"],
                        "final_balance": results["final_balance"],
                        "max_drawdown_pct": results["max_drawdown_pct"],
                    }
                )

    return pd.DataFrame(results_list)


# ------------------------
# Example usage
# ------------------------
strategies_df = pd.DataFrame(
    {
        "strategy_name": [
            # "MA Trend",
            # "MA Pullback",
            # "MAC D",
            # "Trend Pullback"
            # "Bollinger"
            # "schocasticbouncer"
            "adaptive"
        ],
        "strategy_instance": [
            # MATrendStrategy(risk_reward_ratio=3.0),
            # MATrendPullbackStrategy(risk_reward_ratio=3.0),
            MACDTrendStrategy(risk_reward_ratio=2.5),
            # TrendPullbackVolatilityStrategy(risk_reward_ratio=3.0)
            # BollingerReversionStrategy()
            # MicroAccountScalper(risk_reward_ratio=3.0)
            # StochasticBounceScalper(risk_reward_ratio=3.0)
            # StructureBasedStrategy()
            # AdaptiveTrendMomentumStrategy()
        ],
    }
)

symbols = [
    "USDJPYm",  # USD/JPY micro
    "EURUSDm",  # EUR/USD micro
    "GBPUSDm",  # GBP/USD micro
    "GBPJPYm",  # GBP/JPY micro
]
data_provider = MT5DataProvider(mt5_config)

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


yearly_results = backtest_by_year_df(
    strategies_df, symbols, start_year=2022, end_year=2025
)

# Export results
yearly_results.to_csv("yearly_strategy_comparison.csv", index=False)
print(yearly_results)
