import numpy as np
import pandas as pd


class RSITrendStrategy:
    """
    RSI Mean Reversion Strategy with Trend Filter

    Buy Signal: RSI < oversold_level AND price > trend_ma (uptrend)
    Sell Signal: RSI > overbought_level AND price < trend_ma (downtrend)

    Fully modular - can be used with any backtester or live system
    """

    def __init__(
        self,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        trend_ma_period=200,
        atr_period=14,
        atr_multiplier=1.5,
        risk_reward_ratio=2.0,
    ):
        """
        Parameters:
        -----------
        rsi_period : int
            Lookback period for RSI calculation (default: 14)
        rsi_oversold : float
            RSI level considered oversold (default: 30)
        rsi_overbought : float
            RSI level considered overbought (default: 70)
        trend_ma_period : int
            Moving average period for trend filter (default: 200)
        atr_period : int
            Period for ATR calculation (default: 14)
        atr_multiplier : float
            Multiplier for ATR to set stop loss distance (default: 1.5)
        risk_reward_ratio : float
            Take profit distance as multiple of stop loss (default: 2.0)
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_ma_period = trend_ma_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio

        # Minimum bars needed for strategy
        self.min_bars = max(rsi_period, trend_ma_period) + 1

    def calculate_rsi(self, prices):
        """
        Calculate RSI indicator

        Parameters:
        -----------
        prices : np.array
            Array of closing prices

        Returns:
        --------
        float or None
            RSI value or None if insufficient data
        """
        if len(prices) < self.rsi_period + 1:
            return None

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.rsi_period :])
        avg_loss = np.mean(losses[-self.rsi_period :])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_ma(self, prices):
        """
        Calculate simple moving average

        Parameters:
        -----------
        prices : np.array
            Array of closing prices

        Returns:
        --------
        float or None
            Moving average value or None if insufficient data
        """
        if len(prices) < self.trend_ma_period:
            return None

        return np.mean(prices[-self.trend_ma_period :])

    def calculate_atr(self, highs, lows, closes):
        """
        Calculate Average True Range

        Parameters:
        -----------
        highs : np.array
            Array of high prices
        lows : np.array
            Array of low prices
        closes : np.array
            Array of closing prices

        Returns:
        --------
        float or None
            ATR value or None if insufficient data
        """
        if len(closes) < self.atr_period + 1:
            return None

        high_low = highs[-self.atr_period :] - lows[-self.atr_period :]
        high_close = np.abs(
            highs[-self.atr_period :] - closes[-self.atr_period - 1 : -1]
        )
        low_close = np.abs(lows[-self.atr_period :] - closes[-self.atr_period - 1 : -1])

        true_ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.mean(true_ranges)

        return atr

    def generate_signal(self, price_data):
        """
        Generate trading signal based on price data

        Parameters:
        -----------
        price_data : dict
            Dictionary containing:
            - 'close': np.array of closing prices
            - 'high': np.array of high prices (optional)
            - 'low': np.array of low prices (optional)

        Returns:
        --------
        dict
            Signal information containing:
            - 'signal': 'buy', 'sell', or None
            - 'entry_price': float
            - 'stop_loss': float
            - 'take_profit': float
            - 'rsi': float
            - 'ma': float
            - 'atr': float
            - 'reason': str (explanation)
        """
        closes = price_data["close"]
        highs = price_data.get("high", closes)
        lows = price_data.get("low", closes)

        # Check minimum data requirement
        if len(closes) < self.min_bars:
            return {
                "signal": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reason": "Insufficient data",
            }

        # Calculate indicators
        current_price = closes[-1]
        rsi = self.calculate_rsi(closes)
        trend_ma = self.calculate_ma(closes)
        atr = self.calculate_atr(highs, lows, closes)

        # Validate indicators
        if rsi is None or trend_ma is None or atr is None:
            return {
                "signal": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reason": "Indicator calculation failed",
            }

        # Initialize result
        result = {
            "signal": None,
            "entry_price": current_price,
            "stop_loss": None,
            "take_profit": None,
            "rsi": rsi,
            "ma": trend_ma,
            "atr": atr,
            "reason": None,
        }

        # Calculate stop distance
        stop_distance = self.atr_multiplier * atr

        # BUY SIGNAL: Oversold RSI + Price above MA (uptrend)
        if rsi < self.rsi_oversold and current_price > trend_ma:
            result["signal"] = "buy"
            result["stop_loss"] = current_price - stop_distance
            result["take_profit"] = current_price + (
                stop_distance * self.risk_reward_ratio
            )
            result["reason"] = f"RSI oversold ({rsi:.1f}) in uptrend"

        # SELL SIGNAL: Overbought RSI + Price below MA (downtrend)
        elif rsi > self.rsi_overbought and current_price < trend_ma:
            result["signal"] = "sell"
            result["stop_loss"] = current_price + stop_distance
            result["take_profit"] = current_price - (
                stop_distance * self.risk_reward_ratio
            )
            result["reason"] = f"RSI overbought ({rsi:.1f}) in downtrend"

        # No signal - explain why
        else:
            if rsi < self.rsi_oversold and current_price < trend_ma:
                result["reason"] = f"RSI oversold but price below MA (downtrend)"
            elif rsi > self.rsi_overbought and current_price > trend_ma:
                result["reason"] = f"RSI overbought but price above MA (uptrend)"
            else:
                result["reason"] = f"RSI neutral ({rsi:.1f}), no extreme levels"

        return result

    def get_parameters(self):
        """
        Return strategy parameters as dictionary

        Returns:
        --------
        dict
            All strategy parameters
        """
        return {
            "name": "RSI Trend Strategy",
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "trend_ma_period": self.trend_ma_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self):
        """String representation of strategy"""
        params = self.get_parameters()
        return f"RSITrendStrategy(RSI={params['rsi_period']}, MA={params['trend_ma_period']}, RR={params['risk_reward_ratio']})"
