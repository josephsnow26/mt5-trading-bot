"""
MACD + EMA Trend Strategy
Clean, modular implementation with proper error handling and validation.
"""

import pandas as pd
from typing import Dict, Optional, Any

# 15 TF TIMEFRAME START WITH 10 DOLLARS   ABOVE 15 TF USE 100 DOLLARS


class MACDTrendStrategy:
    """
    MACD + 200 EMA Trend-Following Strategy

    Entry Rules:
    - Bullish: Price > 200 EMA AND MACD crosses above Signal
    - Bearish: Price < 200 EMA AND MACD crosses below Signal

    Risk Management:
    - Stop Loss: 200 EMA
    - Take Profit: Risk-Reward ratio applied to stop distance
    """

    def __init__(
        self,
        ema_period: int = 200,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        risk_reward_ratio: float = 3.0,
    ):
        """
        Initialize strategy parameters.

        Args:
            ema_period: Period for trend-following EMA (default: 200)
            macd_fast: Fast EMA period for MACD (default: 12)
            macd_slow: Slow EMA period for MACD (default: 26)
            macd_signal: Signal line period (default: 9)
            risk_reward_ratio: Target profit as multiple of risk (default: 3.0)
        """
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.risk_reward_ratio = risk_reward_ratio

        # Minimum bars needed for calculations
        self.min_bars = max(ema_period, macd_slow + macd_signal) + 2

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_macd(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Calculate MACD and Signal line.

        Returns:
            (macd_line, signal_line)
        """
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd, signal

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data has required columns and sufficient bars.

        Returns:
            (is_valid, error_message)
        """
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        if "close" not in price_data.columns:
            return False, "Missing 'close' column in price data"

        if "time" not in price_data.columns:
            return False, "Missing 'time' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary with reason."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "trend": None,
            "macd_prev": None,
            "macd_now": None,
            "macd_signal_prev": None,
            "macd_signal_now": None,
            "ema_200": None,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_macd_cross(
        self, macd_prev: float, macd_now: float, signal_prev: float, signal_now: float
    ) -> Optional[str]:
        """
        Detect MACD crossover direction.

        Returns:
            'bullish' for cross up, 'bearish' for cross down, None for no cross
        """
        if macd_prev < signal_prev and macd_now > signal_now:
            return "bullish"
        elif macd_prev > signal_prev and macd_now < signal_now:
            return "bearish"
        return None

    def _calculate_stop_and_target(
        self, entry_price: float, stop_price: float, signal_type: str
    ) -> tuple[float, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            stop_price: Stop loss price (200 EMA)
            signal_type: 'buy' or 'sell'

        Returns:
            (stop_loss, take_profit)
        """
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = stop_price  # Below entry
            tp = entry_price + target_distance  # Above entry
        else:  # sell
            sl = stop_price  # Above entry
            tp = entry_price - target_distance  # Below entry

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data and generate trading signal.

        Args:
            price_data: DataFrame with 'close' and 'time' columns

        Returns:
            Dictionary containing:
                - signal: 'buy', 'sell', or None
                - entry_price: Proposed entry price
                - stop_loss: Stop loss level
                - take_profit: Take profit level
                - trend: 'bullish' or 'bearish'
                - macd_prev/now: MACD values for last 2 bars
                - macd_signal_prev/now: Signal line values
                - ema_200: Current 200 EMA value
                - reason: Human-readable explanation
                - entry_date: Timestamp of signal
        """
        # Validate data
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            # Get close prices
            closes = price_data["close"]
            current_price = closes.iloc[-1]

            # Calculate indicators
            ema_200 = self._calculate_ema(closes, self.ema_period).iloc[-1]
            macd, macd_signal = self._calculate_macd(closes)

            # Get last 2 values for crossover detection
            macd_prev, macd_now = macd.iloc[-2], macd.iloc[-1]
            signal_prev, signal_now = macd_signal.iloc[-2], macd_signal.iloc[-1]

            # Determine trend
            trend = "bullish" if current_price > ema_200 else "bearish"

            # Detect MACD crossover
            macd_cross = self._detect_macd_cross(
                macd_prev, macd_now, signal_prev, signal_now
            )

            # Base result structure
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "position_size": 0.02,
                "macd_prev": float(macd_prev),
                "macd_now": float(macd_now),
                "macd_signal_prev": float(signal_prev),
                "macd_signal_now": float(signal_now),
                "ema_200": float(ema_200),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate signal based on trend and MACD cross
            if trend == "bullish" and macd_cross == "bullish":
                result["signal"] = "buy"
                result["reason"] = "MACD bullish cross in uptrend"
                sl, tp = self._calculate_stop_and_target(current_price, ema_200, "buy")
                sl, tp = round(sl, 3), round(sl, 3)
                result["stop_loss"] = sl
                result["take_profit"] = tp

            elif trend == "bearish" and macd_cross == "bearish":
                result["signal"] = "sell"
                result["reason"] = "MACD bearish cross in downtrend"
                sl, tp = self._calculate_stop_and_target(current_price, ema_200, "sell")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                # No signal conditions
                if macd_cross is None:
                    result["reason"] = "No MACD crossover detected"
                elif trend == "bullish" and macd_cross == "bearish":
                    result["reason"] = "MACD bearish cross conflicts with uptrend"
                elif trend == "bearish" and macd_cross == "bullish":
                    result["reason"] = "MACD bullish cross conflicts with downtrend"
                else:
                    result["reason"] = "No signal criteria met"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters as dictionary."""
        return {
            "name": "MACD Trend Strategy",
            "ema_period": self.ema_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"MACDTrendStrategy("
            f"EMA={self.ema_period}, "
            f"MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}, "
            f"RR={self.risk_reward_ratio})"
        )
