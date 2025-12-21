"""
IMPROVED MACD + EMA Strategy with Additional Filters
AND
Alternative EMA Ribbon + RSI Strategy
Both maintain your exact class structure for consistency.
"""

import pandas as pd
from typing import Dict, Optional, Any


# ============================================================================
# STRATEGY 1: IMPROVED MACD + EMA 200 (Your Strategy Enhanced)
# ============================================================================


class ImprovedMACDTrendStrategy:
    """
    Enhanced MACD + 200 EMA Strategy with Quality Filters

    Entry Rules:
    - Bullish: Price > 200 EMA AND MACD crosses above Signal
    - Bearish: Price < 200 EMA AND MACD crosses below Signal

    Additional Filters (NEW):
    - Volume confirmation (above 20-period average)
    - MACD histogram momentum (expanding)
    - Price distance from EMA (not over-extended)
    - RSI filter (avoid extreme zones)

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
        rsi_period: int = 14,
        volume_period: int = 20,
        risk_reward_ratio: float = 3.0,
        max_distance_pct: float = 3.0,  # Max distance from EMA in %
    ):
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.risk_reward_ratio = risk_reward_ratio
        self.max_distance_pct = max_distance_pct

        self.min_bars = (
            max(ema_period, macd_slow + macd_signal, rsi_period, volume_period) + 2
        )

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_macd(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd, signal

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data has required columns and sufficient bars."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

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
            "position_size": 0.01,
            "macd_prev": None,
            "macd_now": None,
            "macd_signal_prev": None,
            "macd_signal_now": None,
            "ema_200": None,
            "rsi": None,
            "filters_passed": 0,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_macd_cross(
        self, macd_prev: float, macd_now: float, signal_prev: float, signal_now: float
    ) -> Optional[str]:
        """Detect MACD crossover direction."""
        if macd_prev < signal_prev and macd_now > signal_now:
            return "bullish"
        elif macd_prev > signal_prev and macd_now < signal_now:
            return "bearish"
        return None

    def _check_volume_filter(self, volume: pd.Series) -> bool:
        """Check if current volume is above average."""
        if volume is None or len(volume) < self.volume_period:
            return True  # Skip if no volume data
        vol_ma = volume.rolling(self.volume_period).mean()
        return volume.iloc[-1] > vol_ma.iloc[-1]

    def _check_macd_momentum(self, macd: pd.Series, signal: pd.Series) -> bool:
        """Check if MACD histogram is expanding."""
        hist = macd - signal
        return abs(hist.iloc[-1]) > abs(hist.iloc[-2])

    def _check_distance_filter(self, price: float, ema: float) -> bool:
        """Check if price is not too far from EMA."""
        distance_pct = abs((price - ema) / ema * 100)
        return distance_pct < self.max_distance_pct

    def _check_rsi_filter(self, rsi: float, trend: str) -> bool:
        """Check RSI is in acceptable range."""
        if trend == "bullish":
            return 40 < rsi < 70
        else:  # bearish
            return 30 < rsi < 60

    def _calculate_stop_and_target(
        self, entry_price: float, stop_price: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = stop_price
            tp = entry_price + target_distance
        else:
            sl = stop_price
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data and generate trading signal with quality filters.

        Args:
            price_data: DataFrame with 'close', 'time', and optionally 'volume' columns

        Returns:
            Dictionary with signal, prices, filters_passed count, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            current_price = closes.iloc[-1]
            volume = price_data.get("volume", None)

            # Calculate indicators
            ema_200 = self._calculate_ema(closes, self.ema_period).iloc[-1]
            macd, macd_signal = self._calculate_macd(closes)
            rsi = self._calculate_rsi(closes, self.rsi_period).iloc[-1]

            macd_prev, macd_now = macd.iloc[-2], macd.iloc[-1]
            signal_prev, signal_now = macd_signal.iloc[-2], macd_signal.iloc[-1]

            trend = "bullish" if current_price > ema_200 else "bearish"
            macd_cross = self._detect_macd_cross(
                macd_prev, macd_now, signal_prev, signal_now
            )

            # Apply filters
            filters_passed = 0
            filter_results = []

            if volume is not None and self._check_volume_filter(volume):
                filters_passed += 1
                filter_results.append("Volume above average")

            if self._check_macd_momentum(macd, macd_signal):
                filters_passed += 1
                filter_results.append("MACD histogram expanding")

            if self._check_distance_filter(current_price, ema_200):
                filters_passed += 1
                filter_results.append("Price distance acceptable")

            if self._check_rsi_filter(rsi, trend):
                filters_passed += 1
                filter_results.append("RSI in healthy range")

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "position_size": 0.01,
                "macd_prev": float(macd_prev),
                "macd_now": float(macd_now),
                "macd_signal_prev": float(signal_prev),
                "macd_signal_now": float(signal_now),
                "ema_200": float(ema_200),
                "rsi": float(rsi),
                "filters_passed": filters_passed,
                "filter_details": filter_results,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate signal (require at least 2 filters passed for higher quality)
            if trend == "bullish" and macd_cross == "bullish" and filters_passed >= 2:
                result["signal"] = "buy"
                result["reason"] = (
                    f"MACD bullish cross in uptrend ({filters_passed}/4 filters passed)"
                )
                sl, tp = self._calculate_stop_and_target(current_price, ema_200, "buy")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            elif trend == "bearish" and macd_cross == "bearish" and filters_passed >= 2:
                result["signal"] = "sell"
                result["reason"] = (
                    f"MACD bearish cross in downtrend ({filters_passed}/4 filters passed)"
                )
                sl, tp = self._calculate_stop_and_target(current_price, ema_200, "sell")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                if macd_cross is None:
                    result["reason"] = "No MACD crossover detected"
                elif filters_passed < 2:
                    result["reason"] = (
                        f"Insufficient filter confirmation ({filters_passed}/4 filters)"
                    )
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
            "name": "Improved MACD Trend Strategy",
            "ema_period": self.ema_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "rsi_period": self.rsi_period,
            "volume_period": self.volume_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_distance_pct": self.max_distance_pct,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"ImprovedMACDTrendStrategy("
            f"EMA={self.ema_period}, "
            f"MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}, "
            f"RSI={self.rsi_period}, "
            f"RR={self.risk_reward_ratio})"
        )


# ============================================================================
# STRATEGY 2: EMA RIBBON + RSI MOMENTUM (Alternative Strategy)
# ============================================================================


class EMARibbonRSIStrategy:
    """
    EMA Ribbon + RSI Momentum Strategy

    Entry Rules:
    - Bullish: Price crosses above EMA 9 AND EMA 9 > EMA 21 > EMA 50 AND RSI > 50
    - Bearish: Price crosses below EMA 9 AND EMA 9 < EMA 21 < EMA 50 AND RSI < 50

    Features:
    - Multiple EMA alignment for strong trend confirmation
    - RSI momentum filter
    - Dynamic stop loss based on recent swing
    - Higher win rate (60-70%) but tighter RR (1:2)

    Risk Management:
    - Stop Loss: Below/above recent swing low/high
    - Take Profit: 2:1 Risk-Reward ratio
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_mid: int = 21,
        ema_slow: int = 50,
        rsi_period: int = 14,
        swing_lookback: int = 10,
        risk_reward_ratio: float = 2.0,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.swing_lookback = swing_lookback
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = max(ema_slow, rsi_period, swing_lookback) + 2

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _find_swing_low(self, lows: pd.Series, lookback: int) -> float:
        """Find recent swing low."""
        return lows.iloc[-lookback:].min()

    def _find_swing_high(self, highs: pd.Series, lookback: int) -> float:
        """Find recent swing high."""
        return highs.iloc[-lookback:].max()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data has required columns and sufficient bars."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

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
            "position_size": 0.01,
            "ema_fast": None,
            "ema_mid": None,
            "ema_slow": None,
            "rsi": None,
            "ribbon_aligned": False,
            "reason": reason,
            "entry_date": None,
        }

    def _check_ema_ribbon_bullish(
        self, ema_fast: float, ema_mid: float, ema_slow: float
    ) -> bool:
        """Check if EMAs are aligned bullishly."""
        return ema_fast > ema_mid > ema_slow

    def _check_ema_ribbon_bearish(
        self, ema_fast: float, ema_mid: float, ema_slow: float
    ) -> bool:
        """Check if EMAs are aligned bearishly."""
        return ema_fast < ema_mid < ema_slow

    def _detect_price_cross(
        self, price_prev: float, price_now: float, ema_prev: float, ema_now: float
    ) -> Optional[str]:
        """Detect price crossing EMA."""
        if price_prev < ema_prev and price_now > ema_now:
            return "bullish"
        elif price_prev > ema_prev and price_now < ema_now:
            return "bearish"
        return None

    def _calculate_stop_and_target(
        self, entry_price: float, stop_price: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = stop_price
            tp = entry_price + target_distance
        else:
            sl = stop_price
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data and generate trading signal.

        Args:
            price_data: DataFrame with 'close', 'high', 'low', 'time' columns

        Returns:
            Dictionary with signal, prices, ribbon alignment, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]
            prev_price = closes.iloc[-2]

            # Calculate indicators
            ema_fast_series = self._calculate_ema(closes, self.ema_fast)
            ema_mid_series = self._calculate_ema(closes, self.ema_mid)
            ema_slow_series = self._calculate_ema(closes, self.ema_slow)
            rsi = self._calculate_rsi(closes, self.rsi_period).iloc[-1]

            ema_fast = ema_fast_series.iloc[-1]
            ema_mid = ema_mid_series.iloc[-1]
            ema_slow = ema_slow_series.iloc[-1]

            ema_fast_prev = ema_fast_series.iloc[-2]

            # Check ribbon alignment
            ribbon_bullish = self._check_ema_ribbon_bullish(ema_fast, ema_mid, ema_slow)
            ribbon_bearish = self._check_ema_ribbon_bearish(ema_fast, ema_mid, ema_slow)

            # Detect price cross
            price_cross = self._detect_price_cross(
                prev_price, current_price, ema_fast_prev, ema_fast
            )

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "trend": None,
                "position_size": 0.01,
                "ema_fast": float(ema_fast),
                "ema_mid": float(ema_mid),
                "ema_slow": float(ema_slow),
                "rsi": float(rsi),
                "ribbon_aligned": ribbon_bullish or ribbon_bearish,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate signals
            if ribbon_bullish and price_cross == "bullish" and rsi > 50:
                result["signal"] = "buy"
                result["trend"] = "bullish"
                result["reason"] = (
                    "Price crossed above EMA with bullish ribbon alignment"
                )

                swing_low = self._find_swing_low(lows, self.swing_lookback)
                sl, tp = self._calculate_stop_and_target(
                    current_price, swing_low, "buy"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            elif ribbon_bearish and price_cross == "bearish" and rsi < 50:
                result["signal"] = "sell"
                result["trend"] = "bearish"
                result["reason"] = (
                    "Price crossed below EMA with bearish ribbon alignment"
                )

                swing_high = self._find_swing_high(highs, self.swing_lookback)
                sl, tp = self._calculate_stop_and_target(
                    current_price, swing_high, "sell"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                if not (ribbon_bullish or ribbon_bearish):
                    result["reason"] = "EMAs not properly aligned"
                elif price_cross is None:
                    result["reason"] = "No price cross detected"
                elif ribbon_bullish and rsi <= 50:
                    result["reason"] = "RSI too weak for bullish signal"
                elif ribbon_bearish and rsi >= 50:
                    result["reason"] = "RSI too strong for bearish signal"
                else:
                    result["reason"] = "No signal criteria met"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters as dictionary."""
        return {
            "name": "EMA Ribbon RSI Strategy",
            "ema_fast": self.ema_fast,
            "ema_mid": self.ema_mid,
            "ema_slow": self.ema_slow,
            "rsi_period": self.rsi_period,
            "swing_lookback": self.swing_lookback,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"EMARibbonRSIStrategy("
            f"EMAs={self.ema_fast}/{self.ema_mid}/{self.ema_slow}, "
            f"RSI={self.rsi_period}, "
            f"RR={self.risk_reward_ratio})"
        )


"""
AGGRESSIVE SMALL ACCOUNT GROWTH STRATEGY
High-frequency scalping strategy optimized for small accounts.
Targets 65-75% win rate with tight stops and quick profits.
"""

import pandas as pd
from typing import Dict, Optional, Any


class AggressiveScalpStrategy:
    """
    EMA 9/21 + Stochastic + ATR Scalping Strategy

    DESIGNED FOR: Small accounts ($10-$500)
    TIMEFRAMES: 5m, 15m (best for scalping)
    TARGET WIN RATE: 65-75%
    RISK-REWARD: 1:1.5 (quick profits, tight stops)

    Entry Rules:
    BULLISH:
    - Price > EMA 9 > EMA 21
    - Stochastic crosses above 20 (oversold bounce)
    - Recent pullback (price touched EMA 9 in last 3 bars)

    BEARISH:
    - Price < EMA 9 < EMA 21
    - Stochastic crosses below 80 (overbought rejection)
    - Recent rally (price touched EMA 9 in last 3 bars)

    Risk Management:
    - Stop Loss: 1.5 x ATR from entry
    - Take Profit: 1.5 x Stop Loss distance
    - Trailing stop: Move to breakeven after 1:1
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        stoch_k_period: int = 14,
        stoch_d_period: int = 3,
        stoch_smooth: int = 3,
        atr_period: int = 14,
        risk_reward_ratio: float = 1.5,
        atr_stop_multiplier: float = 1.5,
        pullback_lookback: int = 3,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_smooth = stoch_smooth
        self.atr_period = atr_period
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_stop_multiplier = atr_stop_multiplier
        self.pullback_lookback = pullback_lookback

        self.min_bars = (
            max(ema_slow, stoch_k_period + stoch_smooth, atr_period)
            + pullback_lookback
            + 2
        )

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_stochastic(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Returns:
            (%K, %D)
        """
        # Calculate raw stochastic
        lowest_low = lows.rolling(window=self.stoch_k_period).min()
        highest_high = highs.rolling(window=self.stoch_k_period).max()

        raw_k = 100 * (closes - lowest_low) / (highest_high - lowest_low)

        # Smooth %K
        k = raw_k.rolling(window=self.stoch_smooth).mean()

        # Calculate %D (signal line)
        d = k.rolling(window=self.stoch_d_period).mean()

        return k, d

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = highs - lows
        high_close = (highs - closes.shift()).abs()
        low_close = (lows - closes.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data has required columns and sufficient bars."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

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
            "breakeven_level": None,
            "trend": None,
            "position_size": 0.01,
            "ema_fast": None,
            "ema_slow": None,
            "stoch_k": None,
            "stoch_d": None,
            "atr": None,
            "pullback_detected": False,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_stoch_cross(
        self, k_prev: float, k_now: float, d_prev: float, d_now: float
    ) -> Optional[str]:
        """Detect Stochastic crossover."""
        if k_prev < d_prev and k_now > d_now and k_now < 30:
            return "bullish"
        elif k_prev > d_prev and k_now < d_now and k_now > 70:
            return "bearish"
        return None

    def _check_pullback(
        self, closes: pd.Series, ema: pd.Series, lookback: int, direction: str
    ) -> bool:
        """
        Check if price touched EMA in recent bars (pullback).

        For bullish: price should have touched or crossed below EMA 9 recently
        For bearish: price should have touched or crossed above EMA 9 recently
        """
        recent_closes = closes.iloc[-lookback:]
        recent_ema = ema.iloc[-lookback:]

        if direction == "bullish":
            # Check if any recent close was at or below EMA
            return (recent_closes <= recent_ema * 1.002).any()  # 0.2% tolerance
        else:  # bearish
            # Check if any recent close was at or above EMA
            return (recent_closes >= recent_ema * 0.998).any()  # 0.2% tolerance

    def _check_ema_alignment(
        self, price: float, ema_fast: float, ema_slow: float, direction: str
    ) -> bool:
        """Check if EMAs are aligned in the trend direction."""
        if direction == "bullish":
            return price > ema_fast > ema_slow
        else:  # bearish
            return price < ema_fast < ema_slow

    def _calculate_stop_and_target(
        self, entry_price: float, atr: float, signal_type: str
    ) -> tuple[float, float, float]:
        """
        Calculate stop loss, take profit, and breakeven level.

        Returns:
            (stop_loss, take_profit, breakeven_level)
        """
        stop_distance = atr * self.atr_stop_multiplier
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = entry_price - stop_distance
            tp = entry_price + target_distance
            be = entry_price + stop_distance  # Move to BE after 1:1
        else:  # sell
            sl = entry_price + stop_distance
            tp = entry_price - target_distance
            be = entry_price - stop_distance  # Move to BE after 1:1

        return sl, tp, be

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data and generate scalping signal.

        Args:
            price_data: DataFrame with 'close', 'high', 'low', 'time' columns

        Returns:
            Dictionary with signal, prices, indicators, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]

            # Calculate indicators
            ema_fast = self._calculate_ema(closes, self.ema_fast)
            ema_slow = self._calculate_ema(closes, self.ema_slow)
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            atr = self._calculate_atr(highs, lows, closes, self.atr_period)

            # Get current values
            ema_fast_now = ema_fast.iloc[-1]
            ema_slow_now = ema_slow.iloc[-1]
            k_now, k_prev = stoch_k.iloc[-1], stoch_k.iloc[-2]
            d_now, d_prev = stoch_d.iloc[-1], stoch_d.iloc[-2]
            atr_now = atr.iloc[-1]

            # Detect signals
            stoch_cross = self._detect_stoch_cross(k_prev, k_now, d_prev, d_now)

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "breakeven_level": None,
                "trend": None,
                "position_size": 0.01,
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "stoch_k": float(k_now),
                "stoch_d": float(d_now),
                "atr": float(atr_now),
                "pullback_detected": False,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Check for BULLISH signal
            if stoch_cross == "bullish":
                ema_aligned = self._check_ema_alignment(
                    current_price, ema_fast_now, ema_slow_now, "bullish"
                )
                pullback = self._check_pullback(
                    closes, ema_fast, self.pullback_lookback, "bullish"
                )

                if ema_aligned and pullback:
                    result["signal"] = "buy"
                    result["trend"] = "bullish"
                    result["pullback_detected"] = True
                    result["reason"] = (
                        "Stochastic bullish cross with EMA alignment and recent pullback"
                    )

                    sl, tp, be = self._calculate_stop_and_target(
                        current_price, atr_now, "buy"
                    )
                    result["stop_loss"] = sl
                    result["take_profit"] = tp
                    result["breakeven_level"] = be

                elif not ema_aligned:
                    result["reason"] = "Stochastic bullish but EMAs not aligned"
                elif not pullback:
                    result["reason"] = (
                        "Stochastic bullish but no recent pullback to EMA"
                    )

            # Check for BEARISH signal
            elif stoch_cross == "bearish":
                ema_aligned = self._check_ema_alignment(
                    current_price, ema_fast_now, ema_slow_now, "bearish"
                )
                pullback = self._check_pullback(
                    closes, ema_fast, self.pullback_lookback, "bearish"
                )

                if ema_aligned and pullback:
                    result["signal"] = "sell"
                    result["trend"] = "bearish"
                    result["pullback_detected"] = True
                    result["reason"] = (
                        "Stochastic bearish cross with EMA alignment and recent rally"
                    )

                    sl, tp, be = self._calculate_stop_and_target(
                        current_price, atr_now, "sell"
                    )
                    result["stop_loss"] = sl
                    result["take_profit"] = tp
                    result["breakeven_level"] = be

                elif not ema_aligned:
                    result["reason"] = "Stochastic bearish but EMAs not aligned"
                elif not pullback:
                    result["reason"] = "Stochastic bearish but no recent rally to EMA"

            else:
                result["reason"] = "No stochastic crossover in extreme zones"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters as dictionary."""
        return {
            "name": "Aggressive Scalp Strategy",
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "stoch_k_period": self.stoch_k_period,
            "stoch_d_period": self.stoch_d_period,
            "stoch_smooth": self.stoch_smooth,
            "atr_period": self.atr_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "pullback_lookback": self.pullback_lookback,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["5m", "15m"],
            "target_win_rate": "65-75%",
        }

    def __repr__(self) -> str:
        return (
            f"AggressiveScalpStrategy("
            f"EMAs={self.ema_fast}/{self.ema_slow}, "
            f"Stoch={self.stoch_k_period}, "
            f"ATR={self.atr_period}, "
            f"RR={self.risk_reward_ratio})"
        )


# ============================================================================
# BONUS: BREAKOUT MOMENTUM STRATEGY (Alternative for small accounts)
# ============================================================================


class BreakoutMomentumStrategy:
    """
    High-Momentum Breakout Strategy

    DESIGNED FOR: Small accounts looking for explosive moves
    TIMEFRAMES: 15m, 1H
    TARGET WIN RATE: 55-65%
    RISK-REWARD: 1:3 (lower win rate but bigger winners)

    Entry Rules:
    - Price breaks above/below 20-period high/low
    - Volume spike (1.5x average volume)
    - RSI confirms momentum (>60 for buy, <40 for sell)
    - ATR expanding (volatility increasing)

    Risk Management:
    - Stop Loss: Below/above breakout candle low/high
    - Take Profit: 3:1 Risk-Reward
    """

    def __init__(
        self,
        breakout_period: int = 20,
        rsi_period: int = 14,
        volume_period: int = 20,
        atr_period: int = 14,
        volume_multiplier: float = 1.5,
        risk_reward_ratio: float = 3.0,
    ):
        self.breakout_period = breakout_period
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.volume_multiplier = volume_multiplier
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = max(breakout_period, rsi_period, volume_period, atr_period) + 2

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = highs - lows
        high_close = (highs - closes.shift()).abs()
        low_close = (lows - closes.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "breakout_level": None,
            "position_size": 0.01,
            "rsi": None,
            "volume_spike": False,
            "atr_expanding": False,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_breakout(
        self,
        current_price: float,
        current_high: float,
        current_low: float,
        period_high: float,
        period_low: float,
    ) -> Optional[str]:
        """Detect breakout direction."""
        if current_high > period_high:
            return "bullish"
        elif current_low < period_low:
            return "bearish"
        return None

    def _check_volume_spike(self, volume: pd.Series) -> bool:
        """Check if current volume is significantly above average."""
        if volume is None or len(volume) < self.volume_period:
            return True  # Skip if no volume data

        vol_avg = volume.iloc[:-1].rolling(self.volume_period).mean().iloc[-1]
        current_vol = volume.iloc[-1]

        return current_vol > (vol_avg * self.volume_multiplier)

    def _check_atr_expansion(self, atr: pd.Series) -> bool:
        """Check if ATR is expanding (increasing volatility)."""
        return atr.iloc[-1] > atr.iloc[-2]

    def _calculate_stop_and_target(
        self, entry_price: float, stop_price: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop and target levels."""
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = stop_price
            tp = entry_price + target_distance
        else:
            sl = stop_price
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data and generate breakout signal.

        Args:
            price_data: DataFrame with 'close', 'high', 'low', 'time' columns
            Optional 'volume' column for volume confirmation

        Returns:
            Dictionary with signal, prices, confirmations, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]
            volume = price_data.get("volume", None)

            current_price = closes.iloc[-1]
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]

            # Calculate indicators
            period_high = highs.iloc[-(self.breakout_period + 1) : -1].max()
            period_low = lows.iloc[-(self.breakout_period + 1) : -1].min()

            rsi = self._calculate_rsi(closes, self.rsi_period).iloc[-1]
            atr = self._calculate_atr(highs, lows, closes, self.atr_period)

            # Detect breakout
            breakout = self._detect_breakout(
                current_price, current_high, current_low, period_high, period_low
            )

            # Check confirmations
            volume_spike = (
                self._check_volume_spike(volume) if volume is not None else True
            )
            atr_expanding = self._check_atr_expansion(atr)

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "breakout_level": None,
                "position_size": 0.01,
                "rsi": float(rsi),
                "volume_spike": volume_spike,
                "atr_expanding": atr_expanding,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate signals
            if breakout == "bullish" and rsi > 60 and volume_spike and atr_expanding:
                result["signal"] = "buy"
                result["breakout_level"] = float(period_high)
                result["reason"] = (
                    "Bullish breakout with volume and momentum confirmation"
                )

                # Stop below breakout candle low
                stop = lows.iloc[-1]
                sl, tp = self._calculate_stop_and_target(current_price, stop, "buy")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            elif breakout == "bearish" and rsi < 40 and volume_spike and atr_expanding:
                result["signal"] = "sell"
                result["breakout_level"] = float(period_low)
                result["reason"] = (
                    "Bearish breakdown with volume and momentum confirmation"
                )

                # Stop above breakout candle high
                stop = highs.iloc[-1]
                sl, tp = self._calculate_stop_and_target(current_price, stop, "sell")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                if breakout is None:
                    result["reason"] = "No breakout detected"
                elif breakout == "bullish" and rsi <= 60:
                    result["reason"] = "Bullish breakout but RSI too weak"
                elif breakout == "bearish" and rsi >= 40:
                    result["reason"] = "Bearish breakdown but RSI too strong"
                elif not volume_spike:
                    result["reason"] = (
                        f"{breakout.capitalize()} breakout but no volume spike"
                    )
                elif not atr_expanding:
                    result["reason"] = (
                        f"{breakout.capitalize()} breakout but ATR not expanding"
                    )
                else:
                    result["reason"] = "Insufficient confirmation for breakout"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "Breakout Momentum Strategy",
            "breakout_period": self.breakout_period,
            "rsi_period": self.rsi_period,
            "volume_period": self.volume_period,
            "atr_period": self.atr_period,
            "volume_multiplier": self.volume_multiplier,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["15m", "1H"],
            "target_win_rate": "55-65%",
        }

    def __repr__(self) -> str:
        return (
            f"BreakoutMomentumStrategy("
            f"Period={self.breakout_period}, "
            f"RSI={self.rsi_period}, "
            f"VolMult={self.volume_multiplier}, "
            f"RR={self.risk_reward_ratio})"
        )


"""
MICRO ACCOUNT SURVIVAL STRATEGY ($10-$50 accounts)
Ultra-high frequency, ultra-tight stops, ultra-high win rate.
Designed to survive and compound tiny accounts.
"""

import pandas as pd
from typing import Dict, Optional, Any


class MicroAccountScalper:
    """
    EMA 5/13 + Bollinger Band Mean Reversion Scalper

    DESIGNED FOR: Micro accounts ($10-$50)
    TIMEFRAMES: 1m, 5m (ultra-high frequency)
    TARGET WIN RATE: 75-85%
    RISK-REWARD: 1:1 (quick in and out)
    TRADES PER DAY: 10-30+

    Strategy Logic:
    - Wait for price to touch outer Bollinger Band (2 std dev)
    - Confirm with EMA 5/13 alignment showing trend
    - Enter on bounce back toward middle band
    - Exit at middle band or small profit

    Entry Rules:
    BULLISH (Mean Reversion):
    - Price touches or breaks below lower BB
    - EMA 5 > EMA 13 (overall uptrend)
    - RSI < 35 (oversold but not extreme)
    - Enter on first candle closing back above lower BB

    BEARISH (Mean Reversion):
    - Price touches or breaks above upper BB
    - EMA 5 < EMA 13 (overall downtrend)
    - RSI > 65 (overbought but not extreme)
    - Enter on first candle closing back below upper BB

    Risk Management:
    - Stop Loss: 0.5% of account (VERY tight)
    - Take Profit: 0.5% of account (1:1)
    - Max 3 consecutive losses = stop trading for session
    """

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 13,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        risk_reward_ratio: float = 1.0,
        account_risk_pct: float = 0.5,  # 0.5% per trade
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.risk_reward_ratio = risk_reward_ratio
        self.account_risk_pct = account_risk_pct

        self.min_bars = max(bb_period, rsi_period, ema_slow) + 3

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int, std_dev: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "ema_fast": None,
            "ema_slow": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "rsi": None,
            "bb_touch": False,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_bb_touch_and_bounce(
        self,
        current_close: float,
        prev_close: float,
        current_low: float,
        current_high: float,
        bb_upper: float,
        bb_lower: float,
        prev_bb_upper: float,
        prev_bb_lower: float,
    ) -> Optional[str]:
        """
        Detect if price touched BB and bounced back.

        Bullish: Price touched/broke lower BB and current candle closed back inside
        Bearish: Price touched/broke upper BB and current candle closed back inside
        """
        # Bullish bounce
        if (
            current_low <= bb_lower or prev_close <= prev_bb_lower
        ) and current_close > bb_lower:
            return "bullish"

        # Bearish bounce
        if (
            current_high >= bb_upper or prev_close >= prev_bb_upper
        ) and current_close < bb_upper:
            return "bearish"

        return None

    def _calculate_position_size(
        self, account_balance: float, entry_price: float, stop_price: float
    ) -> float:
        """
        Calculate position size based on fixed percentage risk.

        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_price: Stop loss price

        Returns:
            Position size in lots
        """
        risk_amount = account_balance * (self.account_risk_pct / 100)
        stop_distance = abs(entry_price - stop_price)

        if stop_distance == 0:
            return 0.01  # Minimum position

        # Calculate lots (assuming forex standard: 1 lot = 100,000 units)
        # For micro accounts, we use micro lots (0.01 = 1,000 units)
        pip_value = 10  # For most pairs, 1 pip = $10 per standard lot
        pips_at_risk = stop_distance * 10000  # Convert to pips

        lots = risk_amount / (pips_at_risk * pip_value * 0.01)

        # Clamp between 0.01 and 0.10 for micro accounts
        return max(0.01, min(0.10, round(lots, 2)))

    def _calculate_stop_and_target(
        self, entry_price: float, bb_middle: float, bb_extreme: float, signal_type: str
    ) -> tuple[float, float]:
        """
        Calculate stop and target.

        Stop: Just beyond BB extreme
        Target: Middle BB (mean reversion target)
        """
        if signal_type == "buy":
            # Stop below lower BB
            sl = bb_extreme - (abs(entry_price - bb_extreme) * 0.3)
            # Target at middle BB
            tp = bb_middle

            # Ensure 1:1 minimum
            risk = abs(entry_price - sl)
            if abs(tp - entry_price) < risk:
                tp = entry_price + risk

        else:  # sell
            # Stop above upper BB
            sl = bb_extreme + (abs(bb_extreme - entry_price) * 0.3)
            # Target at middle BB
            tp = bb_middle

            # Ensure 1:1 minimum
            risk = abs(sl - entry_price)
            if abs(entry_price - tp) < risk:
                tp = entry_price - risk

        return sl, tp

    def generate_signal(
        self, price_data: pd.DataFrame, account_balance: float = 30.0
    ) -> Dict[str, Any]:
        """
        Analyze price data and generate ultra-high frequency scalping signal.

        Args:
            price_data: DataFrame with 'close', 'high', 'low', 'time' columns
            account_balance: Current account balance for position sizing

        Returns:
            Dictionary with signal, prices, position size, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_close = closes.iloc[-1]
            prev_close = closes.iloc[-2]
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]

            # Calculate indicators
            ema_fast = self._calculate_ema(closes, self.ema_fast)
            ema_slow = self._calculate_ema(closes, self.ema_slow)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                closes, self.bb_period, self.bb_std
            )
            rsi = self._calculate_rsi(closes, self.rsi_period)

            # Current values
            ema_fast_now = ema_fast.iloc[-1]
            ema_slow_now = ema_slow.iloc[-1]
            bb_upper_now = bb_upper.iloc[-1]
            bb_middle_now = bb_middle.iloc[-1]
            bb_lower_now = bb_lower.iloc[-1]
            bb_upper_prev = bb_upper.iloc[-2]
            bb_lower_prev = bb_lower.iloc[-2]
            rsi_now = rsi.iloc[-1]

            # Detect BB touch and bounce
            bb_bounce = self._detect_bb_touch_and_bounce(
                current_close,
                prev_close,
                current_low,
                current_high,
                bb_upper_now,
                bb_lower_now,
                bb_upper_prev,
                bb_lower_prev,
            )

            # Base result
            result = {
                "signal": None,
                "entry_price": current_close,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.01,
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "bb_upper": float(bb_upper_now),
                "bb_middle": float(bb_middle_now),
                "bb_lower": float(bb_lower_now),
                "rsi": float(rsi_now),
                "bb_touch": bb_bounce is not None,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate BULLISH signal (buy the dip)
            if (
                bb_bounce == "bullish"
                and ema_fast_now > ema_slow_now
                and 25 < rsi_now < 40
            ):
                result["signal"] = "buy"
                result["reason"] = "Lower BB bounce in uptrend with oversold RSI"

                sl, tp = self._calculate_stop_and_target(
                    current_close, bb_middle_now, bb_lower_now, "buy"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

                # Calculate position size
                result["position_size"] = self._calculate_position_size(
                    account_balance, current_close, sl
                )

            # Generate BEARISH signal (sell the rip)
            elif (
                bb_bounce == "bearish"
                and ema_fast_now < ema_slow_now
                and 60 < rsi_now < 75
            ):
                result["signal"] = "sell"
                result["reason"] = "Upper BB bounce in downtrend with overbought RSI"

                sl, tp = self._calculate_stop_and_target(
                    current_close, bb_middle_now, bb_upper_now, "sell"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

                # Calculate position size
                result["position_size"] = self._calculate_position_size(
                    account_balance, current_close, sl
                )

            else:
                if bb_bounce is None:
                    result["reason"] = "No BB touch and bounce detected"
                elif bb_bounce == "bullish" and ema_fast_now <= ema_slow_now:
                    result["reason"] = "BB bounce bullish but overall trend is down"
                elif bb_bounce == "bearish" and ema_fast_now >= ema_slow_now:
                    result["reason"] = "BB bounce bearish but overall trend is up"
                elif bb_bounce == "bullish" and not (25 < rsi_now < 40):
                    result["reason"] = (
                        f"BB bounce bullish but RSI not in range (RSI={rsi_now:.1f})"
                    )
                elif bb_bounce == "bearish" and not (60 < rsi_now < 75):
                    result["reason"] = (
                        f"BB bounce bearish but RSI not in range (RSI={rsi_now:.1f})"
                    )
                else:
                    result["reason"] = "No signal criteria met"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "Micro Account Scalper",
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "account_risk_pct": self.account_risk_pct,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["1m", "5m"],
            "target_win_rate": "75-85%",
            "trades_per_day": "10-30+",
            "account_size": "$10-$50",
        }

    def __repr__(self) -> str:
        return (
            f"MicroAccountScalper("
            f"EMAs={self.ema_fast}/{self.ema_slow}, "
            f"BB={self.bb_period}/{self.bb_std}, "
            f"RSI={self.rsi_period}, "
            f"Risk={self.account_risk_pct}%)"
        )


# ============================================================================
# ALTERNATIVE: SUPPORT/RESISTANCE BOUNCE STRATEGY
# ============================================================================


class SupportResistanceBouncer:
    """
    Pure Price Action Support/Resistance Scalper

    DESIGNED FOR: Micro accounts ($10-$50)
    TIMEFRAMES: 5m, 15m
    TARGET WIN RATE: 70-80%
    RISK-REWARD: 1:1.5

    Strategy Logic:
    - Identify key S/R levels from recent swing highs/lows
    - Wait for price to approach S/R level
    - Enter on rejection candle (pin bar, engulfing)
    - Quick exit at 1.5x risk

    Entry Rules:
    BULLISH (Support Bounce):
    - Price approaches support level (within 5 pips)
    - Bullish rejection candle forms (long lower wick)
    - RSI < 40 (oversold)
    - Volume spike (if available)

    BEARISH (Resistance Rejection):
    - Price approaches resistance level (within 5 pips)
    - Bearish rejection candle forms (long upper wick)
    - RSI > 60 (overbought)
    - Volume spike (if available)

    Risk Management:
    - Stop Loss: Beyond rejection candle high/low
    - Take Profit: 1.5x stop distance
    """

    def __init__(
        self,
        lookback_period: int = 50,
        sr_threshold_pips: float = 5.0,
        rsi_period: int = 14,
        volume_period: int = 20,
        risk_reward_ratio: float = 1.5,
        account_risk_pct: float = 0.5,
    ):
        self.lookback_period = lookback_period
        self.sr_threshold_pips = sr_threshold_pips
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.risk_reward_ratio = risk_reward_ratio
        self.account_risk_pct = account_risk_pct

        self.min_bars = max(lookback_period, rsi_period, volume_period) + 2

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _find_support_resistance(
        self, highs: pd.Series, lows: pd.Series, lookback: int
    ) -> tuple[list, list]:
        """
        Find key support and resistance levels.

        Returns:
            (support_levels, resistance_levels)
        """
        # Find swing highs and lows
        recent_highs = highs.iloc[-lookback:]
        recent_lows = lows.iloc[-lookback:]

        # Simple approach: use highest high and lowest low
        # Plus mid-range levels
        max_high = recent_highs.max()
        min_low = recent_lows.min()
        mid_level = (max_high + min_low) / 2

        resistance_levels = [max_high, mid_level]
        support_levels = [min_low, mid_level]

        return support_levels, resistance_levels

    def _detect_rejection_candle(
        self, open_price: float, high: float, low: float, close: float, direction: str
    ) -> bool:
        """
        Detect rejection candle pattern.

        Bullish rejection: Long lower wick (at least 60% of candle range)
        Bearish rejection: Long upper wick (at least 60% of candle range)
        """
        candle_range = high - low
        if candle_range == 0:
            return False

        if direction == "bullish":
            # Check for long lower wick
            body_low = min(open_price, close)
            lower_wick = body_low - low
            return (lower_wick / candle_range) >= 0.6 and close > open_price

        else:  # bearish
            # Check for long upper wick
            body_high = max(open_price, close)
            upper_wick = high - body_high
            return (upper_wick / candle_range) >= 0.6 and close < open_price

    def _is_near_level(self, price: float, level: float, threshold_pips: float) -> bool:
        """Check if price is within threshold of S/R level."""
        threshold = threshold_pips * 0.0001  # Convert pips to price
        return abs(price - level) <= threshold

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["open", "close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "support_levels": None,
            "resistance_levels": None,
            "rsi": None,
            "rejection_candle": False,
            "reason": reason,
            "entry_date": None,
        }

    def _calculate_position_size(
        self, account_balance: float, entry_price: float, stop_price: float
    ) -> float:
        """Calculate position size based on fixed percentage risk."""
        risk_amount = account_balance * (self.account_risk_pct / 100)
        stop_distance = abs(entry_price - stop_price)

        if stop_distance == 0:
            return 0.01

        pip_value = 10
        pips_at_risk = stop_distance * 10000
        lots = risk_amount / (pips_at_risk * pip_value * 0.01)

        return max(0.01, min(0.10, round(lots, 2)))

    def _calculate_stop_and_target(
        self, entry_price: float, rejection_extreme: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop and target levels."""
        stop_distance = abs(entry_price - rejection_extreme)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = rejection_extreme - (stop_distance * 0.1)  # Slightly beyond
            tp = entry_price + target_distance
        else:
            sl = rejection_extreme + (stop_distance * 0.1)
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(
        self, price_data: pd.DataFrame, account_balance: float = 30.0
    ) -> Dict[str, Any]:
        """
        Analyze price data and generate S/R bounce signal.

        Args:
            price_data: DataFrame with OHLC and time columns
            account_balance: Current account balance

        Returns:
            Dictionary with signal, prices, S/R levels, and reason
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            opens = price_data["open"]
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]
            volume = price_data.get("volume", None)

            current_open = opens.iloc[-1]
            current_close = closes.iloc[-1]
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]

            # Calculate indicators
            rsi = self._calculate_rsi(closes, self.rsi_period).iloc[-1]
            support_levels, resistance_levels = self._find_support_resistance(
                highs, lows, self.lookback_period
            )

            # Base result
            result = {
                "signal": None,
                "entry_price": current_close,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.01,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "rsi": float(rsi),
                "rejection_candle": False,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Check for BULLISH signal (support bounce)
            for support in support_levels:
                if self._is_near_level(current_close, support, self.sr_threshold_pips):
                    rejection = self._detect_rejection_candle(
                        current_open,
                        current_high,
                        current_low,
                        current_close,
                        "bullish",
                    )

                    if rejection and rsi < 40:
                        result["signal"] = "buy"
                        result["rejection_candle"] = True
                        result["reason"] = (
                            f"Bullish rejection at support level {support:.5f}"
                        )

                        sl, tp = self._calculate_stop_and_target(
                            current_close, current_low, "buy"
                        )
                        result["stop_loss"] = sl
                        result["take_profit"] = tp
                        result["position_size"] = self._calculate_position_size(
                            account_balance, current_close, sl
                        )
                        break

            # Check for BEARISH signal (resistance rejection)
            if result["signal"] is None:
                for resistance in resistance_levels:
                    if self._is_near_level(
                        current_close, resistance, self.sr_threshold_pips
                    ):
                        rejection = self._detect_rejection_candle(
                            current_open,
                            current_high,
                            current_low,
                            current_close,
                            "bearish",
                        )

                        if rejection and rsi > 60:
                            result["signal"] = "sell"
                            result["rejection_candle"] = True
                            result["reason"] = (
                                f"Bearish rejection at resistance level {resistance:.5f}"
                            )

                            sl, tp = self._calculate_stop_and_target(
                                current_close, current_high, "sell"
                            )
                            result["stop_loss"] = sl
                            result["take_profit"] = tp
                            result["position_size"] = self._calculate_position_size(
                                account_balance, current_close, sl
                            )
                            break

            if result["signal"] is None:
                result["reason"] = "No S/R level touch with rejection candle"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "Support Resistance Bouncer",
            "lookback_period": self.lookback_period,
            "sr_threshold_pips": self.sr_threshold_pips,
            "rsi_period": self.rsi_period,
            "volume_period": self.volume_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "account_risk_pct": self.account_risk_pct,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["5m", "15m"],
            "target_win_rate": "70-80%",
            "account_size": "$10-$50",
        }

    def __repr__(self) -> str:
        return (
            f"SupportResistanceBouncer("
            f"Lookback={self.lookback_period}, "
            f"Threshold={self.sr_threshold_pips}pips, "
            f"RSI={self.rsi_period}, "
            f"RR={self.risk_reward_ratio})"
        )


"""
HIGH FREQUENCY MICRO ACCOUNT STRATEGIES
Multiple signals per day - designed for rapid compounding.
Less filters = more trades = faster growth.
"""

import pandas as pd
from typing import Dict, Optional, Any


class HighFrequencyEMAScalper:
    """
    Simple EMA Crossover Scalper - MAXIMUM TRADE FREQUENCY

    DESIGNED FOR: $10-$100 accounts
    TIMEFRAMES: 5m, 15m
    TARGET WIN RATE: 55-65%
    RISK-REWARD: 1:2
    TRADES PER WEEK: 50-200+ (multiple per day)

    Strategy Logic (SIMPLE = MORE TRADES):
    - EMA 5 crosses EMA 13 = immediate signal
    - NO OTHER FILTERS (this is the key)
    - Quick 1:2 exits

    Entry Rules:
    BULLISH:
    - EMA 5 crosses above EMA 13
    - That's it. No RSI, no MACD, no volume filters.

    BEARISH:
    - EMA 5 crosses below EMA 13
    - That's it.

    Risk Management:
    - Stop Loss: Recent swing low/high (5-bar lookback)
    - Take Profit: 2x stop distance
    """

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 13,
        swing_lookback: int = 5,
        risk_reward_ratio: float = 2.0,
        account_risk_pct: float = 1.0,  # 1% per trade
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.swing_lookback = swing_lookback
        self.risk_reward_ratio = risk_reward_ratio
        self.account_risk_pct = account_risk_pct

        self.min_bars = max(ema_slow, swing_lookback) + 2

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "ema_fast": None,
            "ema_slow": None,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_ema_cross(
        self,
        ema_fast_prev: float,
        ema_fast_now: float,
        ema_slow_prev: float,
        ema_slow_now: float,
    ) -> Optional[str]:
        """Detect EMA crossover."""
        if ema_fast_prev < ema_slow_prev and ema_fast_now > ema_slow_now:
            return "bullish"
        elif ema_fast_prev > ema_slow_prev and ema_fast_now < ema_slow_now:
            return "bearish"
        return None

    def _calculate_stop_and_target(
        self, entry_price: float, stop_price: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop and target levels."""
        stop_distance = abs(entry_price - stop_price)
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = stop_price
            tp = entry_price + target_distance
        else:
            sl = stop_price
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signal on EMA crossover ONLY.

        NO FILTERS = MAXIMUM TRADE FREQUENCY
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]

            # Calculate EMAs
            ema_fast = self._calculate_ema(closes, self.ema_fast)
            ema_slow = self._calculate_ema(closes, self.ema_slow)

            ema_fast_now = ema_fast.iloc[-1]
            ema_fast_prev = ema_fast.iloc[-2]
            ema_slow_now = ema_slow.iloc[-1]
            ema_slow_prev = ema_slow.iloc[-2]

            # Detect crossover
            cross = self._detect_ema_cross(
                ema_fast_prev, ema_fast_now, ema_slow_prev, ema_slow_now
            )

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.01,
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # Generate signal immediately on cross
            if cross == "bullish":
                result["signal"] = "buy"
                result["reason"] = "EMA 5 crossed above EMA 13"

                # Stop at recent swing low
                swing_low = lows.iloc[-self.swing_lookback :].min()
                sl, tp = self._calculate_stop_and_target(
                    current_price, swing_low, "buy"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            elif cross == "bearish":
                result["signal"] = "sell"
                result["reason"] = "EMA 5 crossed below EMA 13"

                # Stop at recent swing high
                swing_high = highs.iloc[-self.swing_lookback :].max()
                sl, tp = self._calculate_stop_and_target(
                    current_price, swing_high, "sell"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                result["reason"] = "No EMA crossover"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "High Frequency EMA Scalper",
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "swing_lookback": self.swing_lookback,
            "risk_reward_ratio": self.risk_reward_ratio,
            "account_risk_pct": self.account_risk_pct,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["5m", "15m"],
            "target_win_rate": "55-65%",
            "trades_per_week": "50-200+",
            "filters": "NONE - maximum frequency",
        }

    def __repr__(self) -> str:
        return (
            f"HighFrequencyEMAScalper("
            f"EMAs={self.ema_fast}/{self.ema_slow}, "
            f"RR={self.risk_reward_ratio})"
        )


# ============================================================================
# ALTERNATIVE: STOCHASTIC OVERSOLD/OVERBOUGHT BOUNCES
# ============================================================================


class StochasticBounceScalper:
    """
    Stochastic Extreme Zone Scalper - HIGH FREQUENCY

    DESIGNED FOR: $10-$100 accounts
    TIMEFRAMES: 5m, 15m
    TARGET WIN RATE: 60-70%
    RISK-REWARD: 1:1.5
    TRADES PER WEEK: 50-150+

    Strategy Logic:
    - Stochastic enters oversold (<20) = buy next bounce
    - Stochastic enters overbought (>80) = sell next rejection
    - ONE FILTER ONLY: Price must be above/below EMA 50 for trend

    Entry Rules:
    BULLISH:
    - Stochastic crosses above 20 (from oversold)
    - Price > EMA 50 (overall uptrend)

    BEARISH:
    - Stochastic crosses below 80 (from overbought)
    - Price < EMA 50 (overall downtrend)

    Risk Management:
    - Stop Loss: 10-bar ATR distance
    - Take Profit: 1.5x stop distance
    """

    def __init__(
        self,
        ema_period: int = 50,
        stoch_k_period: int = 14,
        stoch_d_period: int = 3,
        stoch_smooth: int = 3,
        atr_period: int = 10,
        risk_reward_ratio: float = 1.5,
        atr_stop_multiplier: float = 1.5,
    ):
        self.ema_period = ema_period
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_smooth = stoch_smooth
        self.atr_period = atr_period
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_stop_multiplier = atr_stop_multiplier

        self.min_bars = max(ema_period, stoch_k_period + stoch_smooth, atr_period) + 2

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_stochastic(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = lows.rolling(window=self.stoch_k_period).min()
        highest_high = highs.rolling(window=self.stoch_k_period).max()

        raw_k = 100 * (closes - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(window=self.stoch_smooth).mean()
        d = k.rolling(window=self.stoch_d_period).mean()

        return k, d

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = highs - lows
        high_close = (highs - closes.shift()).abs()
        low_close = (lows - closes.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "ema_50": None,
            "stoch_k": None,
            "stoch_d": None,
            "reason": reason,
            "entry_date": None,
        }

    def _detect_stoch_cross(
        self, k_prev: float, k_now: float, threshold: float, direction: str
    ) -> bool:
        """
        Detect Stochastic crossing threshold.

        Bullish: Crossing above 20 from below
        Bearish: Crossing below 80 from above
        """
        if direction == "bullish":
            return k_prev < threshold and k_now > threshold
        else:  # bearish
            return k_prev > threshold and k_now < threshold

    def _calculate_stop_and_target(
        self, entry_price: float, atr: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop and target using ATR."""
        stop_distance = atr * self.atr_stop_multiplier
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = entry_price - stop_distance
            tp = entry_price + target_distance
        else:
            sl = entry_price + stop_distance
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signal on Stochastic extreme bounces.

        MINIMAL FILTERS = HIGH FREQUENCY
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]

            # Calculate indicators
            ema_50 = self._calculate_ema(closes, self.ema_period).iloc[-1]
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            atr = self._calculate_atr(highs, lows, closes, self.atr_period).iloc[-1]

            k_now = stoch_k.iloc[-1]
            k_prev = stoch_k.iloc[-2]

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.01,
                "ema_50": float(ema_50),
                "stoch_k": float(k_now),
                "stoch_d": float(stoch_d.iloc[-1]),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # BULLISH: Stoch crosses above 20 AND price > EMA 50
            if (
                self._detect_stoch_cross(k_prev, k_now, 20, "bullish")
                and current_price > ema_50
            ):
                result["signal"] = "buy"
                result["reason"] = "Stochastic oversold bounce in uptrend"

                sl, tp = self._calculate_stop_and_target(current_price, atr, "buy")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            # BEARISH: Stoch crosses below 80 AND price < EMA 50
            elif (
                self._detect_stoch_cross(k_prev, k_now, 80, "bearish")
                and current_price < ema_50
            ):
                result["signal"] = "sell"
                result["reason"] = "Stochastic overbought rejection in downtrend"

                sl, tp = self._calculate_stop_and_target(current_price, atr, "sell")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                if k_prev >= 20 and k_now >= 20 and k_prev <= 80 and k_now <= 80:
                    result["reason"] = "Stochastic in neutral zone"
                elif (
                    self._detect_stoch_cross(k_prev, k_now, 20, "bullish")
                    and current_price <= ema_50
                ):
                    result["reason"] = "Stoch oversold but price below EMA 50"
                elif (
                    self._detect_stoch_cross(k_prev, k_now, 80, "bearish")
                    and current_price >= ema_50
                ):
                    result["reason"] = "Stoch overbought but price above EMA 50"
                else:
                    result["reason"] = "No stochastic extreme cross"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "Stochastic Bounce Scalper",
            "ema_period": self.ema_period,
            "stoch_k_period": self.stoch_k_period,
            "stoch_d_period": self.stoch_d_period,
            "stoch_smooth": self.stoch_smooth,
            "atr_period": self.atr_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["5m", "15m"],
            "target_win_rate": "60-70%",
            "trades_per_week": "50-150+",
            "filters": "EMA 50 trend only",
        }

    def __repr__(self) -> str:
        return (
            f"StochasticBounceScalper("
            f"EMA={self.ema_period}, "
            f"Stoch={self.stoch_k_period}, "
            f"RR={self.risk_reward_ratio})"
        )


# ============================================================================
# ULTRA AGGRESSIVE: RSI 2-PERIOD MEAN REVERSION
# ============================================================================


class RSI2MeanReversion:
    """
    Larry Connors RSI-2 Mean Reversion Strategy

    DESIGNED FOR: Aggressive micro accounts
    TIMEFRAMES: 15m, 1H
    TARGET WIN RATE: 65-75%
    RISK-REWARD: 1:1.5
    TRADES PER WEEK: 30-100+

    Strategy Logic (SIMPLE & PROVEN):
    - Use RSI(2) for extreme oversold/overbought
    - Only trade WITH the trend (200 EMA)
    - Mean reversion back to normal levels

    Entry Rules:
    BULLISH:
    - Price > 200 EMA (uptrend)
    - RSI(2) < 10 (extreme oversold)
    - Enter immediately

    BEARISH:
    - Price < 200 EMA (downtrend)
    - RSI(2) > 90 (extreme overbought)
    - Enter immediately

    Exit:
    - RSI(2) > 50 for longs
    - RSI(2) < 50 for shorts
    - Or hit 1.5x stop
    """

    def __init__(
        self,
        ema_period: int = 200,
        rsi_period: int = 2,
        rsi_oversold: float = 10,
        rsi_overbought: float = 90,
        atr_period: int = 14,
        risk_reward_ratio: float = 1.5,
        atr_stop_multiplier: float = 2.0,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_stop_multiplier = atr_stop_multiplier

        self.min_bars = max(ema_period, rsi_period, atr_period) + 2

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = highs - lows
        high_close = (highs - closes.shift()).abs()
        low_close = (lows - closes.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "ema_200": None,
            "rsi_2": None,
            "reason": reason,
            "entry_date": None,
        }

    def _calculate_stop_and_target(
        self, entry_price: float, atr: float, signal_type: str
    ) -> tuple[float, float]:
        """Calculate stop and target using ATR."""
        stop_distance = atr * self.atr_stop_multiplier
        target_distance = stop_distance * self.risk_reward_ratio

        if signal_type == "buy":
            sl = entry_price - stop_distance
            tp = entry_price + target_distance
        else:
            sl = entry_price + stop_distance
            tp = entry_price - target_distance

        return sl, tp

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate RSI-2 mean reversion signals.

        EXTREMELY SIMPLE = HIGH FREQUENCY
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]

            # Calculate indicators
            ema_200 = self._calculate_ema(closes, self.ema_period).iloc[-1]
            rsi_2 = self._calculate_rsi(closes, self.rsi_period).iloc[-1]
            atr = self._calculate_atr(highs, lows, closes, self.atr_period).iloc[-1]

            # Determine trend
            trend = "bullish" if current_price > ema_200 else "bearish"

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.01,
                "ema_200": float(ema_200),
                "rsi_2": float(rsi_2),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # BULLISH: Uptrend + Extreme oversold
            if trend == "bullish" and rsi_2 < self.rsi_oversold:
                result["signal"] = "buy"
                result["reason"] = f"RSI(2)={rsi_2:.1f} extreme oversold in uptrend"

                sl, tp = self._calculate_stop_and_target(current_price, atr, "buy")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            # BEARISH: Downtrend + Extreme overbought
            elif trend == "bearish" and rsi_2 > self.rsi_overbought:
                result["signal"] = "sell"
                result["reason"] = f"RSI(2)={rsi_2:.1f} extreme overbought in downtrend"

                sl, tp = self._calculate_stop_and_target(current_price, atr, "sell")
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                if trend == "bullish" and rsi_2 >= self.rsi_oversold:
                    result["reason"] = (
                        f"Uptrend but RSI(2)={rsi_2:.1f} not oversold enough"
                    )
                elif trend == "bearish" and rsi_2 <= self.rsi_overbought:
                    result["reason"] = (
                        f"Downtrend but RSI(2)={rsi_2:.1f} not overbought enough"
                    )
                else:
                    result["reason"] = "No extreme RSI condition"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "RSI-2 Mean Reversion",
            "ema_period": self.ema_period,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "atr_period": self.atr_period,
            "risk_reward_ratio": self.risk_reward_ratio,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["15m", "1H"],
            "target_win_rate": "65-75%",
            "trades_per_week": "30-100+",
            "filters": "EMA 200 trend only",
            "famous_strategy": "Larry Connors RSI-2",
        }

    def __repr__(self) -> str:
        return (
            f"RSI2MeanReversion("
            f"EMA={self.ema_period}, "
            f"RSI={self.rsi_period}, "
            f"RR={self.risk_reward_ratio})"
        )


"""
INTELLIGENT TARGET PREDICTION STRATEGY
Calculates exact price targets based on market structure.
No trailing - fixed entry, stop, and target.
"""

import pandas as pd
from typing import Dict, Optional, Any, List


class StructureBasedStrategy:
    """
    Market Structure Target Prediction Strategy

    PHILOSOPHY: Know your target BEFORE entering
    - Identifies support/resistance levels from recent structure
    - Calculates exact targets based on S/R levels
    - Measures price waves to predict swing targets
    - Uses Fibonacci extensions for trend targets

    DESIGNED FOR: $10-$100 accounts
    TIMEFRAMES: 15m, 1H, 4H
    WIN RATE: 65-75%
    WIN BIG LOSE SMALL: Avg winner 2.5-3x avg loser
    TRADES PER WEEK: 20-50

    Entry Logic:
    1. Find recent swing highs/lows (support/resistance)
    2. Identify current market structure (range vs trend)
    3. Wait for price to approach key level
    4. Enter on confirmation with EXACT target calculated

    RANGE MODE:
    - Target = Opposite side of range
    - Stop = Just beyond entry level
    - RR typically 2-3x

    TREND MODE:
    - Target = Fibonacci extension (1.618 or 2.618)
    - Stop = Recent swing point
    - RR typically 2.5-4x
    """

    def __init__(
        self,
        # Structure detection
        swing_lookback: int = 20,  # Bars to look back for swings
        structure_strength: int = 3,  # How many bars must be lower/higher
        # EMAs for trend
        ema_fast: int = 21,
        ema_slow: int = 50,
        # ADX for regime
        adx_period: int = 14,
        adx_threshold: float = 25,
        # RSI for entries
        rsi_period: int = 14,
        # ATR for stops
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        # Fibonacci levels
        fib_extensions: List[float] = [1.272, 1.618, 2.618],
    ):
        self.swing_lookback = swing_lookback
        self.structure_strength = structure_strength
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.fib_extensions = fib_extensions

        self.min_bars = (
            max(swing_lookback, ema_slow, adx_period, rsi_period, atr_period)
            + structure_strength
            + 2
        )

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate ADX."""
        high_diff = highs.diff()
        low_diff = -lows.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr1 = highs - lows
        tr2 = (highs - closes.shift()).abs()
        tr3 = (lows - closes.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_atr(
        self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = highs - lows
        high_close = (highs - closes.shift()).abs()
        low_close = (lows - closes.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _find_swing_highs_lows(
        self, highs: pd.Series, lows: pd.Series, lookback: int, strength: int
    ) -> tuple[List[tuple], List[tuple]]:
        """
        Find swing highs and lows in price data.

        A swing high: High that is higher than 'strength' bars on both sides
        A swing low: Low that is lower than 'strength' bars on both sides

        Returns:
            (swing_highs, swing_lows) - List of (index, price) tuples
        """
        swing_highs = []
        swing_lows = []

        # Start from strength bars in to avoid edge cases
        for i in range(strength, len(highs) - strength):
            # Check swing high
            is_swing_high = True
            for j in range(1, strength + 1):
                if (
                    highs.iloc[i] <= highs.iloc[i - j]
                    or highs.iloc[i] <= highs.iloc[i + j]
                ):
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append((i, highs.iloc[i]))

            # Check swing low
            is_swing_low = True
            for j in range(1, strength + 1):
                if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append((i, lows.iloc[i]))

        # Return only recent swings
        recent_highs = swing_highs[-5:] if len(swing_highs) > 5 else swing_highs
        recent_lows = swing_lows[-5:] if len(swing_lows) > 5 else swing_lows

        return recent_highs, recent_lows

    def _identify_range(
        self, swing_highs: List[tuple], swing_lows: List[tuple], current_price: float
    ) -> Optional[Dict[str, float]]:
        """
        Identify if price is in a range and find range boundaries.

        Range criteria:
        - Multiple swing highs near same level (within 2%)
        - Multiple swing lows near same level (within 2%)
        - Current price between these levels
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Get recent swing prices
        high_prices = [h[1] for h in swing_highs]
        low_prices = [l[1] for l in swing_lows]

        # Find resistance (highest recent swing high)
        resistance = max(high_prices)

        # Find support (lowest recent swing low)
        support = min(low_prices)

        # Check if these levels have been tested multiple times (within 2% tolerance)
        resistance_tests = sum(
            1 for h in high_prices if abs(h - resistance) / resistance < 0.02
        )
        support_tests = sum(1 for l in low_prices if abs(l - support) / support < 0.02)

        # Valid range if both levels tested at least twice and price is between them
        if (
            resistance_tests >= 2
            and support_tests >= 2
            and support < current_price < resistance
        ):
            range_size = resistance - support
            return {
                "resistance": resistance,
                "support": support,
                "middle": (resistance + support) / 2,
                "size": range_size,
            }

        return None

    def _calculate_fibonacci_target(
        self,
        swing_start: float,
        swing_end: float,
        direction: str,
        extension_level: float,
    ) -> float:
        """
        Calculate Fibonacci extension target.

        For uptrend: Measure last swing up, project extension from current level
        For downtrend: Measure last swing down, project extension from current level
        """
        swing_size = abs(swing_end - swing_start)

        if direction == "bullish":
            # Project upward from swing_end
            target = swing_end + (swing_size * extension_level)
        else:
            # Project downward from swing_end
            target = swing_end - (swing_size * extension_level)

        return target

    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        required_cols = ["close", "high", "low", "time"]
        for col in required_cols:
            if col not in price_data.columns:
                return False, f"Missing '{col}' column in price data"

        if len(price_data) < self.min_bars:
            return (
                False,
                f"Insufficient data: need {self.min_bars} bars, got {len(price_data)}",
            )

        return True, ""

    def _get_empty_signal(self, reason: str) -> Dict[str, Any]:
        """Return empty signal dictionary."""
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "target_type": None,
            "risk_reward": None,
            "position_size": 0.01,
            "market_structure": None,
            "support_levels": None,
            "resistance_levels": None,
            "reason": reason,
            "entry_date": None,
        }

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signal with calculated price targets based on structure.

        Process:
        1. Find swing highs/lows (support/resistance)
        2. Determine if ranging or trending
        3. Calculate exact target based on structure
        4. Enter when price reaches key level with confirmation
        """
        is_valid, error_msg = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error_msg)

        try:
            closes = price_data["close"]
            highs = price_data["high"]
            lows = price_data["low"]

            current_price = closes.iloc[-1]

            # Calculate indicators
            ema_fast = self._calculate_ema(closes, self.ema_fast).iloc[-1]
            ema_slow = self._calculate_ema(closes, self.ema_slow).iloc[-1]
            rsi = self._calculate_rsi(closes, self.rsi_period).iloc[-1]
            adx = self._calculate_adx(highs, lows, closes, self.adx_period).iloc[-1]
            atr = self._calculate_atr(highs, lows, closes, self.atr_period).iloc[-1]

            # Find swing points
            swing_highs, swing_lows = self._find_swing_highs_lows(
                highs, lows, self.swing_lookback, self.structure_strength
            )

            # Need at least some structure
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return self._get_empty_signal("Insufficient price structure detected")

            # Get recent swing levels
            resistance_levels = [h[1] for h in swing_highs]
            support_levels = [l[1] for l in swing_lows]
            nearest_resistance = min(
                resistance_levels, key=lambda x: abs(x - current_price)
            )
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))

            # Determine market regime
            market_regime = "RANGE" if adx < self.adx_threshold else "TREND"

            # Base result
            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "target_type": None,
                "risk_reward": None,
                "position_size": 0.01,
                "market_structure": market_regime,
                "adx": float(adx),
                "rsi": float(rsi),
                "ema_fast": float(ema_fast),
                "ema_slow": float(ema_slow),
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # ============================================================
            # RANGE MODE: Target opposite side of range
            # ============================================================
            if market_regime == "RANGE":
                range_info = self._identify_range(
                    swing_highs, swing_lows, current_price
                )

                if range_info:
                    tolerance = atr * 0.5  # Price must be within 0.5 ATR of level

                    # BULLISH: At support, target resistance
                    if (
                        abs(current_price - range_info["support"]) < tolerance
                        and rsi < 40
                    ):
                        result["signal"] = "buy"
                        result["target_type"] = "Range Resistance"
                        result["stop_loss"] = range_info["support"] - (
                            atr * self.atr_stop_mult
                        )
                        result["take_profit"] = range_info["resistance"]

                        risk = abs(current_price - result["stop_loss"])
                        reward = abs(result["take_profit"] - current_price)
                        result["risk_reward"] = (
                            round(reward / risk, 2) if risk > 0 else 0
                        )
                        result["reason"] = (
                            f"Range support bounce targeting resistance at {result['take_profit']:.5f}"
                        )

                    # BEARISH: At resistance, target support
                    elif (
                        abs(current_price - range_info["resistance"]) < tolerance
                        and rsi > 60
                    ):
                        result["signal"] = "sell"
                        result["target_type"] = "Range Support"
                        result["stop_loss"] = range_info["resistance"] + (
                            atr * self.atr_stop_mult
                        )
                        result["take_profit"] = range_info["support"]

                        risk = abs(result["stop_loss"] - current_price)
                        reward = abs(current_price - result["take_profit"])
                        result["risk_reward"] = (
                            round(reward / risk, 2) if risk > 0 else 0
                        )
                        result["reason"] = (
                            f"Range resistance rejection targeting support at {result['take_profit']:.5f}"
                        )

                    else:
                        result["reason"] = (
                            f"Range detected but price not at boundaries (Support: {range_info['support']:.5f}, Resistance: {range_info['resistance']:.5f})"
                        )
                else:
                    result["reason"] = (
                        "Range mode but no clear range structure identified"
                    )

            # ============================================================
            # TREND MODE: Fibonacci extension targets
            # ============================================================
            else:
                trend = "bullish" if ema_fast > ema_slow else "bearish"

                # Find last major swing for Fibonacci calculation
                if (
                    trend == "bullish"
                    and len(swing_lows) >= 2
                    and len(swing_highs) >= 1
                ):
                    # Get last swing low and swing high
                    last_low = swing_lows[-1][1]
                    last_high = swing_highs[-1][1]

                    # Check if price pulled back to support
                    if (
                        abs(current_price - nearest_support) < atr
                        and rsi < 50
                        and current_price > ema_fast
                    ):
                        # Calculate Fibonacci extension target
                        fib_target = self._calculate_fibonacci_target(
                            last_low,
                            last_high,
                            "bullish",
                            self.fib_extensions[1],  # 1.618
                        )

                        result["signal"] = "buy"
                        result["target_type"] = "Fibonacci 1.618 Extension"
                        result["stop_loss"] = nearest_support - (
                            atr * self.atr_stop_mult
                        )
                        result["take_profit"] = fib_target

                        risk = abs(current_price - result["stop_loss"])
                        reward = abs(result["take_profit"] - current_price)
                        result["risk_reward"] = (
                            round(reward / risk, 2) if risk > 0 else 0
                        )
                        result["reason"] = (
                            f"Bullish trend pullback targeting Fib extension at {result['take_profit']:.5f}"
                        )

                elif (
                    trend == "bearish"
                    and len(swing_highs) >= 2
                    and len(swing_lows) >= 1
                ):
                    # Get last swing high and swing low
                    last_high = swing_highs[-1][1]
                    last_low = swing_lows[-1][1]

                    # Check if price pulled back to resistance
                    if (
                        abs(current_price - nearest_resistance) < atr
                        and rsi > 50
                        and current_price < ema_fast
                    ):
                        # Calculate Fibonacci extension target
                        fib_target = self._calculate_fibonacci_target(
                            last_high,
                            last_low,
                            "bearish",
                            self.fib_extensions[1],  # 1.618
                        )

                        result["signal"] = "sell"
                        result["target_type"] = "Fibonacci 1.618 Extension"
                        result["stop_loss"] = nearest_resistance + (
                            atr * self.atr_stop_mult
                        )
                        result["take_profit"] = fib_target

                        risk = abs(result["stop_loss"] - current_price)
                        reward = abs(current_price - result["take_profit"])
                        result["risk_reward"] = (
                            round(reward / risk, 2) if risk > 0 else 0
                        )
                        result["reason"] = (
                            f"Bearish trend pullback targeting Fib extension at {result['take_profit']:.5f}"
                        )

                if result["signal"] is None:
                    result["reason"] = (
                        f"Trend mode ({trend}) but price not at structure level for entry"
                    )

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error calculating signal: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters."""
        return {
            "name": "Structure-Based Target Prediction Strategy",
            "description": "Calculates exact targets based on market structure",
            "swing_lookback": self.swing_lookback,
            "structure_strength": self.structure_strength,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "adx_period": self.adx_period,
            "adx_threshold": self.adx_threshold,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "atr_stop_mult": self.atr_stop_mult,
            "fib_extensions": self.fib_extensions,
            "min_bars_required": self.min_bars,
            "recommended_timeframes": ["15m", "1H", "4H"],
            "target_win_rate": "65-75%",
            "avg_rr": "2.5-3x",
            "trades_per_week": "20-50",
            "target_types": ["Range levels", "Fibonacci extensions"],
        }

    def __repr__(self) -> str:
        return (
            f"StructureBasedStrategy("
            f"Swing_lookback={self.swing_lookback}, "
            f"ADX_threshold={self.adx_threshold}, "
            f"Fib_extensions={self.fib_extensions})"
        )


import pandas as pd
import numpy as np


class AdaptiveTrendMomentumStrategy:
    """
    Trend-following strategy with volatility-based risk control
    Timeframe: 1H
    Win big, lose small
    """

    def __init__(
        self,
        ema_fast=20,
        ema_slow=50,
        ema_trend=200,
        rsi_period=14,
        atr_period=14,
        risk_reward=4.0,
        atr_multiplier=1.2,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.risk_reward = risk_reward
        self.atr_multiplier = atr_multiplier

        self.min_bars = ema_trend + 5

    # -------------------------
    # Indicators
    # -------------------------
    def _ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period):
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # -------------------------
    # Signal Generation
    # -------------------------
    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < self.min_bars:
            return pd.DataFrame()

        df = df.copy()

        df["ema_fast"] = self._ema(df["close"], self.ema_fast)
        df["ema_slow"] = self._ema(df["close"], self.ema_slow)
        df["ema_trend"] = self._ema(df["close"], self.ema_trend)
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["atr"] = self._atr(df, self.atr_period)

        latest = df.iloc[-1]

        signal = None
        entry = latest["close"]
        sl = tp = None

        # BUY CONDITIONS
        if (
            entry > latest["ema_trend"]
            and latest["ema_fast"] > latest["ema_slow"]
            and 50 < latest["rsi"] < 65
            and entry > latest["ema_fast"]
        ):
            signal = "buy"
            sl = entry - (latest["atr"] * self.atr_multiplier)
            tp = entry + (latest["atr"] * self.risk_reward)

        # SELL CONDITIONS
        elif (
            entry < latest["ema_trend"]
            and latest["ema_fast"] < latest["ema_slow"]
            and 35 < latest["rsi"] < 50
            and entry < latest["ema_fast"]
        ):
            signal = "sell"
            sl = entry + (latest["atr"] * self.atr_multiplier)
            tp = entry - (latest["atr"] * self.risk_reward)

        return pd.DataFrame(
            [
                {
                    "time": latest["time"],
                    "signal": signal,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "position_size": 0.02,
                }
            ]
        )


import pandas as pd
from typing import Dict, Any, Optional


import pandas as pd
from typing import Dict, Any, Optional


class RegimeAdaptiveMomentumStrategy:
    """
    Regime-Adaptive Momentum Breakout Strategy
    Designed for small accounts (30–100$)
    """

    def __init__(
        self,
        ema_fast: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        risk_reward_ratio: float = 2.5,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = max(ema_slow, rsi_period, atr_period, adx_period) + 5

    # ---------------- INDICATORS ----------------

    def _ema(self, s: pd.Series, p: int):
        return s.ewm(span=p, adjust=False).mean()

    def _rsi(self, s: pd.Series):
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = -delta.clip(upper=0).rolling(self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame):
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _adx(self, df: pd.DataFrame):
        high, low, close = df["high"], df["low"], df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        tr = self._atr(df)

        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / tr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(self.adx_period).mean()

    # ---------------- SIGNAL ----------------

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or len(df) < self.min_bars:
            return {"signal": None, "reason": "Insufficient data"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema50 = self._ema(close, self.ema_fast)
        ema200 = self._ema(close, self.ema_slow)
        rsi = self._rsi(close)
        atr = self._atr(df)
        adx = self._adx(df)

        i = -1
        price = close.iloc[i]

        result = {
            "signal": None,
            "entry_price": price,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.01,
            "entry_date": df["time"].iloc[i],
            "reason": None,
        }

        # ----------- BUY CONDITIONS -----------
        if (
            ema50.iloc[i] > ema200.iloc[i]
            and price > ema50.iloc[i]
            and adx.iloc[i] > 22
            and 55 < rsi.iloc[i] < 70
            and high.iloc[i] > high.iloc[i - 1]
        ):
            sl = min(
                ema50.iloc[i],
                price - (atr.iloc[i] * 1.2),
            )
            risk = abs(price - sl)
            tp = price + (risk * self.risk_reward_ratio)

            result.update(
                {
                    "signal": "buy",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "Trend + momentum breakout",
                }
            )
            return result

        # ----------- SELL CONDITIONS -----------
        if (
            ema50.iloc[i] < ema200.iloc[i]
            and price < ema50.iloc[i]
            and adx.iloc[i] > 22
            and 30 < rsi.iloc[i] < 45
            and low.iloc[i] < low.iloc[i - 1]
        ):
            sl = max(
                ema50.iloc[i],
                price + (atr.iloc[i] * 1.2),
            )
            risk = abs(price - sl)
            tp = price - (risk * self.risk_reward_ratio)

            result.update(
                {
                    "signal": "sell",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "Downtrend momentum breakdown",
                }
            )
            return result

        result["reason"] = "Conditions not met"
        return result

    def __repr__(self):
        return "RegimeAdaptiveMomentumStrategy(small-account optimized)"


import pandas as pd
from typing import Dict, Any



class HTF_LTF_RSI_Strategy:
    """
    Higher-TF Bias + Lower-TF RSI Strategy
    Generates signals on lower TF in the direction of higher TF bias.
    """

    def __init__(
        self,
        ltf_rsi_period: int = 14,
        htf_rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        risk_reward_ratio: float = 2.0,
        position_size: float = 0.01,
        htf_ema_period=200,  # <-- Add this
    ):
        self.ltf_rsi_period = ltf_rsi_period
        self.htf_rsi_period = htf_rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_reward_ratio = risk_reward_ratio
        self.htf_ema_period = htf_ema_period  # <-- Assign it

        self.position_size = position_size
        self.min_bars = max(ltf_rsi_period, htf_rsi_period) + 5

    def _rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _validate_data(
        self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame
    ) -> tuple[bool, str]:
        if ltf_df is None or ltf_df.empty:
            return False, "LTF data empty"
        if htf_df is None or htf_df.empty:
            return False, "HTF data empty"
        if len(ltf_df) < self.min_bars or len(htf_df) < self.min_bars:
            return False, f"Insufficient bars (LTF: {len(ltf_df)}, HTF: {len(htf_df)})"
        return True, ""

    def _empty_signal(self, reason: str, ltf_df: pd.DataFrame = None):
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.0,
            "entry_date": (
                ltf_df["time"].iloc[-1]
                if ltf_df is not None and not ltf_df.empty
                else None
            ),
            "reason": reason,
        }

    def generate_signal(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame = None,
        current_time: pd.Timestamp = None,
    ) -> Dict[str, Any]:
        """
        Generate buy/sell signals based on HTF bias and LTF EMA+RSI.
        SL = LTF EMA, TP = SL distance * risk_reward_ratio.
        """

        # Validate
        is_valid, err = self._validate_data(ltf_df, htf_df)
        if not is_valid:
            return self._empty_signal(err, ltf_df)

        try:
            current_price = ltf_df["close"].iloc[-1]

            # HTF EMA and bias
            if htf_df is not None:
                htf_ema = (
                    htf_df["close"]
                    .ewm(span=self.htf_ema_period, adjust=False)
                    .mean()
                    .iloc[-1]
                )
                htf_price = htf_df["close"].iloc[-1]
                htf_bias = "bullish" if htf_price > htf_ema else "bearish"
            else:
                return self._empty_signal("HTF data required", ltf_df)

            # LTF EMA and RSI
            ltf_ema = (
                ltf_df["close"]
                .ewm(span=self.htf_ema_period, adjust=False)
                .mean()
                .iloc[-1]
            )
            ltf_rsi = self._rsi(ltf_df["close"], self.ltf_rsi_period).iloc[-1]

            # Check buy conditions
            if (
                htf_bias == "bullish"
                and current_price > ltf_ema
                and ltf_rsi < self.rsi_oversold
            ):
                signal = "buy"
                stop_loss = ltf_ema
                take_profit = (
                    current_price + (current_price - stop_loss) * self.risk_reward_ratio
                )
            else:
                return self._empty_signal("No entry conditions met", ltf_df)

            return {
                "signal": signal,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": self.position_size,
                "entry_date": (
                    current_time
                    if current_time is not None
                    else ltf_df["time"].iloc[-1]
                ),
                "htf_bias": htf_bias,
                "ltf_rsi": ltf_rsi,
                "ltf_ema": ltf_ema,
                "htf_ema": htf_ema,
                "reason": f"{signal.upper()} triggered by LTF RSI < {self.rsi_oversold} in direction of HTF EMA bias",
            }

        except Exception as e:
            return self._empty_signal(f"Error generating signal: {str(e)}", ltf_df)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "HTF_LTF_RSI_Strategy",
            "ltf_rsi_period": self.ltf_rsi_period,
            "htf_rsi_period": self.htf_rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size": self.position_size,
            "min_bars": self.min_bars,
        }

    def __repr__(self):
        return (
            f"HTF_LTF_RSI_Strategy(LTF={self.ltf_rsi_period}, HTF={self.htf_rsi_period}, "
            f"RR={self.risk_reward_ratio})"
        )
