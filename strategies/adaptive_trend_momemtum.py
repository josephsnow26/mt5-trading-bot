"""
Adaptive Trend Momentum Strategy
Designed for small accounts ($30–$100)
Slow, stable equity growth with strict filtering
"""

import pandas as pd
from typing import Dict, Optional, Any


class AdaptiveTrendMomentumStrategy:
    """
    Trend + Momentum + Structure Strategy

    Conditions:
    - Trend: EMA 200
    - Structure: EMA 50
    - Momentum: RSI
    - Timing: MACD crossover

    Designed to:
    - Avoid chop
    - Trade less
    - Protect small capital
    """

    def __init__(
        self,
        ema_trend: int = 200,
        ema_structure: int = 50,
        rsi_period: int = 14,
        rsi_buy_min: int = 50,
        rsi_sell_max: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        risk_reward_ratio: float = 2.0,
    ):
        self.ema_trend = ema_trend
        self.ema_structure = ema_structure
        self.rsi_period = rsi_period
        self.rsi_buy_min = rsi_buy_min
        self.rsi_sell_max = rsi_sell_max
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = max(
            ema_trend, ema_structure, macd_slow + macd_signal, rsi_period
        ) + 5

    # ---------------- INDICATORS ---------------- #

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd(self, series: pd.Series):
        fast = self._ema(series, self.macd_fast)
        slow = self._ema(series, self.macd_slow)
        macd = fast - slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd, signal

    # ---------------- VALIDATION ---------------- #

    def _validate_data(self, df: pd.DataFrame):
        if df is None or df.empty:
            return False, "Empty price data"

        required = {"close", "high", "low", "time"}
        if not required.issubset(df.columns):
            return False, f"Missing columns: {required - set(df.columns)}"

        if len(df) < self.min_bars:
            return False, f"Need {self.min_bars} bars"

        return True, ""

    def _empty_signal(self, reason: str):
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "trend": None,
            "position_size": 0.01,
            "reason": reason,
            "entry_date": None,
        }

    # ---------------- CORE LOGIC ---------------- #

    def _detect_macd_cross(self, m_prev, m_now, s_prev, s_now):
        if m_prev < s_prev and m_now > s_now:
            return "bullish"
        if m_prev > s_prev and m_now < s_now:
            return "bearish"
        return None

    def _structure_stop(self, df: pd.DataFrame, direction: str):
        lookback = 10
        if direction == "buy":
            return df["low"].iloc[-lookback:].min()
        else:
            return df["high"].iloc[-lookback:].max()

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        valid, msg = self._validate_data(price_data)
        if not valid:
            return self._empty_signal(msg)

        try:
            closes = price_data["close"]
            current_price = closes.iloc[-1]

            ema_200 = self._ema(closes, self.ema_trend).iloc[-1]
            ema_50 = self._ema(closes, self.ema_structure).iloc[-1]
            rsi = self._rsi(closes).iloc[-1]

            macd, macd_signal = self._macd(closes)
            macd_prev, macd_now = macd.iloc[-2], macd.iloc[-1]
            sig_prev, sig_now = macd_signal.iloc[-2], macd_signal.iloc[-1]

            trend = (
                "bullish"
                if current_price > ema_200 and ema_50 > ema_200
                else "bearish"
                if current_price < ema_200 and ema_50 < ema_200
                else "neutral"
            )

            if trend == "neutral":
                return self._empty_signal("Market ranging – no trade")

            macd_cross = self._detect_macd_cross(
                macd_prev, macd_now, sig_prev, sig_now
            )

            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "position_size": 0.01,
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # BUY CONDITIONS
            if (
                trend == "bullish"
                and macd_cross == "bullish"
                and rsi >= self.rsi_buy_min
            ):
                sl = self._structure_stop(price_data, "buy")
                risk = current_price - sl
                tp = current_price + (risk * self.risk_reward_ratio)

                result.update(
                    {
                        "signal": "buy",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "Bull trend + RSI momentum + MACD entry",
                    }
                )

            # SELL CONDITIONS
            elif (
                trend == "bearish"
                and macd_cross == "bearish"
                and rsi <= self.rsi_sell_max
            ):
                sl = self._structure_stop(price_data, "sell")
                risk = sl - current_price
                tp = current_price - (risk * self.risk_reward_ratio)

                result.update(
                    {
                        "signal": "sell",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "Bear trend + RSI momentum + MACD entry",
                    }
                )
            else:
                result["reason"] = "Conditions not aligned"

            return result

        except Exception as e:
            return self._empty_signal(str(e))

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "Adaptive Trend Momentum Strategy",
            "ema_trend": self.ema_trend,
            "ema_structure": self.ema_structure,
            "rsi_period": self.rsi_period,
            "risk_reward_ratio": self.risk_reward_ratio,
        }

    def __repr__(self):
        return "AdaptiveTrendMomentumStrategy(small-account optimized)"
