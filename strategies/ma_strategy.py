
import pandas as pd

# 1 HOUR TIMEFRAME START WITH 100 DOLLARS

class MATrendStrategy:
    """
    Moving Average Crossover Strategy

    Entry Rules:
    - BUY: Fast MA crosses ABOVE Slow MA
    - SELL: Fast MA crosses BELOW Slow MA

    Risk Management:
    - Stop Loss: Slow MA
    - Take Profit: Risk-Reward ratio applied to stop distance
    """

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        ma_type: str = "ema",  # "ema" or "sma"
        risk_reward_ratio: float = 3.0
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.ma_type = ma_type
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = slow_ma + 2

    # -------------------------
    # Indicator calculations
    # -------------------------
    def _calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return prices.ewm(span=period, adjust=False).mean()
        return prices.rolling(period).mean()

    # -------------------------
    # Validation
    # -------------------------
    def _validate_data(self, price_data: pd.DataFrame) -> tuple[bool, str]:
        if price_data is None or price_data.empty:
            return False, "Price data is empty"

        if "close" not in price_data.columns:
            return False, "Missing 'close' column"

        if "time" not in price_data.columns:
            return False, "Missing 'time' column"

        if len(price_data) < self.min_bars:
            return False, f"Need {self.min_bars} bars"

        return True, ""

    def _get_empty_signal(self, reason: str) -> dict:
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "fast_ma_prev": None,
            "fast_ma_now": None,
            "slow_ma_prev": None,
            "slow_ma_now": None,
            "reason": reason,
            "entry_date": None,
        }

    # -------------------------
    # Risk calculation
    # -------------------------
    def _calculate_stop_and_target(
        self,
        entry_price: float,
        stop_price: float,
        signal_type: str
    ) -> tuple[float, float]:

        risk = abs(entry_price - stop_price)
        reward = risk * self.risk_reward_ratio

        if signal_type == "buy":
            return stop_price, entry_price + reward
        else:
            return stop_price, entry_price - reward

    # -------------------------
    # Signal generation
    # -------------------------
    def generate_signal(self, price_data: pd.DataFrame) -> dict:
        is_valid, error = self._validate_data(price_data)
        if not is_valid:
            return self._get_empty_signal(error)

        try:
            closes = price_data["close"]
            current_price = closes.iloc[-1]

            fast_ma = self._calculate_ma(closes, self.fast_ma)
            slow_ma = self._calculate_ma(closes, self.slow_ma)

            fast_prev, fast_now = fast_ma.iloc[-2], fast_ma.iloc[-1]
            slow_prev, slow_now = slow_ma.iloc[-2], slow_ma.iloc[-1]

            result = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.02,
                "fast_ma_prev": float(fast_prev),
                "fast_ma_now": float(fast_now),
                "slow_ma_prev": float(slow_prev),
                "slow_ma_now": float(slow_now),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # BUY crossover
            if fast_prev < slow_prev and fast_now > slow_now:
                result["signal"] = "buy"
                result["reason"] = "Fast MA crossed above Slow MA"
                sl, tp = self._calculate_stop_and_target(
                    current_price, slow_now, "buy"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            # SELL crossover
            elif fast_prev > slow_prev and fast_now < slow_now:
                result["signal"] = "sell"
                result["reason"] = "Fast MA crossed below Slow MA"
                sl, tp = self._calculate_stop_and_target(
                    current_price, slow_now, "sell"
                )
                result["stop_loss"] = sl
                result["take_profit"] = tp

            else:
                result["reason"] = "No MA crossover"

            return result

        except Exception as e:
            return self._get_empty_signal(f"Error: {str(e)}")

    def get_parameters(self) -> dict:
        return {
            "name": "MA Crossover Strategy",
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "ma_type": self.ma_type,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"MATrendStrategy("
            f"FastMA={self.fast_ma}, "
            f"SlowMA={self.slow_ma}, "
            f"Type={self.ma_type}, "
            f"RR={self.risk_reward_ratio})"
        )
