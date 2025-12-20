import pandas as pd


class MATrendPullbackStrategy:
    """
    Trend-Filtered MA Pullback Strategy (1H)

    Indicators:
    - EMA 50 (trend pullback)
    - EMA 200 (trend filter)
    - RSI 14 (momentum filter)
    - ATR 14 (volatility-based SL)

    Risk:
    - Stop Loss = 1.2 * ATR
    - Take Profit = 3R
    """

    def __init__(
        self,
        fast_ma: int = 50,
        slow_ma: int = 200,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier: float = 1.2,
        risk_reward_ratio: float = 3.0,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio

        self.min_bars = max(slow_ma, rsi_period, atr_period) + 2

    # -------------------------
    # Indicator calculations
    # -------------------------
    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # -------------------------
    # Validation
    # -------------------------
    def _validate_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        required = {"time", "open", "high", "low", "close"}
        if df is None or df.empty:
            return False, "Price data is empty"
        if not required.issubset(df.columns):
            return False, f"Missing columns: {required - set(df.columns)}"
        if len(df) < self.min_bars:
            return False, f"Need at least {self.min_bars} bars"
        return True, ""

    def _empty_signal(self, reason: str) -> dict:
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.02,
            "reason": reason,
            "entry_date": None,
        }

    # -------------------------
    # Signal generation
    # -------------------------
    def generate_signal(self, price_data: pd.DataFrame) -> dict:
        is_valid, error = self._validate_data(price_data)
        if not is_valid:
            return self._empty_signal(error)

        try:
            closes = price_data["close"]

            ema_fast = self._ema(closes, self.fast_ma)
            ema_slow = self._ema(closes, self.slow_ma)
            rsi = self._rsi(closes, self.rsi_period)
            atr = self._atr(price_data, self.atr_period)

            price = closes.iloc[-1]
            atr_now = atr.iloc[-1]

            result = {
                "signal": None,
                "entry_price": price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.02,
                "ema_fast": float(ema_fast.iloc[-1]),
                "ema_slow": float(ema_slow.iloc[-1]),
                "rsi": float(rsi.iloc[-1]),
                "atr": float(atr_now),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # -------------------------
            # BUY conditions
            # -------------------------
            if (
                ema_fast.iloc[-1] > ema_slow.iloc[-1] and
                price > ema_fast.iloc[-1] and
                40 <= rsi.iloc[-1] <= 55
            ):
                sl = price - (atr_now * self.atr_multiplier)
                tp = price + (price - sl) * self.risk_reward_ratio

                result.update({
                    "signal": "buy",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "Uptrend pullback with RSI confirmation",
                })

            # -------------------------
            # SELL conditions
            # -------------------------
            elif (
                ema_fast.iloc[-1] < ema_slow.iloc[-1] and
                price < ema_fast.iloc[-1] and
                45 <= rsi.iloc[-1] <= 60
            ):
                sl = price + (atr_now * self.atr_multiplier)
                tp = price - (sl - price) * self.risk_reward_ratio

                result.update({
                    "signal": "sell",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "Downtrend pullback with RSI confirmation",
                })

            else:
                result["reason"] = "No valid setup"

            return result

        except Exception as e:
            return self._empty_signal(f"Error: {str(e)}")

    def get_parameters(self) -> dict:
        return {
            "name": "MA Trend Pullback Strategy",
            "ema_fast": self.fast_ma,
            "ema_slow": self.slow_ma,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars_required": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"MATrendPullbackStrategy("
            f"EMA{self.fast_ma}/EMA{self.slow_ma}, "
            f"RSI={self.rsi_period}, "
            f"ATR={self.atr_period}, "
            f"RR={self.risk_reward_ratio})"
        )
