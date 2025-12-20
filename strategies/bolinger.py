import pandas as pd


class BollingerReversionStrategy:
    """
    Range Reversion Strategy (1H FX)

    Indicators:
    - Bollinger Bands (20, 2.0)
    - RSI (14)
    - ATR (14)

    Logic:
    - Fade volatility extremes
    - Mean reversion to BB mid
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        atr_period: int = 14,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.atr_period = atr_period

        self.min_bars = max(bb_period, rsi_period, atr_period) + 2

    # -------------------------
    # Indicators
    # -------------------------
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

    def _bollinger_bands(self, series: pd.Series):
        mid = series.rolling(self.bb_period).mean()
        std = series.rolling(self.bb_period).std()
        upper = mid + (std * self.bb_std)
        lower = mid - (std * self.bb_std)
        return mid, upper, lower

    # -------------------------
    # Validation
    # -------------------------
    def _validate(self, df: pd.DataFrame):
        required = {"time", "open", "high", "low", "close"}
        if df is None or df.empty:
            return False, "Empty price data"
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
            "position_size": 0.02,
            "reason": reason,
            "entry_date": None,
        }

    # -------------------------
    # Signal generation
    # -------------------------
    def generate_signal(self, price_data: pd.DataFrame) -> dict:
        valid, error = self._validate(price_data)
        if not valid:
            return self._empty_signal(error)

        try:
            close = price_data["close"]
            price = close.iloc[-1]

            rsi = self._rsi(close, self.rsi_period)
            atr = self._atr(price_data, self.atr_period)
            atr_mean = atr.rolling(50).mean()

            mid, upper, lower = self._bollinger_bands(close)

            rsi_now = rsi.iloc[-1]
            atr_now = atr.iloc[-1]

            result = {
                "signal": None,
                "entry_price": price,
                "stop_loss": None,
                "take_profit": None,
                "position_size": 0.02,
                "rsi": float(rsi_now),
                "atr": float(atr_now),
                "reason": None,
                "entry_date": price_data["time"].iloc[-1],
            }

            # BUY (oversold)
            if (
                price < lower.iloc[-1] and
                rsi_now < 30 and
                atr_now > atr_mean.iloc[-1]
            ):
                sl = price - atr_now
                tp = mid.iloc[-1]

                result.update({
                    "signal": "buy",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "BB lower + RSI oversold",
                })

            # SELL (overbought)
            elif (
                price > upper.iloc[-1] and
                rsi_now > 70 and
                atr_now > atr_mean.iloc[-1]
            ):
                sl = price + atr_now
                tp = mid.iloc[-1]

                result.update({
                    "signal": "sell",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "reason": "BB upper + RSI overbought",
                })

            else:
                result["reason"] = "No extreme condition"

            return result

        except Exception as e:
            return self._empty_signal(str(e))

    def __repr__(self):
        return "BollingerReversionStrategy(1H FX)"
