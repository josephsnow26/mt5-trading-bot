import pandas as pd

class TrendPullbackVolatilityStrategy:
    """
    Trend + Pullback + Volatility Strategy

    Entry Rules:
    - Determine trend via EMA (50) vs EMA (200)
    - Pullback entry: price crosses back to EMA(50) within trend
    - Volatility filter: ATR-based stop loss

    Risk Management:
    - Stop Loss: ATR-based
    - Take Profit: Risk-Reward ratio applied to stop distance
    """

    def __init__(
        self,
        fast_ema: int = 50,
        slow_ema: int = 200,
        atr_period: int = 14,
        risk_reward_ratio: float = 2.5,
        ma_type: str = "ema"
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.risk_reward_ratio = risk_reward_ratio
        self.ma_type = ma_type
        self.min_bars = slow_ema + atr_period

    # -------------------------
    # EMA Calculation
    # -------------------------
    def _calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return prices.ewm(span=period, adjust=False).mean()
        return prices.rolling(period).mean()

    # -------------------------
    # ATR Calculation
    # -------------------------
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # -------------------------
    # Signal Generation
    # -------------------------
    def generate_signal(self, price_data: pd.DataFrame) -> dict:
        if price_data is None or price_data.empty or len(price_data) < self.min_bars:
            return {"signal": None}

        closes = price_data['close']
        highs = price_data['high']
        lows = price_data['low']
        current_price = closes.iloc[-1]

        ema_fast = self._calculate_ma(closes, self.fast_ema)
        ema_slow = self._calculate_ma(closes, self.slow_ema)
        atr = self._calculate_atr(price_data, self.atr_period).iloc[-1]

        trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1]

        prev_close = closes.iloc[-2]

        # Entry logic
        signal = None
        stop_loss = None
        take_profit = None

        if trend_up and prev_close < ema_fast.iloc[-2] and current_price > ema_fast.iloc[-1]:
            signal = "buy"
            stop_loss = current_price - atr
            take_profit = current_price + atr * self.risk_reward_ratio

        elif trend_down and prev_close > ema_fast.iloc[-2] and current_price < ema_fast.iloc[-1]:
            signal = "sell"
            stop_loss = current_price + atr
            take_profit = current_price - atr * self.risk_reward_ratio

        return {
            "signal": signal,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": 0.02,
            "fast_ema": float(ema_fast.iloc[-1]),
            "slow_ema": float(ema_slow.iloc[-1]),
            "atr": float(atr),
            "entry_date": price_data["time"].iloc[-1],
        }

    def get_parameters(self) -> dict:
        return {
            "name": "TrendPullbackVolatilityStrategy",
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "atr_period": self.atr_period,
            "risk_reward_ratio": self.risk_reward_ratio,
        }

    def __repr__(self) -> str:
        return (
            f"TrendPullbackVolatilityStrategy("
            f"FastEMA={self.fast_ema}, SlowEMA={self.slow_ema}, "
            f"ATR={self.atr_period}, RR={self.risk_reward_ratio})"
        )
