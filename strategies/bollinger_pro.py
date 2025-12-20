import pandas as pd
from datetime import datetime


class BollingerReversionProStrategy:
    """
    FX Mean-Reversion Strategy (1H)

    Enhancements:
    - London + New York session filter
    - Max 1 trade per day per symbol
    - Risk-based position sizing
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,  # 1% risk
        pip_value_per_lot: float = 10.0,  # standard FX
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.pip_value_per_lot = pip_value_per_lot

        self.min_bars = max(bb_period, rsi_period, atr_period) + 50
        self.last_trade_date = None

    # -------------------------
    # Indicators
    # -------------------------
    def _rsi(self, close, period):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period):
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger(self, close):
        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        return mid, mid + std * self.bb_std, mid - std * self.bb_std

    # -------------------------
    # Filters
    # -------------------------
    def _in_trading_session(self, time_utc: datetime) -> bool:
        hour = time_utc.hour
        return 7 <= hour <= 20  # London + NY overlap

    def _one_trade_per_day(self, time_utc: datetime) -> bool:
        trade_date = time_utc.date()
        if self.last_trade_date == trade_date:
            return False
        self.last_trade_date = trade_date
        return True

    # -------------------------
    # Risk Management
    # -------------------------
    def _calculate_position_size(
        self,
        balance: float,
        stop_distance_pips: float
    ) -> float:
        risk_amount = balance * self.risk_per_trade
        lot_size = risk_amount / (stop_distance_pips * self.pip_value_per_lot)
        return round(max(lot_size, 0.01), 2)

    # -------------------------
    # Signal
    # -------------------------
    def generate_signal(self, df: pd.DataFrame, account_balance: float) -> dict:
        if df is None or len(df) < self.min_bars:
            return {"signal": None, "reason": "Not enough data"}

        now = df["time"].iloc[-1]
        if not self._in_trading_session(now):
            return {"signal": None, "reason": "Outside trading session"}

        if not self._one_trade_per_day(now):
            return {"signal": None, "reason": "Trade already taken today"}

        close = df["close"]
        price = close.iloc[-1]

        rsi = self._rsi(close, self.rsi_period)
        atr = self._atr(df, self.atr_period)
        atr_mean = atr.rolling(50).mean()
        mid, upper, lower = self._bollinger(close)

        atr_now = atr.iloc[-1]
        rsi_now = rsi.iloc[-1]

        # BUY
        if price < lower.iloc[-1] and rsi_now < 30 and atr_now > atr_mean.iloc[-1]:
            sl = price - atr_now
            stop_pips = atr_now * 10000
            lot = self._calculate_position_size(account_balance, stop_pips)

            return {
                "signal": "buy",
                "entry_price": price,
                "stop_loss": sl,
                "take_profit": mid.iloc[-1],
                "position_size": lot,
                "reason": "BB lower + RSI oversold",
                "entry_date": now,
            }

        # SELL
        if price > upper.iloc[-1] and rsi_now > 70 and atr_now > atr_mean.iloc[-1]:
            sl = price + atr_now
            stop_pips = atr_now * 10000
            lot = self._calculate_position_size(account_balance, stop_pips)

            return {
                "signal": "sell",
                "entry_price": price,
                "stop_loss": sl,
                "take_profit": mid.iloc[-1],
                "position_size": lot,
                "reason": "BB upper + RSI overbought",
                "entry_date": now,
            }

        return {"signal": None, "reason": "No valid setup"}

    def __repr__(self):
        return "BollingerReversionProStrategy(1H, Risk-Controlled)"
