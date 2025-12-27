import pandas as pd
from typing import Dict, Any, Optional


class MACDTrendStrategy:
    """
    MACD + EMA Trend-Following Strategy

    Rules:
    - Buy: Price > EMA and MACD crosses above signal
    - Sell: Price < EMA and MACD crosses below signal

    Risk:
    - SL at EMA
    - TP = risk * RR
    """

    def __init__(
        self,
        ema_period: int = 200,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        risk_reward_ratio: float = 3.0,
        allowed_weekdays: Optional[list[int]] = None,
    ):
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.risk_reward_ratio = risk_reward_ratio

        self.allowed_weekdays = allowed_weekdays or list(range(7))
        self.blocked_dates: list[str] = []

        self.min_bars = max(ema_period, macd_slow + macd_signal)

    # ---------------- INDICATORS ---------------- #

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _macd(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        fast = self._ema(series, self.macd_fast)
        slow = self._ema(series, self.macd_slow)
        macd = fast - slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return macd, signal

    # ---------------- VALIDATION ---------------- #

    def _validate_data(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return "Price data is empty"
        if "close" not in df.columns or "time" not in df.columns:
            return "Missing required columns"
        if len(df) < self.min_bars:
            return f"Insufficient bars ({len(df)} < {self.min_bars})"
        return None

    def _empty_signal(self, reason: str) -> Dict[str, Any]:
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "trend": None,
            "reason": reason,
            "entry_date": None,
        }

    # ---------------- HELPERS ---------------- #

    @staticmethod
    def _macd_cross(mp, mn, sp, sn) -> Optional[str]:
        if mp < sp and mn > sn:
            return "bullish"
        if mp > sp and mn < sn:
            return "bearish"
        return None

    def _sl_tp(self, entry: float, ema: float, side: str) -> tuple[float, float]:
        risk = abs(entry - ema)
        reward = risk * self.risk_reward_ratio

        if side == "buy":
            return ema, entry + reward
        else:
            return ema, entry - reward

    def _is_allowed_date(self, timestamp) -> bool:
        dt = pd.to_datetime(timestamp)
        if dt.weekday() not in self.allowed_weekdays:
            return False
        if dt.strftime("%Y-%m-%d") in self.blocked_dates:
            return False
        return True

    # ---------------- MAIN LOGIC ---------------- #

    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        error = self._validate_data(price_data)
        if error:
            return self._empty_signal(error)

        entry_time = price_data["time"].iloc[-1]
        if not self._is_allowed_date(entry_time):
            return self._empty_signal("Trading not allowed on this date")

        try:
            closes = price_data["close"]
            price = closes.iloc[-1]

            ema = self._ema(closes, self.ema_period).iloc[-1]
            macd, signal = self._macd(closes)

            mp, mn = macd.iloc[-2], macd.iloc[-1]
            sp, sn = signal.iloc[-2], signal.iloc[-1]

            trend = "bullish" if price > ema else "bearish" if price < ema else None
            cross = self._macd_cross(mp, mn, sp, sn)

            result = {
                "signal": None,
                "entry_price": price,
                "stop_loss": None,
                "take_profit": None,
                "trend": trend,
                "macd_prev": float(mp),
                "macd_now": float(mn),
                "macd_signal_prev": float(sp),
                "macd_signal_now": float(sn),
                "ema": float(ema),
                "reason": None,
                "entry_date": entry_time,
            }

            if trend == "bullish" and cross == "bullish":
                sl, tp = self._sl_tp(price, ema, "buy")
                result.update(
                    {
                        "signal": "buy",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "MACD bullish cross in uptrend",
                    }
                )

            elif trend == "bearish" and cross == "bearish":
                sl, tp = self._sl_tp(price, ema, "sell")
                result.update(
                    {
                        "signal": "sell",
                        "stop_loss": sl,
                        "take_profit": tp,
                        "reason": "MACD bearish cross in downtrend",
                    }
                )

            else:
                result["reason"] = "No valid MACD + trend alignment"

            return result

        except Exception as e:
            return self._empty_signal(f"Signal error: {e}")

    # ---------------- META ---------------- #

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "MACD Trend Strategy",
            "ema_period": self.ema_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "risk_reward_ratio": self.risk_reward_ratio,
            "min_bars": self.min_bars,
        }

    def __repr__(self) -> str:
        return (
            f"MACDTrendStrategy("
            f"EMA={self.ema_period}, "
            f"MACD={self.macd_fast}/{self.macd_slow}/{self.macd_signal}, "
            f"RR={self.risk_reward_ratio})"
        )


class RestrictiveVolumeStrategy:
    """
    Highly Restrictive MACD + EMA + Volume Strategy

    Rules:
    - Buy: Price > EMA AND MACD crosses above signal AND volume spike AND favorite time/day
    - Sell: Price < EMA AND MACD crosses below signal AND volume spike AND favorite time/day

    Risk:
    - Flexible SL: ATR-based or swing low/high
    - Flexible TP: Multiple methods (fixed RR, ATR-based, or trailing)
    """

    def __init__(
        self,
        # Trend
        ema_period: int = 200,
        # MACD
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Volume
        volume_ma_period: int = 20,
        volume_spike_multiplier: float = 2.0,  # Volume must be 1.5x average
        # Time filters
        favorite_hours: Optional[list[int]] = None,  # e.g., [8, 9, 10] for 8am-10am
        favorite_weekdays: Optional[list[int]] = None,  # 0=Mon, 4=Fri
        # Volatility
        atr_period: int = 14,
        # Risk management
        sl_method: str = "atr",  # "atr", "ema", "swing"
        sl_atr_multiplier: float = 2.0,
        tp_method: str = "atr",  # "fixed_rr", "atr", "swing"
        tp_atr_multiplier: float = 3.0,
        fixed_risk_reward: float = 3.0,
        # Additional filters
        min_distance_from_ema_pct: float = 0.1,  # Price must be >0.1% from EMA
        require_macd_histogram_momentum: bool = True,  # MACD must be accelerating
    ):
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.volume_ma_period = volume_ma_period
        self.volume_spike_multiplier = volume_spike_multiplier

        self.favorite_hours = favorite_hours or list(range(24))
        self.favorite_weekdays = favorite_weekdays or list(range(7))

        self.atr_period = atr_period

        self.sl_method = sl_method
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_method = tp_method
        self.tp_atr_multiplier = tp_atr_multiplier
        self.fixed_risk_reward = fixed_risk_reward

        self.min_distance_from_ema_pct = min_distance_from_ema_pct
        self.require_macd_histogram_momentum = require_macd_histogram_momentum

        self.blocked_dates: list[str] = []
        self.min_bars = (
            max(ema_period, macd_slow + macd_signal, volume_ma_period, atr_period) + 5
        )

    # ---------------- INDICATORS ---------------- #

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _macd(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast = self._ema(series, self.macd_fast)
        slow = self._ema(series, self.macd_slow)
        macd = fast - slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def _volume_ma(self, volume: pd.Series) -> pd.Series:
        return volume.rolling(window=self.volume_ma_period).mean()

    def _find_swing_low(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """Find recent swing low"""
        return df["low"].iloc[-lookback:].min()

    def _find_swing_high(self, df: pd.DataFrame, lookback: int = 10) -> float:
        """Find recent swing high"""
        return df["high"].iloc[-lookback:].max()

    # ---------------- VALIDATION ---------------- #

    def _validate_data(self, df: pd.DataFrame) -> Optional[str]:
        required_cols = ["close", "high", "low", "time", "tick_volume"]

        if df is None or df.empty:
            return "Price data is empty"

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return f"Missing columns: {missing}"

        if len(df) < self.min_bars:
            return f"Insufficient bars ({len(df)} < {self.min_bars})"

        return None

    def _empty_signal(self, reason: str) -> Dict[str, Any]:
        return {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "trend": None,
            "reason": reason,
            "entry_date": None,
            "filters_passed": {},
        }

    # ---------------- FILTERS ---------------- #

    def _is_allowed_datetime(self, timestamp) -> tuple[bool, str]:
        dt = pd.to_datetime(timestamp)

        if dt.weekday() not in self.favorite_weekdays:
            return False, f"Weekday {dt.strftime('%A')} not in favorites"

        if dt.hour not in self.favorite_hours:
            return False, f"Hour {dt.hour} not in favorites"

        if dt.strftime("%Y-%m-%d") in self.blocked_dates:
            return False, "Date is blocked"

        return True, "Time filter passed"

    def _check_volume_spike(
        self, volume: pd.Series, volume_ma: pd.Series
    ) -> tuple[bool, str]:
        current_vol = volume.iloc[-1]
        avg_vol = volume_ma.iloc[-1]

        if pd.isna(avg_vol) or avg_vol == 0:
            return False, "Volume MA not available"

        ratio = current_vol / avg_vol

        if ratio >= self.volume_spike_multiplier:
            return True, f"Volume spike: {ratio:.2f}x average"

        return False, f"Volume too low: {ratio:.2f}x average"

    def _check_ema_distance(self, price: float, ema: float) -> tuple[bool, str]:
        distance_pct = abs(price - ema) / ema * 100

        if distance_pct >= self.min_distance_from_ema_pct:
            return True, f"Price {distance_pct:.2f}% from EMA"

        return False, f"Price too close to EMA: {distance_pct:.2f}%"

    def _check_macd_momentum(self, histogram: pd.Series) -> tuple[bool, str]:
        if not self.require_macd_histogram_momentum:
            return True, "Momentum check disabled"

        h_prev = histogram.iloc[-2]
        h_now = histogram.iloc[-1]

        # For bullish: histogram should be increasing (getting more positive)
        # For bearish: histogram should be decreasing (getting more negative)

        if h_now > h_prev and h_now > 0:
            return True, "Bullish momentum confirmed"

        if h_now < h_prev and h_now < 0:
            return True, "Bearish momentum confirmed"

        return False, "MACD momentum not confirmed"

    @staticmethod
    def _macd_cross(mp, mn, sp, sn) -> Optional[str]:
        if mp < sp and mn > sn:
            return "bullish"
        if mp > sp and mn < sn:
            return "bearish"
        return None

    # ---------------- STOP LOSS / TAKE PROFIT ---------------- #

    def _calculate_stop_loss(
        self, df: pd.DataFrame, entry: float, atr: float, side: str
    ) -> float:
        if self.sl_method == "atr":
            if side == "buy":
                return entry - (atr * self.sl_atr_multiplier)
            else:
                return entry + (atr * self.sl_atr_multiplier)

        elif self.sl_method == "ema":
            ema = self._ema(df["close"], self.ema_period).iloc[-1]
            return ema

        elif self.sl_method == "swing":
            if side == "buy":
                return self._find_swing_low(df)
            else:
                return self._find_swing_high(df)

        else:
            raise ValueError(f"Unknown SL method: {self.sl_method}")

    def _calculate_take_profit(
        self, df: pd.DataFrame, entry: float, stop_loss: float, atr: float, side: str
    ) -> float:
        if self.tp_method == "fixed_rr":
            risk = abs(entry - stop_loss)
            reward = risk * self.fixed_risk_reward

            if side == "buy":
                return entry + reward
            else:
                return entry - reward

        elif self.tp_method == "atr":
            if side == "buy":
                return entry + (atr * self.tp_atr_multiplier)
            else:
                return entry - (atr * self.tp_atr_multiplier)

        elif self.tp_method == "swing":
            # Take profit at previous swing high/low
            if side == "buy":
                return self._find_swing_high(df)
            else:
                return self._find_swing_low(df)

        else:
            raise ValueError(f"Unknown TP method: {self.tp_method}")

    # ---------------- MAIN LOGIC ---------------- #
    def generate_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        error = self._validate_data(price_data)
        if error:
            return self._empty_signal(error)

        df = price_data.copy()
        entry_time = df["time"].iloc[-1]

        filters = {}

        # ---------------- TIME FILTER ----------------
        time_ok, time_msg = self._is_allowed_datetime(entry_time)
        filters["time"] = time_msg

        closes = df["close"]
        volumes = df["tick_volume"]

        price = float(closes.iloc[-1])

        # ---------------- INDICATORS ----------------
        ema = float(self._ema(closes, self.ema_period).iloc[-1])
        macd, signal, histogram = self._macd(closes)
        atr = float(self._atr(df).iloc[-1])
        volume_ma = self._volume_ma(volumes)

        mp, mn = macd.iloc[-2], macd.iloc[-1]
        sp, sn = signal.iloc[-2], signal.iloc[-1]

        # ---------------- TREND ----------------
        trend = "bullish" if price > ema else "bearish" if price < ema else None
        filters["trend"] = f"Trend: {trend}"

        # ---------------- MACD CROSS ----------------
        cross = self._macd_cross(mp, mn, sp, sn)
        filters["macd_cross"] = f"MACD cross: {cross}"

        # ---------------- VOLUME ----------------
        vol_ok, vol_msg = self._check_volume_spike(volumes, volume_ma)
        filters["volume"] = vol_msg

        # ---------------- EMA DISTANCE ----------------
        dist_ok, dist_msg = self._check_ema_distance(price, ema)
        filters["ema_distance"] = dist_msg

        # ---------------- MOMENTUM ----------------
        mom_ok, mom_msg = self._check_macd_momentum(histogram)
        filters["momentum"] = mom_msg

        # ---------------- BASE RESULT (ALWAYS RETURNED) ----------------
        result = {
            "signal": None,
            "entry_price": price,
            "stop_loss": None,
            "take_profit": None,
            "trend": trend,
            "ema": ema,
            "atr": atr,
            "macd_histogram": float(histogram.iloc[-1]),
            "volume_ratio": float(volumes.iloc[-1] / volume_ma.iloc[-1]),
            "entry_date": entry_time,
            "filters_passed": filters,
            "reason": None,
        }

        # ---------------- DECISION LOGIC ----------------
        if not time_ok:
            result["reason"] = "Time filter blocked"
            return result

        if not trend:
            result["reason"] = "No clear trend"
            return result

        if not cross:
            result["reason"] = "No MACD cross"
            return result

        if not vol_ok:
            result["reason"] = "Volume filter failed"
            return result

        if not dist_ok:
            result["reason"] = "EMA distance filter failed"
            return result

        if not mom_ok:
            result["reason"] = "Momentum filter failed"
            return result

        # ---------------- SIGNAL GENERATION ----------------
        if trend == "bullish" and cross == "bullish":
            sl = self._calculate_stop_loss(df, price, atr, "buy")
            tp = self._calculate_take_profit(df, price, sl, atr, "buy")

            result.update(
                {
                    "signal": "buy",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "risk_pips": abs(price - sl),
                    "reward_pips": abs(tp - price),
                    "risk_reward": abs(tp - price) / abs(price - sl),
                    "reason": "All bullish filters passed",
                }
            )

        elif trend == "bearish" and cross == "bearish":
            sl = self._calculate_stop_loss(df, price, atr, "sell")
            tp = self._calculate_take_profit(df, price, sl, atr, "sell")

            result.update(
                {
                    "signal": "sell",
                    "stop_loss": sl,
                    "take_profit": tp,
                    "risk_pips": abs(price - sl),
                    "reward_pips": abs(price - tp),
                    "risk_reward": abs(price - tp) / abs(price - sl),
                    "reason": "All bearish filters passed",
                }
            )

        else:
            result["reason"] = "Trend and MACD direction mismatch"

        return result

    # ---------------- META ---------------- #

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": "Restrictive Volume Strategy",
            "ema_period": self.ema_period,
            "macd": f"{self.macd_fast}/{self.macd_slow}/{self.macd_signal}",
            "volume_spike_multiplier": self.volume_spike_multiplier,
            "favorite_hours": self.favorite_hours,
            "favorite_weekdays": [
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
                for d in self.favorite_weekdays
            ],
            "atr_period": self.atr_period,
            "sl_method": self.sl_method,
            "tp_method": self.tp_method,
            "min_distance_from_ema_pct": self.min_distance_from_ema_pct,
            "require_macd_histogram_momentum": self.require_macd_histogram_momentum,
        }

    def __repr__(self) -> str:
        return (
            f"RestrictiveVolumeStrategy("
            f"EMA={self.ema_period}, "
            f"Vol={self.volume_spike_multiplier}x, "
            f"SL={self.sl_method}, "
            f"TP={self.tp_method})"
        )
