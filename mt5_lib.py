import MetaTrader5
import os, time
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta, timezone
from symbols import get_symbols


symbols_dict = get_symbols().get("symbols_dict")


class TradingStrategy:
    def __init__(self, data = None, balance=10000, risk_per_trade=0.02, strategy="hybrid",symbol=None):
        """
        balance: initial account balance
        risk_per_trade: percentage of balance risked per trade
        strategy: which logic to use ('rsi', 'moving_average', 'hybrid')
        """
        self.data = data
        self.strategy = strategy
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.position = None  # {'type': 'buy' or 'sell', 'entry_price': float, 'sl': float, 'tp': float}
        self.symbol = symbol
    # ------------------------------------------------------------
    # 🧠 INDICATOR CALCULATIONS
    # ------------------------------------------------------------
    def compute_indicators(self):
        """
        Compute indicators for both H1 and M5 (or main) dataframes.
        - H1: SMA(100) for trend direction
        - Main timeframe: RSI, EMA, MACD, Bollinger Bands
        """
        # === H1 Trend Data ===
        data_H1 = self.data.get("data_H1", pd.DataFrame()).copy()
        if not data_H1.empty:
            data_H1["sma_100"] = SMAIndicator(data_H1["close"], window=100).sma_indicator()
            self.data["data_H1"] = data_H1

        # === Main timeframe indicators (e.g., M5) ===
        df = self.data.get("data_M5", pd.DataFrame()).copy()
        if not df.empty:
            df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
            df["ema_fast"] = EMAIndicator(df["close"], window=20).ema_indicator()
            df["ema_slow"] = EMAIndicator(df["close"], window=50).ema_indicator()
            df["ema_8"] = EMAIndicator(df["close"], window=8).ema_indicator()
            df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator()
            df["ema_200"] = EMAIndicator(df["close"], window=200).ema_indicator()
            df["sma_trend"] = df["close"].rolling(window=200).mean()

            macd = MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()

            bb = BollingerBands(df["close"], window=20, window_dev=2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_mid"] = bb.bollinger_mavg()

            self.data["data_M5"] = df

        return df


    # ------------------------------------------------------------
    # ⚙️ STRATEGIES
    # ------------------------------------------------------------

    def get_higher_tf_trend(self):
        """
        Determine the market trend from H1 using SMA100.
        Returns: 'bullish', 'bearish', or None
        """
        data_H1 = self.data.get("data_H1", pd.DataFrame()).copy()

        if data_H1.empty:
            print("⚠️ No H1 data to determine trend.")
            return None

        last = data_H1.iloc[-1]

        if last["close"] > last["sma_100"]:
            return "bullish"
        elif last["close"] < last["sma_100"]:
            return "bearish"
        else:
            return None
        

    def get_swing_points(self, lookback=10):
        """
        Detects the most recent swing high and swing low.
        A swing high: high[n] > high[n-1] and high[n] > high[n+1]
        A swing low:  low[n] < low[n-1] and low[n] < low[n+1]
        """
        df = self.data.get("data_M5", pd.DataFrame()).copy()
        if df.empty or len(df) < lookback + 3:
            return None, None

        df = df.tail(lookback).reset_index(drop=True)

        swing_high = None
        swing_low = None

        for i in range(1, len(df)-1):
            # Swing High
            if df.loc[i, "high"] > df.loc[i-1, "high"] and df.loc[i, "high"] > df.loc[i+1, "high"]:
                swing_high = df.loc[i, "high"]

            # Swing Low
            if df.loc[i, "low"] < df.loc[i-1,"low"] and df.loc[i,"low"] < df.loc[i+1,"low"]:
                swing_low = df.loc[i,"low"]

        return swing_high, swing_low


    def rsi_strategy(self):
        """
        RSI + Trend + Opposite Swing Structure Filter
        """
        self.compute_indicators()

        data_M5 = self.data.get("data_M5", pd.DataFrame()).copy()
        if data_M5.empty:
            print("⚠️ Insufficient M5 data.")
            return None

        last_m5 = data_M5.iloc[-1]
        close_price = last_m5["close"]
        rsi = last_m5["rsi"]

        # === 1️⃣ TREND ===
        trend = self.get_higher_tf_trend()
        if trend is None:
            print("⚠️ Cannot determine trend.")
            return None

        

        # === 3️⃣ SIGNAL FILTER ===
        signal = None

        # BUY: use previous swing HIGH as filter
        if rsi < 30 and trend == "bullish":
            signal = "buy"
           

        # SELL: use previous swing LOW as filter
        elif rsi > 70 and trend == "bearish":
            signal = "sell"
           
        # === 4️⃣ DIAGNOSTICS ===
        print("\n📊 RSI STRATEGY DIAGNOSTICS")
        print(f"Trend: {trend.upper()}")
        print(f"RSI: {rsi:.2f}")
        print(f"Close Price: {close_price}")
        print(f"Signal: {'✅ '+signal if signal else '❌ No signal'}")

        return signal


    def rsi_exit_strategy(self, timeframe=MetaTrader5.TIMEFRAME_M5, rsi_period=14):
        """
        RSI-based exit strategy (independent of self.data)
        - BUY: Exit when RSI > 70 and starts decreasing
        - SELL: Exit when RSI < 30 and starts increasing
        - Fetches live M5 data per symbol for accurate RSI confirmation
        """
    
        positions = MetaTrader5.positions_get()
        if not positions:
            print("⚠️ No open positions to evaluate for RSI exit.")
            return []

        exit_trades = []

        print("\n🧭 RSI EXIT STRATEGY CHECK (live data per symbol)")
        print(f"Total open positions: {len(positions)}\n")

        for pos in positions:
            symbol = pos.symbol
            position_type = "buy" if pos.type == MetaTrader5.POSITION_TYPE_BUY else "sell"

            # === Get live data for this symbol ===
            rates = MetaTrader5.copy_rates_from_pos(symbol, timeframe, 0, 100)
            if rates is None or len(rates) < 2:
                print(f"⚠️ Could not retrieve enough M5 data for {symbol} — skipping.")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # === Compute RSI ===
            df["rsi"] = RSIIndicator(close=df["close"], window=rsi_period).rsi()
            if df["rsi"].isnull().any():
                df = df.dropna(subset=["rsi"])
            if len(df) < 2:
                print(f"⚠️ Not enough valid RSI values for {symbol} — skipping.")
                continue

            rsi_prev = df["rsi"].iloc[-2]
            rsi_now = df["rsi"].iloc[-1]

            should_exit = False
            trend_note = ""

            # === BUY EXIT LOGIC ===
            if position_type == "buy":
                if rsi_prev > 70 and rsi_prev < rsi_now:
                    trend_note = "RSI still rising — holding for extended gains"
                elif rsi_prev >70 and rsi_prev > rsi_now:
                    should_exit = True
                    trend_note = "RSI peaked and now falling — exit to secure profits"

            # === SELL EXIT LOGIC ===
            elif position_type == "sell":
                if rsi_prev < 30 and rsi_prev > rsi_now:
                    trend_note = "RSI still falling — holding for extended downside"
                elif rsi_prev < 30 and rsi_prev < rsi_now:
                    should_exit = True
                    trend_note = "RSI bottomed and now rising — exit to secure profits"

            status = "🚪 EXIT" if should_exit else "⏳ HOLD"
            print(
                f"🔹 {symbol}: {position_type.upper()} | RSI={rsi_now:.2f} (prev {rsi_prev:.2f}) | {status} — {trend_note}"
            )

            if should_exit:
                trade_info = {
                    "symbol": symbol,
                    "ticket": pos.ticket,
                    "position": position_type,
                    "rsi": rsi_now,
                    "exit_reason": "RSI reversal confirmation (70/30)",
                }
                exit_trades.append(trade_info)

        # === Execute exits ===
        if exit_trades:
            print("\n⚙️ Executing RSI-based exits...\n")
            for trade in exit_trades:
                self.close_position(trade)
                print(f"✅ Closed {trade['symbol']} ({trade['position'].upper()}) | RSI={trade['rsi']:.2f}")
        else:
            print("\n✅ All positions still valid — RSI trends not reversed yet.")

        return exit_trades
    

    def ema_crossover_strategy(self):
        """
        EMA Crossover Entry Strategy on M5 with higher timeframe trend filter
        """
        self.compute_indicators()
        df = self.data.get("data_M5", pd.DataFrame())
        if df.empty:
            print("⚠️ No M5 data")
            return None

        trend = self.get_higher_tf_trend()
        if trend is None:
            print("⚠️ Cannot determine trend")
            return None

        last_m5 = df.iloc[-1]
        close_price = last_m5["close"]

        # Get last 2 EMA values to detect crossover
        fast_prev, fast_curr = df["ema_8"].iloc[-2], df["ema_8"].iloc[-1]
        slow_prev, slow_curr = df["ema_21"].iloc[-2], df["ema_21"].iloc[-1]

        # Calculate EMA distance & slope
        ema_distance = abs(fast_curr - slow_curr)
        fast_slope = fast_curr - fast_prev
        min_distance = 0.001 * close_price
        min_slope = 0.0005 * close_price

        signal = None

        print(f"\n📈 EMA Check for latest candle:")
        print(f"   EMA 8 prev: {fast_prev:.5f}, curr: {fast_curr:.5f}")
        print(f"   EMA 21 prev: {slow_prev:.5f}, curr: {slow_curr:.5f}")
        print(f"   EMA distance: {ema_distance:.5f} (min required: {min_distance:.5f})")
        print(f"   EMA slope: {fast_slope:.5f} (min required: {min_slope:.5f})")
        print(f"   Higher TF trend: {trend.upper()}")

        # Check BUY conditions
        buy_conditions = [
            fast_prev <= slow_prev and fast_curr > slow_curr,
            trend == "bullish",
            ema_distance >= min_distance,
            abs(fast_slope) >= min_slope
        ]
        if all(buy_conditions):
            signal = "buy"
        else:
            print("❌ BUY conditions not fully met:")
            print(f"   Buy Crossover: {buy_conditions[0]}")
            print(f"   Trend bullish: {buy_conditions[1]}")
            print(f"   Distance OK: {buy_conditions[2]}")
            print(f"   Slope OK: {buy_conditions[3]}")

        # Check SELL conditions
        sell_conditions = [
            fast_prev >= slow_prev and fast_curr < slow_curr,
            trend == "bearish",
            ema_distance >= min_distance,
            abs(fast_slope) >= min_slope
        ]
        if all(sell_conditions):
            signal = "sell"
        else:
            print("❌ SELL conditions not fully met:")
            print(f"   Sell Crossover: {sell_conditions[0]}")
            print(f"   Trend bearish: {sell_conditions[1]}")
            print(f"   Distance OK: {sell_conditions[2]}")
            print(f"   Slope OK: {sell_conditions[3]}")

        print(f"📊 EMA Crossover Entry Signal: {signal}")
        return signal
    

    def macd_strategy(self):
        """
        MACD + 200 EMA Strategy:
        BUY  = price above 200 EMA + MACD crosses up below zero
        SELL = price below 200 EMA + MACD crosses down above zero

        Returns dict:
            {
                "signal": "buy/sell",
                "sl_pips": int,
                "tp_pips": int
            }
        """
        symbol = self.symbol

        self.compute_indicators()

        df = self.data.get("data_M5", pd.DataFrame()).copy()
        if df.empty:
            print("⚠️ No data available for MACD strategy.")
            return None

        # Ensure symbol pip size
        symbol = (symbol or "")

        # Pip size per symbol
        if symbol == "XAUUSDm":
            PIP = 0.10         # Gold
        elif symbol == "USDJPYm":
            PIP = 0.01         # Yen pairs
        else:
            PIP = 0.0001       # EURUSDm, GBPUSDm, etc.


        # Last two candles (cross detection)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close_price = last["close"]
        ema_200 = last["ema_200"]

        macd_now = last["macd"]
        macd_prev = prev["macd"]
        signal_now = last["macd_signal"]
        signal_prev = prev["macd_signal"]

        # --- Trend from EMA 200 ---
        trend = "bullish" if close_price > ema_200 else "bearish"
        signal = None

        # ---------- BUY ----------
        if trend == "bullish":
            if macd_prev < signal_prev and macd_now > signal_now:
                if macd_prev < 0:  # cross begins below zero
                    signal = "buy"

        # ---------- SELL ----------
        elif trend == "bearish":
            if macd_prev > signal_prev and macd_now < signal_now:
                if macd_prev > 0:  # cross begins above zero
                    signal = "sell"

        # === Diagnostics Output === 
        print("\n📊 MACD STRATEGY DIAGNOSTICS") 
        print(f"Trend: {trend.upper()}") 
        print(f"Close Price: {close_price}") 
        print(f"EMA200: {ema_200}") 
        print(f"MACD(prev→now): {macd_prev:.4f} → {macd_now:.4f}") 
        print(f"Signal(prev→now): {signal_prev:.4f} → {signal_now:.4f}") 
        print(f"Final Signal: {'✅ ' + signal if signal else '❌ No signal'}")

        # No signal → stop
        if not signal:
            return None

        # -----------------------------
        # 📌 SL + TP in pips
        # -----------------------------
        offset_price = 5 * PIP     # 5 pip offset

        if signal == "buy":
            # BUY → SL 5 pips BELOW EMA200
            sl_price = ema_200 - offset_price
            sl_pips = int(abs(close_price - sl_price) / PIP)

        elif signal == "sell":
            # SELL → SL 5 pips ABOVE EMA200
            sl_price = ema_200 + offset_price
            sl_pips = int(abs(sl_price - close_price) / PIP)

        # TP = 1.5 × SL
        tp_pips = int(sl_pips * 1.5)

        # Diagnostic
        print(f"\n📊 {symbol} MACD Strategy")
        print(f"Signal: {signal}")
        print(f"Pip Size: {PIP}")
        print(f"SL (pips): {sl_pips}")
        print(f"TP (pips): {tp_pips}")

        return {
            "signal": signal,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips
        }


    def ema_crossover_exit(self, timeframe=MetaTrader5.TIMEFRAME_M5):
        """
        EMA Crossover Exit Strategy (loop through all open positions)
        - BUY: Exit when EMA(8) crosses below EMA(21)
        - SELL: Exit when EMA(8) crosses above EMA(21)
        - Fetches live M5 data per symbol for accurate EMA confirmation
        """

        positions = MetaTrader5.positions_get()
        if not positions:
            print("⚠️ No open positions to evaluate for EMA exit.")
            return []

        exit_trades = []

        print("\n🧭 EMA CROSSOVER EXIT CHECK (live data per symbol)")
        print(f"Total open positions: {len(positions)}\n")

        for pos in positions:
            symbol = pos.symbol
            position_type = "buy" if pos.type == MetaTrader5.POSITION_TYPE_BUY else "sell"

            # === Get live M5 data for this symbol ===
            rates = MetaTrader5.copy_rates_from_pos(symbol, timeframe, 0, 50)
            if rates is None or len(rates) < 2:
                print(f"⚠️ Not enough M5 data for {symbol} — skipping.")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # === Compute EMAs ===
            df["ema_8"] = df["close"].ewm(span=8, adjust=False).mean()
            df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            if len(df) < 2:
                print(f"⚠️ Not enough EMA data for {symbol} — skipping.")
                continue

            # === Check last crossover ===
            fast_prev, fast_curr = df["ema_8"].iloc[-2], df["ema_8"].iloc[-1]
            slow_prev, slow_curr = df["ema_21"].iloc[-2], df["ema_21"].iloc[-1]

            should_exit = False
            exit_note = ""

            # BUY exit: fast EMA crosses below slow EMA
            if position_type == "buy" and fast_prev >= slow_prev and fast_curr < slow_curr:
                should_exit = True
                exit_note = "EMA(8) crossed below EMA(21) — exit BUY"

            # SELL exit: fast EMA crosses above slow EMA
            elif position_type == "sell" and fast_prev <= slow_prev and fast_curr > slow_curr:
                should_exit = True
                exit_note = "EMA(8) crossed above EMA(21) — exit SELL"

            status = "🚪 EXIT" if should_exit else "⏳ HOLD"
            print(f"🔹 {symbol}: {position_type.upper()} | EMA8={fast_curr:.5f} EMA21={slow_curr:.5f} | {status} — {exit_note}")

            if should_exit:
                trade_info = {
                    "symbol": symbol,
                    "ticket": pos.ticket,
                    "position": position_type,
                    "ema8": fast_curr,
                    "ema21": slow_curr,
                    "exit_reason": exit_note,
                }
                exit_trades.append(trade_info)

        # === Execute exits ===
        if exit_trades:
            print("\n⚙️ Executing EMA-based exits...\n")
            for trade in exit_trades:
                self.close_position(trade)
                print(f"✅ Closed {trade['symbol']} ({trade['position'].upper()}) | EMA8={trade['ema8']:.5f} EMA21={trade['ema21']:.5f}")
        else:
            print("\n✅ All positions still valid — no EMA cross exit yet.")

        return exit_trades


    def close_position(self, trade):
        """
        Close a specific MT5 position by sending a reverse market order.
        Args:
            trade (dict): Contains 'symbol', 'ticket', and 'position' keys
        """
        symbol = trade["symbol"]
        ticket = trade["ticket"]
        position_type = trade["position"]

        # Determine order type for closing
        order_type = (
            MetaTrader5.ORDER_TYPE_SELL
            if position_type == "buy"
            else MetaTrader5.ORDER_TYPE_BUY
        )

        # Retrieve position volume
        positions = MetaTrader5.positions_get(ticket=ticket)
        if not positions:
            print(f"⚠️ Position with ticket {ticket} not found (already closed?)")
            return

        volume = positions[0].volume

        # Prepare close request
        request = {
            "action": MetaTrader5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "deviation": 20,
            "magic": 123456,  # optional ID for your bot
            "comment": "RSI Exit Strategy",
            "type_filling": MetaTrader5.ORDER_FILLING_FOK,
        }

        # Try up to 3 times
        for attempt in range(3):
            result = MetaTrader5.order_send(request)
            if result and result.retcode == MetaTrader5.TRADE_RETCODE_DONE:
                print(f"✅ Closed {symbol} ({position_type.upper()}) successfully | Ticket={ticket}")
                return
            else:
                print(f"⚠️ Attempt {attempt+1} failed to close {symbol} | Code={getattr(result, 'retcode', 'N/A')}")
        
        print(f"❌ Failed to close {symbol} after 3 attempts.")


    
    # ------------------------------------------------------------
    # 🧩 RUNNER
    # ------------------------------------------------------------
    def run_strategy(self):
        """Select and execute the chosen strategy."""
        print(f"\n🚀 Running {self.strategy.upper()} strategy")
        signal = None

        try:
            # Dynamically get the method by its name
            strategy_func = getattr(self, self.strategy)
            signal = strategy_func()  # Call it
        except AttributeError:
            print(f"⚠️ Strategy '{self.strategy}' not found.")
            signal = None

        return signal


class MetaTraderConfig:

    def get_timeframe_duration(self, timeframe):
        mapping = {
            MetaTrader5.TIMEFRAME_M1:  60,
            MetaTrader5.TIMEFRAME_M2:  120,
            MetaTrader5.TIMEFRAME_M3:  180,
            MetaTrader5.TIMEFRAME_M4:  240,
            MetaTrader5.TIMEFRAME_M5:  300,
            MetaTrader5.TIMEFRAME_M15: 900,
            MetaTrader5.TIMEFRAME_M30: 1800,
            MetaTrader5.TIMEFRAME_H1:  3600,
            MetaTrader5.TIMEFRAME_H4:  14400,
            MetaTrader5.TIMEFRAME_D1:  86400,
        }
        return mapping.get(timeframe, 60)  # default 60 sec if not found

    def start_mt5(self, project_settings=None):
        """
        function to start MetaTrader 5
        param project settings: json object with username,pasword,server,file location
        return boolean true started
        """
        username = project_settings["mt5"]["username"]
        username = int(username)
        password = project_settings["mt5"]["password"]
        server = project_settings["mt5"]["server"]
        mt5_pathway = project_settings["mt5"]["mt5_pathway"]

        # Attempt to initialize Mt5
        mt5_init = False
        try:
            mt5_init = MetaTrader5.initialize(
                login=username, password=password, server=server, path=mt5_pathway
            )
        except Exception as e:
            print(f"error initializing MetaTrader 5: {e}")
            mt5_init = False

        self.mt5_init = mt5_init
        self.username = username
        self.password = password
        self.server = server
        self.login_mt5()

    def login_mt5(self):
        mt5_login = False
        if self.mt5_init:
            try:
                mt5_login = MetaTrader5.login(
                    login=self.username, password=self.password, server=self.server
                )
            except Exception as e:
                print(f"Error logging into MeteTrader 5: {e}")
                mt5_login = False

        if mt5_login:
            return True

    def get_market_data_date_range(
        self, timeframe=None, start_date=None, end_date=None, symbol=None, download=None
    ):
        """
        Fetch market data (historical candles) from MetaTrader 5.

        Args:
            timeframe (int, optional): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1).
                                       Defaults to mt5.TIMEFRAME_M1.
            start_date (datetime, optional): Start date for historical data.
                                             Defaults to the most recent 100 candles.
            end_date (datetime, optional): End date for historical data.
                                           Defaults to current UTC time.
            symbol (str, optional): Trading pair symbol (e.g., 'EURUSD').
                                    Defaults to self.symbol.

        Returns:
            pd.DataFrame: Historical OHLCV data with time converted to datetime.
        """
        MetaTrader5.symbol_select(symbol)
        rates = MetaTrader5.copy_rates_range(symbol, timeframe, start_date, end_date)
        data = pd.DataFrame(rates)
        data["time"] = pd.to_datetime(data["time"], unit="s")

        if download:
            self.export_data(
                data, filename=f"{symbol.lower()}_market_data", filetype=download
            )

        return data

    def get_market_data_rate(
        self, timeframe=None, rate=250, symbol=None, download=None
    ):
        """
        Fetch market data (historical candles) from MetaTrader 5.

        Args:
            timeframe (int, optional): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1).
                                    Defaults to mt5.TIMEFRAME_M1.
            start_date (datetime, optional): Start date for historical data.
                                            Defaults to the most recent 100 candles.
            end_date (datetime, optional): End date for historical data.
                                        Defaults to current UTC time.
            symbol (str, optional): Trading pair symbol (e.g., 'EURUSD').
                                    Defaults to self.symbol.

        Returns:
            pd.DataFrame: Historical OHLCV data with time converted to datetime.
        """
        rate = int(rate)
        MetaTrader5.symbol_select(symbol)
        rates = MetaTrader5.copy_rates_from_pos(symbol, timeframe, 0, rate)
        data = pd.DataFrame(rates)
        data["time"] = pd.to_datetime(data["time"], unit="s")
        if download:
            self.export_data(
                data, filename=f"{symbol.lower()}_market_data", filetype=download
            )
        return data

    def get_trade_history(self, start_time, end_time):
        """
        Retrieves closed trades (history) from MT5 in a given time range.
        SL/TP are not available in TradeDeal, so this version omits them.
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        all_data = []

        symbols = get_symbols().get("symbols")
        for sym in symbols:
            history = MetaTrader5.history_deals_get(start_time, end_time, group=sym)
            if history is None:
                continue

            for deal in history:
                if deal.profit != 0:
                    all_data.append(
                        {
                            "ticket": deal.ticket,
                            "symbol": deal.symbol,
                            "time": datetime.fromtimestamp(deal.time, tz=timezone.utc),
                            "type": "buy" if deal.type == 0 else "sell",
                            "volume": deal.volume,
                            "price": deal.price,
                            "profit": deal.profit,
                            "comment": "hit_sl" if "sl" in deal.comment else "hit_tp",
                        }
                    )

        if not all_data:
            print(
                "No closed trades found in this period. Try adjusting the date range or check broker history settings."
            )
            return pd.DataFrame()

        data = pd.DataFrame(all_data)
        self.export_data(data, filename="trade_history", filetype="csv")

    def export_data(self, data, filename="market_data", filetype="csv"):
        """
        Export pandas DataFrame to CSV or Excel.

        Args:
            data (pd.DataFrame): Data to export.
            filename (str): Desired file name without extension.
            filetype (str): 'csv' or 'xlsx' (default = 'csv').

        Returns:
            str: Full path to the saved file.
        """
        if data is None or data.empty:
            print("⚠️ No data to export.")
            return None

        # Create export directory if not exists
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)

        # Clean filename
        filename = filename.replace(" ", "_")

        # Determine full file path
        if filetype.lower() == "xlsx":
            filepath = os.path.join(export_dir, f"{filename}.xlsx")
            data.to_excel(filepath, index=False)
        else:
            filepath = os.path.join(export_dir, f"{filename}.csv")
            data.to_csv(filepath, index=False)

        print(f"✅ File saved successfully: {filepath}")
        return filepath

    def can_trade_symbol(self, symbol, cooldown_minutes=1):
        """
        Determines if a symbol can be traded based on:
        1. No open positions for the symbol
        2. Enough time passed since the last closed trade
        """
        # 1️⃣ Check for open positions
        open_positions = MetaTrader5.positions_get()
        open_symbols = (
            {pos.symbol for pos in open_positions} if open_positions else set()
        )

        if symbol in open_symbols:
            print(f"⚠️ {symbol} already has an open trade — skipping.")
            return False

        # 2️⃣ Check for recent closed trades (cooldown logic)
        now = datetime.now()
        from_time = now - timedelta(days=1)  # fetch last 24h history
        closed_trades = MetaTrader5.history_deals_get(from_time, now, group=symbol)

        if closed_trades:
            last_close_time = max(
                deal.time
                for deal in closed_trades
                if deal.entry == MetaTrader5.DEAL_ENTRY_OUT
            )
            time_diff = now - datetime.fromtimestamp(last_close_time)
            if time_diff < timedelta(minutes=cooldown_minutes):
                print(
                    f"⏳ {symbol} closed {int(time_diff.total_seconds() // 60)} min ago — cooling down."
                )
                return False

        print(f"✅ {symbol} is free to trade.")
        return True

    def execute_trade(
        self,
        symbol,
        signal=None,
        lot=0.01,
        deviation=10,
        sl_pips=None,
        tp_pips=None,
        strategy="rsi_strategy",
    ):
        """Executes a buy or sell order in MetaTrader 5."""
        can_trade = self.can_trade_symbol(symbol, cooldown_minutes=20)

        if not can_trade:
            return

        if signal not in ["buy", "sell"]:
            print(f"⚙️ No trade action for {symbol}")
            return

        tick = MetaTrader5.symbol_info_tick(symbol)
        if tick is None:
            print(f"⚠️ Failed to get tick for {symbol}")
            return

        order_type = (
            MetaTrader5.ORDER_TYPE_BUY
            if signal == "buy"
            else MetaTrader5.ORDER_TYPE_SELL
        )
        price = tick.ask if signal == "buy" else tick.bid
        trade_risk = self.calculate_trade_risk(
            symbol, signal, price, lot=lot, sl_pips=sl_pips, tp_pips=tp_pips
        )
        sl = trade_risk.get("sl_price")
        tp = trade_risk.get("tp_price")
        print(sl, tp, price)

        request = {
            "action": MetaTrader5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": 123456,
            "comment": f"{strategy} auto-bot",
            "type_time": MetaTrader5.ORDER_TIME_GTC,
            "type_filling": MetaTrader5.ORDER_FILLING_IOC,
            "tp": tp,
            "sl": sl,
        }

        result = MetaTrader5.order_send(request)
        if result.retcode != MetaTrader5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed for {symbol}: {result.comment}")
        else:
            print(f"✅ Trade executed for {symbol}: {signal.upper()} at {price}")

    def run_trading_loop(self, symbols, timeframe=MetaTrader5.TIMEFRAME_M5, trail=True):
        """Continuously fetch data, apply strategy, and trade."""
        delay = self.get_timeframe_duration(timeframe=timeframe)

        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if trail:
                self.update_trailing_stop()
            print(f"\n🔄 Checking markets...{now}")
            

            for symbol in symbols:
                print(symbol)

                data_H1 = self.get_market_data_rate(
                    symbol=symbol, timeframe=MetaTrader5.TIMEFRAME_H1
                )
                data_M5 = self.get_market_data_rate(
                    symbol=symbol, timeframe=MetaTrader5.TIMEFRAME_M5
                )

                data = {"data_H1": data_H1, "data_M5": data_M5}
                strategy = "macd_strategy"  # or "rsi_strategy", "hybrid"
                trading_strategy = TradingStrategy(data=data, strategy=strategy,symbol=symbol)
                response = trading_strategy.run_strategy()
                if response is None:
                    print(f"{symbol}: No response from strategy.")
                    continue
                signal = response.get("signal") 
                sl_pips = response.get("sl_pips") 
                tp_pips = response.get("tp_pips") 

                if signal:
                    print(f"{symbol}: Signal = {signal.upper()}")
                    self.execute_trade(
                        symbol,
                        signal,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        strategy=strategy,
                    )
                else:
                    print(f"{symbol}: No trade signal.")

            account_info = MetaTrader5.account_info()
            print("💰 Balance:", f"${account_info.balance}")

            print(f"🕐 Delaying loop for {delay} seconds ({timeframe}) timeframe...\n")
            time.sleep(delay)

    def calculate_trade_risk(
        self, symbol, signal, entry_price, lot=0.01, sl_pips=None, tp_pips=None
    ):
        """
        Calculates SL/TP prices and potential gain/loss in USD, safely for FX, metals, and crypto.

        sl_pips / tp_pips are in pips (for FX) or price units (for non-FX).

        Returns:
            {
                "sl_price": ...,
                "tp_price": ...,
                "potential_loss_usd": ...,
                "potential_gain_usd": ...
            }
        """
        info = MetaTrader5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol info not found for {symbol}")

        point = info.point
        contract_size = info.trade_contract_size
        tick_size = info.trade_tick_size
        min_stop = info.trade_stops_level * point

        # --- Detect pip size automatically ---
        # Forex (EURUSD, GBPUSD) usually 0.0001, JPY pairs 0.01
        if symbol.endswith(("USD", "USDm")) and point == 0.00001:
            pip_size = 0.0001
        elif "JPY" in symbol:
            pip_size = 0.01
        else:
            pip_size = point  # metals/crypto use price unit as pip

        # --- Convert pips to price distance ---
        if sl_pips is not None:
            sl_distance = sl_pips * pip_size
        else:
            sl_distance = min_stop * 2

        if tp_pips is not None:
            tp_distance = tp_pips * pip_size
        else:
            tp_distance = sl_distance * 2  # 2:1 R:R

        # --- SL/TP based on signal ---
        if signal.lower() == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        elif signal.lower() == "sell":
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        else:
            raise ValueError("Signal must be 'buy' or 'sell'")

        # --- Round to tick size ---
        sl_price = round(sl_price / tick_size) * tick_size
        tp_price = round(tp_price / tick_size) * tick_size

        # --- USD risk/gain estimation ---
        pip_value = contract_size * pip_size * lot
        potential_loss_usd = abs(entry_price - sl_price) / pip_size * pip_value
        potential_gain_usd = abs(tp_price - entry_price) / pip_size * pip_value

        return {
            "sl_price": sl_price,
            "tp_price": tp_price,
            "potential_loss_usd": round(potential_loss_usd, 2),
            "potential_gain_usd": round(potential_gain_usd, 2),
        }

    def update_trailing_stop(self, deviation=10):
        """
        Dynamically updates SL for all open trades.

        distance_pips: minimum distance (in pips) between current price and SL before moving
        move_pips: how many pips to move SL forward each time
        """
        print("\n🔧 Updating SL of active trades...")

        positions = MetaTrader5.positions_get()
        if not positions:
            print("⚠️ No active positions found")
            return

        for pos in positions:
            symbol = pos.symbol
            settings = symbols_dict[symbol]
            trail_start = settings["trail_start"]
            trail_step = settings["trail_step"]

            tick = MetaTrader5.symbol_info_tick(symbol)
            info = MetaTrader5.symbol_info(symbol)

            if tick is None or info is None:
                continue

            current_price = (
                tick.bid if pos.type == MetaTrader5.ORDER_TYPE_BUY else tick.ask
            )
            point = info.point
            move_distance = trail_step * 10 * point  # convert pips → price units

            # ---------------- BUY TRADE ----------------
            if pos.type == MetaTrader5.ORDER_TYPE_BUY:
                if pos.sl is None:
                    sl_distance_pips = float("inf")
                else:
                    sl_distance_pips = (current_price - pos.sl) / (10 * point)

                print(
                    f"BUY {symbol} | SL distance = {sl_distance_pips:.1f} pips | "
                    f"SL={pos.sl} | Price={current_price}"
                )

                if sl_distance_pips >= trail_start:
                    new_sl = (pos.sl or current_price) + move_distance  # move SL upward
                    print(
                        f"➡️ Moving BUY SL from {pos.sl} → {new_sl} ({trail_step} pips)"
                    )

                    if pos.sl is None or new_sl > pos.sl:
                        request = {
                            "action": MetaTrader5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": pos.tp,
                            "deviation": deviation,
                            "comment": "Trailing stop update",
                        }
                        result = MetaTrader5.order_send(request)
                        if result.retcode == MetaTrader5.TRADE_RETCODE_DONE:
                            print(f"✅ BUY {symbol}: SL moved up to {new_sl}")
                        else:
                            print(
                                f"❌ Failed to modify SL for BUY {symbol}: {result.comment}"
                            )

            # ---------------- SELL TRADE ----------------
            elif pos.type == MetaTrader5.ORDER_TYPE_SELL:
                if pos.sl is None:
                    sl_distance_pips = float("inf")
                else:
                    sl_distance_pips = (pos.sl - current_price) / (10 * point)

                print(
                    f"SELL {symbol} | SL distance = {sl_distance_pips:.1f} pips | "
                    f"SL={pos.sl} | Price={current_price}"
                )

                if sl_distance_pips >= trail_start:
                    new_sl = (
                        pos.sl or current_price
                    ) - move_distance  # move SL downward
                    print(
                        f"➡️ Moving SELL SL from {pos.sl} → {new_sl} ({trail_step} pips)"
                    )

                    if pos.sl is None or new_sl < pos.sl:
                        request = {
                            "action": MetaTrader5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": pos.tp,
                            "deviation": deviation,
                            "comment": "Trailing stop update",
                        }
                        result = MetaTrader5.order_send(request)
                        if result.retcode == MetaTrader5.TRADE_RETCODE_DONE:
                            print(f"✅ SELL {symbol}: SL moved down to {new_sl}")
                        else:
                            print(
                                f"❌ Failed to modify SL for SELL {symbol}: {result.comment}"
                            )
