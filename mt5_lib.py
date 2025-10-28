import MetaTrader5
import os, time
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta,timezone
from symbols import get_symbols


symbols_dict = get_symbols().get("symbols_dict")


class TradingStrategy:
    def __init__(self, data, balance=10000, risk_per_trade=0.02, strategy="hybrid"):
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

    # ------------------------------------------------------------
    # 🧠 INDICATOR CALCULATIONS
    # ------------------------------------------------------------
    def compute_indicators(self):
        """Compute all indicators needed for decision."""
        df = self.data.copy()
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        df["ema_fast"] = EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["sma_trend"] = df["close"].rolling(window=200).mean()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        return df

    def compute_scalping_indicators(self):
        """Compute fast indicators for scalping."""
        df = self.data.copy()
        df["ema_fast"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["close"], window=21).ema_indicator()
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        return df

    # ------------------------------------------------------------
    # ⚙️ STRATEGIES
    # ------------------------------------------------------------

    def trend_momentum_strategy(self):
        """
        Simple but powerful ensemble:
        - EMA crossover for direction
        - RSI confirms momentum
        - MACD confirms trend
        - Price above/below SMA200 confirms major bias
        - Bollinger breakout filters fake signals
        """
        df = self.compute_indicators()
        if len(df) < 3:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # EMA crossover
        bullish_cross = (
            prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        )
        bearish_cross = (
            prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        )

        # RSI
        rsi = last["rsi"]

        # MACD
        macd_up = last["macd"] > last["macd_signal"]
        macd_down = last["macd"] < last["macd_signal"]

        # Price vs long-term SMA
        above_trend = last["close"] > last["sma_trend"]
        below_trend = last["close"] < last["sma_trend"]

        # Bollinger filter
        breakout_up = last["close"] > last["bb_mid"]
        breakout_down = last["close"] < last["bb_mid"]

        # Combine confirmations
        if bullish_cross and rsi < 60 and macd_up and above_trend and breakout_up:
            return "buy"
        elif bearish_cross and rsi > 40 and macd_down and below_trend and breakout_down:
            return "sell"
        else:
            return None

    def scalping_strategy(self):
        """
        Fast scalping setup using EMA + RSI + Bollinger bounce.
        Works best on M1–M15 with tight SL/TP.
        """
        df = self.compute_scalping_indicators()
        if len(df) < 3:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # EMA crossover
        bullish_cross = (
            prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        )
        bearish_cross = (
            prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        )

        # RSI and Bollinger
        rsi = last["rsi"]
        close = last["close"]
        bb_upper = last["bb_upper"]
        bb_lower = last["bb_lower"]

        # BUY condition
        if close < bb_lower and rsi < 35 and bullish_cross:
            return "buy"

        # SELL condition
        elif close > bb_upper and rsi > 65 and bearish_cross:
            return "sell"

        else:
            return None

    def rsi_strategy(self):
        data = self.compute_indicators()
        last = data.iloc[-1]
        rsi = last["rsi"]

        print(f"RSI: {rsi:.2f}")
        if rsi < 30:
            return "buy"
        elif rsi > 70:
            return "sell"
        else:
            return None

    def moving_average_strategy(self):
        data = self.compute_indicators()
        last = data.iloc[-1]
        prev = data.iloc[-2]

        print(f"EMA Fast: {last['ema_fast']:.5f} | EMA Slow: {last['ema_slow']:.5f}")

        if prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]:
            return "buy"
        elif (
            prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        ):
            return "sell"
        return None

    def hybrid_strategy(self):
        """
        Combines RSI, EMA crossover, and MACD confirmation.
        """
        data = self.compute_indicators()
        last = data.iloc[-1]
        prev = data.iloc[-2]

        signal = None

        # EMA crossover logic
        bullish_crossover = (
            prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        )
        bearish_crossover = (
            prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        )

        # RSI confirmation
        rsi = last["rsi"]
        macd_cross_up = last["macd"] > last["signal"]
        macd_cross_down = last["macd"] < last["signal"]

        if bullish_crossover and rsi < 60 and macd_cross_up:
            signal = "buy"
        elif bearish_crossover and rsi > 40 and macd_cross_down:
            signal = "sell"

        if signal:
            print(
                f"✅ Hybrid signal: {signal.upper()} | RSI={rsi:.2f}, MACD={last['macd']:.4f}"
            )
        else:
            print("⏸ No confirmed signal yet.")

        return signal

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
            MetaTrader5.TIMEFRAME_M1: 60,
            MetaTrader5.TIMEFRAME_M2: 120,
            MetaTrader5.TIMEFRAME_M3: 180,
            MetaTrader5.TIMEFRAME_M4: 240,
            MetaTrader5.TIMEFRAME_M5: 300,
            MetaTrader5.TIMEFRAME_M15: 900,
            MetaTrader5.TIMEFRAME_M30: 1800,
            MetaTrader5.TIMEFRAME_H1: 3600,
            MetaTrader5.TIMEFRAME_H4: 14400,
            MetaTrader5.TIMEFRAME_D1: 86400,
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
        self, timeframe=None, rate=120, symbol=None, download=None
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
                    all_data.append({
                        "ticket": deal.ticket,
                        "symbol": deal.symbol,
                        "time": datetime.fromtimestamp(deal.time, tz=timezone.utc),
                        "type": "buy" if deal.type==0 else "sell",
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": deal.profit,
                        "comment": "hit_sl" if "sl" in deal.comment else "hit_tp"
                    })

        if not all_data:
            print("No closed trades found in this period. Try adjusting the date range or check broker history settings.")
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

    def can_trade_symbol(self, symbol, cooldown_minutes=15):
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

    def run_trading_loop(
        self, symbols, timeframe=MetaTrader5.TIMEFRAME_M5, trail=False
    ):
        """Continuously fetch data, apply strategy, and trade."""
        delay = self.get_timeframe_duration(timeframe=timeframe)

        while True:
            if trail:
                self.update_trailing_stop()
            print("\n🔄 Checking markets...")

            for symbol in symbols:
                settings = symbols_dict[symbol]
                sl_pips = settings["sl_pips"]
                tp_pips = settings["tp_pips"]

                data = self.get_market_data_rate(symbol=symbol, timeframe=timeframe)
                strategy = "rsi_strategy"
                trading_strategy = TradingStrategy(data=data, strategy=strategy)

                signal = trading_strategy.run_strategy()

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
