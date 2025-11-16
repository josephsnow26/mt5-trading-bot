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
    def __init__(self, data = None, balance=10000, risk_per_trade=0.02, strategy="hybrid"):
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
  

    def rsi_strategy(self):
        """
        RSI-based trading strategy:
        - Determines overall trend from H1 using SMA(100)
        - Uses RSI on M5 for entry signals
        """
        # === Compute & prepare indicators ===
        self.compute_indicators()

        data_H1 = self.data.get("data_H1", pd.DataFrame()).copy()
        data_M5 = self.data.get("data_M5", pd.DataFrame()).copy()

        if data_H1.empty or data_M5.empty:
            print("⚠️ Insufficient data for RSI strategy.")
            return None

        # === 1️⃣ Identify trend on H1 ===
        last_h1 = data_H1.iloc[-1]
        trend = "bullish" if last_h1["close"] > last_h1["sma_100"] else "bearish"

        # === 2️⃣ Get RSI from M5 ===
        last_m5 = data_M5.iloc[-1]
        rsi = last_m5["rsi"]

        # === 3️⃣ Determine signal ===
        if rsi < 30 and trend == "bullish":
            signal = "buy"
        elif rsi > 70 and trend == "bearish":
            signal = "sell"
        else:
            signal = None

        # === 4️⃣ Print diagnostic info ===
        print("\n📊 RSI STRATEGY DIAGNOSTICS")
        print(f"📅 Time (M5): {last_m5.name if hasattr(last_m5, 'name') else 'N/A'}")
        print(f"Trend (H1): {trend.upper()}")
        print(f"RSI (M5): {rsi:.2f}")
        # print(f"H1 Close: {last_h1['close']:.5f}")
        # print(f"SMA(100): {last_h1['sma_100']:.5f}")
        print(f"Signal: {'✅ ' + signal.upper() if signal else '❌ No clear signal'}")

        return signal

    def rsi_exit_strategy(self, timeframe=MetaTrader5.TIMEFRAME_M5, rsi_period=14):
        """
        RSI-based exit strategy (independent of self.data)
        - BUY: Exit when RSI > 70 and starts decreasing
        - SELL: Exit when RSI < 30 and starts increasing
        - Fetches live M5 data per symbol for accurate RSI confirmation
        """
        import pandas as pd
        from ta.momentum import RSIIndicator

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



    def compute_retest_strategy(self, use_sma=200):
        """
        Retest strategy (returns only the latest valid signal):
        - H1 for trend direction
        - M5 for execution with retest confirmation
        """

        data_H1 = self.data["data_H1"].copy()
        data_M5 = self.data["data_M5"].copy()

        # === 1️⃣ Identify trend on H1 ===
        data_H1["sma_100"] = SMAIndicator(data_H1["close"], window=100).sma_indicator()
        last_h1 = data_H1.iloc[-1]
        trend = "bullish" if last_h1["close"] > last_h1["sma_100"] else "bearish"
        print(f"[H1] Last Close: {last_h1['close']:.5f}, SMA100: {last_h1['sma_100']:.5f}, Trend: {trend}")

        # === 2️⃣ Compute M5 indicators ===
        data_M5["sma_20"] = SMAIndicator(data_M5["close"], window=20).sma_indicator()
        data_M5["sma_50"] = SMAIndicator(data_M5["close"], window=50).sma_indicator()
        data_M5[f"sma_100"] = SMAIndicator(data_M5["close"], window=100).sma_indicator()


        # Detect SMA crossovers
        data_M5["prev_sma20"] = data_M5["sma_20"].shift(1)
        data_M5["prev_sma50"] = data_M5["sma_50"].shift(1)

        latest_signal = None

        for i in range(1, len(data_M5)):
            row = data_M5.iloc[i]
            prev = data_M5.iloc[i - 1]
            sma_main = row[f"sma_100"]

            bullish_cross = (prev["sma_20"] < prev["sma_50"]) and (row["sma_20"] > row["sma_50"])
            bearish_cross = (prev["sma_20"] > prev["sma_50"]) and (row["sma_20"] < row["sma_50"])

            # print(f"[M5] {data_M5.index[i]} | Close: {row['close']:.5f} | SMA20: {row['sma_20']:.5f} | SMA50: {row['sma_50']:.5f} | SMA{use_sma}: {sma_main:.5f}")
            # print(f"     Bullish Cross: {bullish_cross}, Bearish Cross: {bearish_cross}")

            # === BUY ===
            if trend == "bullish" and bullish_cross:
                if row["close"] < sma_main:
                    # print(f"     ✅ BUY signal possible at {row['close']:.5f} (retest below SMA{use_sma})")
                    latest_signal = {
                        "timestamp": data_M5.index[i],
                        "type": "BUY",
                        "price": row["close"],
                        "trend": trend,
                        "note": f"Retest confirmed below SMA{use_sma}",
                    }

            # === SELL ===
            elif trend == "bearish" and bearish_cross:
                if row["close"] > sma_main:
                    # print(f"     ✅ SELL signal possible at {row['close']:.5f} (retest above SMA{use_sma})")
                    latest_signal = {
                        "timestamp": data_M5.index[i],
                        "type": "SELL",
                        "price": row["close"],
                        "trend": trend,
                        "note": f"Retest confirmed above SMA{use_sma}",
                    }

        if latest_signal:
            print(f"=== Latest Signal: {latest_signal['type']} at {latest_signal['price']:.5f} ===")
            return latest_signal.get("type").lower()
        else:
            print("=== No valid signal found ===")
            return None


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

    def run_trading_loop(self, symbols, timeframe=MetaTrader5.TIMEFRAME_M5, trail=False):
        """Continuously fetch data, apply strategy, and trade."""
        delay = self.get_timeframe_duration(timeframe=timeframe)

        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if trail:
                self.update_trailing_stop()
            print(f"\n🔄 Checking markets...{now}")
            trading_exit_strategy = TradingStrategy()
            trading_exit_strategy.rsi_exit_strategy()

            for symbol in symbols:
                settings = symbols_dict[symbol]
                sl_pips = settings["sl_pips"]
                tp_pips = settings["tp_pips"]

                data_H1 = self.get_market_data_rate(
                    symbol=symbol, timeframe=MetaTrader5.TIMEFRAME_H1
                )
                data_M5 = self.get_market_data_rate(
                    symbol=symbol, timeframe=MetaTrader5.TIMEFRAME_M5
                )

                data = {"data_H1": data_H1, "data_M5": data_M5}
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
