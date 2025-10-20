from ta.momentum import RSIIndicator
import pandas as pd
import MetaTrader5
import os, time


class TradingStrategy:
    def __init__(self, data, strategy = 'moving_average'):
        self.data = data
        self.strategy = strategy

    def rsi_strategy(self):
        """Simple RSI strategy that generates buy/sell signals."""

        data = self.data
        if data.empty:
            return None

        data["rsi"] = RSIIndicator(data["close"], window=14).rsi()
        last = data.iloc[-1]
        rsi = last["rsi"]
        print(rsi)

        if rsi < 30:
            return "buy"
        elif rsi > 70:
            return "sell"
        else:
            return None

    def moving_average_strategy(self, fast_window=30, slow_window=100):
        """Simple Moving Average crossover strategy (with debug prints)."""
        data = self.data
        if data.empty:
            return None

        # Compute moving averages
        data["ma_fast"] = data["close"].rolling(window=fast_window).mean()
        data["ma_slow"] = data["close"].rolling(window=slow_window).mean()

        last = data.iloc[-1]
        prev = data.iloc[-2]

        # ✅ Debug prints
        print(f"MA Fast ({fast_window}): {last['ma_fast']:.5f}")
        print(f"MA Slow ({slow_window}): {last['ma_slow']:.5f}")
        

        # Detect crossover
        if prev["ma_fast"] < prev["ma_slow"] and last["ma_fast"] > last["ma_slow"]:
            print("📈 Crossover detected: BUY signal")
            return "buy"
        elif prev["ma_fast"] > prev["ma_slow"] and last["ma_fast"] < last["ma_slow"]:
            print("📉 Crossover detected: SELL signal")
            return "sell"
        else:
            print("⏸ No crossover signal")
            return None

    def run_strategy(self):
        """Run whichever strategy is selected and print which one."""
        print(f"\n🚀 Running strategy: {self.strategy}")

        if self.strategy == 'rsi':
            return self.rsi_strategy()
        elif self.strategy == 'moving_average':
            return self.moving_average_strategy()
        else:
            print(f"⚠️ Unknown strategy '{self.strategy}'")
            return None

class MetaTraderConfig:

    def __init__(self, project_settings):
        self.project_settings = project_settings

    def start_mt5(self):
        """
        function to start MetaTrader 5
        param project settings: json object with username,pasword,server,file location
        return boolean true started
        """
        project_settings = self.project_settings
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

    def execute_trade(self, symbol, signal=None, lot=0.01, deviation=10,sl_pips=None, tp_pips=None):
        """Executes a buy or sell order in MetaTrader 5."""
        # Get all open trades
        open_positions = MetaTrader5.positions_get()

        # Extract symbols currently open
        open_symbols = {pos.symbol for pos in open_positions} if open_positions else set()

        print("Currently open symbols:", open_symbols)

        if symbol in open_symbols:
            print(f"⚠️ {symbol} already has an open trade")
            return 
        else:
            print(f"✅ Can trade {symbol}")

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
        trade_risk = self.calculate_trade_risk(symbol, signal, price, lot=lot, sl_pips=sl_pips, tp_pips=tp_pips)
        sl = trade_risk.get("sl_price")
        tp = trade_risk.get("tp_price")
        print(sl,tp,price)

        request = {
            "action": MetaTrader5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": 123456,
            "comment": "RSI auto-bot",
            "type_time": MetaTrader5.ORDER_TIME_GTC,
            "type_filling": MetaTrader5.ORDER_FILLING_IOC,
            "tp":tp,
            "sl":sl
        }

        result = MetaTrader5.order_send(request)
        if result.retcode != MetaTrader5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed for {symbol}: {result.comment}")
        else:
            print(f"✅ Trade executed for {symbol}: {signal.upper()} at {price}")

    def run_trading_loop(self, symbols, timeframe=MetaTrader5.TIMEFRAME_M5, delay=50):
        """Continuously fetch data, apply strategy, and trade."""
        while True:
            
            
            print("\n🔧 Updating SL of active trades...")
            self.update_trailing_stop()
            print("\n🔄 Checking markets...")

            for symbol in symbols:
                data = self.get_market_data_rate(symbol=symbol, timeframe=timeframe)
                trading_strategy = TradingStrategy(data=data,strategy="rsi")

                signal = trading_strategy.run_strategy()

                if signal:
                    print(f"{symbol}: Signal = {signal.upper()}")
                    self.execute_trade(symbol, signal,sl_pips=10,tp_pips=40)
                else:
                    print(f"{symbol}: No trade signal.")

            time.sleep(delay)


    def calculate_trade_risk(self, symbol, signal, entry_price, lot=0.01, sl_pips=None, tp_pips=None):
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

    
    def update_trailing_stop(self, distance_pips=25, move_pips=10, deviation=10):
        """
        Dynamically updates SL for all open trades.

        distance_pips: minimum distance (in pips) between current price and SL before moving
        move_pips: how many pips to move SL forward each time
        """
        positions = MetaTrader5.positions_get()
        if not positions:
            print("⚠️ No active positions found")
            return

        for pos in positions:
            symbol = pos.symbol
            tick = MetaTrader5.symbol_info_tick(symbol)
            info = MetaTrader5.symbol_info(symbol)

            if tick is None or info is None:
                continue

            current_price = tick.bid if pos.type == MetaTrader5.ORDER_TYPE_BUY else tick.ask
            point = info.point
            move_distance = move_pips * 10 * point  # convert pips → price units

            # ---------------- BUY TRADE ----------------
            if pos.type == MetaTrader5.ORDER_TYPE_BUY:
                if pos.sl is None:
                    sl_distance_pips = float('inf')
                else:
                    sl_distance_pips = (current_price - pos.sl) / (10 * point)

                print(f"BUY {symbol} | SL distance = {sl_distance_pips:.1f} pips | "
                    f"SL={pos.sl} | Price={current_price}")

                if sl_distance_pips >= distance_pips:
                    new_sl = (pos.sl or current_price) + move_distance  # move SL upward
                    print(f"➡️ Moving BUY SL from {pos.sl} → {new_sl} ({move_pips} pips)")

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
                            print(f"❌ Failed to modify SL for BUY {symbol}: {result.comment}")

            # ---------------- SELL TRADE ----------------
            elif pos.type == MetaTrader5.ORDER_TYPE_SELL:
                if pos.sl is None:
                    sl_distance_pips = float('inf')
                else:
                    sl_distance_pips = (pos.sl - current_price) / (10 * point)

                print(f"SELL {symbol} | SL distance = {sl_distance_pips:.1f} pips | "
                    f"SL={pos.sl} | Price={current_price}")

                if sl_distance_pips >= distance_pips:
                    new_sl = (pos.sl or current_price) - move_distance  # move SL downward
                    print(f"➡️ Moving SELL SL from {pos.sl} → {new_sl} ({move_pips} pips)")

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
                            print(f"❌ Failed to modify SL for SELL {symbol}: {result.comment}")
