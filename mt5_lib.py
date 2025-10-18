from ta.momentum import RSIIndicator
import pandas as pd
import MetaTrader5
import os, time


class TradingStrategy:
    def __init__(self, data):
        self.data = data

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
        self, timeframe=None, rate=100, symbol=None, download=None
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

    def run_trading_loop(self, symbols, timeframe=MetaTrader5.TIMEFRAME_M5, delay=60):
        """Continuously fetch data, apply strategy, and trade."""
        while True:
            
            
            print("\n🔧 Updating SL of active trades...")
            self.update_trailing_stop()
            print("\n🔄 Checking markets...")

            for symbol in symbols:
                data = self.get_market_data_rate(symbol=symbol, timeframe=timeframe)
                trading_strategy = TradingStrategy(data=data)

                signal = trading_strategy.rsi_strategy()

                if signal:
                    print(f"{symbol}: Signal = {signal.upper()}")
                    self.execute_trade(symbol, signal)
                else:
                    print(f"{symbol}: No trade signal.")

            time.sleep(delay)


    def calculate_trade_risk(self, symbol, signal, entry_price, lot=0.01, sl_pips=None, tp_pips=None):
        """
        Calculates Stop Loss and Take Profit prices and potential gain/loss in USD dynamically.
        
        Returns a dict:
            {
                "sl_price": ...,
                "tp_price": ...,
                "potential_loss_usd": ...,
                "potential_gain_usd": ...
            }
        """
        import MetaTrader5 as mt5

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol info not found for {symbol}")

        point = symbol_info.point

        # Dynamic pip value calculation for Forex (assuming USD account)
        # Standard lot = 100,000 units, 1 pip = 0.0001 for most pairs except JPY
        if symbol.endswith("JPY"):
            pip_value_per_standard_lot = 1000  # for JPY pairs
        else:
            pip_value_per_standard_lot = 10  # for most USD pairs

        pip_value = pip_value_per_standard_lot * (lot / 1.0)  # adjust for lot size

        # Calculate absolute SL and TP prices
        if signal.lower() == "buy":
            sl_price = entry_price - sl_pips * point if sl_pips is not None else None
            tp_price = entry_price + tp_pips * point if tp_pips is not None else None
        elif signal.lower() == "sell":
            sl_price = entry_price + sl_pips * point if sl_pips is not None else None
            tp_price = entry_price - tp_pips * point if tp_pips is not None else None
        else:
            raise ValueError("Signal must be 'buy' or 'sell'")

        # Calculate potential gain/loss in USD
        potential_loss_usd = (abs(entry_price - sl_price) / point * pip_value) if sl_price else None
        potential_gain_usd = (abs(tp_price - entry_price) / point * pip_value) if tp_price else None

        return {
            "sl_price": sl_price,
            "tp_price": tp_price,
            "potential_loss_usd": round(potential_loss_usd, 2) if potential_loss_usd else None,
            "potential_gain_usd": round(potential_gain_usd, 2) if potential_gain_usd else None,
        }

    
    def update_trailing_stop(self, distance_pips=20, move_pips=5, deviation=10):
        """
        Updates SL for all open trades dynamically.
        
        distance_pips: minimum number of pips current price must be away from SL before moving it
        move_pips: number of pips to move the SL forward
        """
        positions = MetaTrader5.positions_get()
        if not positions:
            print("⚠️ No active positions found")
            return

        for pos in positions:
            symbol = pos.symbol
            point_multiplier = 0.0001 if not symbol.endswith("JPY") else 0.01

            tick = MetaTrader5.symbol_info_tick(symbol)
            if tick is None:
                continue
            current_price = tick.bid if pos.type == MetaTrader5.ORDER_TYPE_BUY else tick.ask

            # BUY trade
            if pos.type == MetaTrader5.ORDER_TYPE_BUY:
                if pos.sl is None:
                    sl_distance = float('inf')  # force first SL setting
                else:
                    sl_distance = current_price - pos.sl  # distance from current price to SL

                if sl_distance >= distance_pips * point_multiplier:
                    # Move SL forward by move_pips
                    new_sl = (pos.sl or current_price) + move_pips * point_multiplier
                    if new_sl > (pos.sl or 0):
                        result = MetaTrader5.order_modify(
                            ticket=pos.ticket,
                            price=pos.price_open,
                            sl=new_sl,
                            tp=pos.tp,
                            deviation=deviation
                        )
                        if result.retcode == MetaTrader5.TRADE_RETCODE_DONE:
                            print(f"✅ Trailing SL moved up for BUY {symbol} to {new_sl}")
                        else:
                            print(f"❌ Failed to modify SL for BUY {symbol}: {result.comment}")

            # SELL trade
            elif pos.type == MetaTrader5.ORDER_TYPE_SELL:
                if pos.sl is None:
                    sl_distance = float('inf')
                else:
                    sl_distance = pos.sl - current_price  # distance from SL to current price

                if sl_distance >= distance_pips * point_multiplier:
                    new_sl = (pos.sl or current_price) - move_pips * point_multiplier
                    if new_sl < (pos.sl or float('inf')):
                        result = MetaTrader5.order_modify(
                            ticket=pos.ticket,
                            price=pos.price_open,
                            sl=new_sl,
                            tp=pos.tp,
                            deviation=deviation
                        )
                        if result.retcode == MetaTrader5.TRADE_RETCODE_DONE:
                            print(f"✅ Trailing SL moved down for SELL {symbol} to {new_sl}")
                        else:
                            print(f"❌ Failed to modify SL for SELL {symbol}: {result.comment}")
