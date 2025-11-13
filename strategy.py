import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands


class TradingNewStrategy:
    def __init__(self, data, balance=10, risk_per_trade=0.02, strategy="price_action_ma"):
        """
        balance: initial account balance
        risk_per_trade: percentage of balance risked per trade
        strategy: which logic to use ('price_action_ma')
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
        
        # Moving Averages
        df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["sma_200"] = SMAIndicator(df["close"], window=200).sma_indicator()
        
        # RSI
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        
        # MACD (optional for additional confirmation)
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        
        # Bollinger Bands (optional for volatility check)
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        
        return df

    # ------------------------------------------------------------
    # 📊 STEP 1: IDENTIFY TREND
    # ------------------------------------------------------------
    def identify_trend(self, df, idx):
        """
        Determine if market is in uptrend, downtrend, or sideways.
        
        Returns: 'uptrend', 'downtrend', or 'sideways'
        """
        current_price = df["close"].iloc[idx]
        ema_20 = df["ema_20"].iloc[idx]
        ema_50 = df["ema_50"].iloc[idx]
        sma_200 = df["sma_200"].iloc[idx]
        rsi = df["rsi"].iloc[idx]
        
        # Need enough data
        if pd.isna(sma_200) or pd.isna(ema_20) or pd.isna(ema_50):
            return 'sideways'
        
        # UPTREND CONDITIONS
        uptrend_conditions = [
            current_price > sma_200,  # Price above 200 SMA
            ema_20 > ema_50,          # Fast EMA above slow EMA
            rsi > 50                  # RSI above 50
        ]
        
        # DOWNTREND CONDITIONS
        downtrend_conditions = [
            current_price < sma_200,  # Price below 200 SMA
            ema_20 < ema_50,          # Fast EMA below slow EMA
            rsi < 50                  # RSI below 50
        ]
        
        if all(uptrend_conditions):
            return 'uptrend'
        elif all(downtrend_conditions):
            return 'downtrend'
        else:
            return 'sideways'

    # ------------------------------------------------------------
    # 📉 STEP 2: DETECT PULLBACK TO EMA
    # ------------------------------------------------------------
    def check_pullback(self, df, idx, trend):
        """
        Check if price has pulled back to EMA.
        
        Returns: True if pullback detected, False otherwise
        """
        current_price = df["close"].iloc[idx]
        ema_20 = df["ema_20"].iloc[idx]
        ema_50 = df["ema_50"].iloc[idx]
        
        # Allow some tolerance (price within 0.1% of EMA)
        tolerance = 0.001
        
        if trend == 'uptrend':
            # Price should touch or be near 20 or 50 EMA from above
            near_ema_20 = abs(current_price - ema_20) / ema_20 < tolerance
            near_ema_50 = abs(current_price - ema_50) / ema_50 < tolerance
            return near_ema_20 or near_ema_50
        
        elif trend == 'downtrend':
            # Price should touch or be near 20 or 50 EMA from below
            near_ema_20 = abs(current_price - ema_20) / ema_20 < tolerance
            near_ema_50 = abs(current_price - ema_50) / ema_50 < tolerance
            return near_ema_20 or near_ema_50
        
        return False

    # ------------------------------------------------------------
    # 🕯️ STEP 3: DETECT REJECTION CANDLE
    # ------------------------------------------------------------
    def check_rejection_candle(self, df, idx, trend):
        """
        Check if current candle shows rejection (bullish/bearish engulfing or pin bar).
        
        Returns: True if rejection detected, False otherwise
        """
        if idx < 1:
            return False
        
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]
        
        open_price = current["open"]
        close_price = current["close"]
        high_price = current["high"]
        low_price = current["low"]
        
        body = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        
        if trend == 'uptrend':
            # Bullish rejection: long lower wick, closes near high
            bullish_pin = lower_wick > body * 2 and close_price > open_price
            # Bullish engulfing
            bullish_engulf = (close_price > open_price and 
                            previous["close"] < previous["open"] and
                            close_price > previous["open"])
            return bullish_pin or bullish_engulf
        
        elif trend == 'downtrend':
            # Bearish rejection: long upper wick, closes near low
            bearish_pin = upper_wick > body * 2 and close_price < open_price
            # Bearish engulfing
            bearish_engulf = (close_price < open_price and 
                            previous["close"] > previous["open"] and
                            close_price < previous["open"])
            return bearish_pin or bearish_engulf
        
        return False

    # ------------------------------------------------------------
    # ✅ STEP 4: CONFIRMATION CANDLE
    # ------------------------------------------------------------
    def check_confirmation(self, df, idx, trend):
        """
        Check if current candle confirms the direction.
        
        Returns: True if confirmed, False otherwise
        """
        current = df.iloc[idx]
        ema_20 = current["ema_20"]
        ema_50 = current["ema_50"]
        close_price = current["close"]
        
        if trend == 'uptrend':
            # Candle closes above EMA
            return close_price > ema_20 or close_price > ema_50
        
        elif trend == 'downtrend':
            # Candle closes below EMA
            return close_price < ema_20 or close_price < ema_50
        
        return False

    # ------------------------------------------------------------
    # 📈 STEP 5: RSI CONFIRMATION
    # ------------------------------------------------------------
    def check_rsi_confirmation(self, df, idx, trend):
        """
        Check if RSI confirms the signal.
        
        Returns: True if RSI confirms, False otherwise
        """
        if idx < 1:
            return False
        
        current_rsi = df["rsi"].iloc[idx]
        previous_rsi = df["rsi"].iloc[idx - 1]
        
        if pd.isna(current_rsi) or pd.isna(previous_rsi):
            return False
        
        if trend == 'uptrend':
            # RSI should be turning up from 40-50 zone
            return (40 <= previous_rsi <= 55 and current_rsi > previous_rsi)
        
        elif trend == 'downtrend':
            # RSI should be turning down from 50-60 zone
            return (45 <= previous_rsi <= 60 and current_rsi < previous_rsi)
        
        return False

    # ------------------------------------------------------------
    # 🎯 STEP 6: SUPPORT/RESISTANCE CHECK
    # ------------------------------------------------------------
    def check_no_nearby_sr(self, df, idx, trend, lookback=20):
        """
        Check if there's no major support/resistance nearby.
        
        Returns: True if clear, False if SR nearby
        """
        current_price = df["close"].iloc[idx]
        
        # Look back to find recent highs/lows
        start_idx = max(0, idx - lookback)
        recent_data = df.iloc[start_idx:idx]
        
        if len(recent_data) < 5:
            return True  # Not enough data, assume clear
        
        recent_high = recent_data["high"].max()
        recent_low = recent_data["low"].min()
        
        # Check if price is too close to recent high/low (within 0.5%)
        threshold = 0.005
        
        if trend == 'uptrend':
            # Check for resistance above
            distance_to_high = (recent_high - current_price) / current_price
            return distance_to_high > threshold
        
        elif trend == 'downtrend':
            # Check for support below
            distance_to_low = (current_price - recent_low) / current_price
            return distance_to_low > threshold
        
        return True

    # ------------------------------------------------------------
    # 💰 STEP 7: CALCULATE POSITION SIZE
    # ------------------------------------------------------------
    def calculate_position_size(self, stop_loss_pips, pip_value=0.0001):
        """
        Calculate position size based on risk management.
        
        Args:
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value of 1 pip for the pair (default 0.0001 for most pairs)
        
        Returns:
            dict with lot_size and risk_amount
        """
        # Amount willing to risk
        risk_amount = self.balance * self.risk_per_trade
        
        # For forex: 1 standard lot = 100,000 units
        # 1 mini lot = 10,000 units
        # 1 micro lot = 1,000 units
        # 1 nano lot = 100 units
        
        # Calculate lot size
        # Formula: Lot Size = Risk Amount / (Stop Loss in Pips × Pip Value × Contract Size)
        
        # For micro lots (0.01 = 1,000 units)
        # Pip value for 0.01 lot EUR/USD = $0.10 per pip
        pip_value_per_micro_lot = 0.10
        
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_micro_lot)
        
        # Round to 2 decimal places (0.01 micro lot minimum)
        lot_size = round(lot_size, 2)
        
        # Ensure minimum lot size
        if lot_size < 0.01:
            lot_size = 0.01
        
        return {
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'position_value': lot_size * 1000  # in units
        }

    # ------------------------------------------------------------
    # 🛑 STEP 8: CALCULATE STOP LOSS
    # ------------------------------------------------------------
    def calculate_stop_loss(self, df, idx, signal_type):
        """
        Calculate stop loss level based on recent swing points.
        
        Args:
            df: DataFrame with indicators
            idx: Current index
            signal_type: 'buy' or 'sell'
        
        Returns:
            dict with stop_loss_price and stop_loss_pips
        """
        current_price = df["close"].iloc[idx]
        lookback = 20
        
        # Get recent data
        start_idx = max(0, idx - lookback)
        recent_data = df.iloc[start_idx:idx + 1]
        
        if signal_type == 'buy':
            # For buys: SL below recent swing low
            swing_low = recent_data["low"].min()
            buffer_pips = 10  # 10 pip buffer
            buffer = buffer_pips * 0.0001
            stop_loss_price = swing_low - buffer
            
        else:  # sell
            # For sells: SL above recent swing high
            swing_high = recent_data["high"].max()
            buffer_pips = 10
            buffer = buffer_pips * 0.0001
            stop_loss_price = swing_high + buffer
        
        # Calculate pips distance
        stop_loss_pips = abs(current_price - stop_loss_price) / 0.0001
        
        # Ensure SL is within reasonable range (15-30 pips)
        if stop_loss_pips < 15:
            stop_loss_pips = 15
            if signal_type == 'buy':
                stop_loss_price = current_price - (15 * 0.0001)
            else:
                stop_loss_price = current_price + (15 * 0.0001)
        
        if stop_loss_pips > 30:
            stop_loss_pips = 30
            if signal_type == 'buy':
                stop_loss_price = current_price - (30 * 0.0001)
            else:
                stop_loss_price = current_price + (30 * 0.0001)
        
        return {
            'stop_loss_price': round(stop_loss_price, 5),
            'stop_loss_pips': round(stop_loss_pips, 1)
        }

    # ------------------------------------------------------------
    # 🎯 STEP 9: CALCULATE TAKE PROFIT
    # ------------------------------------------------------------
    def calculate_take_profit(self, entry_price, stop_loss_price, signal_type, risk_reward_ratio=2.0):
        """
        Calculate take profit levels based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            signal_type: 'buy' or 'sell'
            risk_reward_ratio: Desired R:R (default 2.0 means 1:2)
        
        Returns:
            dict with tp1, tp2, and their pip distances
        """
        # Calculate risk in pips
        risk_pips = abs(entry_price - stop_loss_price) / 0.0001
        
        # TP1 at 1.5x risk
        tp1_pips = risk_pips * 1.5
        # TP2 at 2x or 3x risk
        tp2_pips = risk_pips * risk_reward_ratio
        
        if signal_type == 'buy':
            tp1_price = entry_price + (tp1_pips * 0.0001)
            tp2_price = entry_price + (tp2_pips * 0.0001)
        else:  # sell
            tp1_price = entry_price - (tp1_pips * 0.0001)
            tp2_price = entry_price - (tp2_pips * 0.0001)
        
        return {
            'tp1_price': round(tp1_price, 5),
            'tp1_pips': round(tp1_pips, 1),
            'tp2_price': round(tp2_price, 5),
            'tp2_pips': round(tp2_pips, 1)
        }

    # ------------------------------------------------------------
    # 🚀 MAIN STRATEGY EXECUTION
    # ------------------------------------------------------------
    def generate_signal(self):
        """
        Main function that executes all steps and returns signal.
        
        Returns: 'buy', 'sell', or 'hold'
        """
        # Compute all indicators
        df = self.compute_indicators()
        
        # Use the last complete candle (not current forming one)
        idx = len(df) - 1
        
        # Need enough data to make decision
        if idx < 200:
            return 'hold'
        
        # STEP 1: Identify trend
        trend = self.identify_trend(df, idx)
        
        if trend == 'sideways':
            return 'hold'
        
        # STEP 2: Check for pullback
        has_pullback = self.check_pullback(df, idx, trend)
        
        if not has_pullback:
            return 'hold'
        
        # STEP 3: Check for rejection candle
        has_rejection = self.check_rejection_candle(df, idx, trend)
        
        if not has_rejection:
            return 'hold'
        
        # STEP 4: Check for confirmation
        has_confirmation = self.check_confirmation(df, idx, trend)
        
        if not has_confirmation:
            return 'hold'
        
        # STEP 5: RSI confirmation
        rsi_confirms = self.check_rsi_confirmation(df, idx, trend)
        
        if not rsi_confirms:
            return 'hold'
        
        # STEP 6: Check for nearby support/resistance
        sr_clear = self.check_no_nearby_sr(df, idx, trend)
        
        if not sr_clear:
            return 'hold'
        
        # ALL CONDITIONS MET - Generate signal
        if trend == 'uptrend':
            return 'buy'
        elif trend == 'downtrend':
            return 'sell'
        
        return 'hold'
    
    # ------------------------------------------------------------
    # 📋 COMPLETE TRADE SETUP
    # ------------------------------------------------------------
    def get_complete_trade_setup(self):
        """
        Get complete trade setup with entry, SL, TP, and position size.
        
        Returns:
            dict with all trade parameters or None if no signal
        """
        # Get signal
        signal = self.generate_signal()
        
        if signal == 'hold':
            return None
        
        # Get data with indicators
        df = self.compute_indicators()
        idx = len(df) - 1
        entry_price = df["close"].iloc[idx]
        
        # Calculate stop loss
        sl_info = self.calculate_stop_loss(df, idx, signal)
        
        # Calculate position size
        position_info = self.calculate_position_size(sl_info['stop_loss_pips'])
        
        # Calculate take profit
        tp_info = self.calculate_take_profit(
            entry_price, 
            sl_info['stop_loss_price'], 
            signal
        )
        
        # Calculate potential profit/loss
        potential_loss = position_info['risk_amount']
        potential_profit_tp1 = potential_loss * 1.5
        potential_profit_tp2 = potential_loss * 2.0
        
        return {
            'signal': signal,
            'entry_price': round(entry_price, 5),
            'stop_loss': sl_info['stop_loss_price'],
            'stop_loss_pips': sl_info['stop_loss_pips'],
            'take_profit_1': tp_info['tp1_price'],
            'take_profit_1_pips': tp_info['tp1_pips'],
            'take_profit_2': tp_info['tp2_price'],
            'take_profit_2_pips': tp_info['tp2_pips'],
            'lot_size': position_info['lot_size'],
            'position_value': position_info['position_value'],
            'risk_amount': round(potential_loss, 2),
            'potential_profit_tp1': round(potential_profit_tp1, 2),
            'potential_profit_tp2': round(potential_profit_tp2, 2),
            'risk_reward_tp1': '1:1.5',
            'risk_reward_tp2': '1:2.0'
        }


# ------------------------------------------------------------
# 📝 EXAMPLE USAGE
# ------------------------------------------------------------
if __name__ == "__main__":
    # Sample data structure (you would get this from your data source)
    # Data should have columns: open, high, low, close, volume
    
    # Creating sample data for demonstration
    dates = pd.date_range(start='2024-01-01', periods=300, freq='1H')
    np.random.seed(42)
    
    # Simulate price movement
    close_prices = 1.0850 + np.cumsum(np.random.randn(300) * 0.0005)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(300) * 0.0002,
        'high': close_prices + abs(np.random.randn(300) * 0.0003),
        'low': close_prices - abs(np.random.randn(300) * 0.0003),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 300)
    })
    
    # Initialize strategy
    strategy = TradingNewStrategy(
        data=sample_data,
        balance=10,
        risk_per_trade=0.02,
        strategy="price_action_ma"
    )
    
    # Get complete trade setup
    trade_setup = strategy.get_complete_trade_setup()
    
    if trade_setup:
        print("=" * 60)
        print("📊 COMPLETE TRADE SETUP")
        print("=" * 60)
        print(f"🎯 Signal: {trade_setup['signal']}")
        print(f"💰 Account Balance: ${strategy.balance}")
        print(f"⚠️  Risk Per Trade: {strategy.risk_per_trade * 100}%")
        print()
        print("📈 ENTRY & EXIT LEVELS:")
        print(f"   Entry Price: {trade_setup['entry_price']}")
        print(f"   Stop Loss: {trade_setup['stop_loss']} ({trade_setup['stop_loss_pips']} pips)")
        print(f"   Take Profit 1: {trade_setup['take_profit_1']} ({trade_setup['take_profit_1_pips']} pips)")
        print(f"   Take Profit 2: {trade_setup['take_profit_2']} ({trade_setup['take_profit_2_pips']} pips)")
        print()
        print("💼 POSITION SIZING:")
        print(f"   Lot Size: {trade_setup['lot_size']} micro lots")
        print(f"   Position Value: {trade_setup['position_value']} units")
        print()
        print("💵 RISK & REWARD:")
        print(f"   Risk Amount: ${trade_setup['risk_amount']}")
        print(f"   Potential Profit (TP1): ${trade_setup['potential_profit_tp1']} | R:R {trade_setup['risk_reward_tp1']}")
        print(f"   Potential Profit (TP2): ${trade_setup['potential_profit_tp2']} | R:R {trade_setup['risk_reward_tp2']}")
        print("=" * 60)
        print()
        print("📝 TRADE EXECUTION PLAN:")
        print("   1. Enter at current price")
        print("   2. Set Stop Loss immediately")
        print("   3. Set Take Profit 1 & 2")
        print("   4. When TP1 hits: Close 50% position, move SL to breakeven")
        print("   5. Let TP2 run with trailing stop")
    else:
        print("⏸️  No trade signal - Conditions not met. HOLD.")