"""
Strategy Builder for Strategy Tester

Creates configurable strategy classes based on user selections.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtester import Strategy
from typing import Optional, Dict


class ConfigurableStrategy(Strategy):
    """Base class for configurable strategies."""
    
    def __init__(self, entry_type: str, exit_type: str, entry_params: Dict, exit_params: Dict, direction_mode: str = "Both"):
        super().__init__()
        self.entry_type = entry_type
        self.exit_type = exit_type
        self.entry_params = entry_params
        self.exit_params = exit_params
        self.direction_mode = direction_mode  # "Long", "Short", or "Both"
        self.atr = None
        self.trailing_stop_price = None
        self.current_trade_direction = None
    
    def init(self):
        """Pre-calculate indicators."""
        # Calculate ATR if needed for exits
        if self.exit_type in ['Fixed TP/SL (ATR)', 'ATR Trailing Stop']:
            atr_length = self.exit_params.get('atr_length', 14)
            self.atr = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_length)
        
        # Calculate entry indicators
        if self.entry_type == 'SMA Crossover':
            fast = self.entry_params.get('fast', 10)
            slow = self.entry_params.get('slow', 50)
            self.data['sma_fast'] = ta.sma(self.data['close'], length=fast)
            self.data['sma_slow'] = ta.sma(self.data['close'], length=slow)
        
        elif self.entry_type == 'RSI Threshold':
            length = self.entry_params.get('length', 14)
            self.data['rsi'] = ta.rsi(self.data['close'], length=length)
        
        elif self.entry_type == 'MACD Cross':
            fast = self.entry_params.get('fast', 12)
            slow = self.entry_params.get('slow', 26)
            signal = self.entry_params.get('signal', 9)
            macd = ta.macd(self.data['close'], fast=fast, slow=slow, signal=signal)
            
            # pandas_ta returns columns with format: MACD_12_26_9, MACDs_12_26_9 (lowercase 's'), MACDh_12_26_9
            if macd is not None and not macd.empty:
                # Use the exact column names that pandas_ta returns
                macd_col = f'MACD_{fast}_{slow}_{signal}'
                signal_col = f'MACDs_{fast}_{slow}_{signal}'  # lowercase 's'
                hist_col = f'MACDh_{fast}_{slow}_{signal}'
                
                # Check if columns exist, if not try to find them dynamically
                if macd_col in macd.columns:
                    self.data['macd'] = macd[macd_col]
                else:
                    # Fallback: find column that starts with MACD_ but not MACDs_ or MACDh_
                    macd_cols = [c for c in macd.columns if c.startswith('MACD_') and not c.startswith('MACDs_') and not c.startswith('MACDh_')]
                    if macd_cols:
                        self.data['macd'] = macd[macd_cols[0]]
                
                if signal_col in macd.columns:
                    self.data['macd_signal'] = macd[signal_col]
                else:
                    # Fallback: find column that starts with MACDs_
                    signal_cols = [c for c in macd.columns if c.startswith('MACDs_')]
                    if signal_cols:
                        self.data['macd_signal'] = macd[signal_cols[0]]
                
                if hist_col in macd.columns:
                    self.data['macd_hist'] = macd[hist_col]
                else:
                    # Fallback: find column that starts with MACDh_
                    hist_cols = [c for c in macd.columns if c.startswith('MACDh_')]
                    if hist_cols:
                        self.data['macd_hist'] = macd[hist_cols[0]]
            else:
                self.data['macd'] = np.nan
                self.data['macd_signal'] = np.nan
                self.data['macd_hist'] = np.nan
    
    def next(self, i, record):
        """Main strategy logic."""
        if i < 50:  # Need enough data for indicators
            return
        
        # Check exit conditions first - use broker's active_trades instead of self.orders
        has_active_trades_before = len(self.broker.active_trades) > 0 if self.broker else False
        if has_active_trades_before:
            self._check_exit(i, record)
        
        # Reset trailing stop if trade was closed (check after exit check)
        has_active_trades_after = len(self.broker.active_trades) > 0 if self.broker else False
        if has_active_trades_before and not has_active_trades_after:
            self.trailing_stop_price = None
            self.current_trade_direction = None
        
        # Check entry conditions - use broker's active_trades instead of self.orders
        if not self.broker or len(self.broker.active_trades) == 0:  # Only enter if no open position
            self._check_entry(i, record)
    
    def _get_sl_tp(self, i, price, direction='long'):
        """Calculate SL/TP based on exit strategy."""
        sl = 0.0
        tp = 0.0
        
        if self.exit_type == 'Fixed TP/SL (ATR)':
            if self.atr is not None and i < len(self.atr):
                atr_value = self.atr.iloc[i]
                sl_mult = self.exit_params.get('sl_atr_mult', 1.0)
                tp_mult = self.exit_params.get('tp_atr_mult', 2.0)
                if direction == 'long':
                    sl = price - (atr_value * sl_mult)
                    tp = price + (atr_value * tp_mult)
                else:  # short
                    sl = price + (atr_value * sl_mult)
                    tp = price - (atr_value * tp_mult)
        
        return sl, tp
    
    def _check_entry(self, i, record):
        """Check entry conditions based on entry_type."""
        price = record['close']
        
        if self.entry_type == 'SMA Crossover':
            fast = self.data['sma_fast'].iloc[i]
            slow = self.data['sma_slow'].iloc[i]
            fast_prev = self.data['sma_fast'].iloc[i-1] if i > 0 else fast
            slow_prev = self.data['sma_slow'].iloc[i-1] if i > 0 else slow
            
            # Crossover: fast crosses above slow (Long entry)
            if fast > slow and fast_prev <= slow_prev:
                if self.direction_mode in ["Long", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'long')
                    self.current_trade_direction = 'long'
                    self.trailing_stop_price = None  # Reset trailing stop
                    self.buy(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
            
            # Reverse crossover: fast crosses below slow (Short entry)
            if fast < slow and fast_prev >= slow_prev:
                if self.direction_mode in ["Short", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'short')
                    self.current_trade_direction = 'short'
                    self.trailing_stop_price = None  # Reset trailing stop
                    self.sell(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
        
        elif self.entry_type == 'RSI Threshold':
            rsi = self.data['rsi'].iloc[i]
            oversold = self.entry_params.get('oversold', 30)
            overbought = self.entry_params.get('overbought', 70)
            mode = self.entry_params.get('mode', 'mean_reversion')  # 'mean_reversion' or 'momentum'
            
            if mode == 'mean_reversion':
                if rsi < oversold and self.direction_mode in ["Long", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'long')
                    self.current_trade_direction = 'long'
                    self.trailing_stop_price = None
                    self.buy(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
                elif rsi > overbought and self.direction_mode in ["Short", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'short')
                    self.current_trade_direction = 'short'
                    self.trailing_stop_price = None
                    self.sell(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
            else:  # momentum
                rsi_prev = self.data['rsi'].iloc[i-1] if i > 0 else rsi
                if rsi > 50 and rsi_prev <= 50 and self.direction_mode in ["Long", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'long')
                    self.current_trade_direction = 'long'
                    self.trailing_stop_price = None
                    self.buy(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
                elif rsi < 50 and rsi_prev >= 50 and self.direction_mode in ["Short", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'short')
                    self.current_trade_direction = 'short'
                    self.trailing_stop_price = None
                    self.sell(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
        
        elif self.entry_type == 'MACD Cross':
            macd = self.data['macd'].iloc[i]
            signal = self.data['macd_signal'].iloc[i]
            hist = self.data['macd_hist'].iloc[i]
            hist_prev = self.data['macd_hist'].iloc[i-1] if i > 0 else hist
            
            # MACD crosses above signal OR histogram crosses above 0 (Long entry)
            entry_mode = self.entry_params.get('mode', 'histogram_cross')
            if entry_mode == 'histogram_cross':
                if hist > 0 and hist_prev <= 0 and self.direction_mode in ["Long", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'long')
                    self.current_trade_direction = 'long'
                    self.trailing_stop_price = None
                    self.buy(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
                elif hist < 0 and hist_prev >= 0 and self.direction_mode in ["Short", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'short')
                    self.current_trade_direction = 'short'
                    self.trailing_stop_price = None
                    self.sell(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
            else:  # signal_cross
                if macd > signal and (i == 0 or self.data['macd'].iloc[i-1] <= self.data['macd_signal'].iloc[i-1]) and self.direction_mode in ["Long", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'long')
                    self.current_trade_direction = 'long'
                    self.trailing_stop_price = None
                    self.buy(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
                elif macd < signal and (i == 0 or self.data['macd'].iloc[i-1] >= self.data['macd_signal'].iloc[i-1]) and self.direction_mode in ["Short", "Both"]:
                    sl, tp = self._get_sl_tp(i, price, 'short')
                    self.current_trade_direction = 'short'
                    self.trailing_stop_price = None
                    self.sell(volume=self.entry_params.get('position_size', 1.0), sl=sl, tp=tp)
    
    def _check_exit(self, i, record):
        """Check exit conditions based on exit_type."""
        # Use broker's active_trades instead of self.orders
        if not self.broker or len(self.broker.active_trades) == 0:
            return
        
        price = record['close']
        high = record['high']
        low = record['low']
        
        if self.exit_type == 'Fixed TP/SL (ATR)':
            # SL/TP are set when opening trade, handled by engine
            pass
        
        elif self.exit_type == 'ATR Trailing Stop':
            if self.atr is not None and i < len(self.atr):
                atr_value = self.atr.iloc[i]
                atr_multiplier = self.exit_params.get('atr_multiplier', 2.0)
                trail_distance = atr_value * atr_multiplier
                
                # Get current trade direction from broker
                if len(self.orders) > 0:
                    # Get direction from the first active order (simplified - assumes one trade)
                    # In real implementation, we'd check broker.active_trades
                    if self.current_trade_direction is None:
                        # Try to infer from entry - this is a limitation, ideally broker would tell us
                        self.current_trade_direction = 'long'  # Default assumption
                    
                    # Update trailing stop
                    if self.current_trade_direction == 'long':
                        new_stop = high - trail_distance
                        if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                            self.trailing_stop_price = new_stop
                        
                        if low <= self.trailing_stop_price:
                            self.close_all()
                            self.trailing_stop_price = None
                            self.current_trade_direction = None
                    else:  # short
                        new_stop = low + trail_distance
                        if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                            self.trailing_stop_price = new_stop
                        
                        if high >= self.trailing_stop_price:
                            self.close_all()
                            self.trailing_stop_price = None
                            self.current_trade_direction = None
        
        elif self.exit_type == 'SMA Cross Back':
            if self.entry_type == 'SMA Crossover':
                fast = self.data['sma_fast'].iloc[i]
                slow = self.data['sma_slow'].iloc[i]
                fast_prev = self.data['sma_fast'].iloc[i-1] if i > 0 else fast
                slow_prev = self.data['sma_slow'].iloc[i-1] if i > 0 else slow
                
                # Reverse crossover: fast crosses below slow
                if fast < slow and fast_prev >= slow_prev:
                    self.close_all()
            
            elif self.entry_type == 'MACD Cross':
                macd = self.data['macd'].iloc[i]
                signal = self.data['macd_signal'].iloc[i]
                macd_prev = self.data['macd'].iloc[i-1] if i > 0 else macd
                signal_prev = self.data['macd_signal'].iloc[i-1] if i > 0 else signal
                
                # MACD crosses below signal
                if macd < signal and macd_prev >= signal_prev:
                    self.close_all()


def create_strategy_with_sl_tp(strategy: ConfigurableStrategy, price: float, direction: str) -> tuple:
    """
    Calculate SL/TP based on exit strategy and current price.
    
    Returns:
        (sl, tp) tuple
    """
    if strategy.exit_type == 'Fixed TP/SL (ATR)':
        atr_value = strategy.atr.iloc[-1] if strategy.atr is not None else price * 0.01
        sl_mult = strategy.exit_params.get('sl_atr_mult', 1.0)
        tp_mult = strategy.exit_params.get('tp_atr_mult', 2.0)
        
        if direction == 'long':
            sl = price - (atr_value * sl_mult)
            tp = price + (atr_value * tp_mult)
        else:  # short
            sl = price + (atr_value * sl_mult)
            tp = price - (atr_value * tp_mult)
        
        return (sl, tp)
    
    return (0.0, 0.0)  # No fixed SL/TP

