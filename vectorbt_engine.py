"""
VectorBT-based Backtesting Engine

This module uses vectorbt for high-performance backtesting with proper indicator calculation.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
from typing import Dict, Tuple, Optional


class VectorBTStrategy:
    """Wrapper class for vectorbt-based strategies."""
    
    def __init__(self, data: pd.DataFrame, entry_type: str, exit_type: str, 
                 entry_params: Dict, exit_params: Dict, direction_mode: str = "Both"):
        self.data = data.copy()
        self.entry_type = entry_type
        self.exit_type = exit_type
        self.entry_params = entry_params
        self.exit_params = exit_params
        self.direction_mode = direction_mode
        
        # Store indicators for plotting
        self.indicators = {}
        
        # Prepare data
        self.data.set_index('time', inplace=True)
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
        self.open = self.data['open']
        
    def generate_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry signals based on strategy configuration.
        Returns: (long_entries, short_entries) as boolean Series
        """
        long_entries = pd.Series(False, index=self.data.index)
        short_entries = pd.Series(False, index=self.data.index)
        
        if self.entry_type == 'SMA Crossover':
            long_entries, short_entries = self._sma_crossover_signals()
        elif self.entry_type == 'RSI Threshold':
            long_entries, short_entries = self._rsi_threshold_signals()
        elif self.entry_type == 'MACD Cross':
            long_entries, short_entries = self._macd_cross_signals()
        
        # Filter by direction mode
        if self.direction_mode == "Long":
            short_entries = pd.Series(False, index=self.data.index)
        elif self.direction_mode == "Short":
            long_entries = pd.Series(False, index=self.data.index)
        
        return long_entries, short_entries
    
    def _sma_crossover_signals(self) -> Tuple[pd.Series, pd.Series]:
        """SMA Crossover entry signals."""
        fast = self.entry_params.get('fast', 10)
        slow = self.entry_params.get('slow', 50)
        
        # Calculate SMAs using vectorbt
        sma_fast = vbt.MA.run(self.close, window=fast, short_name='fast').ma
        sma_slow = vbt.MA.run(self.close, window=slow, short_name='slow').ma
        
        # Store for plotting
        self.indicators['sma_fast'] = sma_fast
        self.indicators['sma_slow'] = sma_slow
        
        # Generate crossover signals
        long_entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        short_entries = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
        
        return long_entries.fillna(False), short_entries.fillna(False)
    
    def _rsi_threshold_signals(self) -> Tuple[pd.Series, pd.Series]:
        """RSI Threshold entry signals."""
        length = self.entry_params.get('length', 14)
        mode = self.entry_params.get('mode', 'mean_reversion')
        
        # Calculate RSI using vectorbt
        rsi = vbt.RSI.run(self.close, window=length).rsi
        
        # Store for plotting
        self.indicators['rsi'] = rsi
        
        if mode == 'mean_reversion':
            oversold = self.entry_params.get('oversold', 30)
            overbought = self.entry_params.get('overbought', 70)
            
            long_entries = rsi < oversold
            short_entries = rsi > overbought
        else:  # momentum
            crossing_threshold = self.entry_params.get('crossing_threshold', 50)
            
            long_entries = (rsi > crossing_threshold) & (rsi.shift(1) <= crossing_threshold)
            short_entries = (rsi < crossing_threshold) & (rsi.shift(1) >= crossing_threshold)
        
        return long_entries.fillna(False), short_entries.fillna(False)
    
    def _macd_cross_signals(self) -> Tuple[pd.Series, pd.Series]:
        """MACD Cross entry signals."""
        fast = self.entry_params.get('fast', 12)
        slow = self.entry_params.get('slow', 26)
        signal = self.entry_params.get('signal', 9)
        mode = self.entry_params.get('mode', 'histogram_cross')
        
        # Calculate MACD using vectorbt
        macd_ind = vbt.MACD.run(self.close, fast_window=fast, slow_window=slow, signal_window=signal)
        macd = macd_ind.macd
        macd_signal = macd_ind.signal
        macd_hist = macd_ind.hist
        
        # Store for plotting
        self.indicators['macd'] = macd
        self.indicators['macd_signal'] = macd_signal
        self.indicators['macd_hist'] = macd_hist
        
        if mode == 'histogram_cross':
            long_entries = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
            short_entries = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
        else:  # signal_cross
            long_entries = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
            short_entries = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
        
        return long_entries.fillna(False), short_entries.fillna(False)
    
    def generate_exits(self) -> Tuple[pd.Series, pd.Series]:
        """
        Generate exit signals based on exit strategy.
        Returns: (long_exits, short_exits) as boolean Series
        """
        long_exits = pd.Series(False, index=self.data.index)
        short_exits = pd.Series(False, index=self.data.index)
        
        if self.exit_type == 'SMA Cross Back':
            # Only valid with SMA Crossover entry
            if self.entry_type == 'SMA Crossover':
                long_exits, short_exits = self._sma_crossback_exits()
        elif self.exit_type == 'Time based':
            long_exits, short_exits = self._time_based_exits()
        # ATR-based exits are handled via TP/SL in the portfolio
        
        return long_exits.fillna(False), short_exits.fillna(False)
    
    def _sma_crossback_exits(self) -> Tuple[pd.Series, pd.Series]:
        """Exit when SMA crosses back (reverse of entry)."""
        # Use already calculated SMAs
        sma_fast = self.indicators.get('sma_fast')
        sma_slow = self.indicators.get('sma_slow')
        
        if sma_fast is None or sma_slow is None:
            return pd.Series(False, index=self.data.index), pd.Series(False, index=self.data.index)
        
        # Exit long when fast crosses below slow
        long_exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
        # Exit short when fast crosses above slow
        short_exits = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        
        return long_exits, short_exits
    
    def _time_based_exits(self) -> Tuple[pd.Series, pd.Series]:
        """Exit at end of each trading day."""
        # Create a boolean series that's True at the last bar of each day
        dates = self.data.index.date
        date_changes = pd.Series(dates, index=self.data.index) != pd.Series(dates, index=self.data.index).shift(-1)
        
        # Both long and short exit at EOD
        return date_changes.fillna(False), date_changes.fillna(False)
    
    def get_sl_tp(self, entries: pd.Series, direction: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Calculate stop loss and take profit levels.
        Returns: (sl_stops, tp_stops) as Series or None
        """
        if self.exit_type == 'Fixed TP/SL (ATR)':
            atr_length = self.exit_params.get('atr_length', 14)
            sl_mult = self.exit_params.get('sl_atr_mult', 1.0)
            tp_mult = self.exit_params.get('tp_atr_mult', 2.0)
            
            # Calculate ATR using vectorbt
            atr = vbt.ATR.run(self.high, self.low, self.close, window=atr_length).atr
            
            # Store for plotting
            if 'atr' not in self.indicators:
                self.indicators['atr'] = atr
            
            # Calculate SL and TP distances
            if direction == 'long':
                sl_stop = self.close - (atr * sl_mult)
                tp_stop = self.close + (atr * tp_mult)
            else:  # short
                sl_stop = self.close + (atr * sl_mult)
                tp_stop = self.close - (atr * tp_mult)
            
            # Only set SL/TP where we have entries
            sl_stop = sl_stop.where(entries, np.nan)
            tp_stop = tp_stop.where(entries, np.nan)
            
            return sl_stop, tp_stop
        
        elif self.exit_type == 'ATR Trailing Stop':
            atr_length = self.exit_params.get('atr_length', 14)
            atr_mult = self.exit_params.get('atr_multiplier', 2.0)
            
            # Calculate ATR
            atr = vbt.ATR.run(self.high, self.low, self.close, window=atr_length).atr
            
            # Store for plotting
            if 'atr' not in self.indicators:
                self.indicators['atr'] = atr
            
            # Calculate trailing stop distance
            if direction == 'long':
                sl_stop = self.high - (atr * atr_mult)
            else:  # short
                sl_stop = self.low + (atr * atr_mult)
            
            # Use vectorbt's trailing stop mechanism
            # Only set where we have entries
            sl_stop = sl_stop.where(entries, np.nan)
            
            return sl_stop, None  # No TP for trailing stop
        
        return None, None
    
    def run_backtest(self, initial_cash: float = 10000, commission: float = 0.001) -> vbt.Portfolio:
        """
        Run the backtest using vectorbt.
        
        Args:
            initial_cash: Starting capital
            commission: Commission as fraction (e.g., 0.001 = 0.1%)
        
        Returns:
            vectorbt Portfolio object
        """
        # Generate entry signals
        long_entries, short_entries = self.generate_signals()
        
        # Calculate Exits
        # Determine strict exits vs usage of opposite entries as exits
        exits = None
        short_exits = None
        
        if self.exit_type in ['SMA Cross Back', 'Time based']:
            # Calculate custom exits
            long_custom_exits, short_custom_exits = self.generate_exits()
            
            if self.direction_mode in ["Long", "Both"]:
                exits = long_custom_exits
            if self.direction_mode in ["Short", "Both"]:
                short_exits = short_custom_exits
        else:
            # Default behavior: use opposite entries as exits if Both directions enabled
            if self.direction_mode == "Both":
                exits = short_entries
                short_exits = long_entries
        
        # Get SL/TP for longs and shorts
        long_sl, long_tp = self.get_sl_tp(long_entries, 'long')
        short_sl, short_tp = self.get_sl_tp(short_entries, 'short')
        
        # Build portfolio (Run once!)
        pf = vbt.Portfolio.from_signals(
            close=self.close,
            entries=long_entries,
            exits=exits,
            short_entries=short_entries if self.direction_mode in ["Short", "Both"] else None,
            short_exits=short_exits,
            init_cash=initial_cash,
            fees=commission,
            sl_stop=long_sl if long_sl is not None else short_sl,
            tp_stop=long_tp if long_tp is not None else short_tp,
            size=self.entry_params.get('position_size', 1.0),
            size_type='amount',
            freq='T'
        )
        
        return pf
    
    def get_indicator_plot_data(self) -> Dict:
        """
        Get indicator data for plotting.
        
        Returns:
            Dictionary with indicator names and their values
        """
        return self.indicators


def run_vectorbt_backtest(data: pd.DataFrame, entry_type: str, exit_type: str,
                          entry_params: Dict, exit_params: Dict, direction_mode: str = "Both",
                          initial_cash: float = 10000, commission: float = 0.001) -> Tuple[vbt.Portfolio, Dict]:
    """
    Convenience function to run a backtest with vectorbt.
    
    Args:
        data: OHLC DataFrame with 'time', 'open', 'high', 'low', 'close' columns
        entry_type: Entry strategy type
        exit_type: Exit strategy type
        entry_params: Entry parameters dictionary
        exit_params: Exit parameters dictionary
        direction_mode: "Long", "Short", or "Both"
        initial_cash: Starting capital
        commission: Commission as fraction
    
    Returns:
        Tuple of (Portfolio object, indicators dictionary)
    """
    strategy = VectorBTStrategy(data, entry_type, exit_type, entry_params, exit_params, direction_mode)
    portfolio = strategy.run_backtest(initial_cash, commission)
    indicators = strategy.get_indicator_plot_data()
    
    return portfolio, indicators
