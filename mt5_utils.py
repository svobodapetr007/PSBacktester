"""
MT5 Utilities for Backtester

This module provides MetaTrader5 integration for fetching historical OHLC data.
Set ENABLE_MT5 = False to disable MT5 features for public deployment.
"""

# --- CONFIGURATION ---
# Set to False to disable MT5 features (for public deployment)
ENABLE_MT5 = True

# --- IMPORTS ---
if ENABLE_MT5:
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        MT5_AVAILABLE = False
        print("Warning: MetaTrader5 package not installed. MT5 features disabled.")
else:
    MT5_AVAILABLE = False

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple


# --- TIMEFRAME MAPPING ---
# Initialize timeframe map after checking MT5 availability
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }
else:
    TIMEFRAME_MAP = {
        'M1': None, 'M5': None, 'M15': None, 'M30': None,
        'H1': None, 'H4': None, 'D1': None, 'W1': None, 'MN1': None,
    }


# --- CONNECTION FUNCTIONS ---
def initialize_mt5() -> Tuple[bool, str]:
    """
    Initialize MetaTrader5 connection.
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    if not ENABLE_MT5:
        return False, "MT5 features are disabled. Set ENABLE_MT5 = True in mt5_utils.py"
    
    if not MT5_AVAILABLE:
        return False, "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
    
    if not mt5.initialize():
        error_code = mt5.last_error()
        return False, f"MT5 initialization failed. Error code: {error_code}"
    
    return True, "MT5 connected successfully!"


def shutdown_mt5():
    """Close MetaTrader5 connection."""
    if MT5_AVAILABLE and ENABLE_MT5:
        mt5.shutdown()


def is_mt5_connected() -> bool:
    """Check if MT5 is initialized and connected."""
    if not ENABLE_MT5 or not MT5_AVAILABLE:
        return False
    try:
        return mt5.terminal_info() is not None
    except:
        return False


# --- DATA FETCHING FUNCTIONS ---
def get_ohlc_history(
    symbol: str,
    timeframe: str,
    date_from: datetime,
    date_to: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch OHLC historical data from MetaTrader5.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
        timeframe: Timeframe string (e.g., 'H1', 'M15', 'D1')
        date_from: Start date
        date_to: End date (defaults to now if None)
    
    Returns:
        DataFrame with columns: ['time', 'open', 'high', 'low', 'close']
    
    Raises:
        ValueError: If MT5 is not available or timeframe is invalid
    """
    if not ENABLE_MT5:
        raise ValueError("MT5 features are disabled. Set ENABLE_MT5 = True in mt5_utils.py")
    
    if not MT5_AVAILABLE:
        raise ValueError("MetaTrader5 package not installed.")
    
    if not is_mt5_connected():
        raise ValueError("MT5 is not connected. Call initialize_mt5() first.")
    
    # Convert timeframe string to MT5 constant
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Invalid timeframe: {timeframe}. Available: {list(TIMEFRAME_MAP.keys())}")
    
    mt5_timeframe = TIMEFRAME_MAP[timeframe]
    
    # Default end date to now
    if date_to is None:
        date_to = datetime.now()
    
    # Fetch data
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, date_from, date_to)
    
    if rates is None or len(rates) == 0:
        error_code = mt5.last_error()
        raise ValueError(f"No data retrieved. Error code: {error_code}. Check symbol name and date range.")
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    # MT5 returns UTC timestamps, convert to timezone-naive UTC
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    
    # Select columns - include volume if available
    # MT5 typically provides 'tick_volume' (number of ticks) and sometimes 'real_volume' (actual traded volume)
    base_columns = ['time', 'open', 'high', 'low', 'close']
    
    # Check for volume columns and rename to 'volume' for consistency
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
        base_columns.append('volume')
    elif 'real_volume' in df.columns:
        df = df.rename(columns={'real_volume': 'volume'})
        base_columns.append('volume')
    elif 'volume' in df.columns:
        base_columns.append('volume')
    
    # Return columns in correct order
    return df[base_columns].copy()


def get_available_symbols() -> list:
    """
    Get list of available symbols from MT5.
    
    Returns:
        List of symbol strings
    """
    if not ENABLE_MT5 or not MT5_AVAILABLE or not is_mt5_connected():
        return []
    
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]
    except:
        return []

