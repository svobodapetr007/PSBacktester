"""
Data Utilities for Backtester

Handles data normalization, CSV parsing, and instrument parameter management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import time


# --- INSTRUMENT PARAMETERS ---
INSTRUMENT_PARAMS = {
    # Forex pairs
    'EURUSD': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'quote_currency': 'USD'},
    'GBPUSD': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'quote_currency': 'USD'},
    'USDJPY': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.01, 'pip_value_per_lot': 10.0, 'quote_currency': 'JPY'},
    'AUDUSD': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'quote_currency': 'USD'},
    'USDCAD': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'quote_currency': 'CAD'},
    
    # Crypto
    'BTCUSD': {'type': 'crypto', 'contract_size': 1, 'pip_size': 1.0, 'pip_value_per_lot': 1.0, 'quote_currency': 'USD'},
    'ETHUSD': {'type': 'crypto', 'contract_size': 1, 'pip_size': 1.0, 'pip_value_per_lot': 1.0, 'quote_currency': 'USD'},
    'BTCUSDT': {'type': 'crypto', 'contract_size': 1, 'pip_size': 1.0, 'pip_value_per_lot': 1.0, 'quote_currency': 'USDT'},
    
    # Default forex (fallback)
    'DEFAULT_FOREX': {'type': 'forex', 'contract_size': 100000, 'pip_size': 0.0001, 'pip_value_per_lot': 10.0, 'quote_currency': 'USD'},
    # Default crypto (fallback)
    'DEFAULT_CRYPTO': {'type': 'crypto', 'contract_size': 1, 'pip_size': 1.0, 'pip_value_per_lot': 1.0, 'quote_currency': 'USD'},
}


def get_instrument_params(symbol: str) -> Dict:
    """
    Get instrument parameters for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
    
    Returns:
        Dictionary with instrument parameters
    """
    symbol_upper = symbol.upper()
    
    # Check exact match first
    if symbol_upper in INSTRUMENT_PARAMS:
        return INSTRUMENT_PARAMS[symbol_upper].copy()
    
    # Try to detect from symbol name
    if 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'CRYPTO' in symbol_upper:
        return INSTRUMENT_PARAMS['DEFAULT_CRYPTO'].copy()
    
    # Default to forex
    return INSTRUMENT_PARAMS['DEFAULT_FOREX'].copy()


def normalize_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CSV data to standard format.
    Handles various column name formats and ensures required columns exist.
    
    Args:
        df: Raw DataFrame from CSV
    
    Returns:
        Normalized DataFrame with columns: ['time', 'open', 'high', 'low', 'close']
    
    Raises:
        ValueError: If required columns cannot be found or created
    """
    df = df.copy()
    
    # Column name mapping (case-insensitive)
    column_mapping = {
        'time': ['time', 'datetime', 'date', 'timestamp', 't'],
        'open': ['open', 'o', 'openprice'],
        'high': ['high', 'h', 'highprice'],
        'low': ['low', 'l', 'lowprice'],
        'close': ['close', 'c', 'closeprice', 'price'],
    }
    
    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()
    
    # Map columns
    normalized_df = pd.DataFrame()
    for target_col, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in df.columns:
                normalized_df[target_col] = df[name]
                found = True
                break
        
        if not found:
            raise ValueError(f"Required column '{target_col}' not found. Available columns: {list(df.columns)}")
    
    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(normalized_df['time']):
        normalized_df['time'] = pd.to_datetime(normalized_df['time'], errors='coerce')
    
    # Ensure numeric columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
    
    # Remove rows with NaN values
    normalized_df = normalized_df.dropna()
    
    # Sort by time
    normalized_df = normalized_df.sort_values('time').reset_index(drop=True)
    
    # Ensure timezone-naive (convert to UTC if timezone-aware, then remove timezone)
    if normalized_df['time'].dt.tz is not None:
        normalized_df['time'] = normalized_df['time'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    return normalized_df[['time', 'open', 'high', 'low', 'close']].copy()


def apply_spread_slippage(
    price: float,
    direction: str,
    spread_pips: float,
    slippage_pips: float,
    instrument_params: Dict,
    is_entry: bool
) -> float:
    """
    Apply spread and slippage to a price.
    
    Args:
        price: Base price (open or close)
        direction: 'Long' or 'Short'
        spread_pips: Spread in pips
        slippage_pips: Additional slippage in pips
        instrument_params: Instrument parameters dictionary
        is_entry: True for entry, False for exit
    
    Returns:
        Adjusted price with spread and slippage
    """
    pip_size = instrument_params['pip_size']
    spread_price = spread_pips * pip_size
    slippage_price = slippage_pips * pip_size
    
    if direction == "Long":
        if is_entry:
            # Long entry: buy at ask (higher price)
            return price + (spread_price / 2) + slippage_price
        else:
            # Long exit: sell at bid (lower price)
            return price - (spread_price / 2) - slippage_price
    else:  # Short
        if is_entry:
            # Short entry: sell at bid (lower price)
            return price - (spread_price / 2) - slippage_price
        else:
            # Short exit: buy at ask (higher price)
            return price + (spread_price / 2) + slippage_price


def calculate_profit(
    entry_price: float,
    exit_price: float,
    direction: str,
    position_size: float,
    instrument_params: Dict,
    commission_per_side: float
) -> float:
    """
    Calculate profit in quote currency.
    
    Args:
        entry_price: Entry price (already adjusted for spread/slippage)
        exit_price: Exit price (already adjusted for spread/slippage)
        direction: 'Long' or 'Short'
        position_size: Position size in lots
        instrument_params: Instrument parameters dictionary
        commission_per_side: Commission per lot per side (not round-turn)
    
    Returns:
        Profit in quote currency (dollars for USD pairs)
    """
    pip_size = instrument_params['pip_size']
    pip_value_per_lot = instrument_params['pip_value_per_lot']
    
    # Calculate price difference
    if direction == "Long":
        price_diff = exit_price - entry_price
    else:  # Short
        price_diff = entry_price - exit_price
    
    # Calculate profit: (price_diff / pip_size) * pip_value_per_lot * position_size
    profit = (price_diff / pip_size) * pip_value_per_lot * position_size
    
    # Subtract commission (round-turn = 2 sides)
    commission_total = commission_per_side * position_size * 2
    
    return profit - commission_total


def generate_trades_dt(
    df: pd.DataFrame,
    start_hour: time,
    end_hour: time,
    selected_days: list,
    direction: str,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    instrument_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate day trading trades (one per day).
    Handles cross-midnight scenarios (e.g., start=22:00, end=02:00).
    
    Args:
        df: Filtered DataFrame with day_of_week column
        start_hour: Start hour for entry
        end_hour: End hour for exit
        selected_days: List of selected day names (e.g., ['MON', 'TUE'])
        direction: 'Long' or 'Short'
    
    Returns:
        DataFrame with trades
    """
    df = df.sort_values('time').reset_index(drop=True).copy()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['time_of_day'] = df['hour'] * 60 + df['minute']
    
    start_minutes = start_hour.hour * 60 + start_hour.minute
    end_minutes = end_hour.hour * 60 + end_hour.minute
    
    # Handle cross-midnight case (end < start)
    crosses_midnight = end_minutes < start_minutes
    
    trades = []
    dates = sorted(df['date'].unique())
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date]
        
        # Find entry bar (first bar at or after start hour)
        entry_bars = day_data[day_data['time_of_day'] >= start_minutes]
        
        if len(entry_bars) == 0:
            continue
        
        entry_bar = entry_bars.iloc[0]
        
        # Find exit bar
        if crosses_midnight:
            # Exit can be on current day (if time_of_day >= end) OR next day (if time_of_day < end)
            exit_bars_same_day = day_data[
                (day_data['time_of_day'] >= end_minutes) & 
                (day_data['time'] > entry_bar['time'])
            ]
            
            if len(exit_bars_same_day) > 0:
                exit_bar = exit_bars_same_day.iloc[0]
            else:
                # Look in next day
                if i + 1 < len(dates):
                    next_day = dates[i + 1]
                    next_day_data = df[df['date'] == next_day]
                    exit_bars_next_day = next_day_data[
                        (next_day_data['time_of_day'] < end_minutes) &
                        (next_day_data['time'] > entry_bar['time'])
                    ]
                    if len(exit_bars_next_day) > 0:
                        exit_bar = exit_bars_next_day.iloc[0]
                    else:
                        # Use last bar of current day if no exit found
                        bars_after_entry = day_data[day_data['time'] > entry_bar['time']]
                        if len(bars_after_entry) > 0:
                            exit_bar = bars_after_entry.iloc[-1]
                        else:
                            continue
                else:
                    bars_after_entry = day_data[day_data['time'] > entry_bar['time']]
                    if len(bars_after_entry) > 0:
                        exit_bar = bars_after_entry.iloc[-1]
                    else:
                        continue
        else:
            # Normal case: exit on same day
            exit_bars_same_day = day_data[
                (day_data['time_of_day'] >= end_minutes) & 
                (day_data['time'] > entry_bar['time'])
            ]
            
            if len(exit_bars_same_day) > 0:
                exit_bar = exit_bars_same_day.iloc[0]
            else:
                bars_after_entry = day_data[day_data['time'] > entry_bar['time']]
                if len(bars_after_entry) > 0:
                    exit_bar = bars_after_entry.iloc[-1]
                else:
                    continue
        
        # Apply spread and slippage to entry and exit prices
        if instrument_params is not None:
            entry_price = apply_spread_slippage(
                entry_bar['open'], direction, spread_pips, slippage_pips,
                instrument_params, is_entry=True
            )
            exit_price = apply_spread_slippage(
                exit_bar['close'], direction, spread_pips, slippage_pips,
                instrument_params, is_entry=False
            )
        else:
            entry_price = entry_bar['open']
            exit_price = exit_bar['close']
        
        # Calculate price difference
        if direction == "Long":
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price
        
        trades.append({
            'date': date,
            'entry_time': entry_bar['time'],
            'exit_time': exit_bar['time'],
            'open_price': entry_price,
            'close_price': exit_price,
            'price_diff': price_diff
        })
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(trades)


def generate_trades_swing(
    df: pd.DataFrame,
    direction: str,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    instrument_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate swing trading trades (one per day, open to close).
    
    Args:
        df: Filtered DataFrame
        direction: 'Long' or 'Short'
        spread_pips: Spread in pips
        slippage_pips: Additional slippage in pips
        instrument_params: Instrument parameters dictionary
    
    Returns:
        DataFrame with trades
    """
    df = df.sort_values('time').reset_index(drop=True).copy()
    df['date'] = df['time'].dt.date
    
    trades = []
    for date, day_data in df.groupby('date'):
        if len(day_data) == 0:
            continue
        
        first_bar = day_data.iloc[0]
        last_bar = day_data.iloc[-1]
        
        # Apply spread and slippage to entry and exit prices
        if instrument_params is not None:
            entry_price = apply_spread_slippage(
                first_bar['open'], direction, spread_pips, slippage_pips,
                instrument_params, is_entry=True
            )
            exit_price = apply_spread_slippage(
                last_bar['close'], direction, spread_pips, slippage_pips,
                instrument_params, is_entry=False
            )
        else:
            entry_price = first_bar['open']
            exit_price = last_bar['close']
        
        # Calculate price difference
        if direction == "Long":
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price
        
        trades.append({
            'date': date,
            'entry_time': first_bar['time'],
            'exit_time': last_bar['time'],
            'open_price': entry_price,
            'close_price': exit_price,
            'price_diff': price_diff
        })
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(trades)

