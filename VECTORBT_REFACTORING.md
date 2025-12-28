# VectorBT Refactoring Summary

## Overview
Successfully refactored the PSBacktester to use **vectorbt** as the backtesting engine instead of the custom implementation.

## Key Changes

### 1. New Files Created
- **vectorbt_engine.py**: Complete vectorbt-based backtesting engine
  - `VectorBTStrategy` class for strategy management
  - Entry signal generators (SMA Crossover, RSI Threshold, MACD Cross)
  - Exit signal generators (Fixed TP/SL ATR, ATR Trailing Stop, SMA Cross Back, Time based)
  - Automatic SL/TP calculation based on ATR
  - Indicator storage for plotting

### 2. Updated Files
- **dashboard.py**: 
  - Integrated vectorbt Portfolio object
  - Replaced old backtest results with vectorbt stats
  - Added comprehensive indicator plotting (SMA, RSI, MACD)
  - New view modes: "Equity Curve", "Price Chart with Indicators", "Trades Table", "Stats"
  
- **requirements.txt**: Added vectorbt>=0.28.0

### 3. Features Implemented

#### Entry Strategies
- **SMA Crossover**: Fast/Slow MA crossover with visual indicators
- **RSI Threshold**: Mean reversion or momentum modes with RSI levels
- **MACD Cross**: Histogram or signal line cross with full MACD plotting

#### Exit Strategies  
- **Fixed TP/SL (ATR)**: ATR-based stop loss and take profit
- **ATR Trailing Stop**: Dynamic trailing stops based on ATR
- **SMA Cross Back**: Only works with SMA Crossover entry (as requested)
- **Time based**: End-of-day exits

#### Visualizations
1. **Equity Curve**: Portfolio value over time with drawdown chart
2. **Price Chart with Indicators**: 
   - Candlestick chart with price action
   - SMA lines overlaid (if SMA strategy)
   - RSI subplot with 30/70 levels (if RSI strategy)
   - MACD subplot with histogram (if MACD strategy)
   - Entry/exit markers colored by P&L
3. **Trades Table**: Detailed trade-by-trade breakdown
4. **Stats**: Full vectorbt statistics (Sharpe, Sortino, Max DD, etc.)

#### Performance Metrics
- Total Trades
- Win Rate %
- Total Return %
- Max Drawdown %
- Sharpe Ratio
- Best/Worst trades
- Average Win/Loss

### 4. Logic Improvements
- **SMA Cross Back** exit only available when entry is SMA Crossover
- Proper vectorbt signal generation (boolean Series)
- Efficient indicator calculation using vectorbt's built-in functions
- WebGL-accelerated plotting for large datasets

## How to Use

1. **Select Data Source**: MT5, CSV, or Yahoo Finance
2. **Configure Strategy**:
   - Choose entry type (SMA Crossover, RSI, MACD)
   - Choose exit type (Fixed TP/SL, Trailing Stop, Cross Back, Time)
   - Set direction (Long, Short, Both)
   - Configure parameters
3. **Run Backtest**: Click "Run Backtest" button
4. **View Results**: Switch between Equity Curve, Price Chart, Trades, and Stats

## Benefits of VectorBT

1. **Performance**: Vectorized operations, much faster than loop-based backtesting
2. **Features**: Built-in stats (Sharpe, Sortino, Calmar, etc.)
3. **Flexibility**: Easy to add new indicators and strategies
4. **Visualization**: Native plotting support with proper indicator overlays
5. **Accuracy**: Professional-grade backtesting engine

## Next Steps

- Test all strategies thoroughly
- Add more entry/exit strategies as needed
- Optimize performance for very large datasets
- Add portfolio optimization features
- Implement walk-forward analysis

## Notes
- Commission is converted from per-lot to fraction for vectorbt  
- All indicators are pre-calculated and stored for plotting
- Data is indexed by time for vectorbt compatibility
- Exit logic properly handles direction mode (Long/Short/Both)
