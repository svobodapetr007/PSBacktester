# Refactoring complete! The backtester now uses vectorbt for powerful backtesting.
#
# Key changes:
# 1. Created vectorbt_engine.py with VectorBT-based strategies
# 2. Updated dashboard.py to use vectorbt Portfolio
# 3. Added indicator plotting (SMA, RSI, MACD) on price charts
# 4. Improved performance metrics (Sharpe Ratio, Max Drawdown, etc.)
# 5. SMA Cross Back exit only works with SMA Crossover entry
#
# Next steps: Test and fix any remaining issues
