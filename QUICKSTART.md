# Quick Start Guide - VectorBT Backtester

## ✅ Installation Complete!

Your backtester has been successfully refactored to use **vectorbt**.

## 🚀 Quick Start

1. **Restart Streamlit** (the current session is running the old code):
   ```bash
   # Stop the current streamlit (Ctrl+C in terminal)
   # Then restart:
   streamlit run dashboard.py
   ```

2. **Load Data** - Go to the "Data" tab:
   - Select "Yahoo Finance" 
   - Enter a symbol (e.g., SPY, AAPL, BTC-USD)
   - Choose interval (1h recommended for testing)
   - Click "Fetch YF Data"

3. **Configure Strategy** - Go to "Strategy Tester" tab:
   - **Entry**: Choose "SMA Crossover" (Fast=10, Slow=50)
   - **Exit**: Choose "SMA Cross Back" 
   - **Direction**: Both
   - **Position Size**: 1.0
   - **Starting Balance**: $10,000 (Adjustable)
   - Click "Run Backtest"

4. **View Results**:
   - **Equity Curve**: See portfolio growth and drawdown
   - **Price Chart with Indicators**: See SMA lines, entry/exit markers
   - **Trades Table**: Detailed trade breakdown
   - **Analytics**: Analyze trades by Direction, Day of Week, and Hour
   - **Stats**: Full performance metrics

## 📊 Example Strategies to Try

### 1. SMA Crossover (Trend Following)
- Entry: SMA Crossover (10/50)
- Exit: SMA Cross Back
- Direction: Both
- Works well in trending markets

### 2. RSI Mean Reversion  
- Entry: RSI Threshold (Mean Reversion, 30/70)
- Exit: Fixed TP/SL (ATR) (SL=1.0, TP=2.0)
- Direction: Both
- Works well in ranging markets

### 3. MACD Momentum
- Entry: MACD Cross (12/26/9, Histogram Cross)
- Exit: Time based (EOD)
- Direction: Both
- Intraday momentum strategy

## 🎨 What's New

### Indicators on charts!
- SMA lines overlaid on price
- RSI subplot with levels (30/50/70)
- MACD subplot with histogram
- Entry/exit markers (green/red-blue)

### Better Metrics
- Sharpe Ratio
- Max Drawdown %
- Total Return %
- Win Rate %
- Best/Worst trades

### Faster Performance
- Vectorized calculations
- WebGL-accelerated charts
- No more slow loops!

##  Common Issues

### Issue: "ModuleNotFoundError: No module named 'vectorbt'"
**Solution**: Install vectorbt in Python 3.13:
```bash
python3.13 -m pip install vectorbt
```

### Issue: Dashboard shows old results
**Solution**: Clear streamlit cache:
- Click the "⋮" menu (top right)
- Select "Clear cache"
- Reload the page

### Issue: No trades generated
**Solution**:
- Check if data is loaded (go to Data tab)
- Try different parameters (smaller SMA periods, wider RSI thresholds)
- Ensure enough bars for indicators to calculate

## 📝 Notes

- Yahoo Finance data is free and doesn't require API keys
- For crypto, use format: BTC-USD, ETH-USD
- For stocks: SPY, AAPL, TSLA, etc.
- Recommended: Start with 1h or 1d intervals
- Min bars needed: ~50 for indicators to calculate

## 🎯 Next Steps

1. Try different entry/exit combos
2. Backtest on different symbols
3. Compare strategies side-by-side (run multiple backtests)
4. Optimize parameters for better performance

Enjoy your new vectorbt-powered backtester! 🚀
