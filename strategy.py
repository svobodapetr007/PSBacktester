from backtester import Strategy, BacktestEngine
import pandas_ta as ta # Ensure you pip install pandas_ta
import numpy as np

class MyPerfectStrategy(Strategy):
    def init(self):
        # Pre-calculate indicators using pandas-ta for speed
        self.data['sma_fast'] = ta.sma(self.data['close'], length=10)
        self.data['sma_slow'] = ta.sma(self.data['close'], length=50)
        self.data['rsi'] = ta.rsi(self.data['close'], length=14)

    def next(self, i, record):
        # Simple Crossing Logic
        if i < 50: return # Not enough data yet

        price = record['close']
        fast = self.data['sma_fast'].iloc[i]
        slow = self.data['sma_slow'].iloc[i]
        rsi = self.data['rsi'].iloc[i]

        # Entry Logic
        if fast > slow and rsi < 70:
            if not self.orders: # Only 1 trade at a time
                # Dynamic SL/TP calculation
                sl = price - 0.0050 
                tp = price + 0.0100
                self.buy(volume=1.0, sl=sl, tp=tp)
        
        # Exit Logic (Simple Filter)
        elif fast < slow:
            self.close_all()

# --- Helper to run it locally without GUI ---
if __name__ == "__main__":
    import pandas as pd
    # Load your CSV data here
    # df = pd.read_csv("EURUSD_H1.csv") 
    
    # Mock data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
    df = pd.DataFrame({
        'time': dates,
        'open': 1.05 + np.random.randn(500)/100,
        'high': 1.06 + np.random.randn(500)/100,
        'low': 1.04 + np.random.randn(500)/100,
        'close': 1.05 + np.random.randn(500)/100
    })
    
    engine = BacktestEngine(df, MyPerfectStrategy())
    trades = engine.run()
    print(trades.tail())