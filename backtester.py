import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable

# --- 1. DATA STRUCTURES ---
@dataclass
class Trade:
    ticket: int
    symbol: str
    direction: str  # 'long' or 'short'
    open_time: pd.Timestamp
    open_price: float
    volume: float
    sl: float
    tp: float
    close_time: pd.Timestamp = None
    close_price: float = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    exit_reason: str = None

# --- 2. THE STRATEGY INTERFACE ---
class Strategy:
    """
    Users inherit from this class to create their own strategies.
    """
    def __init__(self):
        self.data = None
        self.broker = None
        self.orders = []
    
    def init(self):
        """Pre-calculate indicators here (runs once)."""
        pass

    def next(self, i, record):
        """Runs on every new candle.
        i: current index
        record: current row data (Open, High, Low, Close, Time)
        """
        pass

    def buy(self, volume=0.1, sl=0.0, tp=0.0):
        self.broker.new_order('long', volume, sl, tp)

    def sell(self, volume=0.1, sl=0.0, tp=0.0):
        self.broker.new_order('short', volume, sl, tp)

    def close_all(self):
        self.broker.close_all_positions()

# --- 3. THE ENGINE ---
class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_balance=10000, commission=2.5):
        # Prepare Data
        self.data = data.copy()
        if 'time' in self.data.columns:
            self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.reset_index(drop=True, inplace=True)
        
        self.strategy = strategy
        self.balance = initial_balance
        self.commission = commission # per lot per side
        self.spread = 0.00010 # Simulated spread (1 pip)
        
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Next Open Order Queue
        self.pending_orders = [] 
        self.close_requests = False

    def new_order(self, direction, volume, sl, tp):
        self.pending_orders.append({
            'direction': direction, 
            'volume': volume, 
            'sl': sl, 
            'tp': tp
        })

    def close_all_positions(self):
        self.close_requests = True

    def run(self):
        # Link strategy to engine
        self.strategy.data = self.data
        self.strategy.broker = self
        self.strategy.init()

        # Convert to records for fast iteration
        records = self.data.to_dict('records')
        total_bars = len(records)

        print(f"Running backtest on {total_bars} bars...")

        for i in range(total_bars):
            bar = records[i]
            
            # --- A. EXECUTION PHASE (Orders from previous bar execute at OPEN) ---
            
            # 1. Close Requests (Market execution at Open)
            if self.close_requests:
                for trade in self.active_trades[:]:
                    self._close_trade(trade, bar['open'], bar['time'], 'signal_close')
                self.close_requests = False

            # 2. Check SL/TP for holding trades (Intra-bar simulation)
            for trade in self.active_trades[:]:
                self._check_sl_tp(trade, bar)

            # 3. Open Pending Orders (Executes at OPEN of current bar)
            for order in self.pending_orders:
                price = bar['open'] 
                # Add spread cost simulation if needed
                if order['direction'] == 'long':
                    price += self.spread
                
                self._open_trade(order, price, bar['time'])
            
            self.pending_orders.clear() # Cleared after execution

            # --- B. STRATEGY PHASE (Think based on Close) ---
            # Strategy decides now, but execution happens next loop (Next Open)
            if i < total_bars - 1:
                self.strategy.next(i, bar)

        # End of data: Close all
        last_bar = records[-1]
        for trade in self.active_trades[:]:
            self._close_trade(trade, last_bar['close'], last_bar['time'], 'end_of_data')

        return self._generate_report()

    def _open_trade(self, order, price, time):
        self.trade_counter += 1
        t = Trade(
            ticket=self.trade_counter,
            symbol="TEST",
            direction=order['direction'],
            open_time=time,
            open_price=price,
            volume=order['volume'],
            sl=order['sl'],
            tp=order['tp'],
            commission=self.commission * order['volume'] * 2 # Round turn
        )
        self.active_trades.append(t)

    def _close_trade(self, trade, price, time, reason):
        trade.close_time = time
        trade.close_price = price
        trade.exit_reason = reason
        
        # Calculate Profit
        multiplier = 1 if trade.direction == 'long' else -1
        trade.profit = (trade.close_price - trade.open_price) * multiplier * trade.volume * 100000 # Standard Lot size
        trade.profit -= trade.commission # Deduct commission
        
        self.balance += trade.profit
        self.active_trades.remove(trade)
        self.closed_trades.append(trade)

    def _check_sl_tp(self, trade, bar):
        # Simple High/Low check
        if trade.direction == 'long':
            if trade.sl > 0 and bar['low'] <= trade.sl:
                self._close_trade(trade, trade.sl, bar['time'], 'sl')
            elif trade.tp > 0 and bar['high'] >= trade.tp:
                self._close_trade(trade, trade.tp, bar['time'], 'tp')
        else: # Short
            if trade.sl > 0 and bar['high'] >= trade.sl:
                self._close_trade(trade, trade.sl, bar['time'], 'sl')
            elif trade.tp > 0 and bar['low'] <= trade.tp:
                self._close_trade(trade, trade.tp, bar['time'], 'tp')

    def _generate_report(self):
        df = pd.DataFrame([vars(t) for t in self.closed_trades])
        if df.empty:
            return pd.DataFrame()
        
        df['cum_profit'] = df['profit'].cumsum() + 10000 # initial balance
        return df