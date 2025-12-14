import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, time
from backtester import BacktestEngine
from strategy import MyPerfectStrategy

# Try to import MT5 utils (will work if ENABLE_MT5 = True)
try:
    from mt5_utils import (
        ENABLE_MT5, initialize_mt5, shutdown_mt5, is_mt5_connected,
        get_ohlc_history, get_available_symbols, TIMEFRAME_MAP
    )
    MT5_ENABLED = ENABLE_MT5
except ImportError:
    MT5_ENABLED = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Backtester", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR TOP MENU ---
st.markdown("""
    <style>
    .top-menu {
        display: flex;
        gap: 20px;
        padding: 2px 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 5px;
    }
    .menu-item {
        padding: 4px 12px;
        cursor: pointer;
        border-radius: 4px;
        font-weight: 500;
    }
    .menu-item:hover {
        background-color: rgba(31, 119, 180, 0.1);
    }
    .menu-item.active {
        background-color: #1f77b4;
        color: white;
    }
    button[kind="secondary"], button[kind="primary"] {
        font-size: 0.7rem !important;
        padding: 0.3rem 0.5rem !important;
        white-space: nowrap !important;
        letter-spacing: 0 !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Data'
if 'mt5_connected' not in st.session_state:
    st.session_state['mt5_connected'] = False

# --- TOP NAVIGATION MENU ---
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 5])
with col1:
    if st.button("Data", use_container_width=True, type="primary" if st.session_state['current_tab'] == 'Data' else "secondary"):
        st.session_state['current_tab'] = 'Data'
        st.rerun()
with col2:
    if st.button("Market Mapper", use_container_width=True, type="primary" if st.session_state['current_tab'] == 'Market Mapper' else "secondary"):
        st.session_state['current_tab'] = 'Market Mapper'
        st.rerun()

# --- TAB 1: DATA (MT5/CSV Connection + OHLC Chart) ---
if st.session_state['current_tab'] == 'Data':
    col_left, col_right = st.columns([0.5, 2.5])
    
    with col_left:
        st.subheader("Data Source")
        
        # Data source selection
        data_source = st.radio(
            "Choose data source:",
            ["MT5 (MetaTrader5)", "CSV Upload", "Simulation Data"],
            index=2 if not MT5_ENABLED else 0,
            label_visibility="collapsed"
        )
        
        df = None
        
        # --- MT5 DATA SOURCE ---
        if data_source == "MT5 (MetaTrader5)" and MT5_ENABLED:
            st.subheader("MT5 Connection")
            
            if st.session_state['mt5_connected']:
                st.success("✅ MT5 Connected")
                if st.button("Disconnect MT5", use_container_width=True):
                    shutdown_mt5()
                    st.session_state['mt5_connected'] = False
                    st.rerun()
            else:
                if st.button("Connect to MT5", type="primary", use_container_width=True):
                    success, message = initialize_mt5()
                    if success:
                        st.session_state['mt5_connected'] = True
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            # MT5 Data Fetching
            if st.session_state.get('mt5_connected', False):
                st.subheader("Fetch Data")
                
                symbol = st.text_input("Symbol", value="BTCUSD", help="e.g., EURUSD, BTCUSD, GBPUSD")
                
                timeframe = st.selectbox(
                    "Timeframe",
                    options=list(TIMEFRAME_MAP.keys()),
                    index=4,  # Default to H1
                    help="Select the chart timeframe"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now())
                
                if st.button("Fetch from MT5", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Fetching data from MT5..."):
                            df = get_ohlc_history(
                                symbol=symbol,
                                timeframe=timeframe,
                                date_from=datetime.combine(start_date, datetime.min.time()),
                                date_to=datetime.combine(end_date, datetime.max.time())
                            )
                            st.session_state['data'] = df
                            st.session_state['data_symbol'] = symbol
                            st.session_state['data_timeframe'] = timeframe
                            st.success(f"✅ Fetched {len(df)} bars")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
                
                if 'data' in st.session_state and not st.session_state['data'].empty:
                    df = st.session_state['data']
        
        # --- CSV UPLOAD ---
        elif data_source == "CSV Upload":
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader(
                "Upload OHLC CSV",
                type=['csv'],
                help="CSV should contain columns: time, open, high, low, close",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                    st.session_state['data'] = df
                    st.session_state['data_symbol'] = uploaded_file.name
                    st.success(f"✅ Loaded {len(df)} rows")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading CSV: {str(e)}")
            
            if 'data' in st.session_state and not st.session_state['data'].empty:
                df = st.session_state['data']
                st.info(f"📊 Loaded: {len(df)} bars")
        
        # --- SIMULATION DATA ---
        else:
            st.subheader("Simulation Data")
            st.info("Generate synthetic data for testing")
            rows = st.slider("Data Points", 100, 5000, 1000)
            volatility = st.slider("Volatility", 0.001, 0.010, 0.005, step=0.001)
            
            if st.button("Generate Data", use_container_width=True):
                dates = pd.date_range(start='2023-01-01', periods=rows, freq='H')
                price_walk = 1.10 + np.cumsum(np.random.normal(0, volatility, rows))
                df = pd.DataFrame({
                    'time': dates,
                    'open': price_walk,
                    'high': price_walk + volatility/2,
                    'low': price_walk - volatility/2,
                    'close': price_walk + np.random.normal(0, volatility/10, rows)
                })
                st.session_state['data'] = df
                st.session_state['data_symbol'] = "SIMULATION"
                st.success(f"✅ Generated {len(df)} bars")
                st.rerun()
            
            if 'data' in st.session_state and not st.session_state['data'].empty:
                df = st.session_state['data']
                st.info(f"📊 Loaded: {len(df)} bars")
    
    with col_right:
        # Display chart if data is available
        if 'data' in st.session_state and not st.session_state['data'].empty:
            df = st.session_state['data']
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df['time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            )])
            
            symbol_name = st.session_state.get('data_symbol', 'Data')
            fig.update_layout(
                title=f"{symbol_name} - OHLC Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                height=600,
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data info
            st.caption(f"Total bars: {len(df)} | Period: {df['time'].min()} to {df['time'].max()}")
        else:
            st.info("👈 Load data from the left panel to view the chart")

# --- TAB 2: MARKET MAPPER (Filters + Equity Curve) ---
elif st.session_state['current_tab'] == 'Market Mapper':
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.warning("⚠️ Please load data first in the 'Data' tab")
        st.info("Go to 📊 Data tab to connect MT5 or upload CSV file")
    else:
        df = st.session_state['data'].copy()
        
        col_left, col_right = st.columns([0.5, 2.5])
        
        with col_left:
            st.subheader("General Parameters")
            
            col_mode_label, col_mode_radio = st.columns([1.5, 2.5])
            with col_mode_label:
                st.write("Mode:")
            with col_mode_radio:
                mode = st.radio("", ["DT", "Swing"], horizontal=True, label_visibility="collapsed", help="DT = Day Trading, Swing = Swing Trading")
            
            col_dir_label, col_dir_radio = st.columns([1.5, 2.5])
            with col_dir_label:
                st.write("Direction:")
            with col_dir_radio:
                direction = st.radio("", ["Long", "Short"], horizontal=True, label_visibility="collapsed", help="Long = Buy, Short = Sell")
            
            col_view_label, col_view_radio = st.columns([1.5, 2.5])
            with col_view_label:
                st.write("View:")
            with col_view_radio:
                view_mode = st.radio("", ["Chart", "Table"], horizontal=True, label_visibility="collapsed", help="Display mode for results")
            
            col_pos_label, col_pos_input = st.columns([1, 2])
            with col_pos_label:
                st.write("Position Size (Lots):")
            with col_pos_input:
                position_size = st.number_input("", min_value=0.01, max_value=100.0, value=1.0, step=0.01, label_visibility="collapsed", help="Number of lots per trade")
            
            col_comm_label, col_comm_input = st.columns([1, 2])
            with col_comm_label:
                st.write("Commission per Lot:")
            with col_comm_input:
                commission_per_lot = st.number_input("", min_value=0.0, max_value=100.0, value=0.0, step=0.1, label_visibility="collapsed", help="Commission per lot per side (round turn = 2x)")
            
            st.markdown("---")
            
            if mode == "DT":
                st.subheader("DT Parameters")
                
                col_start_label, col_start_input = st.columns([1, 2])
                with col_start_label:
                    st.write("Start Hour:")
                with col_start_input:
                    start_hour = st.time_input("", value=time(8, 30), label_visibility="collapsed", help="Entry will only occur if a bar exists within this hour")
                
                col_end_label, col_end_input = st.columns([1, 2])
                with col_end_label:
                    st.write("End Hour:")
                with col_end_input:
                    end_hour = st.time_input("", value=time(15, 30), label_visibility="collapsed", help="Exit at first available bar ≥ this time")
                
                st.write("Days of Week")
                days_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
                
                if 'selected_days' not in st.session_state:
                    st.session_state['selected_days'] = ['MON', 'TUE', 'WED', 'THU', 'FRI'].copy()
                
                cols_row1 = st.columns(4)
                with cols_row1[0]:
                    if st.button("ALL", use_container_width=True, key="btn_all_days"):
                        st.session_state['selected_days'] = days_of_week.copy()
                        st.rerun()
                
                for i, day in enumerate(['MON', 'TUE', 'WED']):
                    with cols_row1[i+1]:
                        is_selected = day in st.session_state['selected_days']
                        if st.button(
                            day,
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                            key=f"btn_{day}"
                        ):
                            if is_selected:
                                st.session_state['selected_days'].remove(day)
                            else:
                                st.session_state['selected_days'].append(day)
                            st.rerun()
                
                cols_row2 = st.columns(4)
                for i, day in enumerate(['THU', 'FRI', 'SAT', 'SUN']):
                    with cols_row2[i]:
                        is_selected = day in st.session_state['selected_days']
                        if st.button(
                            day,
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                            key=f"btn_{day}"
                        ):
                            if is_selected:
                                st.session_state['selected_days'].remove(day)
                            else:
                                st.session_state['selected_days'].append(day)
                            st.rerun()
                
                # Get selected days from session state
                selected_days = st.session_state.get('selected_days', ['MON', 'TUE', 'WED', 'THU', 'FRI'])
            else:
                # Swing mode - no time restrictions, but still filter by days
                start_hour = None
                end_hour = None
                # Initialize selected days if not set
                if 'selected_days' not in st.session_state:
                    st.session_state['selected_days'] = ['MON', 'TUE', 'WED', 'THU', 'FRI']
                selected_days = st.session_state.get('selected_days', ['MON', 'TUE', 'WED', 'THU', 'FRI'])
        
        with col_right:
            # Apply filters to data
            filtered_df = df.copy()
            
            # Filter by days of week (applies to both DT and Swing)
            day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
            filtered_df['day_of_week'] = filtered_df['time'].dt.dayofweek
            selected_day_nums = [day_map[d] for d in selected_days if d in day_map]
            filtered_df = filtered_df[filtered_df['day_of_week'].isin(selected_day_nums)]
            
            # Calculate trades based on mode
            if mode == "DT":
                # Day Trading: One trade per day
                # Entry: First bar at or after start hour within the day
                # Exit: First bar at or after end hour (can be same day or next day)
                filtered_df = filtered_df.sort_values('time').reset_index(drop=True)
                filtered_df['date'] = filtered_df['time'].dt.date
                filtered_df['hour'] = filtered_df['time'].dt.hour
                filtered_df['minute'] = filtered_df['time'].dt.minute
                filtered_df['time_of_day'] = filtered_df['hour'] * 60 + filtered_df['minute']
                
                start_minutes = start_hour.hour * 60 + start_hour.minute
                end_minutes = end_hour.hour * 60 + end_hour.minute
                
                # Group by date and find entry/exit bars
                trades = []
                dates = sorted(filtered_df['date'].unique())
                
                for i, date in enumerate(dates):
                    day_data = filtered_df[filtered_df['date'] == date]
                    
                    # Find entry bar: first bar at or after start hour
                    # Entry will only occur if a bar exists within or after the start hour
                    entry_bars = day_data[day_data['time_of_day'] >= start_minutes]
                    if len(entry_bars) == 0:
                        continue  # No entry bar found for this day
                    entry_bar = entry_bars.iloc[0]
                    
                    # Find exit bar (first bar >= end hour, must be after entry)
                    # First try same day
                    exit_bars_same_day = day_data[
                        (day_data['time_of_day'] >= end_minutes) & 
                        (day_data['time'] > entry_bar['time'])
                    ]
                    
                    if len(exit_bars_same_day) > 0:
                        exit_bar = exit_bars_same_day.iloc[0]
                    else:
                        # If no exit bar found on same day at or after end hour,
                        # use the last bar of the day (if it's after entry)
                        bars_after_entry = day_data[day_data['time'] > entry_bar['time']]
                        if len(bars_after_entry) > 0:
                            exit_bar = bars_after_entry.iloc[-1]  # Last bar of the day
                        else:
                            continue  # No bar found after entry on this day
                    
                    # Calculate profit in dollars
                    if direction == "Long":
                        price_diff = exit_bar['close'] - entry_bar['open']
                    else:  # Short
                        price_diff = entry_bar['open'] - exit_bar['close']
                    
                    # Convert price difference to dollars
                    # For forex: 1 lot = 100,000 units, 1 pip (0.0001) = $10 per lot
                    # For crypto: 1 lot = 1 unit (e.g., 1 BTC)
                    # Use pip value calculation: price_diff * pip_value_multiplier * position_size
                    # pip_value_multiplier = 100000 for forex (0.0001 * 100000 = $10 per pip per lot)
                    # pip_value_multiplier = 1 for crypto (1 * 1 = $1 per $1 move per lot)
                    
                    # Detect if crypto (price > 1000) or forex (price < 10)
                    avg_price = (entry_bar['open'] + exit_bar['close']) / 2
                    if avg_price > 1000:  # Likely crypto
                        pip_value_multiplier = 1  # 1 lot = 1 unit
                    else:  # Likely forex
                        pip_value_multiplier = 100000  # 1 lot = 100,000 units
                    
                    profit_dollars = (price_diff * pip_value_multiplier * position_size) - (commission_per_lot * position_size * 2)  # Round turn commission
                    
                    trades.append({
                        'date': date,
                        'entry_time': entry_bar['time'],
                        'exit_time': exit_bar['time'],
                        'open_price': entry_bar['open'],
                        'close_price': exit_bar['close'],
                        'profit': profit_dollars
                    })
                
                if len(trades) == 0:
                    st.warning("No trades found matching the selected filters")
                    filtered_df = pd.DataFrame()
                else:
                    # Create DataFrame from trades
                    filtered_df = pd.DataFrame(trades)
                    filtered_df['cumulative_equity'] = filtered_df['profit'].cumsum()
                    filtered_df['trade_number'] = range(1, len(filtered_df) + 1)
            
            else:
                # Swing Trading: One trade per day (open to close of same day)
                filtered_df = filtered_df.sort_values('time').reset_index(drop=True)
                filtered_df['date'] = filtered_df['time'].dt.date
                
                # Group by date and get first open and last close of each day
                trades = []
                for date, day_data in filtered_df.groupby('date'):
                    if len(day_data) == 0:
                        continue
                    
                    first_bar = day_data.iloc[0]
                    last_bar = day_data.iloc[-1]
                    
                    # Calculate profit in dollars
                    if direction == "Long":
                        price_diff = last_bar['close'] - first_bar['open']
                    else:  # Short
                        price_diff = first_bar['open'] - last_bar['close']
                    
                    # Convert price difference to dollars
                    # Detect if crypto (price > 1000) or forex (price < 10)
                    avg_price = (first_bar['open'] + last_bar['close']) / 2
                    if avg_price > 1000:  # Likely crypto
                        pip_value_multiplier = 1  # 1 lot = 1 unit
                    else:  # Likely forex
                        pip_value_multiplier = 100000  # 1 lot = 100,000 units
                    
                    profit_dollars = (price_diff * pip_value_multiplier * position_size) - (commission_per_lot * position_size * 2)  # Round turn commission
                    
                    trades.append({
                        'date': date,
                        'entry_time': first_bar['time'],
                        'exit_time': last_bar['time'],
                        'open_price': first_bar['open'],
                        'close_price': last_bar['close'],
                        'profit': profit_dollars
                    })
                
                if len(trades) == 0:
                    st.warning("No trades found matching the selected filters")
                    filtered_df = pd.DataFrame()
                else:
                    # Create DataFrame from trades
                    filtered_df = pd.DataFrame(trades)
                    filtered_df['cumulative_equity'] = filtered_df['profit'].cumsum()
                    filtered_df['trade_number'] = range(1, len(filtered_df) + 1)
            
            if len(filtered_df) == 0:
                st.warning("No data matches the selected filters")
            else:
                col_time_label, col_time_radio = st.columns([1, 4])
                with col_time_label:
                    st.write("Time Range:")
                with col_time_radio:
                    time_ranges = ["All History", "3 Years", "1 Year", "6 Months", "3 Months"]
                    selected_range = st.radio("", time_ranges, horizontal=True, index=0, label_visibility="collapsed")
                
                # Filter by time range
                if selected_range != "All History":
                    end_date = filtered_df['exit_time'].max()
                    if selected_range == "3 Years":
                        start_date = end_date - pd.DateOffset(years=3)
                    elif selected_range == "1 Year":
                        start_date = end_date - pd.DateOffset(years=1)
                    elif selected_range == "6 Months":
                        start_date = end_date - pd.DateOffset(months=6)
                    elif selected_range == "3 Months":
                        start_date = end_date - pd.DateOffset(months=3)
                    
                    range_filtered = filtered_df[filtered_df['exit_time'] >= start_date].copy()
                    # Recalculate trade numbers and cumulative equity for filtered range
                    range_filtered = range_filtered.reset_index(drop=True)
                    range_filtered['trade_number'] = range_filtered.index + 1
                    range_filtered['cumulative_equity'] = range_filtered['profit'].cumsum()
                else:
                    range_filtered = filtered_df.copy()
                
                # Display based on view mode
                if view_mode == "Chart":
                    # Create equity curve chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=range_filtered['trade_number'],
                        y=range_filtered['cumulative_equity'],
                        mode='lines',
                        name='All Trades',
                        line=dict(color='#1f77b4', width=2),
                        hovertemplate='Trade: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    symbol_name = st.session_state.get('data_symbol', 'Data')
                    if mode == "DT":
                        mode_text = f"Day Trading: {start_hour.strftime('%H:%M')} to {end_hour.strftime('%H:%M')}"
                    else:
                        mode_text = "Swing Trading"
                    
                    fig.update_layout(
                        title=f"Equity Curves - {selected_range} ({symbol_name})<br><sub>{mode_text} | {direction}</sub>",
                        xaxis_title="Trade Number",
                        yaxis_title="Cumulative Equity ($)",
                        height=600,
                        template="plotly_dark",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate metrics
                    total_trades = len(range_filtered)
                    if total_trades > 0:
                        winning_trades = len(range_filtered[range_filtered['profit'] > 0])
                        win_rate = (winning_trades / total_trades) * 100
                        
                        # Calculate max drawdown
                        cumulative_max = range_filtered['cumulative_equity'].cummax()
                        drawdown = range_filtered['cumulative_equity'] - cumulative_max
                        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
                        
                        total_pnl = range_filtered['cumulative_equity'].iloc[-1]
                        average_trade = range_filtered['profit'].mean()
                    else:
                        win_rate = 0
                        max_drawdown = 0
                        total_pnl = 0
                        average_trade = 0
                    
                    # Display metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Trades", f"{total_trades}")
                    with col2:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with col3:
                        st.metric("Max Drawdown", f"${max_drawdown:,.2f}")
                    with col4:
                        st.metric("Average Trade", f"${average_trade:,.2f}")
                    with col5:
                        st.metric("Total P&L", f"${total_pnl:,.2f}")
                else:
                    # Table view
                    display_df = range_filtered[['entry_time', 'exit_time', 'open_price', 'close_price', 'profit', 'cumulative_equity']].copy()
                    display_df.columns = ['Entry Time', 'Exit Time', 'Open Price', 'Close Price', 'Profit ($)', 'Cumulative Equity ($)']
                    # Format currency columns
                    display_df['Profit ($)'] = display_df['Profit ($)'].apply(lambda x: f"${x:,.2f}")
                    display_df['Cumulative Equity ($)'] = display_df['Cumulative Equity ($)'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(display_df, use_container_width=True, height=600)
