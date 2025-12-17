import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time
from backtester import BacktestEngine
from strategy import MyPerfectStrategy
from data_utils import (
    normalize_csv_data, get_instrument_params, calculate_profit,
    generate_trades_dt, generate_trades_swing
)
from strategy_builder import ConfigurableStrategy

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

# --- CACHING ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_get_ohlc_history(symbol, timeframe, date_from, date_to):
    """Cached MT5 data fetching."""
    return get_ohlc_history(symbol, timeframe, date_from, date_to)

# --- INITIALIZE SESSION STATE ---
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Data'
if 'mt5_connected' not in st.session_state:
    st.session_state['mt5_connected'] = False
if 'selected_days' not in st.session_state:
    st.session_state['selected_days'] = ['MON', 'TUE', 'WED', 'THU', 'FRI']

# --- TOP NAVIGATION MENU ---
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 4])
with col1:
    if st.button("Data", use_container_width=True, type="primary" if st.session_state['current_tab'] == 'Data' else "secondary", key="nav_data"):
        st.session_state['current_tab'] = 'Data'
with col2:
    if st.button("Market Mapper", use_container_width=True, type="primary" if st.session_state['current_tab'] == 'Market Mapper' else "secondary", key="nav_mapper"):
        st.session_state['current_tab'] = 'Market Mapper'
with col3:
    if st.button("Strategy Tester", use_container_width=True, type="primary" if st.session_state['current_tab'] == 'Strategy Tester' else "secondary", key="nav_strategy"):
        st.session_state['current_tab'] = 'Strategy Tester'

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
                if st.button("Disconnect MT5", use_container_width=True, key="disconnect_mt5"):
                    shutdown_mt5()
                    st.session_state['mt5_connected'] = False
            else:
                if st.button("Connect to MT5", type="primary", use_container_width=True, key="connect_mt5"):
                    success, message = initialize_mt5()
                    if success:
                        st.session_state['mt5_connected'] = True
                        st.success(message)
                    else:
                        st.error(message)
            
            # MT5 Data Fetching
            if st.session_state.get('mt5_connected', False):
                st.subheader("Fetch Data")
                
                # Get available symbols from MT5
                if 'mt5_symbols' not in st.session_state:
                    with st.spinner("Loading available symbols..."):
                        st.session_state['mt5_symbols'] = get_available_symbols()
                
                symbols_list = st.session_state.get('mt5_symbols', [])
                
                if len(symbols_list) > 0:
                    # Use selectbox if symbols are available
                    default_idx = 0
                    if 'BTCUSD' in symbols_list:
                        default_idx = symbols_list.index('BTCUSD')
                    elif 'EURUSD' in symbols_list:
                        default_idx = symbols_list.index('EURUSD')
                    
                    col_symbol_select, col_symbol_refresh = st.columns([3, 1])
                    with col_symbol_select:
                        symbol = st.selectbox(
                            "Symbol",
                            options=symbols_list,
                            index=default_idx,
                            help="Select symbol from available MT5 symbols"
                        )
                    with col_symbol_refresh:
                        st.write("")  # Spacer
                        if st.button("🔄", help="Refresh symbol list", key="refresh_symbols"):
                            st.session_state['mt5_symbols'] = get_available_symbols()
                else:
                    # Fallback to text input if no symbols loaded
                    symbol = st.text_input("Symbol", value="BTCUSD", help="e.g., EURUSD, BTCUSD, GBPUSD")
                    if st.button("Refresh Symbols", use_container_width=True, key="refresh_symbols_fallback"):
                        st.session_state['mt5_symbols'] = get_available_symbols()
                
                timeframe = st.selectbox(
                    "Timeframe",
                    options=list(TIMEFRAME_MAP.keys()),
                    index=4,  # Default to H1
                    help="Select the chart timeframe"
                )
                
                col_date_start, col_date_end = st.columns(2)
                with col_date_start:
                    start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
                with col_date_end:
                    end_date = st.date_input("End Date", value=datetime.now())
                
                if st.button("Fetch from MT5", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Fetching data from MT5..."):
                            df_raw = cached_get_ohlc_history(
                                symbol=symbol,
                                timeframe=timeframe,
                                date_from=datetime.combine(start_date, datetime.min.time()),
                                date_to=datetime.combine(end_date, datetime.max.time())
                            )
                            # Normalize all data sources for consistency
                            df = normalize_csv_data(df_raw)
                            st.session_state['data'] = df
                            st.session_state['data_symbol'] = symbol
                            st.session_state['data_timeframe'] = timeframe
                            st.success(f"✅ Fetched {len(df)} bars")
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
                    df_raw = pd.read_csv(uploaded_file)
                    df = normalize_csv_data(df_raw)
                    st.session_state['data'] = df
                    st.session_state['data_symbol'] = uploaded_file.name.replace('.csv', '')
                    st.success(f"✅ Loaded {len(df)} rows")
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
                try:
                    dates = pd.date_range(start='2023-01-01', periods=rows, freq='H')
                    price_walk = 1.10 + np.cumsum(np.random.normal(0, volatility, rows))
                    df_raw = pd.DataFrame({
                        'time': dates,
                        'open': price_walk,
                        'high': price_walk + volatility/2,
                        'low': price_walk - volatility/2,
                        'close': price_walk + np.random.normal(0, volatility/10, rows)
                    })
                    # Normalize all data sources for consistency
                    df = normalize_csv_data(df_raw)
                    st.session_state['data'] = df
                    st.session_state['data_symbol'] = "SIMULATION"
                    st.success(f"✅ Generated {len(df)} bars")
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
            
            if 'data' in st.session_state and not st.session_state['data'].empty:
                df = st.session_state['data']
                st.info(f"📊 Loaded: {len(df)} bars")
    
    with col_right:
        # Display chart if data is available
        if 'data' in st.session_state and not st.session_state['data'].empty:
            df = st.session_state['data']
            
            # Check if volume is available
            has_volume = 'volume' in df.columns
            
            symbol_name = st.session_state.get('data_symbol', 'Data')
            
            # Create candlestick chart
            if has_volume:
                # Create subplots: price on top, volume on bottom
                from plotly.subplots import make_subplots
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.8, 0.2],
                    subplot_titles=(f"{symbol_name} - OHLC Chart", None)
                )
                
                # Add candlestick to first subplot
                fig.add_trace(
                    go.Candlestick(
                        x=df['time'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add volume bars to second subplot matching OHLC candle colors (vectorized)
                colors = np.where(df['close'] < df['open'], '#ef5350', '#26a69a')
                fig.add_trace(
                    go.Bar(
                        x=df['time'],
                        y=df['volume'],
                        name="Volume",
                        marker=dict(
                            color=colors,
                            line=dict(width=0),
                            opacity=1.0  # Full opacity
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=700,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
            else:
                # Single chart without volume
                fig = go.Figure(data=[go.Candlestick(
                    x=df['time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                )])
                
                fig.update_layout(
                    title=f"{symbol_name} - OHLC Chart",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data info with volume availability
            volume_info = "" if has_volume else " | (Volume not available)"
            st.caption(f"Total bars: {len(df)} | Period: {df['time'].min()} to {df['time'].max()}{volume_info}")
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
                st.write("Commission per Lot (per side):")
            with col_comm_input:
                commission_per_lot = st.number_input("", min_value=0.0, max_value=100.0, value=0.0, step=0.1, label_visibility="collapsed", help="Commission per lot per side. Round-turn = 2x this value")
            
            # Advanced instrument parameters (collapsible)
            with st.expander("Advanced: Instrument Parameters"):
                symbol_name = st.session_state.get('data_symbol', 'DEFAULT')
                instrument_params = get_instrument_params(symbol_name)
                
                st.write(f"**Detected Symbol:** {symbol_name}")
                st.write(f"**Type:** {instrument_params['type']}")
                st.write(f"**Contract Size:** {instrument_params['contract_size']:,}")
                st.write(f"**Pip Size:** {instrument_params['pip_size']}")
                st.write(f"**Pip Value per Lot:** ${instrument_params['pip_value_per_lot']:.2f}")
                st.write(f"**Quote Currency:** {instrument_params['quote_currency']}")
                st.caption("Parameters are auto-detected from symbol. Edit data_utils.py to customize.")
            
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
                
                days_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
                selected_days = st.multiselect(
                    "Days of Week",
                    days_of_week,
                    default=st.session_state.get('selected_days', ['MON', 'TUE', 'WED', 'THU', 'FRI']),
                    key="days_multiselect"
                )
                st.session_state['selected_days'] = selected_days
            else:
                # Swing mode - no time restrictions, but still filter by days
                start_hour = None
                end_hour = None
                days_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
                selected_days = st.multiselect(
                    "Days of Week",
                    days_of_week,
                    default=st.session_state.get('selected_days', ['MON', 'TUE', 'WED', 'THU', 'FRI']),
                    key="days_multiselect_swing"
                )
                st.session_state['selected_days'] = selected_days
            
            # Execution Costs (optional, collapsible)
            with st.expander("Execution Costs (Spread/Slippage)"):
                col_spread_label, col_spread_input = st.columns([1, 2])
                with col_spread_label:
                    st.write("Spread (pips):")
                with col_spread_input:
                    spread_pips = st.number_input("", min_value=0.0, max_value=1000.0, value=st.session_state.get('spread_pips', 0.0), step=0.1, label_visibility="collapsed", key="spread_pips", help="Bid-ask spread in pips. Applied to both entry and exit")
                
                col_slippage_label, col_slippage_input = st.columns([1, 2])
                with col_slippage_label:
                    st.write("Slippage (pips):")
                with col_slippage_input:
                    slippage_pips = st.number_input("", min_value=0.0, max_value=1000.0, value=st.session_state.get('slippage_pips', 0.0), step=0.1, label_visibility="collapsed", key="slippage_pips", help="Additional slippage in pips beyond spread. Applied to both entry and exit")
            
            # Ensure values are available even when expander is collapsed
            spread_pips = st.session_state.get('spread_pips', 0.0)
            slippage_pips = st.session_state.get('slippage_pips', 0.0)
        
        with col_right:
            # Apply filters to data
            filtered_df = df.copy()
            
            # Filter by days of week (applies to both DT and Swing)
            day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
            filtered_df['day_of_week'] = filtered_df['time'].dt.dayofweek
            selected_day_nums = [day_map[d] for d in selected_days if d in day_map]
            filtered_df = filtered_df[filtered_df['day_of_week'].isin(selected_day_nums)]
            
            # Get instrument parameters for profit calculation
            symbol_name = st.session_state.get('data_symbol', 'DEFAULT')
            instrument_params = get_instrument_params(symbol_name)
            
            # Generate trades based on mode (with spread/slippage applied)
            # Note: selected_days filtering is done upstream, so we don't pass it to generate_trades_dt
            if mode == "DT":
                trades_df = generate_trades_dt(
                    filtered_df, start_hour, end_hour, direction,
                    spread_pips, slippage_pips, instrument_params
                )
            else:
                trades_df = generate_trades_swing(
                    filtered_df, direction,
                    spread_pips, slippage_pips, instrument_params
                )
            
            if len(trades_df) == 0:
                st.warning("No trades found matching the selected filters")
                filtered_df = pd.DataFrame()
            else:
                # Calculate profit for each trade (prices already include spread/slippage)
                trades_df['profit'] = trades_df.apply(
                    lambda row: calculate_profit(
                        row['open_price'],
                        row['close_price'],
                        direction,
                        position_size,
                        instrument_params,
                        commission_per_lot
                    ),
                    axis=1
                )
                
                filtered_df = trades_df[['date', 'entry_time', 'exit_time', 'open_price', 'close_price', 'profit']].copy()
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

# --- TAB 3: STRATEGY TESTER ---
elif st.session_state['current_tab'] == 'Strategy Tester':
    if 'data' not in st.session_state or st.session_state['data'].empty:
        st.warning("⚠️ Please load data first in the 'Data' tab")
        st.info("Go to Data tab to connect MT5 or upload CSV file")
    else:
        df = st.session_state['data'].copy()
        
        col_left, col_right = st.columns([0.7, 2.3])
        
        with col_left:
            st.subheader("Strategy Configuration")
            
            # Session selector
            session_mode = st.radio(
                "Session",
                ["Use mapped session", "Use all market time"],
                help="Use mapped session applies day/time filters from Market Mapper, otherwise use all data"
            )
            
            st.markdown("---")
            
            # Direction selector
            direction_mode = st.radio(
                "Direction:",
                ["Long", "Short", "Both"],
                index=2,  # Default to "Both"
                horizontal=True,
                key="strategy_direction"
            )
            
            st.markdown("---")
            
            # Entry Strategy
            st.write("**Entry Strategy**")
            entry_type = st.selectbox(
                "Entry",
                ["SMA Crossover", "RSI Threshold", "MACD Cross"],
                key="entry_strategy"
            )
            
            entry_params = {}
            
            if entry_type == "SMA Crossover":
                col_fast, col_slow = st.columns(2)
                with col_fast:
                    entry_params['fast'] = st.number_input("Fast SMA", min_value=1, max_value=200, value=10, step=1, key="sma_fast")
                with col_slow:
                    entry_params['slow'] = st.number_input("Slow SMA", min_value=1, max_value=200, value=50, step=1, key="sma_slow")
            
            elif entry_type == "RSI Threshold":
                entry_mode = st.radio("Mode", ["Mean Reversion", "Momentum"], key="rsi_mode")
                entry_params['mode'] = entry_mode.lower().replace(' ', '_')
                col_len, col_os, col_ob = st.columns(3)
                with col_len:
                    entry_params['length'] = st.number_input("RSI Length", min_value=2, max_value=50, value=14, step=1, key="rsi_length")
                with col_os:
                    entry_params['oversold'] = st.number_input("Oversold", min_value=0, max_value=50, value=30, step=1, key="rsi_oversold")
                with col_ob:
                    entry_params['overbought'] = st.number_input("Overbought", min_value=50, max_value=100, value=70, step=1, key="rsi_overbought")
            
            elif entry_type == "MACD Cross":
                col_fast, col_slow, col_sig = st.columns(3)
                with col_fast:
                    entry_params['fast'] = st.number_input("Fast", min_value=1, max_value=50, value=12, step=1, key="macd_fast")
                with col_slow:
                    entry_params['slow'] = st.number_input("Slow", min_value=1, max_value=50, value=26, step=1, key="macd_slow")
                with col_sig:
                    entry_params['signal'] = st.number_input("Signal", min_value=1, max_value=50, value=9, step=1, key="macd_signal")
                entry_mode = st.radio("Entry Mode", ["Histogram Cross", "Signal Cross"], key="macd_entry_mode")
                entry_params['mode'] = entry_mode.lower().replace(' ', '_')
            
            st.markdown("---")
            
            # Exit Strategy
            st.write("**Exit Strategy**")
            exit_options = ["Fixed TP/SL (ATR)", "ATR Trailing Stop", "SMA Cross Back"]
            
            # Auto-select SMA Cross Back exit when SMA entry is selected
            if entry_type == "SMA Crossover":
                if 'exit_strategy' not in st.session_state or st.session_state.get('exit_strategy') != "SMA Cross Back":
                    st.session_state['exit_strategy'] = "SMA Cross Back"
                default_exit_index = 2  # SMA Cross Back
            else:
                # Keep current selection if it exists, otherwise default to first option
                current_exit = st.session_state.get('exit_strategy', exit_options[0])
                default_exit_index = exit_options.index(current_exit) if current_exit in exit_options else 0
            
            exit_type = st.selectbox(
                "Exit",
                exit_options,
                key="exit_strategy",
                index=default_exit_index
            )
            
            exit_params = {}
            
            if exit_type == "Fixed TP/SL (ATR)":
                col_atr_len, col_sl, col_tp = st.columns(3)
                with col_atr_len:
                    exit_params['atr_length'] = st.number_input("ATR Length", min_value=1, max_value=50, value=14, step=1, key="atr_length")
                with col_sl:
                    exit_params['sl_atr_mult'] = st.number_input("SL (ATR)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="sl_atr")
                with col_tp:
                    exit_params['tp_atr_mult'] = st.number_input("TP (ATR)", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="tp_atr")
            
            elif exit_type == "ATR Trailing Stop":
                col_atr_len, col_mult = st.columns(2)
                with col_atr_len:
                    exit_params['atr_length'] = st.number_input("ATR Length", min_value=1, max_value=50, value=14, step=1, key="atr_length_trail")
                with col_mult:
                    exit_params['atr_multiplier'] = st.number_input("ATR Multiplier", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="atr_mult")
            
            elif exit_type == "SMA Cross Back":
                st.info("Uses reverse of entry strategy (e.g., SMA cross back for SMA entry)")
            
            st.markdown("---")
            
            # Position and Costs
            st.write("**Position & Costs**")
            position_size = st.number_input("Position Size (Lots)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key="strat_pos_size")
            commission_per_lot = st.number_input("Commission per Lot (per side)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="strat_commission")
            
            with st.expander("Execution Costs (Spread/Slippage)"):
                spread_pips = st.number_input("Spread (pips)", min_value=0.0, max_value=1000.0, value=st.session_state.get('spread_pips', 0.0), step=0.1, key="strat_spread")
                slippage_pips = st.number_input("Slippage (pips)", min_value=0.0, max_value=1000.0, value=st.session_state.get('slippage_pips', 0.0), step=0.1, key="strat_slippage")
            
            st.markdown("---")
            
            # Run button
            run_backtest_clicked = st.button("Run Backtest", type="primary", use_container_width=True, key="run_backtest")
        
        with col_right:
            # Apply session filter if needed
            if session_mode == "Use mapped session":
                if 'selected_days' in st.session_state:
                    day_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
                    selected_day_nums = [day_map[d] for d in st.session_state['selected_days'] if d in day_map]
                    df['day_of_week'] = df['time'].dt.dayofweek
                    df = df[df['day_of_week'].isin(selected_day_nums)].copy()
            
            # Run backtest if button was clicked
            if run_backtest_clicked:
                
                try:
                    with st.spinner("Running backtest..."):
                        # Add position size to entry params
                        entry_params['position_size'] = position_size
                        
                        # Get instrument parameters for profit calculation
                        symbol_name = st.session_state.get('data_symbol', 'DEFAULT')
                        instrument_params = get_instrument_params(symbol_name)
                        
                        # Create strategy with direction mode
                        strategy = ConfigurableStrategy(entry_type, exit_type, entry_params, exit_params, direction_mode)
                        
                        # Create and run backtest engine
                        engine = BacktestEngine(
                            df,
                            strategy,
                            initial_balance=10000,
                            commission=commission_per_lot,
                            instrument_params=instrument_params
                        )
                        
                        # Run backtest
                        results_df = engine.run()
                        
                        # Store results
                        st.session_state['backtest_results'] = {
                            'trades': results_df,
                            'engine': engine,
                            'data': df
                        }
                        
                        st.success("✅ Backtest completed!")
                
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Display results
            if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
                results = st.session_state['backtest_results']
                trades_df = results['trades']
                engine = results['engine']
                data_df = results['data']
                
                if len(trades_df) > 0:
                    # Calculate metrics
                    total_trades = len(trades_df)
                    winning_trades = len(trades_df[trades_df['profit'] > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    total_pnl = trades_df['profit'].sum()
                    
                    # Calculate cumulative equity
                    trades_df = trades_df.copy()
                    trades_df['cumulative_equity'] = 10000 + trades_df['profit'].cumsum()
                    
                    # Display metrics
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric("Total Trades", total_trades)
                    with col_met2:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with col_met3:
                        st.metric("Total P&L", f"${total_pnl:,.2f}")
                    
                    st.markdown("---")
                    
                    # View switcher
                    view_mode = st.radio(
                        "View:",
                        ["Graph", "Table", "Distribution"],
                        horizontal=True,
                        key="strategy_view_mode"
                    )
                    
                    st.markdown("---")
                    
                    if view_mode == "Graph":
                        # Equity curve
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trades_df.index,
                            y=trades_df['cumulative_equity'],
                            mode='lines',
                            name='Equity Curve',
                            line=dict(color='#26a69a', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Equity Curve",
                            xaxis_title="Trade Number",
                            yaxis_title="Equity ($)",
                            height=400,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Price chart with trades (only in Graph view)
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Candlestick(
                            x=data_df['time'],
                            open=data_df['open'],
                            high=data_df['high'],
                            low=data_df['low'],
                            close=data_df['close'],
                            name='Price'
                        ))
                        
                        # Add trade markers on price chart
                        for idx, trade in trades_df.iterrows():
                            entry_time = trade['open_time']
                            exit_time = trade['close_time']
                            
                            fig_price.add_trace(go.Scatter(
                                x=[entry_time],
                                y=[trade['open_price']],
                                mode='markers',
                                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='white')),
                                name="Entry",
                                showlegend=False
                            ))
                            
                            exit_color = 'red' if trade['profit'] < 0 else 'blue'
                            fig_price.add_trace(go.Scatter(
                                x=[exit_time],
                                y=[trade['close_price']],
                                mode='markers',
                                marker=dict(symbol='triangle-down', size=12, color=exit_color, line=dict(width=2, color='white')),
                                name="Exit",
                                showlegend=False
                            ))
                        
                        fig_price.update_layout(
                            title="Price Chart with Trade Markers",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            height=500,
                            template="plotly_dark",
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig_price, use_container_width=True)
                    
                    elif view_mode == "Table":
                        # Prepare table data
                        display_df = trades_df.copy()
                        display_df['Entry Time'] = pd.to_datetime(display_df['open_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_df['Exit Time'] = pd.to_datetime(display_df['close_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_df['Direction'] = display_df['direction'].str.capitalize()
                        display_df['Entry Price'] = display_df['open_price'].round(2)
                        display_df['Exit Price'] = display_df['close_price'].round(2)
                        display_df['P&L'] = display_df['profit'].round(2)
                        display_df['Exit Reason'] = display_df['exit_reason']
                        
                        # Select columns to display
                        table_cols = ['Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price', 'P&L', 'Exit Reason']
                        display_df = display_df[table_cols]
                        
                        st.dataframe(display_df, use_container_width=True, height=600)
                        
                        # Show summary statistics below table
                        st.markdown("---")
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        with col_sum1:
                            st.metric("Total Trades", total_trades)
                        with col_sum2:
                            st.metric("Winning Trades", winning_trades)
                        with col_sum3:
                            st.metric("Losing Trades", total_trades - winning_trades)
                        with col_sum4:
                            avg_profit = trades_df['profit'].mean()
                            st.metric("Average P&L", f"${avg_profit:,.2f}")
                    
                    elif view_mode == "Distribution":
                        # Prepare distribution data
                        dist_df = trades_df.copy()
                        dist_df['open_time'] = pd.to_datetime(dist_df['open_time'])
                        dist_df['is_win'] = dist_df['profit'] > 0
                        dist_df['direction_label'] = dist_df['direction'].str.capitalize()
                        dist_df['day_of_week'] = dist_df['open_time'].dt.day_name()
                        dist_df['hour'] = dist_df['open_time'].dt.hour
                        
                        # Create subplots
                        fig_dist = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=('By Buy/Sell', 'By Day of Week', 'By Trading Hour'),
                            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
                        )
                        
                        # 1. Distribution by Buy/Sell
                        buy_sell_counts = dist_df.groupby(['direction_label', 'is_win']).size().reset_index(name='count')
                        
                        directions = ['Long', 'Short']
                        x_buysell = []
                        y_win = []
                        y_loss = []
                        
                        for direction in directions:
                            win_count = buy_sell_counts[
                                (buy_sell_counts['direction_label'] == direction) & 
                                (buy_sell_counts['is_win'] == True)
                            ]['count'].sum()
                            loss_count = buy_sell_counts[
                                (buy_sell_counts['direction_label'] == direction) & 
                                (buy_sell_counts['is_win'] == False)
                            ]['count'].sum()
                            
                            x_buysell.append(direction)
                            y_win.append(win_count)
                            y_loss.append(loss_count)
                        
                        fig_dist.add_trace(
                            go.Bar(x=x_buysell, y=y_win, name='Win', marker_color='green', showlegend=True),
                            row=1, col=1
                        )
                        fig_dist.add_trace(
                            go.Bar(x=x_buysell, y=y_loss, name='Loss', marker_color='red', showlegend=True),
                            row=1, col=1
                        )
                        
                        # 2. Distribution by Day of Week
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = dist_df.groupby(['day_of_week', 'is_win']).size().reset_index(name='count')
                        
                        x_days = []
                        y_win_days = []
                        y_loss_days = []
                        
                        for day in day_order:
                            win_count = day_counts[
                                (day_counts['day_of_week'] == day) & (day_counts['is_win'] == True)
                            ]['count'].sum()
                            loss_count = day_counts[
                                (day_counts['day_of_week'] == day) & (day_counts['is_win'] == False)
                            ]['count'].sum()
                            
                            if win_count > 0 or loss_count > 0:
                                x_days.append(day[:3])
                                y_win_days.append(win_count)
                                y_loss_days.append(loss_count)
                        
                        if x_days:
                            fig_dist.add_trace(
                                go.Bar(x=x_days, y=y_win_days, name='Win', marker_color='green', showlegend=False),
                                row=1, col=2
                            )
                            fig_dist.add_trace(
                                go.Bar(x=x_days, y=y_loss_days, name='Loss', marker_color='red', showlegend=False),
                                row=1, col=2
                            )
                        
                        # 3. Distribution by Trading Hour
                        hour_counts = dist_df.groupby(['hour', 'is_win']).size().reset_index(name='count')
                        
                        x_hours = []
                        y_win_hours = []
                        y_loss_hours = []
                        
                        for hour in range(24):
                            win_count = hour_counts[
                                (hour_counts['hour'] == hour) & (hour_counts['is_win'] == True)
                            ]['count'].sum()
                            loss_count = hour_counts[
                                (hour_counts['hour'] == hour) & (hour_counts['is_win'] == False)
                            ]['count'].sum()
                            
                            if win_count > 0 or loss_count > 0:
                                x_hours.append(f"{hour:02d}:00")
                                y_win_hours.append(win_count)
                                y_loss_hours.append(loss_count)
                        
                        if x_hours:
                            fig_dist.add_trace(
                                go.Bar(x=x_hours, y=y_win_hours, name='Win', marker_color='green', showlegend=False),
                                row=1, col=3
                            )
                            fig_dist.add_trace(
                                go.Bar(x=x_hours, y=y_loss_hours, name='Loss', marker_color='red', showlegend=False),
                                row=1, col=3
                            )
                        
                        fig_dist.update_layout(
                            height=500,
                            template="plotly_dark",
                            barmode='group',
                            showlegend=True
                        )
                        
                        fig_dist.update_xaxes(title_text="", row=1, col=1)
                        fig_dist.update_xaxes(title_text="Day", row=1, col=2)
                        fig_dist.update_xaxes(title_text="Hour", row=1, col=3)
                        fig_dist.update_yaxes(title_text="Number of Trades", row=1, col=1)
                        fig_dist.update_yaxes(title_text="Number of Trades", row=1, col=2)
                        fig_dist.update_yaxes(title_text="Number of Trades", row=1, col=3)
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                else:
                    st.warning("No trades generated by the strategy")
            else:
                st.info("Configure strategy and click 'Run Backtest' to see results")
