import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time, timedelta
import yfinance as yf
from backtester import BacktestEngine
from strategy import MyPerfectStrategy
from data_utils import (
    normalize_csv_data, get_instrument_params, calculate_profit,
    generate_trades_dt, generate_trades_swing
)
from strategy_builder import ConfigurableStrategy
from vectorbt_engine import run_vectorbt_backtest
import vectorbt as vbt

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
            ["MT5 (MetaTrader5)", "CSV Upload", "Yahoo Finance"],
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
        # --- YAHOO FINANCE ---
        else:
            st.subheader("Yahoo Finance Data")
            st.info("Fetch market data from Yahoo Finance")
            
            col_yf_sym, col_yf_int = st.columns([2, 1])
            with col_yf_sym:
                yf_symbol = st.text_input("Symbol", value="SPY", help="e.g. SPY, AAPL, BTC-USD")
            with col_yf_int:
                yf_interval = st.selectbox(
                    "Interval",
                    options=["1d", "1h", "15m", "5m", "1m"],
                    index=1
                )
            
            col_yf_start, col_yf_end = st.columns(2)
            with col_yf_start:
                yf_start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365), key="yf_start")
            with col_yf_end:
                yf_end_date = st.date_input("End Date", value=datetime.now(), key="yf_end")
            
            if st.button("Fetch YF Data", use_container_width=True):
                try:
                    with st.spinner(f"Fetching {yf_symbol} data..."):
                        # Fetch data
                        df_raw = yf.download(
                            tickers=yf_symbol,
                            start=yf_start_date,
                            end=yf_end_date + timedelta(days=1),
                            interval=yf_interval,
                            progress=False
                        )
                        
                        if df_raw.empty:
                            st.error(f"No data found for {yf_symbol}")
                        else:
                            # Handle MultiIndex columns if present (yfinance 0.2+)
                            if isinstance(df_raw.columns, pd.MultiIndex):
                                # If we have a MultiIndex with (Price, Ticker), we just want the Price part
                                # But if multiple tickers were downloaded it might be different. 
                                # Here we only download one ticker.
                                try:
                                    df_raw.columns = df_raw.columns.get_level_values(0)
                                except:
                                    pass
                            
                            # Reset index to get Date/Datetime as column
                            df_raw = df_raw.reset_index()
                            
                            # Normalize all data sources for consistency
                            df = normalize_csv_data(df_raw)
                            st.session_state['data'] = df
                            st.session_state['data_symbol'] = yf_symbol
                            st.session_state['data_timeframe'] = yf_interval
                            st.success(f"✅ Fetched {len(df)} bars")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
            
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
                
                # Add candlestick to first subplot - optimize for performance by downsampling
                # Downsample data if too large for better performance
                display_df = df.copy()
                if len(display_df) > 5000:
                    # Downsample to max 5000 points while preserving OHLC structure
                    step = len(display_df) // 5000
                    display_df = display_df.iloc[::step].copy()
                
                fig.add_trace(
                    go.Candlestick(
                        x=display_df['time'],
                        open=display_df['open'],
                        high=display_df['high'],
                        low=display_df['low'],
                        close=display_df['close'],
                        name="Price",
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ),
                    row=1, col=1
                )
                
                # Add volume bars to second subplot matching OHLC candle colors (vectorized)
                # Use Scattergl for better performance with large datasets
                # Use same downsampled data for volume to keep alignment
                colors = np.where(display_df['close'] < display_df['open'], '#ef5350', '#26a69a')
                fig.add_trace(
                    go.Scattergl(
                        x=display_df['time'],
                        y=display_df['volume'],
                        mode='markers',
                        name="Volume",
                        marker=dict(
                            color=colors,
                            size=3,
                            opacity=0.8,
                            line=dict(width=0)
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=700,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    showlegend=False,
                    uirevision='ohlc_chart',  # Prevents unnecessary redraws
                    hovermode='x unified',  # More efficient hover
                    dragmode='pan'  # Default to pan for better performance
                )
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
            else:
                # Single chart without volume - optimize for performance by downsampling
                # Downsample data if too large for better performance
                display_df = df.copy()
                if len(display_df) > 5000:
                    # Downsample to max 5000 points while preserving OHLC structure
                    step = len(display_df) // 5000
                    display_df = display_df.iloc[::step].copy()
                
                fig = go.Figure(data=[go.Candlestick(
                    x=display_df['time'],
                    open=display_df['open'],
                    high=display_df['high'],
                    low=display_df['low'],
                    close=display_df['close'],
                    name='OHLC',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                )])
                
                fig.update_layout(
                    title=f"{symbol_name} - OHLC Chart",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    uirevision='ohlc_single',  # Prevents unnecessary redraws
                    hovermode='x unified',  # More efficient hover
                    dragmode='pan'  # Default to pan for better performance
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
                    # Create equity curve chart - use Scattergl for performance
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scattergl(
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
                        hovermode='x unified',
                        uirevision='market_mapper_equity',  # Prevents unnecessary redraws
                        dragmode='pan'  # Default to pan for better performance
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
            
            # Track previous entry strategy to detect changes
            prev_entry_strategy = st.session_state.get('prev_entry_strategy', None)
            entry_strategy_changed = (prev_entry_strategy != entry_type)
            if entry_strategy_changed:
                st.session_state['prev_entry_strategy'] = entry_type
            
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
                col_len = st.columns(1)[0]
                with col_len:
                    entry_params['length'] = st.number_input("RSI Length", min_value=2, max_value=50, value=14, step=1, key="rsi_length")
                
                if entry_mode == "Mean Reversion":
                    col_os, col_ob = st.columns(2)
                    with col_os:
                        entry_params['oversold'] = st.number_input("Oversold", min_value=0, max_value=50, value=30, step=1, key="rsi_oversold")
                    with col_ob:
                        entry_params['overbought'] = st.number_input("Overbought", min_value=50, max_value=100, value=70, step=1, key="rsi_overbought")
                else:  # Momentum
                    col_cross = st.columns(1)[0]
                    with col_cross:
                        entry_params['crossing_threshold'] = st.number_input("RSI Crossing", min_value=0, max_value=100, value=50, step=1, key="rsi_crossing")
            
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
            
            # Exit Strategy - dynamically filtered based on entry
            st.write("**Exit Strategy**")
            
            # Define which exit strategies are compatible with each entry strategy
            exit_compatibility = {
                "SMA Crossover": ["SMA Cross Back", "Fixed TP/SL (ATR)", "ATR Trailing Stop", "Time based"],
                "RSI Threshold": ["Fixed TP/SL (ATR)", "ATR Trailing Stop", "Time based"],
                "MACD Cross": ["Fixed TP/SL (ATR)", "ATR Trailing Stop", "Time based"]
            }
            
            # Get available exit options for current entry
            available_exit_options = exit_compatibility.get(entry_type, ["Fixed TP/SL (ATR)", "ATR Trailing Stop", "Time based"])
            
            # Auto-select exit strategy only when entry strategy changes
            if entry_strategy_changed:
                if entry_type == "SMA Crossover":
                    st.session_state['exit_strategy'] = "SMA Cross Back"
                else:
                    st.session_state['exit_strategy'] = "Fixed TP/SL (ATR)"
            
            # Get current exit selection, ensure it's in available options
            current_exit = st.session_state.get('exit_strategy', available_exit_options[0])
            if current_exit not in available_exit_options:
                current_exit = available_exit_options[0]
                st.session_state['exit_strategy'] = current_exit
            
            default_exit_index = available_exit_options.index(current_exit)
            
            exit_type = st.selectbox(
                "Exit",
                available_exit_options,
                key="exit_strategy",
                index=default_exit_index,
                help="Available exit strategies depend on your entry strategy"
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
                if entry_type == "SMA Crossover":
                    st.info("✓ Exits when fast SMA crosses back below/above slow SMA (opposite of entry)")
                else:
                    st.warning("⚠️ SMA Cross Back only works with SMA Crossover entry")
            
            elif exit_type == "Time based":
                exit_time_mode = st.selectbox(
                    "Time Mode",
                    ["EOD"],
                    key="exit_time_mode"
                )
                exit_params['time_mode'] = exit_time_mode.lower()
                st.info("Closes all positions at the end of each trading day")
            
            st.markdown("---")
            
            # Position and Costs
            st.write("**Position & Costs**")
            
            # Starting balance
            initial_balance = st.number_input(
                "Starting Balance ($)", 
                min_value=100.0, 
                max_value=1000000.0, 
                value=10000.0, 
                step=100.0, 
                key="initial_balance",
                help="Initial account balance for backtesting"
            )
            
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
                    with st.spinner("Running backtest with VectorBT..."):
                        # Add position size to entry params
                        entry_params['position_size'] = position_size
                        
                        # Run vectorbt backtest
                        portfolio, indicators = run_vectorbt_backtest(
                            df,
                            entry_type,
                            exit_type,
                            entry_params,
                            exit_params,
                            direction_mode,
                            initial_cash=initial_balance,
                            commission=commission_per_lot / 10000  # Convert to fraction
                        )
                        
                        # Store results
                        st.session_state['backtest_results'] = {
                            'portfolio': portfolio,
                            'indicators': indicators,
                            'data': df,
                            'entry_type': entry_type,
                            'exit_type': exit_type
                        }
                        
                        st.success("✅ Backtest completed with VectorBT!")
                
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Display results
            if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
                results = st.session_state['backtest_results']
                portfolio = results['portfolio']
                indicators = results['indicators']
                data_df = results['data'].copy()
                entry_type = results.get('entry_type', '')
                exit_type = results.get('exit_type', '')
                
                # Get portfolio stats
                stats = portfolio.stats()
                
                # Display key metrics
                col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
                with col_met1:
                    st.metric("Total Trades", int(stats['Total Trades']))
                with col_met2:
                    st.metric("Win Rate", f"{stats['Win Rate [%]']:.1f}%")
                with col_met3:
                    st.metric("Total Return", f"{stats['Total Return [%]']:.2f}%")
                with col_met4:
                    st.metric("Max Drawdown", f"{stats['Max Drawdown [%]']:.2f}%")
                with col_met5:
                    st.metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0):.2f}")
                
                st.markdown("---")
                
                # View switcher
                view_mode = st.radio(
                    "View:",
                    ["Equity Curve", "Price Chart with Indicators", "Trades Table", "Analytics", "Stats"],
                    horizontal=True,
                    key="strategy_view_mode"
                )
                
                st.markdown("---")
                
                if view_mode == "Equity Curve":
                    # Use vectorbt's built-in plotting
                    st.subheader("Portfolio Value Over Time")
                    
                    # Get equity data
                    equity = portfolio.value()
                    
                    # Create plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity.index,
                        y=equity.values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#26a69a', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(38, 166, 154, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Equity Curve",
                        xaxis_title="Time",
                        yaxis_title="Portfolio Value ($)",
                        height=500,
                        template="plotly_dark",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Drawdown chart
                    st.subheader("Drawdown")
                    drawdowns = portfolio.drawdown() * 100  # Convert to percentage
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=drawdowns.index,
                        y=drawdowns.values,
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='#ef5350', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(239, 83, 80, 0.1)'
                    ))
                    
                    fig_dd.update_layout(
                        title="Drawdown Over Time",
                        xaxis_title="Time",
                        yaxis_title="Drawdown (%)",
                        height=300,
                        template="plotly_dark",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                elif view_mode == "Price Chart with Indicators":
                    st.subheader(f"Price Chart with {entry_type} Signals")
                    
                    # Set time index for data_df (vectorbt uses time index)
                    data_df_indexed = data_df.set_index('time')
                    
                    # Create subplots based on indicators
                    num_subplots = 1
                    subplot_titles = ["Price"]
                    
                    # Determine if we need additional subplots
                    if 'rsi' in indicators:
                        num_subplots += 1
                        subplot_titles.append("RSI")
                    if 'macd' in indicators:
                        num_subplots += 1
                        subplot_titles.append("MACD")
                    
                    # Create figure with subplots
                    row_heights = [0.6] + [0.2] * (num_subplots - 1)
                    fig = make_subplots(
                        rows=num_subplots,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=row_heights,
                        subplot_titles=subplot_titles
                    )
                    
                    # Downsample if needed
                    display_df = data_df_indexed.copy()
                    if len(display_df) > 5000:
                        step = len(display_df) // 5000
                        display_df = display_df.iloc[::step]
                    
                    # Add candlestick chart
                    fig.add_trace(
                        go.Candlestick(
                            x=display_df.index,
                            open=display_df['open'],
                            high=display_df['high'],
                            low=display_df['low'],
                            close=display_df['close'],
                            name='Price',
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ),
                        row=1, col=1
                    )
                    
                    # Add SMAs if present
                    if 'sma_fast' in indicators and 'sma_slow' in indicators:
                        sma_fast = indicators['sma_fast']
                        sma_slow = indicators['sma_slow']
                        
                        # Align with display_df
                        if len(display_df) < len(sma_fast):
                            sma_fast_display = sma_fast.loc[display_df.index]
                            sma_slow_display = sma_slow.loc[display_df.index]
                        else:
                            sma_fast_display = sma_fast
                            sma_slow_display = sma_slow
                        
                        fig.add_trace(
                            go.Scatter(
                                x=sma_fast_display.index,
                                y=sma_fast_display.values,
                                mode='lines',
                                name='Fast SMA',
                                line=dict(color='cyan', width=1)
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=sma_slow_display.index,
                                y=sma_slow_display.values,
                                mode='lines',
                                name='Slow SMA',
                                line=dict(color='orange', width=1)
                            ),
                            row=1, col=1
                        )
                    
                    # Add RSI subplot if present
                    current_row = 2
                    if 'rsi' in indicators:
                        rsi = indicators['rsi']
                        
                        # Align with display_df
                        if len(display_df) < len(rsi):
                            rsi_display = rsi.loc[display_df.index]
                        else:
                            rsi_display = rsi
                        
                        fig.add_trace(
                            go.Scatter(
                                x=rsi_display.index,
                                y=rsi_display.values,
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple', width=2)
                            ),
                            row=current_row, col=1
                        )
                        
                        # Add RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, opacity=0.5)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, opacity=0.5)
                        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1, opacity=0.3)
                        
                        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
                        current_row += 1
                    
                    # Add MACD subplot if present
                    if 'macd' in indicators:
                        macd = indicators['macd']
                        macd_signal = indicators['macd_signal']
                        macd_hist = indicators['macd_hist']
                        
                        # Align with display_df
                        if len(display_df) < len(macd):
                            macd_display = macd.loc[display_df.index]
                            signal_display = macd_signal.loc[display_df.index]
                            hist_display = macd_hist.loc[display_df.index]
                        else:
                            macd_display = macd
                            signal_display = macd_signal
                            hist_display = macd_hist
                        
                        fig.add_trace(
                            go.Scatter(
                                x=macd_display.index,
                                y=macd_display.values,
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue', width=1.5)
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=signal_display.index,
                                y=signal_display.values,
                                mode='lines',
                                name='Signal',
                                line=dict(color='orange', width=1.5)
                            ),
                            row=current_row, col=1
                        )
                        
                        # Histogram as bar chart
                        colors = ['green' if val > 0 else 'red' for val in hist_display.values]
                        fig.add_trace(
                            go.Bar(
                                x=hist_display.index,
                                y=hist_display.values,
                                name='Histogram',
                                marker_color=colors,
                                opacity=0.5
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                    
                    # Add trade entry/exit markers
                    trades = portfolio.trades.records_readable
                    if len(trades) > 0:
                        # Entry points - vectorbt uses different column names
                        entry_times = pd.to_datetime(trades['Entry Timestamp'])
                        # Try different possible column names for entry price
                        if 'Avg Entry Price' in trades.columns:
                            entry_prices = trades['Avg Entry Price']
                        elif 'Entry Price' in trades.columns:
                            entry_prices = trades['Entry Price']
                        else:
                            # Fallback: use the first price-related column
                            entry_prices = trades.iloc[:, 3]  # Usually the 4th column
                        
                        fig.add_trace(
                            go.Scatter(
                                x=entry_times,
                                y=entry_prices,
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=10,
                                    color='lime',
                                    line=dict(width=1, color='white')
                                ),
                                name='Entry',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                        
                        # Exit points
                        exit_times = pd.to_datetime(trades['Exit Timestamp'])
                        # Try different possible column names for exit price
                        if 'Avg Exit Price' in trades.columns:
                            exit_prices = trades['Avg Exit Price']
                        elif 'Exit Price' in trades.columns:
                            exit_prices = trades['Exit Price']
                        else:
                            # Fallback: use the price-related column
                            exit_prices = trades.iloc[:, 4]  # Usually the 5th column
                        
                        exit_colors = ['red' if pnl < 0 else 'blue' for pnl in trades['PnL']]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=exit_times,
                                y=exit_prices,
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=10,
                                    color=exit_colors,
                                    line=dict(width=1, color='white')
                                ),
                                name='Exit',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    
                    # Update layout
                    fig.update_layout(
                        height=600 + (num_subplots - 1) * 200,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    fig.update_xaxes(title_text="Time", row=num_subplots, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif view_mode == "Trades Table":
                    # Get trades from portfolio
                    trades = portfolio.trades.records_readable
                    
                    if len(trades) > 0:
                        # Format trades for display
                        display_trades = trades.copy()
                        display_trades['Entry Time'] = pd.to_datetime(display_trades['Entry Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_trades['Exit Time'] = pd.to_datetime(display_trades['Exit Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_trades['Return %'] = display_trades['Return'] * 100
                        
                        # Select columns - handle different vectorbt column names
                        cols_to_show = ['Entry Time', 'Exit Time', 'Direction', 'Size']
                        
                        # Add entry price column (different names in different vectorbt versions)
                        if 'Avg Entry Price' in display_trades.columns:
                            display_trades['Entry Price'] = display_trades['Avg Entry Price']
                        elif 'Entry Price' in display_trades.columns: # Fallback for older versions
                            display_trades['Entry Price'] = display_trades['Entry Price']
                        if 'Entry Price' in display_trades.columns: # Only add if it exists after checks
                            cols_to_show.append('Entry Price')
                        
                        # Add exit price column
                        if 'Avg Exit Price' in display_trades.columns:
                            display_trades['Exit Price'] = display_trades['Avg Exit Price']
                        elif 'Exit Price' in display_trades.columns: # Fallback for older versions
                            display_trades['Exit Price'] = display_trades['Exit Price']
                        if 'Exit Price' in display_trades.columns: # Only add if it exists after checks
                            cols_to_show.append('Exit Price')
                        
                        # Add PnL and other columns
                        cols_to_show.extend(['PnL', 'Return %'])
                        
                        # Add Status if available
                        if 'Status' in display_trades.columns:
                            cols_to_show.append('Status')
                        
                        # Only select columns that exist
                        cols_to_show = [col for col in cols_to_show if col in display_trades.columns]
                        display_trades = display_trades[cols_to_show]
                        
                        st.dataframe(display_trades, use_container_width=True, height=600)
                        
                        # Summary stats
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Win", f"${display_trades[display_trades['PnL'] > 0]['PnL'].mean():.2f}")
                        with col2:
                            st.metric("Avg Loss", f"${display_trades[display_trades['PnL'] < 0]['PnL'].mean():.2f}")
                        with col3:
                            st.metric("Best Trade", f"${display_trades['PnL'].max():.2f}")
                        with col4:
                            st.metric("Worst Trade", f"${display_trades['PnL'].min():.2f}")
                    else:
                        st.warning("No trades executed")
                
                elif view_mode == "Analytics":
                    st.subheader("Trade Analytics & Distribution")
                    
                    # Get trades from portfolio
                    trades = portfolio.trades.records_readable
                    
                    if len(trades) > 0:
                        # Prepare analytics data
                        analytics_df = trades.copy()
                        analytics_df['Entry Timestamp'] = pd.to_datetime(analytics_df['Entry Timestamp'])
                        analytics_df['is_win'] = analytics_df['PnL'] > 0
                        analytics_df['day_of_week'] = analytics_df['Entry Timestamp'].dt.day_name()
                        analytics_df['hour'] = analytics_df['Entry Timestamp'].dt.hour
                        
                        # Create subplots
                        from plotly.subplots import make_subplots
                        import plotly.graph_objects as go
                        
                        fig_analytics = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=('By Direction (Long/Short)', 'By Day of Week', 'By Trading Hour'),
                            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
                        )
                        
                        # 1. Distribution by Long/Short
                        direction_counts = analytics_df.groupby(['Direction', 'is_win']).size().reset_index(name='count')
                        
                        directions = ['Long', 'Short']
                        x_direction = []
                        y_win = []
                        y_loss = []
                        
                        for direction in directions:
                            win_count = direction_counts[
                                (direction_counts['Direction'] == direction) & 
                                (direction_counts['is_win'] == True)
                            ]['count'].sum()
                            loss_count = direction_counts[
                                (direction_counts['Direction'] == direction) & 
                                (direction_counts['is_win'] == False)
                            ]['count'].sum()
                            
                            if win_count > 0 or loss_count > 0:
                                x_direction.append(direction)
                                y_win.append(win_count)
                                y_loss.append(loss_count)
                        
                        if x_direction:
                            fig_analytics.add_trace(
                                go.Bar(x=x_direction, y=y_win, name='Win', marker_color='#26a69a', showlegend=True),
                                row=1, col=1
                            )
                            fig_analytics.add_trace(
                                go.Bar(x=x_direction, y=y_loss, name='Loss', marker_color='#ef5350', showlegend=True),
                                row=1, col=1
                            )
                        
                        # 2. Distribution by Day of Week
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = analytics_df.groupby(['day_of_week', 'is_win']).size().reset_index(name='count')
                        
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
                                x_days.append(day[:3])  # Use 3-letter abbreviation
                                y_win_days.append(win_count)
                                y_loss_days.append(loss_count)
                        
                        if x_days:
                            fig_analytics.add_trace(
                                go.Bar(x=x_days, y=y_win_days, name='Win', marker_color='#26a69a', showlegend=False),
                                row=1, col=2
                            )
                            fig_analytics.add_trace(
                                go.Bar(x=x_days, y=y_loss_days, name='Loss', marker_color='#ef5350', showlegend=False),
                                row=1, col=2
                            )
                        
                        # 3. Distribution by Trading Hour
                        hour_counts = analytics_df.groupby(['hour', 'is_win']).size().reset_index(name='count')
                        
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
                            fig_analytics.add_trace(
                                go.Bar(x=x_hours, y=y_win_hours, name='Win', marker_color='#26a69a', showlegend=False),
                                row=1, col=3
                            )
                            fig_analytics.add_trace(
                                go.Bar(x=x_hours, y=y_loss_hours, name='Loss', marker_color='#ef5350', showlegend=False),
                                row=1, col=3
                            )
                        
                        # Update layout
                        fig_analytics.update_layout(
                            height=500,
                            template="plotly_dark",
                            barmode='group',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        fig_analytics.update_xaxes(title_text="Direction", row=1, col=1)
                        fig_analytics.update_xaxes(title_text="Day", row=1, col=2)
                        fig_analytics.update_xaxes(title_text="Hour", row=1, col=3)
                        fig_analytics.update_yaxes(title_text="Number of Trades", row=1, col=1)
                        fig_analytics.update_yaxes(title_text="Number of Trades", row=1, col=2)
                        fig_analytics.update_yaxes(title_text="Number of Trades", row=1, col=3)
                        
                        st.plotly_chart(fig_analytics, use_container_width=True)
                        
                        # Add summary statistics for each category
                        st.markdown("---")
                        st.subheader("Detailed Breakdown")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**By Direction**")
                            for direction in ['Long', 'Short']:
                                dir_trades = analytics_df[analytics_df['Direction'] == direction]
                                if len(dir_trades) > 0:
                                    win_rate = (dir_trades['is_win'].sum() / len(dir_trades)) * 100
                                    avg_pnl = dir_trades['PnL'].mean()
                                    st.metric(
                                        f"{direction}",
                                        f"{len(dir_trades)} trades",
                                        f"WR: {win_rate:.1f}% | Avg: ${avg_pnl:.2f}"
                                    )
                        
                        with col2:
                            st.write("**By Day (Best/Worst)**")
                            day_stats = analytics_df.groupby('day_of_week').agg({
                                'PnL': ['sum', 'count', 'mean'],
                                'is_win': 'sum'
                            }).round(2)
                            day_stats.columns = ['Total PnL', 'Trades', 'Avg PnL', 'Wins']
                            day_stats['Win Rate %'] = (day_stats['Wins'] / day_stats['Trades'] * 100).round(1)
                            
                            # Sort by total PnL to find best/worst
                            day_stats_sorted = day_stats.sort_values('Total PnL', ascending=False)
                            
                            if len(day_stats_sorted) > 0:
                                best_day = day_stats_sorted.index[0]
                                best_pnl = day_stats_sorted.iloc[0]['Total PnL']
                                st.metric(f"Best: {best_day[:3]}", f"${best_pnl:.2f}", f"{day_stats_sorted.iloc[0]['Trades']:.0f} trades")
                                
                                if len(day_stats_sorted) > 1:
                                    worst_day = day_stats_sorted.index[-1]
                                    worst_pnl = day_stats_sorted.iloc[-1]['Total PnL']
                                    st.metric(f"Worst: {worst_day[:3]}", f"${worst_pnl:.2f}", f"{day_stats_sorted.iloc[-1]['Trades']:.0f} trades")
                        
                        with col3:
                            st.write("**By Hour (Most Active)**")
                            hour_stats = analytics_df.groupby('hour').agg({
                                'PnL': ['sum', 'count', 'mean'],
                                'is_win': 'sum'
                            }).round(2)
                            hour_stats.columns = ['Total PnL', 'Trades', 'Avg PnL', 'Wins']
                            hour_stats['Win Rate %'] = (hour_stats['Wins'] / hour_stats['Trades'] * 100).round(1)
                            
                            # Sort by number of trades to find most active hours
                            hour_stats_sorted = hour_stats.sort_values('Trades', ascending=False)
                            
                            for i in range(min(3, len(hour_stats_sorted))):
                                hour = hour_stats_sorted.index[i]
                                trades = hour_stats_sorted.iloc[i]['Trades']
                                avg_pnl = hour_stats_sorted.iloc[i]['Avg PnL']
                                wr = hour_stats_sorted.iloc[i]['Win Rate %']
                                st.metric(
                                    f"{hour:02d}:00",
                                    f"{trades:.0f} trades",
                                    f"WR: {wr:.1f}% | Avg: ${avg_pnl:.2f}"
                                )
                    else:
                        st.warning("No trades to analyze")
                
                elif view_mode == "Stats":
                    st.subheader("Detailed Performance Statistics")
                    
                    # Display full stats
                    stats_df = pd.DataFrame({
                        'Metric': stats.index,
                        'Value': stats.values
                    })
                    
                    st.dataframe(stats_df, use_container_width=True, height=600)
            else:
                st.info("Configure strategy and click 'Run Backtest' to see results")
