import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta
from typing import Union

# --- Page Configuration ---
st.set_page_config(
    page_title="Nifty50 Quant Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching ---
# Cache data fetching for performance. Cache for 1 hour.
@st.cache_data(ttl=3600)
def get_nifty50_tickers():
    """
    Fetches the list of Nifty50 tickers from Wikipedia.
    Returns a list of Yahoo Finance compatible tickers (e.g., 'RELIANCE.NS').
    """
    try:
        url = 'https://en.wikipedia.org/wiki/NIFTY_50'
        tables = pd.read_html(url, attrs={'id': 'constituents'})
        if not tables:
            st.error("Could not find Nifty50 constituents table. Using a static list.")
            return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']
        
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [t + ".NS" for t in tickers]
    except Exception as e:
        st.error(f"Error fetching Nifty50 tickers: {e}. Using a static list.")
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    """
    Fetches OHLCV data from Yahoo Finance.
    Renames columns for compatibility with backtesting.py.
    """
    # yfinance timeframe mapping
    tf_map = {
        "15m": "15m",
        "1H": "60m",
        "1D": "1d"
    }
    
    # yfinance intraday data is limited (max 730 days)
    if timeframe in ["15m", "1H"]:
        start_dt = pd.to_datetime(start_date)
        if (pd.to_datetime(end_date) - start_dt).days > 720:
            start_date = (pd.to_datetime(end_date) - timedelta(days=720)).strftime('%Y-%m-%d')
            st.sidebar.warning(f"Intraday data limited to 720 days. Start date adjusted to {start_date}.")

    data = yf.download(ticker, start=start_date, end=end_date, interval=tf_map[timeframe])
    
    if data.empty:
        return pd.DataFrame()
    
    # Rename for backtesting.py compatibility
    data.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    
    # Ensure datetime index is timezone-naive for backtesting.py
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
        
    return data


# --- Vectorized Indicator Functions ---
# These functions will be wrapped by self.I() in the strategy
# This is far more efficient than calculating in a loop.

def calculate_cpr(high, low, close):
    """Calculates Pivot, TC, and BC (Vectorized)"""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc

def is_narrow_cpr(tc, bc, threshold=0.06):
    """Checks for narrow CPR (Vectorized)"""
    cpr_width = np.abs(tc - bc)
    cpr_percentage = (cpr_width / bc) * 100
    return cpr_percentage < threshold

def is_near_sma(close, sma, threshold=0.15):
    """Checks if price is near SMA (Vectorized)"""
    diff_percentage = np.abs(close - sma) / sma * 100
    return diff_percentage <= threshold

def candle_signals(open_price, high_price, low_price, close_price):
    """Calculates the specific candle patterns (Vectorized)"""
    is_green_candle = close_price > open_price
    is_red_candle = close_price < open_price

    # Buy Signal Logic
    prev1_red = is_red_candle.shift(1)
    prev2_red = is_red_candle.shift(2)
    break_prev1_high = high_price > high_price.shift(1)
    break_prev2_high = high_price > high_price.shift(2)
    buy_candle = is_green_candle & ((prev1_red & break_prev1_high) | (prev2_red & break_prev2_high))
    
    # Sell Signal Logic
    prev1_green = is_green_candle.shift(1)
    prev2_green = is_green_candle.shift(2)
    break_prev1_low = low_price < low_price.shift(1)
    break_prev2_low = low_price < low_price.shift(2)
    sell_candle = is_red_candle & ((prev1_green & break_prev1_low) | (prev2_green & break_prev2_low))

    return buy_candle, sell_candle

# --- The Strategy (using backtesting.py) ---

class CprSmaStrategy(Strategy):
    """
    Implements the trading strategy within the backtesting.py framework.
    Parameters are passed from the Streamlit UI.
    """
    # --- Strategy Parameters (will be set from UI) ---
    sma_short_len = 20
    sma_long_len = 200
    cpr_threshold = 0.06
    sma_threshold = 0.15
    sma_lookback = 6
    tp_pct = 0.002  # 0.2%
    sl_pct = 0.001  # 0.1%
    leverage_val = 10

    def init(self):
        """
        Initialize the strategy.
        Pre-calculate all indicators for maximum speed.
        """
        # --- SMA ---
        self.sma_short = self.I(ta.sma, pd.Series(self.data.Close), length=self.sma_short_len)
        self.sma_long = self.I(ta.sma, pd.Series(self.data.Close), length=self.sma_long_len)
        
        # --- CPR ---
        self.pivot, self.tc, self.bc = self.I(
            calculate_cpr, self.data.High, self.data.Low, self.data.Close
        )
        self.narrow_cpr = self.I(is_narrow_cpr, self.tc, self.bc, threshold=self.cpr_threshold)
        
        # --- Conditions ---
        self.near_sma = self.I(is_near_sma, self.data.Close, self.sma_short, threshold=self.sma_threshold)
        self.sma_rising = self.I(lambda x: x > x.shift(self.sma_lookback), self.sma_short)
        self.sma_declining = self.I(lambda x: x < x.shift(self.sma_lookback), self.sma_short)
        
        # --- Candle Patterns ---
        self.buy_candle, self.sell_candle = self.I(
            candle_signals, self.data.Open, self.data.High, self.data.Low, self.data.Close
        )

    def next(self):
        """
        The main strategy logic, executed on each bar.
        """
        price = self.data.Close[-1]
        
        # --- Exit Logic ---
        # backtesting.py handles exits automatically if 'sl' and 'tp' are set on the trade.
        # We don't need manual exit logic.
        
        # --- Entry Logic ---
        if not self.position:  # Only enter if we don't have a position
            
            # --- Buy Signal ---
            if (self.narrow_cpr[-1] and
                self.near_sma[-1] and
                self.sma_rising[-1] and
                self.buy_candle[-1]):
                
                # Calculate SL and TP
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
                
                # Place buy order
                self.buy(sl=sl, tp=tp, size=self.leverage_val)

            # --- Sell Signal ---
            elif (self.narrow_cpr[-1] and
                  self.near_sma[-1] and
                  self.sma_declining[-1] and
                  self.sell_candle[-1]):
                
                # Calculate SL and TP
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)
                
                # Place sell (short) order
                self.sell(sl=sl, tp=tp, size=self.leverage_val)


# --- Plotting Function ---

def plot_backtest_chart(data: pd.DataFrame, stats: pd.Series, trades: pd.DataFrame):
    """
    Creates an interactive Plotly chart with OHLC, indicators, and trades.
    """
    fig = go.Figure()

    # 1. OHLC Data
    fig.add_trace(go.Ohlc(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))

    # 2. Indicators (from the strategy stats)
    indicators = stats._strategy.indicators
    for indicator in indicators:
        # Avoid plotting boolean arrays
        if indicator.data.dtype == 'bool':
            continue
        # Only plot key indicators
        if indicator.name in ["sma_short", "sma_long", "pivot", "tc", "bc"]:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=indicator.data,
                name=indicator.name,
                line=dict(width=1, dash='dot' if "cpr" in indicator.name else 'solid')
            ))

    # 3. Trade Markers
    if not trades.empty:
        # Separate buys and sells
        buys = trades[trades['Size'] > 0]
        sells = trades[trades['Size'] < 0] # Covers both short entries and long exits

        fig.add_trace(go.Scatter(
            x=buys['EntryTime'],
            y=buys['EntryPrice'],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Entry'
        ))
        
        fig.add_trace(go.Scatter(
            x=trades['ExitTime'],
            y=trades['ExitPrice'],
            mode='markers',
            marker=dict(color='gray', symbol='x', size=8),
            name='Trade Exit'
        ))
        
        fig.add_trace(go.Scatter(
            x=sells['EntryTime'],
            y=sells['EntryPrice'],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Entry'
        ))

    fig.update_layout(
        title=f"Backtest Results for {stats['Symbol']}",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark"
    )
    return fig


# --- Streamlit UI ---

def main():
    """Defines the Streamlit application UI."""
    
    st.title("📈 Nifty50 Quantitative Strategy Backtester")
    st.markdown("Test the **Narrow CPR & SMA Crossover** strategy on Nifty50 stocks.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # --- Stock and Timeframe ---
        nifty_tickers = get_nifty50_tickers()
        symbol = st.selectbox("Select Stock", options=nifty_tickers, index=nifty_tickers.index("RELIANCE.NS"))
        
        timeframe = st.selectbox("Select Timeframe", options=["1D", "1H", "15m"], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=3 * 365))
        with col2:
            end_date = st.date_input("End Date", datetime.now())

        # --- Strategy Parameters ---
        st.subheader("Strategy Parameters")
        
        with st.expander("Indicator Settings"):
            sma_short_len = st.number_input("Short SMA Period", min_value=1, max_value=100, value=20)
            sma_long_len = st.number_input("Long SMA Period", min_value=50, max_value=500, value=200)
            sma_lookback = st.number_input("SMA Direction Lookback", min_value=1, max_value=20, value=6)
            cpr_threshold = st.slider("Narrow CPR Threshold (%)", 0.01, 1.0, 0.06, 0.01)
            sma_threshold = st.slider("SMA Proximity Threshold (%)", 0.01, 5.0, 0.15, 0.01)

        with st.expander("Trade & Risk Settings"):
            leverage = st.number_input("Leverage", min_value=1, max_value=20, value=10)
            tp_pct_ui = st.number_input("Take Profit (%)", min_value=0.1, max_value=5.0, value=0.2, step=0.1)
            sl_pct_ui = st.number_input("Stop Loss (%)", min_value=0.1, max_value=5.0, value=0.1, step=0.1)
            commission = st.number_input("Commission (% per trade)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            initial_cash = st.number_input("Initial Cash", min_value=1000, value=100_000, step=1000)

        # --- Run Button ---
        run_button = st.button("Run Backtest", use_container_width=True, type="primary")

    # --- Main Panel for Results ---
    if run_button:
        with st.spinner(f"Fetching {symbol} data and running backtest..."):
            # 1. Fetch Data
            data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), timeframe)
            
            if data.empty:
                st.error(f"No data found for {symbol}. Please try a different stock or date range.")
                return

            # 2. Configure Strategy
            # Update the strategy class with UI parameters
            CprSmaStrategy.sma_short_len = sma_short_len
            CprSmaStrategy.sma_long_len = sma_long_len
            CprSmaStrategy.cpr_threshold = cpr_threshold
            CprSmaStrategy.sma_threshold = sma_threshold
            CprSmaStrategy.sma_lookback = sma_lookback
            CprSmaStrategy.tp_pct = tp_pct_ui / 100.0
            CprSmaStrategy.sl_pct = sl_pct_ui / 100.0
            CprSmaStrategy.leverage_val = leverage

            # 3. Run Backtest
            bt = Backtest(
                data,
                CprSmaStrategy,
                cash=initial_cash,
                commission=commission / 100.0,  # Convert % to decimal
                exclusive_orders=True
            )
            
            try:
                stats = bt.run()
                stats['Symbol'] = symbol # Add symbol for plotting
                trades = stats._trades
            except Exception as e:
                st.error(f"An error occurred during backtest: {e}")
                st.info("This can happen if there is not enough data for the indicators (e.g., SMA 200). Try a longer date range.")
                return

        st.success("Backtest complete!")

        # 4. Display Results
        st.header("Backtest Results")
        
        # --- Key Metrics ---
        st.subheader("Key Performance Indicators (KPIs)")
        
        # Add a check for 'Return (Ann.) [%]' which may not exist for short periods
        return_metric = stats.get('Return [%]', 0.0)
        
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Return [%]", f"{return_metric:.2f}")
        kpi_cols[1].metric("Win Rate [%]", f"{stats.get('Win Rate [%]', 0.0):.2f}")
        kpi_cols[2].metric("# of Trades", f"{stats.get('# Trades', 0)}")
        kpi_cols[3].metric("Profit Factor", f"{stats.get('Profit Factor', 0.0):.2f}")
        kpi_cols[4].metric("Max Drawdown [%]", f"{stats.get('Max. Drawdown [%]', 0.0):.2f}")
        kpi_cols[5].metric("Sharpe Ratio", f"{stats.get('Sharpe Ratio', 0.0):.2f}")

        # --- Tabs for Charts and Logs ---
        tab1, tab2, tab3 = st.tabs(["📈 Interactive Chart", "📋 Trade Log", "📊 Full Statistics"])

        with tab1:
            st.subheader("Equity Curve & Trades")
            # Generate and display the interactive chart
            fig = plot_backtest_chart(data, stats, trades)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Trade Log")
            st.dataframe(trades, use_container_width=True)

        with tab3:
            st.subheader("Full Backtest Statistics")
            st.dataframe(pd.Series(stats).to_frame('Value'), use_container_width=True)

    else:
        st.info("Please select your parameters in the sidebar and click **Run Backtest**.")

if __name__ == "__main__":
    main()
