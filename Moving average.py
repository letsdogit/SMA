import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import requests
from io import StringIO
import ta

warnings.filterwarnings('ignore')

# Nifty 50 stocks with their Yahoo Finance symbols
NIFTY50_SYMBOLS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ICICIBANK.NS': 'ICICI Bank',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC',
    'LT.NS': 'Larsen & Toubro',
    'SBIN.NS': 'State Bank of India',
    'ASIANPAINT.NS': 'Asian Paints',
    'HCLTECH.NS': 'HCL Technologies',
    'AXISBANK.NS': 'Axis Bank',
    'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'TITAN.NS': 'Titan Company',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'WIPRO.NS': 'Wipro',
    'NESTLEIND.NS': 'Nestle India',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'ONGC.NS': 'ONGC',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC',
    'M&M.NS': 'Mahindra & Mahindra',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'ADANIPORTS.NS': 'Adani Ports',
    'TECHM.NS': 'Tech Mahindra',
    'BRITANNIA.NS': 'Britannia Industries',
    'HDFC.NS': 'HDFC',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'CIPLA.NS': 'Cipla',
    'GRASIM.NS': 'Grasim Industries',
    'COALINDIA.NS': 'Coal India',
    'JSWSTEEL.NS': 'JSW Steel',
    'TATAMOTORS.NS': 'Tata Motors',
    'SBILIFE.NS': 'SBI Life Insurance',
    'HINDALCO.NS': 'Hindalco Industries',
    'UPL.NS': 'UPL',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'TATASTEEL.NS': 'Tata Steel',
    'DIVISLAB.NS': 'Divis Laboratories',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'SHREECEM.NS': 'Shree Cement',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    'EICHERMOT.NS': 'Eicher Motors',
    'BPCL.NS': 'BPCL',
    'ADANIENT.NS': 'Adani Enterprises'
}

class EnhancedTradingStrategy:
    def __init__(self):
        self.commission_rate = 0.001  # 0.1% per trade
        self.slippage = 0.0005  # 0.05% slippage
        
    def calculate_advanced_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR for volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df
    
    def calculate_cpr(self, data):
        """Calculate Central Pivot Range with enhanced logic"""
        pivot = (data['high'].shift(1) + data['low'].shift(1) + data['close'].shift(1)) / 3
        bc = (data['high'].shift(1) + data['low'].shift(1)) / 2
        tc = (pivot - bc) + pivot
        return pivot, tc, bc
    
    def is_narrow_cpr(self, tc, bc, threshold=0.08):
        """Enhanced CPR narrow condition"""
        if pd.isna(tc) or pd.isna(bc) or bc == 0:
            return False
        cpr_width = abs(tc - bc)
        cpr_percentage = (cpr_width / bc) * 100
        return cpr_percentage < threshold
    
    def generate_signals(self, data):
        """Generate enhanced trading signals"""
        df = data.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        for i in range(200, len(df)):
            current = df.iloc[i]
            
            # Skip if missing data
            if any(pd.isna([current['sma_20'], current['sma_50'], current['rsi'], 
                           current['macd'], current['bb_upper']])):
                continue
            
            # Bullish conditions
            bullish_conditions = 0
            total_conditions = 0
            
            # Price above key SMAs
            if current['close'] > current['sma_20']:
                bullish_conditions += 1
            total_conditions += 1
            
            if current['close'] > current['sma_50']:
                bullish_conditions += 1
            total_conditions += 1
            
            # RSI not overbought
            if current['rsi'] < 70:
                bullish_conditions += 1
            total_conditions += 1
            
            # MACD bullish
            if current['macd'] > current['macd_signal']:
                bullish_conditions += 1
            total_conditions += 1
            
            # Volume confirmation
            if current['volume_ratio'] > 1.0:
                bullish_conditions += 1
            total_conditions += 1
            
            # Calculate signal strength
            signal_strength = bullish_conditions / total_conditions
            
            # Generate signals based on strength
            if signal_strength >= 0.6 and current['narrow_cpr']:
                if current['close'] > current['sma_20'] and current['macd_histogram'] > 0:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_strength'] = signal_strength
                    
            elif signal_strength <= 0.3 and current['narrow_cpr']:
                if current['close'] < current['sma_20'] and current['macd_histogram'] < 0:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_strength'] = signal_strength
        
        return df
    
    def calculate_position_size(self, equity, price, risk_per_trade=0.02, stop_loss_pct=0.02):
        """Calculate position size based on risk management"""
        risk_amount = equity * risk_per_trade
        stop_loss_amount = price * stop_loss_pct
        position_size = risk_amount / stop_loss_amount if stop_loss_amount > 0 else 0
        return min(position_size, equity * 0.1 / price)  # Max 10% per trade
    
    def run_backtest(self, data, initial_capital=100000, leverage=1):
        """Enhanced backtest with professional risk management"""
        df = data.copy()
        
        # Calculate indicators
        df = self.calculate_advanced_indicators(df)
        df['pivot'], df['tc'], df['bc'] = self.calculate_cpr(df)
        df['narrow_cpr'] = df.apply(lambda x: self.is_narrow_cpr(x['tc'], x['bc']), axis=1)
        df = self.generate_signals(df)
        
        # Initialize tracking columns
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        df['equity'] = float(initial_capital)
        df['drawdown'] = 0.0
        df['position_size'] = 0
        df['daily_return'] = 0.0
        
        position = 0
        entry_price = 0
        equity = initial_capital
        peak_equity = initial_capital
        total_trades = 0
        winning_trades = 0
        
        for i in range(200, len(df)):
            current = df.iloc[i]
            
            # Calculate daily return
            if i > 200:
                prev_equity = df.iloc[i-1]['equity']
                df.loc[df.index[i], 'daily_return'] = (equity - prev_equity) / prev_equity * 100
            
            # Exit conditions
            if position != 0:
                if position == 1:  # Long position
                    # Take profit at 2R (risk:reward 1:2)
                    take_profit = entry_price * (1 + (entry_price * 0.02 * 2))
                    stop_loss = entry_price * (1 - 0.02)  # 2% stop loss
                    
                    if current['high'] >= take_profit or current['low'] <= stop_loss:
                        exit_price = take_profit if current['high'] >= take_profit else stop_loss
                        pnl = (exit_price - entry_price) * position_size
                        commission = entry_price * position_size * self.commission_rate * 2
                        net_pnl = pnl - commission
                        
                        df.loc[df.index[i], 'exit_price'] = exit_price
                        df.loc[df.index[i], 'pnl'] = net_pnl
                        equity += net_pnl
                        
                        if net_pnl > 0:
                            winning_trades += 1
                        total_trades += 1
                        
                        position = 0
                        entry_price = 0
                
                elif position == -1:  # Short position
                    take_profit = entry_price * (1 - (entry_price * 0.02 * 2))
                    stop_loss = entry_price * (1 + 0.02)
                    
                    if current['low'] <= take_profit or current['high'] >= stop_loss:
                        exit_price = take_profit if current['low'] <= take_profit else stop_loss
                        pnl = (entry_price - exit_price) * position_size
                        commission = entry_price * position_size * self.commission_rate * 2
                        net_pnl = pnl - commission
                        
                        df.loc[df.index[i], 'exit_price'] = exit_price
                        df.loc[df.index[i], 'pnl'] = net_pnl
                        equity += net_pnl
                        
                        if net_pnl > 0:
                            winning_trades += 1
                        total_trades += 1
                        
                        position = 0
                        entry_price = 0
            
            # Entry conditions
            if position == 0 and current['signal'] != 0:
                position_size = self.calculate_position_size(equity, current['close'])
                
                if position_size > 0:
                    if current['signal'] == 1:  # Buy signal
                        position = 1
                        entry_price = current['close'] * (1 + self.slippage)
                        df.loc[df.index[i], 'position'] = 1
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'position_size'] = position_size
                    
                    elif current['signal'] == -1:  # Sell signal
                        position = -1
                        entry_price = current['close'] * (1 - self.slippage)
                        df.loc[df.index[i], 'position'] = -1
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'position_size'] = position_size
            
            # Update equity and drawdown
            df.loc[df.index[i], 'equity'] = equity
            df.loc[df.index[i], 'cumulative_pnl'] = equity - initial_capital
            
            if equity > peak_equity:
                peak_equity = equity
            
            current_drawdown = (peak_equity - equity) / peak_equity * 100
            df.loc[df.index[i], 'drawdown'] = current_drawdown
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(df, initial_capital, total_trades, winning_trades)
        
        return df, performance
    
    def calculate_performance_metrics(self, df, initial_capital, total_trades, winning_trades):
        """Calculate comprehensive performance metrics"""
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Sharpe Ratio (annualized)
        daily_returns = df['daily_return'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        max_drawdown = df['drawdown'].max()
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        cagr = ((final_equity / initial_capital) ** (365/days) - 1) * 100 if days > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'final_equity': final_equity,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

def fetch_market_data(symbol, period='2y', interval='1d'):
    """Fetch real market data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        
        # Reset index and rename columns
        data = data.reset_index()
        data = data.rename(columns={'Date': 'timestamp', 'Open': 'open', 
                                  'High': 'high', 'Low': 'low', 
                                  'Close': 'close', 'Volume': 'volume'})
        
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_dashboard():
    """Create the main Streamlit dashboard"""
    st.set_page_config(
        page_title="Nifty50 Trading Strategy Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive { color: #00aa00; }
    .negative { color: #ff0000; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏛️ Nifty50 Quantitative Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Strategy Configuration")
    
    # Stock selection
    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        options=list(NIFTY50_SYMBOLS.keys()),
        format_func=lambda x: f"{NIFTY50_SYMBOLS[x]} ({x})"
    )
    
    # Time period selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=['1d', '1h', '4h', '1wk'],
        index=0
    )
    
    # Strategy parameters
    st.sidebar.subheader("Risk Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=100000, min_value=10000, step=10000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100
    cpr_threshold = st.sidebar.slider("CPR Narrow Threshold (%)", 0.01, 0.2, 0.08)
    
    # Main content
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Fetching market data and running backtest..."):
            # Fetch data
            period = '2y'  # Default period, will be filtered by dates
            data = fetch_market_data(selected_symbol, period=period, interval=timeframe)
            
            if data is not None:
                # Filter by selected date range
                data = data[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]
                
                if len(data) > 200:
                    # Initialize and run strategy
                    strategy = EnhancedTradingStrategy()
                    strategy.is_narrow_cpr = lambda tc, bc: strategy.is_narrow_cpr(tc, bc, cpr_threshold)
                    
                    results, performance = strategy.run_backtest(
                        data, 
                        initial_capital=initial_capital,
                        leverage=1
                    )
                    
                    # Display results
                    display_results(selected_symbol, results, performance, strategy)
                else:
                    st.error("Insufficient data for selected period. Please choose a longer time period.")
            else:
                st.error("Failed to fetch market data. Please try again.")

def display_results(symbol, results, performance, strategy):
    """Display comprehensive backtest results"""
    
    st.header(f"Backtest Results: {NIFTY50_SYMBOLS[symbol]} ({symbol})")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return", 
            f"₹{performance['final_equity']:,.0f}",
            f"{performance['total_return']:.2f}%"
        )
    
    with col2:
        st.metric(
            "CAGR",
            f"{performance['cagr']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{performance['sharpe_ratio']:.2f}"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{performance['max_drawdown']:.2f}%"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Win Rate",
            f"{performance['win_rate']:.1f}%"
        )
    
    with col6:
        st.metric(
            "Total Trades",
            f"{performance['total_trades']}"
        )
    
    with col7:
        st.metric(
            "Profit Factor",
            f"{performance['profit_factor']:.2f}"
        )
    
    with col8:
        color = "positive" if performance['gross_profit'] > performance['gross_loss'] else "negative"
        st.markdown(f"<div class='metric-card'>Gross P/L: <span class='{color}'>₹{performance['gross_profit']:,.0f}/₹{performance['gross_loss']:,.0f}</span></div>", unsafe_allow_html=True)
    
    # Charts
    create_interactive_charts(results, performance)
    
    # Trade analysis
    st.subheader("Trade Analysis")
    display_trade_analysis(results)
    
    # Download results
    csv = convert_df_to_csv(results)
    st.download_button(
        label="Download Backtest Results",
        data=csv,
        file_name=f"backtest_results_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def create_interactive_charts(results, performance):
    """Create interactive Plotly charts"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Price with Trading Signals', 
            'Equity Curve',
            'Indicator: RSI',
            'Indicator: MACD',
            'Drawdown',
            'Daily Returns Distribution'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # 1. Price with signals
    fig.add_trace(
        go.Scatter(x=results.index, y=results['close'], name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = results[results['signal'] == 1]
    fig.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['close'], 
                  mode='markers', name='Buy', marker=dict(color='green', size=8, symbol='triangle-up')),
        row=1, col=1
    )
    
    # Add sell signals
    sell_signals = results[results['signal'] == -1]
    fig.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['close'], 
                  mode='markers', name='Sell', marker=dict(color='red', size=8, symbol='triangle-down')),
        row=1, col=1
    )
    
    # 2. Equity curve
    fig.add_trace(
        go.Scatter(x=results.index, y=results['equity'], name='Equity', line=dict(color='green')),
        row=1, col=2
    )
    
    # 3. RSI
    fig.add_trace(
        go.Scatter(x=results.index, y=results['rsi'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # 4. MACD
    fig.add_trace(
        go.Scatter(x=results.index, y=results['macd'], name='MACD', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=results.index, y=results['macd_signal'], name='Signal', line=dict(color='red')),
        row=2, col=2
    )
    
    # 5. Drawdown
    fig.add_trace(
        go.Scatter(x=results.index, y=results['drawdown'], name='Drawdown', 
                  fill='tozeroy', line=dict(color='red')),
        row=3, col=1
    )
    
    # 6. Daily returns distribution
    daily_returns = results['daily_return'].dropna()
    fig.add_trace(
        go.Histogram(x=daily_returns, name='Returns Distribution', nbinsx=50),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, showlegend=True, title_text="Comprehensive Strategy Analysis")
    st.plotly_chart(fig, use_container_width=True)

def display_trade_analysis(results):
    """Display detailed trade analysis"""
    
    trades = results[(results['signal'] != 0) | (results['pnl'] != 0)].copy()
    
    if len(trades) > 0:
        # Create trade log
        trade_log = []
        current_trade = None
        
        for i, row in trades.iterrows():
            if row['signal'] != 0 and current_trade is None:
                current_trade = {
                    'entry_date': i,
                    'entry_price': row['entry_price'],
                    'position': 'Long' if row['signal'] == 1 else 'Short',
                    'signal_strength': row.get('signal_strength', 0)
                }
            elif row['pnl'] != 0 and current_trade is not None:
                current_trade.update({
                    'exit_date': i,
                    'exit_price': row['exit_price'],
                    'pnl': row['pnl'],
                    'holding_period': (i - current_trade['entry_date']).days
                })
                trade_log.append(current_trade)
                current_trade = None
        
        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            st.dataframe(trade_df.style.format({
                'entry_price': '{:.2f}',
                'exit_price': '{:.2f}',
                'pnl': '₹{:,.2f}',
                'signal_strength': '{:.2f}'
            }))

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv().encode('utf-8')

def main():
    """Main application"""
    create_dashboard()

if __name__ == "__main__":
    main()
