# nifty_backtest_dashboard.py
# Streamlit dashboard to pick a Nifty50 stock (including Reliance) and run the provided backtest
# Save this file and run: `streamlit run nifty_backtest_dashboard.py`

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.set_page_config(layout='wide', page_title='Nifty50 Backtest Dashboard')

# -------------------------- Helper functions (strategy code adapted) --------------------------

def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()


def calculate_cpr_daily(data):
    """Calculate CPR (Pivot, TC, BC) using previous calendar day's OHLC.
    Requires daily data (interval='1d')."""
    pivot = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
    bc = (data['High'].shift(1) + data['Low'].shift(1)) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc


def is_narrow_cpr(tc, bc, threshold=0.06):
    if pd.isna(tc) or pd.isna(bc) or bc == 0:
        return False
    cpr_width = abs(tc - bc)
    cpr_percentage = (cpr_width / bc) * 100
    return cpr_percentage < threshold


def is_near_sma(price, sma, threshold=0.15):
    if pd.isna(sma) or sma == 0:
        return False
    diff_percentage = abs(price - sma) / sma * 100
    return diff_percentage <= threshold


def is_sma_rising(sma_current, sma_past):
    if pd.isna(sma_current) or pd.isna(sma_past):
        return False
    return sma_current > sma_past


def is_sma_declining(sma_current, sma_past):
    if pd.isna(sma_current) or pd.isna(sma_past):
        return False
    return sma_current < sma_past


def check_buy_signal(df, i):
    current = df.iloc[i]
    if current['Close'] <= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['Close'] < prev['Open']:
                if current['High'] > prev['High']:
                    return True
    return False


def check_sell_signal(df, i):
    current = df.iloc[i]
    if current['Close'] >= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['Close'] > prev['Open']:
                if current['Low'] < prev['Low']:
                    return True
    return False


def run_strategy(df, leverage=10, commission_rate=0.001):
    df = df.copy().reset_index(drop=False)
    # Expect columns: Date (or index), Open, High, Low, Close
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # indicators
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_200'] = calculate_sma(df, 200)
    df['pivot'], df['tc'], df['bc'] = calculate_cpr_daily(df)
    df['narrow_cpr'] = df.apply(lambda r: is_narrow_cpr(r['tc'], r['bc']), axis=1)

    # trade tracking
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['pnl'] = 0.0
    df['position'] = ''
    df['equity'] = 10000.0
    df['drawdown'] = 0.0

    position = None
    entry_price = 0.0
    equity = 10000.0

    # ensure we have at least 200 rows for SMA200
    start_i = 200 if len(df) > 200 else 0

    for i in range(start_i, len(df)):
        current = df.iloc[i]

        # Exit rules
        if position == 'long':
            take_profit = entry_price * 1.002
            stop_loss = entry_price * 0.999
            if current['High'] >= take_profit:
                df.at[i, 'exit_price'] = take_profit
                price_change_pct = ((take_profit - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity = equity * (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_long'
                position = None
            elif current['Low'] <= stop_loss:
                df.at[i, 'exit_price'] = stop_loss
                price_change_pct = ((stop_loss - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity = equity * (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_long'
                position = None

        elif position == 'short':
            take_profit = entry_price * 0.998
            stop_loss = entry_price * 1.001
            if current['Low'] <= take_profit:
                df.at[i, 'exit_price'] = take_profit
                price_change_pct = ((entry_price - take_profit) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity = equity * (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_short'
                position = None
            elif current['High'] >= stop_loss:
                df.at[i, 'exit_price'] = stop_loss
                price_change_pct = ((entry_price - stop_loss) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity = equity * (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_short'
                position = None

        df.at[i, 'equity'] = equity
        peak_equity = df.loc[:i, 'equity'].max()
        drawdown = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        df.at[i, 'drawdown'] = drawdown

        # Entry rules
        if position is None and current['narrow_cpr']:
            if is_near_sma(current['Close'], current['sma_20']):
                if i >= 6 and is_sma_rising(df.at[i, 'sma_20'], df.at[i - 6, 'sma_20']) and check_buy_signal(df, i):
                    df.at[i, 'signal'] = 1
                    df.at[i, 'entry_price'] = current['Close']
                    df.at[i, 'position'] = 'long'
                    position = 'long'
                    entry_price = current['Close']

                elif i >= 6 and is_sma_declining(df.at[i, 'sma_20'], df.at[i - 6, 'sma_20']) and check_sell_signal(df, i):
                    df.at[i, 'signal'] = -1
                    df.at[i, 'entry_price'] = current['Close']
                    df.at[i, 'position'] = 'short'
                    position = 'short'
                    entry_price = current['Close']

    return df

# -------------------------- Data fetching utility --------------------------
@st.cache_data(ttl=3600)
def fetch_yahoo(ticker, period='2y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.reset_index(inplace=True)
    return df

# -------------------------- Nifty50 list (yahoo tickers) --------------------------
NIFTY50 = {
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'Infosys': 'INFY.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'HDFC': 'HDFC.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Kotak Bank': 'KOTAKBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'State Bank of India': 'SBIN.NS',
    'ITC': 'ITC.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Oil & Natural Gas': 'ONGC.NS',
    'Bharat Petroleum': 'BPCL.NS',
    'Larsen & Toubro': 'LT.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Titan Company': 'TITAN.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS'
}

# -------------------------- Streamlit UI --------------------------
st.title('Nifty50 Backtest Dashboard')
st.markdown('Select a stock (default: Reliance) and timeframe, then run the backtest. Data is fetched from Yahoo Finance via yfinance.')

col1, col2 = st.columns([2, 1])
with col1:
    ticker_name = st.selectbox('Pick a stock', options=list(NIFTY50.keys()), index=0)
    ticker = NIFTY50[ticker_name]
    period = st.selectbox('History period', options=['6mo', '1y', '2y', '5y'], index=2)
    interval = st.selectbox('Interval', options=['1d'], index=0)  # restricting to daily for CPR correctness
    leverage = st.number_input('Leverage (x)', min_value=1, max_value=50, value=10)
    commission = st.number_input('Commission per side (as decimal)', min_value=0.0, max_value=0.01, value=0.001, step=0.0005)
    run_btn = st.button('Run Backtest')

with col2:
    st.write('Quick info')
    st.write(f'Chosen ticker: **{ticker}**')
    st.write('Note: CPR is calculated using previous daily OHLC, so daily data is recommended.')

# default convenience: quick button for Reliance as well
if st.button('Show Reliance backtest'):
    ticker_name = 'Reliance Industries'
    ticker = NIFTY50[ticker_name]
    period = '2y'
    interval = '1d'
    leverage = 10
    commission = 0.001
    run_btn = True

if run_btn:
    with st.spinner('Fetching data and running backtest...'):
        df = fetch_yahoo(ticker, period=period, interval=interval)
        if df.empty or len(df) < 60:
            st.error('Not enough data fetched. Try a longer period or check ticker.')
        else:
            result = run_strategy(df, leverage=leverage, commission_rate=commission)

            # Summary statistics
            total_trades = len(result[result['signal'] != 0])
            winning_trades = len(result[result['pnl'] > 0])
            losing_trades = len(result[result['pnl'] < 0])
            total_pnl = result['pnl'].sum()
            final_equity = result['equity'].iloc[-1]
            roi = ((final_equity - 10000) / 10000) * 100
            max_drawdown = result['drawdown'].min()
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            st.subheader(f'Backtest results — {ticker_name} ({ticker})')
            st.metric('Final Equity', f'${final_equity:,.2f}', delta=f'{roi:.2f}%')
            st.metric('Total Trades', total_trades)
            st.metric('Win Rate', f'{win_rate:.2f}%')
            st.metric('Max Drawdown', f'{max_drawdown:.2f}%')

            # Plots
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(result['timestamp'], result['Close'], label='Close', linewidth=1)
            ax.plot(result['timestamp'], result['sma_20'], label='SMA 20')
            ax.plot(result['timestamp'], result['sma_200'], label='SMA 200')
            buys = result[result['signal'] == 1]
            sells = result[result['signal'] == -1]
            ax.scatter(buys['timestamp'], buys['Close'], marker='^', s=80, label='Buy')
            ax.scatter(sells['timestamp'], sells['Close'], marker='v', s=80, label='Sell')
            ax.set_title('Price with SMAs and Signals')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(result['timestamp'], result['equity'], linewidth=2)
            ax2.set_title('Equity Curve')
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Trade log
            entries = result[result['signal'] != 0][['timestamp', 'signal', 'entry_price', 'position', 'sma_20']]
            exits = result[result['pnl'] != 0][['timestamp', 'exit_price', 'pnl', 'position']]

            st.subheader('Trade Log')
            if not entries.empty:
                colA, colB = st.columns(2)
                with colA:
                    st.write('Entries')
                    st.dataframe(entries.reset_index(drop=True))
                with colB:
                    st.write('Exits / PnL')
                    st.dataframe(exits.reset_index(drop=True))
            else:
                st.info('No trades executed during this backtest period.')

            # Download results
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button('Download full backtest CSV', data=csv, file_name=f'backtest_{ticker}.csv')

st.markdown('---')
st.write('Tips: Use daily data for accurate CPR. For intraday CPR you would need to compute previous calendar-day OHLC and map it to intraday bars.')

# End of file
