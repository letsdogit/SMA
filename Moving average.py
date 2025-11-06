# nifty_backtest_dashboard.py
# Streamlit dashboard to pick a Nifty50 stock (including Reliance) and run the provided backtest
# Uses DAILY data only (CPR based on previous day's OHLC). Avoids ValueError by vectorizing CPR narrow check.
# Run:  streamlit run nifty_backtest_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

st.set_page_config(layout='wide', page_title='Nifty50 Backtest Dashboard')

# -------------------------- Strategy helpers --------------------------

def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
    return data['Close'].rolling(window=period).mean()


def calculate_cpr_daily(df_daily: pd.DataFrame):
    """Calculate CPR (Pivot, TC, BC) using previous calendar day's OHLC.
    Expects DAILY data (interval='1d') with columns: Open, High, Low, Close.
    """
    pivot = (df_daily['High'].shift(1) + df_daily['Low'].shift(1) + df_daily['Close'].shift(1)) / 3
    bc = (df_daily['High'].shift(1) + df_daily['Low'].shift(1)) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc


def is_near_sma(price: float, sma: float, threshold: float = 0.15) -> bool:
    if pd.isna(sma) or sma == 0:
        return False
    diff_percentage = abs(price - sma) / sma * 100
    return diff_percentage <= threshold


def is_sma_rising(sma_current: float, sma_past: float) -> bool:
    if pd.isna(sma_current) or pd.isna(sma_past):
        return False
    return sma_current > sma_past


def is_sma_declining(sma_current: float, sma_past: float) -> bool:
    if pd.isna(sma_current) or pd.isna(sma_past):
        return False
    return sma_current < sma_past


def check_buy_signal(df: pd.DataFrame, i: int) -> bool:
    current = df.iloc[i]
    if current['Close'] <= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['Close'] < prev['Open'] and current['High'] > prev['High']:
                return True
    return False


def check_sell_signal(df: pd.DataFrame, i: int) -> bool:
    current = df.iloc[i]
    if current['Close'] >= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['Close'] > prev['Open'] and current['Low'] < prev['Low']:
                return True
    return False


def run_strategy(df_daily: pd.DataFrame, leverage: int = 10, commission_rate: float = 0.001) -> pd.DataFrame:
    """Run backtest on DAILY data. Returns dataframe with signals, equity, logs."""
    df = df_daily.copy().reset_index(drop=False)
    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)  # first column is the date index from yfinance

    # indicators
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_200'] = calculate_sma(df, 200)
    df['pivot'], df['tc'], df['bc'] = calculate_cpr_daily(df)

    # Vectorized narrow CPR (avoid per-row apply and ambiguous Series truth-value)
    denom = df['bc'].replace(0, np.nan)
    cpr_pct = (df['tc'] - df['bc']).abs() / denom * 100
    df['narrow_cpr'] = (cpr_pct < 0.06).fillna(False)

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

    start_i = 200 if len(df) > 200 else 0  # wait for SMA200

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
                equity *= (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_long'
                position = None
            elif current['Low'] <= stop_loss:
                df.at[i, 'exit_price'] = stop_loss
                price_change_pct = ((stop_loss - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity *= (1 + net_pnl / 100)
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
                equity *= (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_short'
                position = None
            elif current['High'] >= stop_loss:
                df.at[i, 'exit_price'] = stop_loss
                price_change_pct = ((entry_price - stop_loss) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                df.at[i, 'pnl'] = net_pnl
                equity *= (1 + net_pnl / 100)
                df.at[i, 'position'] = 'exit_short'
                position = None

        df.at[i, 'equity'] = equity
        peak_equity = df.loc[:i, 'equity'].max()
        drawdown = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        df.at[i, 'drawdown'] = drawdown

        # Entry rules (daily CPR + SMA+ pattern)
        # Safe check for narrow_cpr
        narrow_flag = bool(df.at[i, 'narrow_cpr']) if 'narrow_cpr' in df.columns and not pd.isna(df.at[i, 'narrow_cpr']) else False
        if position is None and narrow_flag:
            if is_near_sma(current['Close'], df.at[i, 'sma_20']):
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

# -------------------------- Data fetching --------------------------
@st.cache_data(ttl=3600)
def fetch_yahoo_daily(ticker: str, period: str = '2y') -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval='1d', progress=False)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.reset_index(inplace=True)  # Date column becomes first column
    return df

# -------------------------- Nifty50 (subset, editable) --------------------------
NIFTY50 = {
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'Infosys': 'INFY.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'State Bank of India': 'SBIN.NS',
    'ITC': 'ITC.NS',
    'Larsen & Toubro': 'LT.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Oil & Natural Gas': 'ONGC.NS',
    'Bharat Petroleum': 'BPCL.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Titan Company': 'TITAN.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Maruti Suzuki': 'MARUTI.NS'
}

# -------------------------- Streamlit UI --------------------------
st.title('Nifty50 Backtest Dashboard — Daily CPR')
st.caption('Data source: Yahoo Finance via yfinance. CPR is computed from previous daily OHLC. Reliance is included by default.')

with st.sidebar:
    st.header('Parameters')
    ticker_name = st.selectbox('Pick a Nifty50 stock', options=list(NIFTY50.keys()), index=0)
    ticker = NIFTY50[ticker_name]
    period = st.selectbox('History period', options=['6mo', '1y', '2y', '5y', '10y', 'max'], index=2)
    leverage = st.number_input('Leverage (x)', min_value=1, max_value=50, value=10)
    commission = st.number_input('Commission per side (decimal)', min_value=0.0, max_value=0.01, value=0.001, step=0.0005)
    run_btn = st.button('Run Backtest', use_container_width=True)

# Quick action button
if st.button('Run Reliance sample (2y daily)'):
    ticker_name = 'Reliance Industries'
    ticker = NIFTY50[ticker_name]
    period = '2y'
    leverage = 10
    commission = 0.001
    run_btn = True

if run_btn:
    with st.spinner(f'Fetching {ticker_name} ({ticker}) daily data and running backtest...'):
        df = fetch_yahoo_daily(ticker, period=period)
        if df.empty or len(df) < 220:  # need enough bars for SMA200
            st.error('Not enough data fetched. Try a longer period or verify ticker.')
        else:
            result = run_strategy(df, leverage=leverage, commission_rate=commission)

            # ---- Summary ----
            total_trades = int((result['signal'] != 0).sum())
            winning_trades = int((result['pnl'] > 0).sum())
            losing_trades = int((result['pnl'] < 0).sum())
            total_pnl = float(result['pnl'].sum())
            final_equity = float(result['equity'].iloc[-1])
            roi = ((final_equity - 10000) / 10000) * 100
            max_drawdown = float(result['drawdown'].min())
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            st.subheader(f'Backtest — {ticker_name} ({ticker})')
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Final Equity', f"${final_equity:,.2f}", delta=f"{roi:.2f}%")
            m2.metric('Total Trades', total_trades)
            m3.metric('Win Rate', f"{win_rate:.2f}%")
            m4.metric('Max Drawdown', f"{max_drawdown:.2f}%")

            # ---- Charts ----
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(result['timestamp'], result['Close'], label='Close', linewidth=1)
            ax.plot(result['timestamp'], result['sma_20'], label='SMA 20')
            ax.plot(result['timestamp'], result['sma_200'], label='SMA 200')
            buys = result[result['signal'] == 1]
            sells = result[result['signal'] == -1]
            ax.scatter(buys['timestamp'], buys['Close'], marker='^', s=70, label='Buy')
            ax.scatter(sells['timestamp'], sells['Close'], marker='v', s=70, label='Sell')
            ax.set_title('Price with SMAs and Signals (Daily)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(result['timestamp'], result['equity'], linewidth=2)
            ax2.axhline(y=10000, linestyle='--', linewidth=0.8)
            ax2.set_title('Equity Curve')
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            # ---- Trade logs ----
            entries = result[result['signal'] != 0][['timestamp', 'signal', 'entry_price', 'position', 'sma_20']]
            exits = result[result['pnl'] != 0][['timestamp', 'exit_price', 'pnl', 'position']]
            st.subheader('Trade Log')
            if not entries.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.write('Entries')
                    st.dataframe(entries.reset_index(drop=True))
                with c2:
                    st.write('Exits / PnL')
                    st.dataframe(exits.reset_index(drop=True))
            else:
                st.info('No trades executed during this backtest period.')

            # ---- Download ----
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button('Download full backtest CSV', data=csv, file_name=f'backtest_{ticker}.csv', mime='text/csv')

st.markdown('---')
st.write('Notes:')
st.write('- CPR is computed only on daily data using \*previous day\* OHLC. For intraday CPR mapping, we can extend this.')
st.write('- Commission is per side (entry and exit both charged).')
st.write('- If you need more Nifty tickers added, tell me and I will expand the list.')
