import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ==============================================================
# Utility functions
# ==============================================================

def fetch_market_data(symbol, period="6mo", interval="1h"):
    """Fetch historical data from Yahoo Finance"""
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    data.reset_index(inplace=True)
    data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }, inplace=True)
    return data[['Datetime', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'Datetime': 'timestamp'})

def calculate_sma(data, period):
    """Simple Moving Average"""
    return data['close'].rolling(window=period).mean()

def calculate_cpr(data):
    """Central Pivot Range (using previous day’s data)"""
    pivot = (data['high'].shift(1) + data['low'].shift(1) + data['close'].shift(1)) / 3
    bc = (data['high'].shift(1) + data['low'].shift(1)) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc

def is_narrow_cpr(tc, bc, threshold=0.06):
    if pd.isna(tc) or pd.isna(bc) or bc == 0:
        return False
    cpr_width = abs(tc - bc)
    return (cpr_width / bc * 100) < threshold

def is_near_sma(price, sma, threshold=0.15):
    if pd.isna(sma) or sma == 0:
        return False
    return abs(price - sma) / sma * 100 <= threshold

def is_sma_rising(curr, past): return curr > past if pd.notna(curr) and pd.notna(past) else False
def is_sma_declining(curr, past): return curr < past if pd.notna(curr) and pd.notna(past) else False

def check_buy_signal(df, i):
    current = df.iloc[i]
    if current['close'] <= current['open']: return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['close'] < prev['open'] and current['high'] > prev['high']:
                return True
    return False

def check_sell_signal(df, i):
    current = df.iloc[i]
    if current['close'] >= current['open']: return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev['close'] > prev['open'] and current['low'] < prev['low']:
                return True
    return False

# ==============================================================
# Main Strategy Logic
# ==============================================================

def run_strategy(data):
    df = data.copy().reset_index(drop=True)
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_200'] = calculate_sma(df, 200)
    df['pivot'], df['tc'], df['bc'] = calculate_cpr(df)
    df['narrow_cpr'] = df.apply(lambda x: is_narrow_cpr(x['tc'], x['bc']), axis=1)

    df['signal'] = 0
    df['pnl'] = 0.0
    df['equity'] = 10000.0
    df['position'] = ''
    df['drawdown'] = 0.0

    position = None
    entry_price = 0
    equity = 10000.0
    leverage = 10
    commission = 0.001  # 0.1%

    for i in range(200, len(df)):
        curr = df.iloc[i]

        # Exit logic
        if position == 'long':
            tp = entry_price * 1.002
            sl = entry_price * 0.999
            if curr['high'] >= tp or curr['low'] <= sl:
                exit_price = tp if curr['high'] >= tp else sl
                price_change = ((exit_price - entry_price) / entry_price) * 100
                pnl = price_change * leverage - (2 * commission * 100)
                equity *= (1 + pnl / 100)
                df.at[i, 'pnl'] = pnl
                position = None

        elif position == 'short':
            tp = entry_price * 0.998
            sl = entry_price * 1.001
            if curr['low'] <= tp or curr['high'] >= sl:
                exit_price = tp if curr['low'] <= tp else sl
                price_change = ((entry_price - exit_price) / entry_price) * 100
                pnl = price_change * leverage - (2 * commission * 100)
                equity *= (1 + pnl / 100)
                df.at[i, 'pnl'] = pnl
                position = None

        df.at[i, 'equity'] = equity
        peak = df.loc[:i, 'equity'].max()
        df.at[i, 'drawdown'] = ((equity - peak) / peak) * 100 if peak > 0 else 0

        # Entry logic
        if position is None and curr['narrow_cpr']:
            if is_near_sma(curr['close'], curr['sma_20']):
                if i >= 6 and is_sma_rising(df.at[i, 'sma_20'], df.at[i - 6, 'sma_20']) and check_buy_signal(df, i):
                    position = 'long'
                    entry_price = curr['close']
                    df.at[i, 'signal'] = 1
                elif i >= 6 and is_sma_declining(df.at[i, 'sma_20'], df.at[i - 6, 'sma_20']) and check_sell_signal(df, i):
                    position = 'short'
                    entry_price = curr['close']
                    df.at[i, 'signal'] = -1
    return df

# ==============================================================
# Streamlit Dashboard
# ==============================================================

def create_dashboard():
    st.title("📈 NIFTY 50 Trading Strategy Backtest Dashboard")

    nifty_stocks = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "ITC": "ITC.NS", "KOTAKBANK": "KOTAKBANK.NS",
        "AXISBANK": "AXISBANK.NS", "LT": "LT.NS", "ADANIENT": "ADANIENT.NS", "BHARTIARTL": "BHARTIARTL.NS",
        "BAJFINANCE": "BAJFINANCE.NS", "HINDUNILVR": "HINDUNILVR.NS", "MARUTI": "MARUTI.NS"
    }

    stock = st.sidebar.selectbox("📊 Select Stock", list(nifty_stocks.keys()))
    period = st.sidebar.selectbox("⏳ Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.sidebar.selectbox("🕐 Interval", ["15m", "1h", "4h", "1d"], index=1)

    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    if st.sidebar.button("Run Backtest 🚀"):
        symbol = nifty_stocks[stock]
        st.write(f"### Running backtest for **{stock} ({symbol})**")

        with st.spinner("Fetching data..."):
            data = fetch_market_data(symbol, period=period, interval=interval)

            if data.empty:
                st.error("No data available.")
                return

            # ✅ Convert timestamp and filter
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            data = data.loc[(data.index >= start_date) & (data.index <= end_date)]

        with st.spinner("Running strategy..."):
            result = run_strategy(data)

        st.success("✅ Backtest completed!")
        final_equity = result['equity'].iloc[-1]
        total_pnl = result['pnl'].sum()
        st.write(f"**Final Equity:** ${final_equity:.2f}")
        st.write(f"**Total P&L:** {total_pnl:.2f}%")

        # Plot equity curve
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(result.index, result['equity'], label='Equity', color='blue')
        ax.set_title("Equity Curve")
        st.pyplot(fig)

        # Price with signals
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(result.index, result['close'], label='Close', alpha=0.6)
        buys = result[result['signal'] == 1]
        sells = result[result['signal'] == -1]
        ax2.scatter(buys.index, buys['close'], color='green', label='Buy', marker='^')
        ax2.scatter(sells.index, sells['close'], color='red', label='Sell', marker='v')
        ax2.legend()
        ax2.set_title("Price with Buy/Sell Signals")
        st.pyplot(fig2)

# ==============================================================
# Run App
# ==============================================================

if __name__ == "__main__":
    create_dashboard()
