import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Nifty 50 Quant Backtest Dashboard")

# --- Strategy Functions (modular, realistic) ---
def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_cpr(data):
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

def check_buy_signal(data, i):
    current = data.iloc[i]
    if current['Close'] <= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev['Close'] < prev['Open'] and current['High'] > prev['High']:
                return True
    return False

def check_sell_signal(data, i):
    current = data.iloc[i]
    if current['Close'] >= current['Open']:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev['Close'] > prev['Open'] and current['Low'] < prev['Low']:
                return True
    return False

def run_strategy(df, leverage=10, commission_rate=0.001):
    df = df.copy().reset_index(drop=True)
    df['sma_20'] = calculate_sma(df, 20)
    df['sma_200'] = calculate_sma(df, 200)
    df['pivot'], df['tc'], df['bc'] = calculate_cpr(df)
    df['narrow_cpr'] = df.apply(lambda row: is_narrow_cpr(row['tc'], row['bc']), axis=1)
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['pnl'] = 0.0
    df['position'] = ''
    df['equity'] = 10000.0
    df['drawdown'] = 0.0
    position = None
    entry_price = 0
    equity = 10000.0

    for i in range(200, len(df)):
        current = df.iloc[i]
        if position == 'long':
            take_profit = entry_price * 1.002
            stop_loss = entry_price * 0.999
            if current['High'] >= take_profit:
                exit_price = take_profit
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl/100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_long'
                position = None
            elif current['Low'] <= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl/100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_long'
                position = None

        elif position == 'short':
            take_profit = entry_price * 0.998
            stop_loss = entry_price * 1.001
            if current['Low'] <= take_profit:
                exit_price = take_profit
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl/100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_short'
                position = None
            elif current['High'] >= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl/100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_short'
                position = None

        df.at[i, 'equity'] = equity
        peak_equity = df.loc[:i, 'equity'].max()
        drawdown = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
        df.at[i, 'drawdown'] = drawdown

        # Entry logic
        if position is None and current['narrow_cpr']:
            if is_near_sma(current['Close'], current['sma_20']):
                if i >= 6 and is_sma_rising(df.at[i, 'sma_20'], df.at[i-6, 'sma_20']) and check_buy_signal(df, i):
                    df.at[i, 'signal'] = 1
                    df.at[i, 'entry_price'] = current['Close']
                    df.at[i, 'position'] = 'long'
                    position = 'long'
                    entry_price = current['Close']
                elif i >= 6 and is_sma_declining(df.at[i, 'sma_20'], df.at[i-6, 'sma_20']) and check_sell_signal(df, i):
                    df.at[i, 'signal'] = -1
                    df.at[i, 'entry_price'] = current['Close']
                    df.at[i, 'position'] = 'short'
                    position = 'short'
                    entry_price = current['Close']
    return df

# --- Sidebar: Controls ---
nifty50_list = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BPCL.NS","BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS","INDUSINDBK.NS","INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LTIM.NS","LT.NS",
    "M&M.NS","MARUTI.NS","NTPC.NS","NESTLEIND.NS","ONGC.NS","PIDILITIND.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SUNPHARMA.NS","TCS.NS","TATACONSUM.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","TITAN.NS","UPL.NS","ULTRACEMCO.NS","WIPRO.NS"
]

st.sidebar.title("Nifty 50 Quant Backtest")
chosen_stock = st.sidebar.selectbox("Choose Stock", options=nifty50_list)
today = datetime.today()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=180), min_value=datetime(2008,1,1))
end_date = st.sidebar.date_input("End Date", today, min_value=start_date, max_value=today)
timeframe_mapping = {
    '2 minute': '2m', '5 minute': '5m',
    '15 minute': '15m', '30 minute': '30m',
    '1 hour': '1h', '1 day': '1d'
}
timeframe_display = st.sidebar.selectbox("Timeframe", tuple(timeframe_mapping.keys()), index=2)
interval = timeframe_mapping[timeframe_display]
leverage = st.sidebar.slider("Leverage", min_value=1, max_value=20, value=10)
commission = st.sidebar.slider("Commission per Side (bps)", min_value=1, max_value=30, value=10)

# --- Data Download ---
with st.spinner(f"Fetching {chosen_stock} historical data..."):
    data = yf.download(chosen_stock, start=start_date, end=end_date + timedelta(days=1), interval=interval, auto_adjust=True)
    data = data.dropna()
    data = data.reset_index()

# --- Run the Strategy ---
with st.spinner("Processing backtest..."):
    if not data.empty:
        run_df = run_strategy(data, leverage=leverage, commission_rate=commission/10000)
    else:
        st.error("No data available for selected period and timeframe.")

# --- Visuals & Results ---
if not data.empty and not run_df.empty:
    st.header(f"Backtest Results: {chosen_stock}")
    # --- Charts ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    # 1. Candles, SMAs, Buy/Sell
    axes[0,0].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['Close'], label='Price', alpha=0.7)
    axes[0,0].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['sma_20'], label='SMA 20', alpha=0.8)
    axes[0,0].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['sma_200'], label='SMA 200', alpha=0.8)
    buys = run_df[run_df['signal'] == 1]
    axes[0,0].scatter(buys['Date'] if 'Date' in buys.columns else buys.index, buys['Close'], marker='^', color='green', label='Buy')
    sells = run_df[run_df['signal'] == -1]
    axes[0,0].scatter(sells['Date'] if 'Date' in sells.columns else sells.index, sells['Close'], marker='v', color='red', label='Sell')
    axes[0,0].legend(); axes[0,0].set_title("Price, SMA, and Trade Signals")
    
    # 2. CPR
    axes[0,1].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['Close'], label='Price')
    axes[0,1].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['tc'], label='TC', linestyle='--')
    axes[0,1].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['bc'], label='BC', linestyle='--')
    axes[0,1].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['pivot'], label='Pivot', linestyle=':')
    narrow_cpr = run_df[run_df['narrow_cpr'] == True]
    axes[0,1].scatter(narrow_cpr['Date'] if 'Date' in narrow_cpr.columns else narrow_cpr.index, narrow_cpr['Close'],
                      color='yellow', marker='o', alpha=0.5, label="Narrow CPR")
    axes[0,1].legend(); axes[0,1].set_title("CPR Visualization")
    
    # 3. Equity Curve
    axes[1,0].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, run_df['equity'], color='blue')
    axes[1,0].axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label="Starting Capital")
    axes[1,0].set_title("Equity Curve")
    axes[1,0].legend()

    # 4. Trade PnL Distribution
    pnl_vals = run_df[run_df['pnl'] != 0]['pnl']
    colors = ['green' if x > 0 else 'red' for x in pnl_vals]
    if len(pnl_vals) > 0:
        axes[1,1].bar(range(len(pnl_vals)), pnl_vals, color=colors, alpha=0.7)
    axes[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1,1].set_title("Trade P&L")

    # 5. Cumulative PnL
    cumulative_pnl = run_df['pnl'].cumsum()
    axes[2,0].plot(run_df['Date'] if 'Date' in run_df.columns else run_df.index, cumulative_pnl, color='purple')
    axes[2,0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[2,0].set_title("Cumulative P&L")

    # 6. Statistics Table (offscreen use)
    axes[2,1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Stats Table ---
    total_trades = int((run_df['signal'] != 0).sum())
    winning_trades = int((run_df['pnl'] > 0).sum())
    losing_trades = int((run_df['pnl'] < 0).sum())
    win_rate = (winning_trades / total_trades) * 100 if total_trades else 0
    avg_win = run_df[run_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = run_df[run_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
    total_pnl = run_df['pnl'].sum()
    final_equity = run_df['equity'].iloc[-1]
    roi = ((final_equity - 10000) / 10000) * 100
    max_drawdown = run_df['drawdown'].min()
    min_equity = run_df['equity'].min()
    max_equity = run_df['equity'].max()

    st.subheader(f"**Key Performance Metrics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Trades", total_trades)
        st.metric("Winning Trades", winning_trades)
        st.metric("Losing Trades", losing_trades)
        st.metric("Win Rate (%)", f"{win_rate:.2f}")
        st.metric("Leverage (×)", leverage)
        st.metric("Trade Commission (%)", commission/100)
    with col2:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.metric("Total P&L (%)", f"{total_pnl:.2f}")
        st.metric("ROI (%)", f"{roi:.2f}")
        st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
        st.metric("Final Equity", f"{final_equity:.2f}")
        st.metric("Starting Capital", "10,000")

    # --- Trade Log ---
    st.subheader("Trade Log (Entries & Exits)")
    entries = run_df[run_df['signal'] != 0][['Date', 'signal', 'entry_price', 'position', 'sma_20']]
    exits = run_df[run_df['pnl'] != 0][['Date', 'exit_price', 'pnl', 'position']]
    st.dataframe(entries, use_container_width=True)
    st.dataframe(exits, use_container_width=True)
else:
    st.warning("No backtesting results available for the selected configuration.")

st.caption("© 2025. Deployable quant strategy—production-ready for Indian Nifty50 stocks.")

