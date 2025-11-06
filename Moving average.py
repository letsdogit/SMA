import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="NIFTY50 Backtest (CPR+SMA)", layout="wide")

# -----------------------------------------
# ✅ SAFE Yahoo Finance Data Fetcher
# -----------------------------------------
def fetch_market_data(symbol, period="6mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    except:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Use Datetime or Date as timestamp
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    else:
        df["timestamp"] = pd.to_datetime(df.index)

    # Standardize names
    df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "close", "Volume": "volume"
    }, inplace=True)

    # Convert time to IST → make naive
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Keep only needed columns
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

# -----------------------------------------
# ✅ Technical Indicator Functions
# -----------------------------------------
def sma(data, period):
    return data['close'].rolling(period).mean()

def calculate_cpr(df):
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    bc = (df['high'].shift(1) + df['low'].shift(1)) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc

def is_narrow_cpr(tc, bc, threshold=0.06):
    if pd.isna(tc) or pd.isna(bc) or bc == 0:
        return False
    return abs(tc - bc)/bc * 100 < threshold

def is_near_sma(price, sma, threshold=0.15):
    if pd.isna(sma) or sma == 0:
        return False
    return abs(price - sma)/sma * 100 <= threshold

def is_sma_rising(curr, past):
    return pd.notna(curr) and pd.notna(past) and curr > past

def is_sma_declining(curr, past):
    return pd.notna(curr) and pd.notna(past) and curr < past

def check_buy(df, i):
    cur = df.iloc[i]
    if cur['close'] <= cur['open']:
        return False
    for back in [1,2]:
        if i >= back:
            prev = df.iloc[i-back]
            if prev['close'] < prev['open'] and cur['high'] > prev['high']:
                return True
    return False

def check_sell(df, i):
    cur = df.iloc[i]
    if cur['close'] >= cur['open']:
        return False
    for back in [1,2]:
        if i >= back:
            prev = df.iloc[i-back]
            if prev['close'] > prev['open'] and cur['low'] < prev['low']:
                return True
    return False

# -----------------------------------------
# PART 2 — Strategy, Backtest, Streamlit UI
# -----------------------------------------

def run_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the CPR + SMA + price-action strategy.
    Input: DataFrame with columns ['timestamp','open','high','low','close','volume']
    Returns: original frame with added columns: sma_20, sma_200, pivot, tc, bc, narrow_cpr,
             signal, entry_price, exit_price, pnl, position, equity, drawdown
    """
    df = data.copy().reset_index(drop=True)
    # Indicators
    df['sma_20'] = sma(df, 20)
    df['sma_200'] = sma(df, 200)
    df['pivot'], df['tc'], df['bc'] = calculate_cpr(df)
    df['narrow_cpr'] = df.apply(lambda row: is_narrow_cpr(row['tc'], row['bc']), axis=1)

    # Init columns
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
    leverage = 10
    commission_rate = 0.001  # 0.1% per side

    # Start after enough bars for SMA200
    start_idx = 200 if len(df) > 200 else 0

    for i in range(start_idx, len(df)):
        current = df.iloc[i]

        # Exit logic
        if position == 'long':
            take_profit = entry_price * 1.002
            stop_loss = entry_price * 0.999

            if current['high'] >= take_profit:
                exit_price = take_profit
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity *= (1 + net_pnl / 100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_long'
                position = None

            elif current['low'] <= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity *= (1 + net_pnl / 100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_long'
                position = None

        elif position == 'short':
            take_profit = entry_price * 0.998
            stop_loss = entry_price * 1.001

            if current['low'] <= take_profit:
                exit_price = take_profit
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity *= (1 + net_pnl / 100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_short'
                position = None

            elif current['high'] >= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity *= (1 + net_pnl / 100)
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'pnl'] = net_pnl
                df.at[i, 'position'] = 'exit_short'
                position = None

        # Update equity and drawdown on every step
        df.at[i, 'equity'] = equity
        peak_equity = df.loc[:i, 'equity'].max()
        df.at[i, 'drawdown'] = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

        # Entry logic
        if position is None and current.get('narrow_cpr', False):
            if is_near_sma(current['close'], current['sma_20']):
                # Long entry
                if i >= 6 and is_sma_rising(df.at[i, 'sma_20'], df.at[i-6, 'sma_20']) and check_buy(df, i):
                    df.at[i, 'signal'] = 1
                    df.at[i, 'entry_price'] = current['close']
                    df.at[i, 'position'] = 'long'
                    position = 'long'
                    entry_price = current['close']
                # Short entry
                elif i >= 6 and is_sma_declining(df.at[i, 'sma_20'], df.at[i-6, 'sma_20']) and check_sell(df, i):
                    df.at[i, 'signal'] = -1
                    df.at[i, 'entry_price'] = current['close']
                    df.at[i, 'position'] = 'short'
                    position = 'short'
                    entry_price = current['close']

    return df


# -----------------------------------------
# Streamlit App UI
# -----------------------------------------
def create_dashboard():
    st.title("📈 NIFTY50 Backtest Dashboard — CPR + SMA + Price Action")
    st.write("Select a stock, timeframe and date range, then click **Run Backtest**.")

    nifty_50 = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "ITC": "ITC.NS", "KOTAKBANK": "KOTAKBANK.NS",
        "AXISBANK": "AXISBANK.NS", "LT": "LT.NS", "ADANIENT": "ADANIENT.NS", "BHARTIARTL": "BHARTIARTL.NS",
        "BAJFINANCE": "BAJFINANCE.NS", "HINDUNILVR": "HINDUNILVR.NS", "MARUTI": "MARUTI.NS",
        "TITAN": "TITAN.NS", "SUNPHARMA": "SUNPHARMA.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
        "POWERGRID": "POWERGRID.NS", "ADANIPORTS": "ADANIPORTS.NS"
    }

    col_left, col_right = st.columns([2, 1])
    with col_left:
        stock = st.selectbox("Select stock", list(nifty_50.keys()), index=0)
        symbol = nifty_50[stock]
        st.caption(f"Yahoo symbol: {symbol}")
    with col_right:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)

    # Date inputs (defaults)
    default_end = pd.to_datetime("today").date()
    default_start = (pd.to_datetime("today") - pd.Timedelta(days=180)).date()

    start_date = st.date_input("Start Date (IST)", value=default_start)
    end_date = st.date_input("End Date (IST)", value=default_end)

    if st.button("Run Backtest 🚀", use_container_width=True):
        # Fetch data
        with st.spinner("Fetching data from Yahoo Finance..."):
            raw = fetch_market_data(symbol, period=period, interval=interval)

        if raw.empty:
            st.error("No data fetched. Try a different Period/Interval (note: intraday data has limited history).")
            return

        # Filter by date range safely
        df = raw.copy()
        df.set_index('timestamp', inplace=True)

        # Convert Streamlit date inputs to timestamps in same naive IST timezone
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            st.warning("No data in selected date range. Try widening the range or changing Period/Interval.")
            return

        # Run strategy
        with st.spinner("Running strategy..."):
            result = run_strategy(df.reset_index())

        st.success("Backtest finished ✅")

        # Metrics
        total_trades = int((result['signal'] != 0).sum())
        winning_trades = int((result['pnl'] > 0).sum())
        losing_trades = int((result['pnl'] < 0).sum())
        total_pnl = float(result['pnl'].sum())
        final_equity = float(result['equity'].iloc[-1])
        max_dd = float(result['drawdown'].min())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", total_trades)
        c2.metric("Winning Trades", winning_trades)
        c3.metric("Total P&L (%)", f"{total_pnl:.2f}")
        c4.metric("Final Equity ($)", f"{final_equity:,.2f}")

        # Equity chart
        fig_eq, ax_eq = plt.subplots(figsize=(10, 3.5))
        ax_eq.plot(result['timestamp'], result['equity'], linewidth=1.5)
        ax_eq.set_title("Equity Curve")
        ax_eq.set_xlabel("Time (IST)")
        ax_eq.set_ylabel("Equity ($)")
        ax_eq.grid(alpha=0.3)
        st.pyplot(fig_eq)

        # Price + signals chart
        fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
        ax_p.plot(result['timestamp'], result['close'], label='Close', alpha=0.7)
        buys = result[result['signal'] == 1]
        sells = result[result['signal'] == -1]
        if not buys.empty:
            ax_p.scatter(buys['timestamp'], buys['close'], marker='^', s=70, label='Buy', zorder=5)
        if not sells.empty:
            ax_p.scatter(sells['timestamp'], sells['close'], marker='v', s=70, label='Sell', zorder=5)
        ax_p.set_title("Price with Buy / Sell Signals")
        ax_p.legend()
        ax_p.grid(alpha=0.3)
        st.pyplot(fig_p)

        # Show trades & exits
        entries = result[result['signal'] != 0][['timestamp', 'signal', 'entry_price', 'position', 'sma_20']].reset_index(drop=True)
        exits = result[result['pnl'] != 0][['timestamp', 'exit_price', 'pnl', 'position']].reset_index(drop=True)

        with st.expander("View entry signals"):
            st.dataframe(entries, height=250)

        with st.expander("View exits / trade PnL"):
            st.dataframe(exits, height=250)

        # Download buttons
        csv_buffer = BytesIO()
        result.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button("Download full backtest CSV", data=csv_buffer, file_name=f"{symbol}_backtest.csv", mime="text/csv")

        trades_summary = pd.concat([entries, exits], axis=1)
        st.download_button("Download trades summary CSV", data=trades_summary.to_csv(index=False), file_name=f"{symbol}_trades_summary.csv", mime="text/csv")


if __name__ == "__main__":
    create_dashboard()
