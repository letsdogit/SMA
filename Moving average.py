import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="NIFTY50 Backtest (CPR+SMA+PA)", layout="wide")


# -----------------------------
# Robust Yahoo fetcher
# -----------------------------
def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance and return a dataframe with columns:
    ['timestamp','open','high','low','close','volume'] where timestamp is naive IST.
    Returns empty DataFrame on failure/empty data.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Reset index if index is datetime-like -> get it into a column
    df = df.reset_index()

    # Pick timestamp column name (Datetime for intraday, Date for daily) or fallback to index
    if "Datetime" in df.columns:
        ts_col = "Datetime"
    elif "Date" in df.columns:
        ts_col = "Date"
    else:
        # fallback: use first column if it's datetime-like, otherwise use index
        possible = df.columns[0]
        ts_col = possible

    # Standardize
    df = df.rename(columns={ts_col: "timestamp",
                            "Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Adj Close": "close", "Volume": "volume"})

    if "timestamp" not in df.columns:
        # last resort: try to create from index
        df["timestamp"] = pd.to_datetime(df.index, errors="coerce")

    # Convert to datetime (UTC-aware), drop rows where that fails
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)

    # Convert to Asia/Kolkata then remove tzinfo (make naive) for easy comparison with Streamlit dates
    try:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    except Exception:
        # If tz_convert fails (already naive), try assuming UTC then convert
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Ensure OHLC columns exist and are numeric
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])

    return df.reset_index(drop=True)


# -----------------------------
# Indicators / Strategy helpers
# -----------------------------
def calculate_sma(data: pd.DataFrame, period: int):
    return data["close"].rolling(window=period).mean()


def calculate_cpr(data: pd.DataFrame):
    pivot = (data["high"].shift(1) + data["low"].shift(1) + data["close"].shift(1)) / 3
    bc = (data["high"].shift(1) + data["low"].shift(1)) / 2
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
    if current["close"] <= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev["close"] < prev["open"] and current["high"] > prev["high"]:
                return True
    return False


def check_sell_signal(data, i):
    current = data.iloc[i]
    if current["close"] >= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev["close"] > prev["open"] and current["low"] < prev["low"]:
                return True
    return False


# -----------------------------
# Core backtest
# -----------------------------
def run_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Input: dataframe with columns ['timestamp','open','high','low','close','volume'] (timestamp can be column)
    Returns: dataframe with added strategy columns and equity/pnl info
    """
    df = data.copy().reset_index(drop=True)
    df["sma_20"] = calculate_sma(df, 20)
    df["sma_200"] = calculate_sma(df, 200)
    df["pivot"], df["tc"], df["bc"] = calculate_cpr(df)
    df["narrow_cpr"] = df.apply(lambda row: is_narrow_cpr(row["tc"], row["bc"]), axis=1)

    # initialize result cols
    df["signal"] = 0
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df["pnl"] = 0.0
    df["position"] = ""
    df["equity"] = 10000.0
    df["drawdown"] = 0.0

    position = None
    entry_price = 0.0
    equity = 10000.0
    leverage = 10  # 10x
    commission_rate = 0.001  # 0.1% per side

    for i in range(200, len(df)):
        current = df.iloc[i]

        # Exit existing position first
        if position == "long":
            take_profit = entry_price * 1.002
            stop_loss = entry_price * 0.999

            if current["high"] >= take_profit:
                exit_price = take_profit
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = net_pnl
                df.at[i, "position"] = "exit_long"
                position = None

            elif current["low"] <= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = net_pnl
                df.at[i, "position"] = "exit_long"
                position = None

        elif position == "short":
            take_profit = entry_price * 0.998
            stop_loss = entry_price * 1.001

            if current["low"] <= take_profit:
                exit_price = take_profit
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = net_pnl
                df.at[i, "position"] = "exit_short"
                position = None

            elif current["high"] >= stop_loss:
                exit_price = stop_loss
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                leveraged_pnl = price_change_pct * leverage
                commission_cost = 2 * commission_rate * 100
                net_pnl = leveraged_pnl - commission_cost
                equity = equity * (1 + net_pnl / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = net_pnl
                df.at[i, "position"] = "exit_short"
                position = None

        # update equity/drawdown
        df.at[i, "equity"] = equity
        peak_equity = df.loc[:i, "equity"].max()
        df.at[i, "drawdown"] = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

        # Entry logic (only when no position)
        if position is None and current["narrow_cpr"]:
            if is_near_sma(current["close"], current["sma_20"]):
                if i >= 6 and is_sma_rising(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_buy_signal(df, i):
                    df.at[i, "signal"] = 1
                    df.at[i, "entry_price"] = current["close"]
                    df.at[i, "position"] = "long"
                    position = "long"
                    entry_price = current["close"]
                elif i >= 6 and is_sma_declining(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_sell_signal(df, i):
                    df.at[i, "signal"] = -1
                    df.at[i, "entry_price"] = current["close"]
                    df.at[i, "position"] = "short"
                    position = "short"
                    entry_price = current["close"]

    return df


# -----------------------------
# Streamlit UI / Dashboard
# -----------------------------
def create_dashboard():
    st.title("📈 NIFTY 50 Backtest Dashboard — CPR + SMA + Price Action")
    st.markdown("Select a stock, timeframe and dates; click **Run Backtest** to execute.")

    # NIFTY50 subset (expand as needed)
    nifty_50 = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "ITC": "ITC.NS", "KOTAKBANK": "KOTAKBANK.NS",
        "AXISBANK": "AXISBANK.NS", "LT": "LT.NS", "ADANIENT": "ADANIENT.NS", "BHARTIARTL": "BHARTIARTL.NS",
        "BAJFINANCE": "BAJFINANCE.NS", "HINDUNILVR": "HINDUNILVR.NS", "MARUTI": "MARUTI.NS",
        "TITAN": "TITAN.NS", "SUNPHARMA": "SUNPHARMA.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
        "POWERGRID": "POWERGRID.NS", "ADANIPORTS": "ADANIPORTS.NS"
    }

    left, right = st.columns([2, 1])
    with left:
        stock = st.selectbox("Select Stock", list(nifty_50.keys()), index=0)
        symbol = nifty_50[stock]
        st.caption(f"Yahoo symbol: {symbol}")
    with right:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)

    col1, col2 = st.columns(2)
    # Default start/end: last 6 months-ish
    default_end = pd.to_datetime("today").date()
    default_start = (pd.to_datetime("today") - pd.Timedelta(days=180)).date()
    with col1:
        start_date = st.date_input("Start Date (IST)", value=default_start)
    with col2:
        end_date = st.date_input("End Date (IST)", value=default_end)

    if st.button("Run Backtest 🚀", use_container_width=True):
        with st.spinner("Fetching data from Yahoo Finance..."):
            raw = fetch_market_data(symbol, period=period, interval=interval)

        if raw.empty:
            st.error("No data fetched — try a different Period/Interval (note: 15m has limited history).")
            return

        # set index to timestamp (naive IST)
        df = raw.copy()
        df.set_index("timestamp", inplace=True)

        # Convert date inputs to timestamps (naive IST) and include entire end date
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

        # Filter
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        if df.empty:
            st.warning("No data available in the selected date range. Try widening the dates or changing period/interval.")
            return

        # Run backtest
        with st.spinner("Running strategy..."):
            result = run_strategy(df.reset_index())

        # Metrics
        total_trades = int((result["signal"] != 0).sum())
        winning_trades = int((result["pnl"] > 0).sum())
        losing_trades = int((result["pnl"] < 0).sum())
        total_pnl = float(result["pnl"].sum())
        final_equity = float(result["equity"].iloc[-1])
        max_dd = float(result["drawdown"].min())

        st.success("Backtest complete ✅")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Winning Trades", winning_trades)
        m3.metric("Total P&L (%)", f"{total_pnl:.2f}")
        m4.metric("Final Equity ($)", f"{final_equity:,.2f}")

        # Equity curve
        fig_eq, ax_eq = plt.subplots(figsize=(10, 3.5))
        ax_eq.plot(result["timestamp"], result["equity"], linewidth=1.6)
        ax_eq.axhline(10000, linestyle="--", alpha=0.6)
        ax_eq.set_title("Equity Curve")
        ax_eq.set_xlabel("Time (IST)")
        ax_eq.set_ylabel("Equity ($)")
        ax_eq.grid(alpha=0.3)
        st.pyplot(fig_eq)

        # Price + signals
        fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
        ax_p.plot(result["timestamp"], result["close"], label="Close", alpha=0.7)
        buys = result[result["signal"] == 1]
        sells = result[result["signal"] == -1]
        if not buys.empty:
            ax_p.scatter(buys["timestamp"], buys["close"], marker="^", s=60, label="Buy", zorder=5)
        if not sells.empty:
            ax_p.scatter(sells["timestamp"], sells["close"], marker="v", s=60, label="Sell", zorder=5)
        ax_p.set_title("Price with Buy / Sell signals")
        ax_p.legend()
        ax_p.grid(alpha=0.3)
        st.pyplot(fig_p)

        # Show recent trade rows (entry+exit)
        trades_df = result.loc[result["signal"] != 0, ["timestamp", "signal", "entry_price", "position", "sma_20"]]
        exits_df = result.loc[result["pnl"] != 0, ["timestamp", "exit_price", "pnl", "position"]]

        with st.expander("View trade entries (signals)"):
            st.dataframe(trades_df.sort_values("timestamp").reset_index(drop=True), height=250)

        with st.expander("View trade exits (pnl)"):
            st.dataframe(exits_df.sort_values("timestamp").reset_index(drop=True), height=250)

        # Download complete backtest CSV
        csv_buffer = BytesIO()
        result.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button("Download full backtest CSV", data=csv_buffer, file_name=f"{symbol}_backtest.csv", mime="text/csv")

        # Download trades-only CSV
        trades_only = pd.concat([trades_df.reset_index(drop=True), exits_df.reset_index(drop=True)], axis=1)
        csv2 = trades_only.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades summary CSV", data=csv2, file_name=f"{symbol}_trades.csv", mime="text/csv")


if __name__ == "__main__":
    create_dashboard()
