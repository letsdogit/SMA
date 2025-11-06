# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="NIFTY50 Backtest (CPR + SMA + Price Action)", layout="wide")

# =========================================================
# Fixed Yahoo Finance fetcher
# =========================================================
def fetch_market_data(symbol: str, period: str = "6mo", interval: str = "1h") -> pd.DataFrame:
    """
    Safely fetch OHLCV from Yahoo Finance and return standardized DataFrame.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"⚠️ yfinance error: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        st.warning(f"No data found for {symbol}")
        return pd.DataFrame()

    # Reset index to make Date/Datetime a column
    df = df.reset_index()
    
    # Find the timestamp column
    time_col = None
    for col in df.columns:
        if str(col).lower() in ("datetime", "date"):
            time_col = col
            break
    
    if time_col:
        df = df.rename(columns={time_col: "timestamp"})
    else:
        # If no timestamp column found, create one from index
        df["timestamp"] = pd.to_datetime(df.index, errors="coerce")

    # Standardize column names - FIXED: Handle different column naming conventions
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "open":
            column_mapping[col] = "open"
        elif col_lower == "high":
            column_mapping[col] = "high"
        elif col_lower == "low":
            column_mapping[col] = "low"
        elif col_lower in ["close", "adj close"]:
            column_mapping[col] = "close"
        elif col_lower == "volume":
            column_mapping[col] = "volume"
    
    df = df.rename(columns=column_mapping)

    # Ensure timestamp is properly formatted
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    
    if df.empty:
        return pd.DataFrame()

    # Convert to IST and make timezone-naive
    try:
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        else:
            # Assume UTC if no timezone, convert to IST
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    except Exception:
        # If timezone conversion fails, proceed without it
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure required columns exist
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Convert numeric columns safely - FIXED: Check if column exists before conversion
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and reorder columns
    final_columns = ["timestamp"] + numeric_columns
    df = df[final_columns]
    
    # Drop rows where essential price data is missing
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    
    return df

# =======================================
# Indicators & helper functions (unchanged)
# =======================================
def calculate_sma(data: pd.DataFrame, period: int):
    return data["close"].rolling(window=period).mean()

def calculate_cpr(data: pd.DataFrame):
    """Central Pivot Range using previous bar (works for intraday/daily)"""
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

def check_buy_signal(data: pd.DataFrame, i: int):
    """Green bar takes out red bar (1–2 lookback) and closes green"""
    current = data.iloc[i]
    if current["close"] <= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev["close"] < prev["open"] and current["high"] > prev["high"]:
                return True
    return False

def check_sell_signal(data: pd.DataFrame, i: int):
    """Red bar takes out green bar (1–2 lookback) and closes red"""
    current = data.iloc[i]
    if current["close"] >= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = data.iloc[i - lookback]
            if prev["close"] > prev["open"] and current["low"] < prev["low"]:
                return True
    return False

# ============================
# Core backtest / strategy
# ============================
def run_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with ['timestamp','open','high','low','close','volume']
    Output: Adds SMA/CPR, signals, positions, equity, PnL, drawdown.
    """
    df = data.copy().reset_index(drop=True)

    # Indicators
    df["sma_20"] = calculate_sma(df, 20)
    df["sma_200"] = calculate_sma(df, 200)
    df["pivot"], df["tc"], df["bc"] = calculate_cpr(df)
    df["narrow_cpr"] = df.apply(lambda row: is_narrow_cpr(row["tc"], row["bc"]), axis=1)

    # Tracking columns
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
    leverage = 10            # 10x leverage
    commission_rate = 0.001  # 0.1% per side (entry + exit = 0.2%)

    # Start after we have enough SMA200 history
    start_idx = 200 if len(df) > 200 else 0

    for i in range(start_idx, len(df)):
        current = df.iloc[i]

        # 1) Exit logic (priority)
        if position == "long":
            tp = entry_price * 1.002
            sl = entry_price * 0.999

            if current["high"] >= tp:
                exit_price = tp
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_pct = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl_pct / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = pnl_pct
                df.at[i, "position"] = "exit_long"
                position = None

            elif current["low"] <= sl:
                exit_price = sl
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl_pct = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl_pct / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = pnl_pct
                df.at[i, "position"] = "exit_long"
                position = None

        elif position == "short":
            tp = entry_price * 0.998
            sl = entry_price * 1.001

            if current["low"] <= tp:
                exit_price = tp
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                pnl_pct = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl_pct / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = pnl_pct
                df.at[i, "position"] = "exit_short"
                position = None

            elif current["high"] >= sl:
                exit_price = sl
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                pnl_pct = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl_pct / 100)
                df.at[i, "exit_price"] = exit_price
                df.at[i, "pnl"] = pnl_pct
                df.at[i, "position"] = "exit_short"
                position = None

        # Update equity & drawdown every bar
        df.at[i, "equity"] = equity
        peak_equity = df.loc[:i, "equity"].max()
        df.at[i, "drawdown"] = ((equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

        # 2) Entry logic (only if flat)
        if position is None and current["narrow_cpr"]:
            if is_near_sma(current["close"], current["sma_20"]):
                # Long condition
                if i >= 6 and is_sma_rising(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_buy_signal(df, i):
                    df.at[i, "signal"] = 1
                    df.at[i, "entry_price"] = current["close"]
                    df.at[i, "position"] = "long"
                    position = "long"
                    entry_price = current["close"]

                # Short condition
                elif i >= 6 and is_sma_declining(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_sell_signal(df, i):
                    df.at[i, "signal"] = -1
                    df.at[i, "entry_price"] = current["close"]
                    df.at[i, "position"] = "short"
                    position = "short"
                    entry_price = current["close"]

    return df

# ============================
# Streamlit Dashboard
# ============================
def create_dashboard():
    st.title("📈 NIFTY 50 Backtest — CPR + SMA + Price Action")
    st.caption("Pick a stock, timeframe, and dates. Click **Run Backtest** to simulate.")

    # NIFTY 50 symbols
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

    # Defaults: last ~6 months
    default_end = pd.to_datetime("today").date()
    default_start = (pd.to_datetime("today") - pd.Timedelta(days=180)).date()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date (IST)", value=default_start)
    with c2:
        end_date = st.date_input("End Date (IST)", value=default_end)

    if st.button("Run Backtest 🚀", use_container_width=True):
        # Fetch data
        with st.spinner("Fetching data from Yahoo Finance..."):
            raw = fetch_market_data(symbol, period=period, interval=interval)

        if raw.empty:
            st.error("No data fetched. Try changing Period/Interval (note: intraday has limited history).")
            return

        # Date filter (both sides naive IST)
        df = raw.copy()
        df.set_index("timestamp", inplace=True)
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        if df.empty:
            st.warning("No rows in the selected date range. Widen the range or change Period/Interval.")
            return

        # Run strategy
        with st.spinner("Running strategy..."):
            result = run_strategy(df.reset_index())

        st.success("Backtest complete ✅")

        # ---- KPIs
        total_trades = int((result["signal"] != 0).sum())
        winning_trades = int((result["pnl"] > 0).sum())
        losing_trades = int((result["pnl"] < 0).sum())
        total_pnl = float(result["pnl"].sum())
        final_equity = float(result["equity"].iloc[-1])
        max_dd = float(result["drawdown"].min())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trades", total_trades)
        k2.metric("Winning Trades", winning_trades)
        k3.metric("Total P&L (%)", f"{total_pnl:.2f}")
        k4.metric("Final Equity ($)", f"{final_equity:,.2f}")

        # ---- Equity curve
        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        ax1.plot(result["timestamp"], result["equity"], linewidth=1.6)
        ax1.axhline(10000, linestyle="--", alpha=0.5)
        ax1.set_title("Equity Curve")
        ax1.set_xlabel("Time (IST)")
        ax1.set_ylabel("Equity ($)")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

        # ---- Price + signals
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(result["timestamp"], result["close"], label="Close", alpha=0.75)
        buys = result[result["signal"] == 1]
        sells = result[result["signal"] == -1]
        if not buys.empty:
            ax2.scatter(buys["timestamp"], buys["close"], marker="^", s=70, color="green", label="Buy", zorder=5)
        if not sells.empty:
            ax2.scatter(sells["timestamp"], sells["close"], marker="v", s=70, color="red", label="Sell", zorder=5)
        ax2.set_title("Price with Buy/Sell Signals")
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        # ---- Tables
        entries = result.loc[result["signal"] != 0, ["timestamp", "signal", "entry_price", "position", "sma_20"]]
        exits = result.loc[result["pnl"] != 0, ["timestamp", "exit_price", "pnl", "position"]]
        
        with st.expander("View entry signals"):
            if not entries.empty:
                st.dataframe(entries.sort_values("timestamp").reset_index(drop=True), height=260)
            else:
                st.info("No entry signals generated during this period")

        with st.expander("View exits / trade P&L"):
            if not exits.empty:
                st.dataframe(exits.sort_values("timestamp").reset_index(drop=True), height=260)
            else:
                st.info("No trade exits during this period")

        # ---- Downloads
        full_csv_buf = BytesIO()
        result.to_csv(full_csv_buf, index=False)
        full_csv_buf.seek(0)
        st.download_button(
            "⬇ Download full backtest CSV",
            data=full_csv_buf,
            file_name=f"{symbol}_backtest.csv",
            mime="text/csv",
        )

        # Create trades summary
        if not entries.empty and not exits.empty:
            trades_summary = pd.concat(
                [entries.reset_index(drop=True), exits.reset_index(drop=True)],
                axis=1
            )
            st.download_button(
                "⬇ Download trades summary CSV",
                data=trades_summary.to_csv(index=False),
                file_name=f"{symbol}_trades_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    create_dashboard()
