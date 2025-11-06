import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------
# Data fetch (Yahoo Finance)
# ------------------------------
def fetch_market_data(symbol: str, period="6mo", interval="1h") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Ensures we return columns: timestamp, open, high, low, close, volume
    """
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return df

    # Reset to column format
    df = df.reset_index()

    # Handle different datetime column names from yfinance
    # For intraday: "Datetime"; for daily: "Date"
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df.rename(
        columns={
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Ensure timestamp is pandas datetime (tz-aware → convert to IST → make naive)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # Convert UTC → Asia/Kolkata, then drop tz so comparisons with Streamlit dates work
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # Final tidy columns
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ------------------------------
# Indicators & helpers (your logic)
# ------------------------------
def calculate_sma(data, period):
    return data["close"].rolling(window=period).mean()

def calculate_cpr(data):
    pivot = (data["high"].shift(1) + data["low"].shift(1) + data["close"].shift(1)) / 3
    bc = (data["high"].shift(1) + data["low"].shift(1)) / 2
    tc = (pivot - bc) + pivot
    return pivot, tc, bc

def is_narrow_cpr(tc, bc, threshold=0.06):
    if pd.isna(tc) or pd.isna(bc) or bc == 0:
        return False
    cpr_width = abs(tc - bc)
    return (cpr_width / bc) * 100 < threshold

def is_near_sma(price, sma, threshold=0.15):
    if pd.isna(sma) or sma == 0:
        return False
    return abs(price - sma) / sma * 100 <= threshold

def is_sma_rising(curr, past):
    return bool(pd.notna(curr) and pd.notna(past) and curr > past)

def is_sma_declining(curr, past):
    return bool(pd.notna(curr) and pd.notna(past) and curr < past)

def check_buy_signal(df, i):
    current = df.iloc[i]
    if current["close"] <= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev["close"] < prev["open"] and current["high"] > prev["high"]:
                return True
    return False

def check_sell_signal(df, i):
    current = df.iloc[i]
    if current["close"] >= current["open"]:
        return False
    for lookback in [1, 2]:
        if i >= lookback:
            prev = df.iloc[i - lookback]
            if prev["close"] > prev["open"] and current["low"] < prev["low"]:
                return True
    return False


# ------------------------------
# Strategy
# ------------------------------
def run_strategy(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)

    df["sma_20"] = calculate_sma(df, 20)
    df["sma_200"] = calculate_sma(df, 200)
    df["pivot"], df["tc"], df["bc"] = calculate_cpr(df)
    df["narrow_cpr"] = df.apply(lambda x: is_narrow_cpr(x["tc"], x["bc"]), axis=1)

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
    leverage = 10
    commission_rate = 0.001  # 0.1% per trade (each side)

    for i in range(200, len(df)):
        curr = df.iloc[i]

        # --- Exit rules ---
        if position == "long":
            tp = entry_price * 1.002
            sl = entry_price * 0.999
            if curr["high"] >= tp:
                exit_price = tp
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl / 100)
                df.at[i, "pnl"] = pnl
                df.at[i, "exit_price"] = exit_price
                df.at[i, "position"] = "exit_long"
                position = None
            elif curr["low"] <= sl:
                exit_price = sl
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100
                pnl = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl / 100)
                df.at[i, "pnl"] = pnl
                df.at[i, "exit_price"] = exit_price
                df.at[i, "position"] = "exit_long"
                position = None

        elif position == "short":
            tp = entry_price * 0.998
            sl = entry_price * 1.001
            if curr["low"] <= tp:
                exit_price = tp
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                pnl = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl / 100)
                df.at[i, "pnl"] = pnl
                df.at[i, "exit_price"] = exit_price
                df.at[i, "position"] = "exit_short"
                position = None
            elif curr["high"] >= sl:
                exit_price = sl
                price_change_pct = ((entry_price - exit_price) / entry_price) * 100
                pnl = price_change_pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + pnl / 100)
                df.at[i, "pnl"] = pnl
                df.at[i, "exit_price"] = exit_price
                df.at[i, "position"] = "exit_short"
                position = None

        df.at[i, "equity"] = equity
        peak = df.loc[:i, "equity"].max()
        df.at[i, "drawdown"] = ((equity - peak) / peak) * 100 if peak > 0 else 0

        # --- Entry rules ---
        if position is None and curr["narrow_cpr"]:
            if is_near_sma(curr["close"], curr["sma_20"]):
                if i >= 6 and is_sma_rising(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_buy_signal(df, i):
                    position = "long"
                    entry_price = curr["close"]
                    df.at[i, "signal"] = 1
                    df.at[i, "entry_price"] = entry_price
                    df.at[i, "position"] = "long"
                elif i >= 6 and is_sma_declining(df.at[i, "sma_20"], df.at[i - 6, "sma_20"]) and check_sell_signal(df, i):
                    position = "short"
                    entry_price = curr["close"]
                    df.at[i, "signal"] = -1
                    df.at[i, "entry_price"] = entry_price
                    df.at[i, "position"] = "short"

    return df


# ------------------------------
# Streamlit Dashboard
# ------------------------------
def create_dashboard():
    st.title("📈 NIFTY 50 Backtest Dashboard (CPR + SMA + PA)")

    nifty50 = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "ITC": "ITC.NS", "KOTAKBANK": "KOTAKBANK.NS",
        "AXISBANK": "AXISBANK.NS", "LT": "LT.NS", "ADANIENT": "ADANIENT.NS", "BHARTIARTL": "BHARTIARTL.NS",
        "BAJFINANCE": "BAJFINANCE.NS", "HINDUNILVR": "HINDUNILVR.NS", "MARUTI": "MARUTI.NS",
        "TITAN": "TITAN.NS", "SUNPHARMA": "SUNPHARMA.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
        "POWERGRID": "POWERGRID.NS", "ADANIPORTS": "ADANIPORTS.NS"
    }

    colA, colB = st.columns(2)
    with colA:
        stock = st.selectbox("Select Stock", list(nifty50.keys()), index=0)
    with colB:
        symbol = nifty50[stock]
        st.write(f"Chosen: **{stock} ({symbol})**")

    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)

    # Default dates cover all of fetched data; user may override
    start_date = st.sidebar.date_input("Start Date (IST)", value=pd.to_datetime("today").date() - pd.Timedelta(days=180))
    end_date = st.sidebar.date_input("End Date (IST)", value=pd.to_datetime("today").date())

    if st.button("Run Backtest 🚀", use_container_width=True):
        with st.spinner("Fetching data..."):
            raw = fetch_market_data(symbol, period=period, interval=interval)
            if raw.empty:
                st.error("No data returned from Yahoo. Try a different period/interval.")
                return

            # Index prep (naive datetime in IST)
            df = raw.copy()
            df.set_index("timestamp", inplace=True)

            # Convert sidebar dates to pandas Timestamps (naive IST) and make end inclusive
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

            # SAFE filter (no tz mismatch now)
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

            if df.empty:
                st.warning("No rows in the selected date range.")
                return

        with st.spinner("Running strategy..."):
            result = run_strategy(df.reset_index())

        st.success("Done ✅")

        # Summary
        total_trades = int((result["signal"] != 0).sum())
        total_pnl = float(result["pnl"].sum())
        final_equity = float(result["equity"].iloc[-1])
        max_dd = float(result["drawdown"].min())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Total P&L (%)", f"{total_pnl:.2f}")
        m3.metric("Final Equity ($)", f"{final_equity:,.2f}")
        m4.metric("Max Drawdown (%)", f"{max_dd:.2f}")

        # Equity curve
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(result["timestamp"], result["equity"], label="Equity")
        ax1.set_title("Equity Curve")
        ax1.set_xlabel("Time (IST)")
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        st.pyplot(fig1)

        # Price + signals
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(result["timestamp"], result["close"], label="Close", alpha=0.7)
        buys = result[result["signal"] == 1]
        sells = result[result["signal"] == -1]
        ax2.scatter(buys["timestamp"], buys["close"], marker="^", s=70, label="Buy")
        ax2.scatter(sells["timestamp"], sells["close"], marker="v", s=70, label="Sell")
        ax2.set_title("Price with Buy/Sell Signals")
        ax2.set_xlabel("Time (IST)")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)


if __name__ == "__main__":
    create_dashboard()
