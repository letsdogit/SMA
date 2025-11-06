import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.set_page_config(page_title="Nifty50 Strategy Dashboard", layout="wide")

# ======================== Helper Functions ========================

def calculate_sma(df, period):
    return df["Close"].rolling(window=period).mean()

def calculate_cpr_daily(df):
    pivot = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    bc = (df["High"].shift(1) + df["Low"].shift(1)) / 2
    tc = pivot + (pivot - bc)
    return pivot, tc, bc

def is_near_sma(price, sma, threshold=0.15):
    if pd.isna(sma) or sma == 0:
        return False
    return abs(price - sma) / sma * 100 <= threshold

def check_buy_signal(df, i):
    current = df.iloc[i]
    if current["Close"] <= current["Open"]:
        return False
    for lookback in [1, 2]:
        prev = df.iloc[i - lookback]
        if prev["Close"] < prev["Open"] and current["High"] > prev["High"]:
            return True
    return False

def check_sell_signal(df, i):
    current = df.iloc[i]
    if current["Close"] >= current["Open"]:
        return False
    for lookback in [1, 2]:
        prev = df.iloc[i - lookback]
        if prev["Close"] > prev["Open"] and current["Low"] < prev["Low"]:
            return True
    return False

# ======================== Strategy Function ========================

def run_strategy(df, leverage=10, commission_rate=0.001):
    df = df.copy().reset_index()
    df.rename(columns={"Date": "timestamp"}, inplace=True)

    df["sma_20"] = calculate_sma(df, 20)
    df["sma_200"] = calculate_sma(df, 200)
    df["pivot"], df["tc"], df["bc"] = calculate_cpr_daily(df)

    cpr_width = abs(df["tc"] - df["bc"])
    cpr_pct = cpr_width / df["bc"].replace(0, np.nan) * 100
    df["narrow_cpr"] = (cpr_pct < 0.06).fillna(False)

    df["signal"] = 0
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df["pnl"] = 0.0
    df["position"] = ""
    df["equity"] = 10000.0
    df["drawdown"] = 0.0

    position = None
    entry_price = 0
    equity = 10000

    for i in range(200, len(df)):
        current = df.iloc[i]

        # Exit Long
        if position == "long":
            tp = entry_price * 1.002
            sl = entry_price * 0.999
            if current["High"] >= tp or current["Low"] <= sl:
                exit_px = tp if current["High"] >= tp else sl
                df.loc[i, "exit_price"] = exit_px
                pct = (exit_px - entry_price) / entry_price * 100
                df.loc[i, "pnl"] = pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + df.loc[i, "pnl"] / 100)
                df.loc[i, "position"] = "exit_long"
                position = None

        # Exit Short
        elif position == "short":
            tp = entry_price * 0.998
            sl = entry_price * 1.001
            if current["Low"] <= tp or current["High"] >= sl:
                exit_px = tp if current["Low"] <= tp else sl
                df.loc[i, "exit_price"] = exit_px
                pct = (entry_price - exit_px) / entry_price * 100
                df.loc[i, "pnl"] = pct * leverage - (2 * commission_rate * 100)
                equity *= (1 + df.loc[i, "pnl"] / 100)
                df.loc[i, "position"] = "exit_short"
                position = None

        df.loc[i, "equity"] = equity
        peak = df["equity"][:i+1].max()
        df.loc[i, "drawdown"] = (equity - peak) / peak * 100 if peak != 0 else 0

        # Entry Logic
        if position is None and df["narrow_cpr"].iloc[i]:
            if is_near_sma(current["Close"], df["sma_20"].iloc[i]):
                # Long
                if i >= 6 and df["sma_20"].iloc[i] > df["sma_20"].iloc[i-6] and check_buy_signal(df, i):
                    df.loc[i, "signal"] = 1
                    df.loc[i, "entry_price"] = current["Close"]
                    df.loc[i, "position"] = "long"
                    position = "long"
                    entry_price = current["Close"]
                # Short
                elif i >= 6 and df["sma_20"].iloc[i] < df["sma_20"].iloc[i-6] and check_sell_signal(df, i):
                    df.loc[i, "signal"] = -1
                    df.loc[i, "entry_price"] = current["Close"]
                    df.loc[i, "position"] = "short"
                    position = "short"
                    entry_price = current["Close"]

    return df

# ======================== Data Fetch ========================

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()

# ======================== NIFTY50 List ========================

NIFTY50 = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS"
}

# ======================== Streamlit UI ========================

st.title("📊 Nifty50 Stock Backtest Dashboard (CPR + SMA Strategy)")
ticker_name = st.selectbox("Select Stock", list(NIFTY50.keys()))
ticker = NIFTY50[ticker_name]
period = st.selectbox("Select Period", ["6mo", "1y", "2y", "5y"], index=2)
leverage = st.slider("Leverage (x)", 1, 50, 10)
commission = st.slider("Commission per side", 0.0, 0.01, 0.001, step=0.0005)

if st.button("Run Backtest 🚀"):
    df = fetch_data(ticker, period)
    result = run_strategy(df, leverage, commission)

    # Metrics
    st.subheader(f"Results for {ticker_name}")
    st.metric("Final Equity", f"${result['equity'].iloc[-1]:.2f}")
    st.metric("Total Trades", int((result['signal'] != 0).sum()))
    st.metric("Max Drawdown", f"{result['drawdown'].min():.2f}%")

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(result["timestamp"], result["Close"], label="Close")
    ax[0].plot(result["timestamp"], result["sma_20"], label="SMA 20")
    ax[0].plot(result["timestamp"], result["sma_200"], label="SMA 200")
    ax[0].legend(); ax[0].set_title("Price with SMAs")

    ax[1].plot(result["timestamp"], result["equity"], label="Equity Curve", color="green")
    ax[1].set_title("Equity Curve")
    st.pyplot(fig)

