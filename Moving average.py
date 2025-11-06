# ===================== Data Fetch =====================

@st.cache_data(ttl=3600)
def fetch_yahoo_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    # Keep standard columns; Date index stays (run_strategy handles reset)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# ===================== Nifty50 (editable subset) =====================

NIFTY50 = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank": "AXISBANK.NS",
    "Oil & Natural Gas": "ONGC.NS",
    "Bharat Petroleum": "BPCL.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Maruti Suzuki": "MARUTI.NS",
}


# ===================== Streamlit App =====================

st.title("Nifty50 Backtest Dashboard — Daily CPR")
st.caption(
    "Uses Yahoo Finance (yfinance). CPR is computed from the **previous day** OHLC. "
    "Reliance is included by default. No intraday CPR in this version."
)

with st.sidebar:
    st.header("Parameters")
    ticker_name = st.selectbox("Pick a Nifty50 stock", options=list(NIFTY50.keys()), index=0)
    ticker = NIFTY50[ticker_name]
    period = st.selectbox("History period", options=["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
    leverage = st.number_input("Leverage (x)", min_value=1, max_value=50, value=10)
    commission = st.number_input(
        "Commission per side (decimal)", min_value=0.0, max_value=0.01, value=0.001, step=0.0005
    )
    run_btn = st.button("Run Backtest", use_container_width=True)

# Quick action button
if st.button("Run Reliance sample (2y daily)"):
    ticker_name = "Reliance Industries"
    ticker = NIFTY50[ticker_name]
    period = "2y"
    leverage = 10
    commission = 0.001
    run_btn = True

if run_btn:
    with st.spinner(f"Fetching {ticker_name} ({ticker}) daily data and running backtest..."):
        df = fetch_yahoo_daily(ticker, period=period)
        if df.empty or len(df) < 220:  # enough bars for SMA200
            st.error("Not enough data fetched. Try a longer period or verify ticker.")
        else:
            result = run_strategy(df, leverage=leverage, commission_rate=commission)

            # -------- Summary --------
            total_trades = int((result["signal"] != 0).sum())
            winning_trades = int((result["pnl"] > 0).sum())
            losing_trades = int((result["pnl"] < 0).sum())
            total_pnl = float(result["pnl"].sum())
            final_equity = float(result["equity"].iloc[-1])
            roi = ((final_equity - 10000) / 10000) * 100
            max_drawdown = float(result["drawdown"].min())
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            st.subheader(f"Backtest — {ticker_name} ({ticker})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Equity", f"${final_equity:,.2f}", delta=f"{roi:.2f}%")
            m2.metric("Total Trades", total_trades)
            m3.metric("Win Rate", f"{win_rate:.2f}%")
            m4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

            # -------- Charts --------
            fig = plt.figure(figsize=(14, 8))

            ax = fig.add_subplot(2, 1, 1)
            ax.plot(result["timestamp"], result["Close"], label="Close", linewidth=1)
            ax.plot(result["timestamp"], result["sma_20"], label="SMA 20")
            ax.plot(result["timestamp"], result["sma_200"], label="SMA 200")

            buys = result[result["signal"] == 1]
            sells = result[result["signal"] == -1]
            ax.scatter(buys["timestamp"], buys["Close"], marker="^", s=70, label="Buy")
            ax.scatter(sells["timestamp"], sells["Close"], marker="v", s=70, label="Sell")

            ax.set_title("Price with SMAs and Signals (Daily)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(result["timestamp"], result["equity"], linewidth=2)
            ax2.axhline(y=10000, linestyle="--", linewidth=0.8)
            ax2.set_title("Equity Curve")
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            # -------- Trade logs --------
            entries = result[result["signal"] != 0][
                ["timestamp", "signal", "entry_price", "position", "sma_20"]
            ]
            exits = result[result["pnl"] != 0][["timestamp", "exit_price", "pnl", "position"]]
            st.subheader("Trade Log")
            if not entries.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Entries")
                    st.dataframe(entries.reset_index(drop=True))
                with c2:
                    st.write("Exits / PnL")
                    st.dataframe(exits.reset_index(drop=True))
            else:
                st.info("No trades executed during this backtest period.")

            # -------- Download --------
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full backtest CSV",
                data=csv,
                file_name=f"backtest_{ticker}.csv",
                mime="text/csv",
            )

st.markdown("---")
st.write("Notes:")
st.write("- CPR is computed only on **daily** data using the *previous day* OHLC.")
st.write("- Commission is per side (entry and exit both charged).")
st.write("- Want intraday CPR mapping (15m/1h) or Plotly charts? I can add that next.")
