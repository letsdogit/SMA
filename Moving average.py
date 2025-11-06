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

    for i in range(200, len(df)):  # ensure SMA200 exists
        current = df.iloc[i]

        # Exit existing trades
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

        narrow_flag = df["narrow_cpr"].iloc[i] == True

        if position is None and narrow_flag and is_near_sma(current["Close"], current["sma_20"]):
            if i >= 6 and df["sma_20"].iloc[i] > df["sma_20"].iloc[i-6] and check_buy_signal(df, i):
                df.loc[i, "signal"] = 1
                df.loc[i, "entry_price"] = current["Close"]
                df.loc[i, "position"] = "long"
                position = "long"
                entry_price = current["Close"]
            elif i >= 6 and df["sma_20"].iloc[i] < df["sma_20"].iloc[i-6] and check_sell_signal(df, i):
                df.loc[i, "signal"] = -1
                df.loc[i, "entry_price"] = current["Close"]
                df.loc[i, "position"] = "short"
                position = "short"
                entry_price = current["Close"]

    return df
