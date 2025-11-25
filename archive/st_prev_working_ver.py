import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ========================================================
# 1. DATA LOADING
# ========================================================
@st.cache_data
def load_fx(timeframe, start_date, end_date):
    interval = {
        "Daily": "1d",
        "4H": "4h",
        "1H": "1h",
        "Weekly": "1wk"
    }[timeframe]

    df = yf.download(
        "EURJPY=X",
        interval=interval,
        start=start_date,
        end=end_date,
        group_by='ticker'
    )
    df = df.dropna(axis=1, how="all")  # remove empty columns

    # FIX: force close column extraction
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance intraday returns multi-index columns
        df = df["EURJPY=X"]  # select the ticker level

    # NOW safely take Close
    df = df[["Close"]].dropna()
    df.columns = ["price"]

    prices = df["price"].astype(float)
    return df


# ========================================================
# 2. EXPONENTIAL SMOOTHING
# ========================================================
def exp_smooth(series, lam):
    es = np.zeros(len(series))
    es[0] = series.iloc[0]
    for t in range(1, len(series)):
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t-1]
    return es


# ========================================================
# 3. SIGNAL GENERATION
# ========================================================
def generate_signals(es_alpha, es_beta, index, x=0, exit_rule="cross", theta=0):

    d = es_beta - es_alpha
    raw = np.zeros(len(d))

    for t in range(1, len(d)):
        if (d[t-1] < 0) and (d[t] > x):
            raw[t] = 1     # long entry signal
        elif (d[t-1] > 0) and (d[t] < -x):
            raw[t] = -1    # short entry signal

        if exit_rule == "deceleration":
            slope = d[t] - d[t-1]
            if slope < -theta:
                raw[t] = 0    # exit signal

    raw_signals = pd.Series(raw, index=index)

    # positions = executed trades (forward filled)
    positions = raw_signals.replace(0, np.nan).ffill().fillna(0)

    return raw_signals, positions


# ========================================================
# 4. BACKTEST
# ========================================================
def backtest(prices, signals, cost=0.0001, starting_capital=10000):

    prices = prices.astype(float)
    returns = prices.pct_change().fillna(0)

    # align signals
    signals = signals.reindex(returns.index).astype(float)

    # strategy returns (percentage)
    strat_returns_pct = signals.shift(1) * returns

    # transaction cost
    turnover = signals.diff().abs().fillna(0)
    strat_returns_pct -= turnover * cost

    # portfolio value
    equity = (1 + strat_returns_pct).cumprod() * starting_capital

    # --- CREATE TRADE LOG ---
    trades = []
    position = 0
    entry_price = None
    entry_time = None

    for i in range(1, len(signals)):
        # if position changes → trade event
        if raw_signals.iloc[i] != 0:

            # close previous trade
            if position != 0:
                exit_price = prices.iloc[i]
                exit_time = prices.index[i]
                pnl = (exit_price - entry_price) * position
                ret = pnl / entry_price

                trades.append({
                    "Entry Time": entry_time,
                    "Entry Price": entry_price,
                    "Exit Time": exit_time,
                    "Exit Price": exit_price,
                    "Direction": "Long" if position==1 else "Short",
                    "PnL": pnl,
                    "Return %": ret * 100
                })

            # open new trade
            if signals.iloc[i] != 0:
                position = signals.iloc[i]
                entry_price = prices.iloc[i]
                entry_time = prices.index[i]
            else:
                position = 0
                entry_price = None
                entry_time = None

    trades_df = pd.DataFrame(trades)

    # --- METRICS ---
    sr = np.sign(strat_returns_pct).fillna(0).to_numpy()
    sg = np.sign(signals.shift(1)).fillna(0).to_numpy()
    hit_rate = float(np.mean(sr == sg))

    total_return = float(equity.iloc[-1] - starting_capital)
    annualized_return = float(strat_returns_pct.mean() * 252)
    vol = float(strat_returns_pct.std() * np.sqrt(252))
    sharpe = float(annualized_return / vol) if vol > 0 else 0

    drawdown = equity.cummax() - equity
    max_dd = float(drawdown.max())

    metrics = {
        "Starting Capital": starting_capital,
        "Final Portfolio Value": float(equity.iloc[-1]),
        "Net Profit": total_return,
        "Total Return %": (total_return / starting_capital) * 100,
        "Annualized Return": annualized_return,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Hit Rate (direction)": hit_rate,
        "Number of Trades": len(trades_df)
    }

    return strat_returns_pct, equity, metrics, trades_df

# ========================================================
# 5. STREAMLIT UI
# ========================================================
st.title("📈 EUR/JPY Exponential Smoothing Trading Strategy")

# Sidebar parameters
st.sidebar.header("Parameters")

from datetime import datetime, timedelta

st.sidebar.subheader("Backtest Date Range")

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2010, 1, 1)       # default can be changed
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now().date()
)

# Ensure valid order
if start_date >= end_date:
    st.sidebar.error("End Date must be after Start Date.")
    st.stop()

starting_capital = st.sidebar.number_input(
    "Starting Capital", 
    min_value=1000, 
    max_value=10_000_000, 
    value=10000
)

timeframe = st.sidebar.selectbox("Select timeframe", ["Daily", "4H", "1H", "Weekly"])

alpha = st.sidebar.slider("α (slow smoothing)", 0.01, 0.5, 0.1, 0.01)
beta = st.sidebar.slider("β (fast smoothing)", alpha + 0.01, 0.9, 0.3, 0.01)

threshold_mode = st.sidebar.selectbox("Threshold type", ["None", "Fixed Pip"])
x = st.sidebar.number_input("x threshold", value=0.0)

exit_rule = st.sidebar.radio("Exit rule", ["cross", "deceleration"])
theta = st.sidebar.slider("θ (deceleration)", 0.0, 0.005, 0.001)

cost = st.sidebar.number_input("Transaction cost (per turnover)", value=0.0001)

# Load data
df = load_fx(timeframe, start_date, end_date)
prices = df["price"].astype(float)

# Compute ES
es_alpha = exp_smooth(prices, alpha)
es_beta = exp_smooth(prices, beta)

# Generate signals
raw_signals, positions = generate_signals(
    es_alpha, es_beta, prices.index, x=x, exit_rule=exit_rule, theta=theta
)

# Backtest
strat_returns, equity, metrics, trades_df = backtest(
    prices, positions, cost, starting_capital
)

# Show ES plot
st.subheader("Exponential Smoothing")
st.line_chart(pd.DataFrame({
    "Price": prices,
    f"ES(alpha={alpha})": es_alpha,
    f"ES(beta={beta})": es_beta
}, index=prices.index))

st.subheader("Price with Trade Signals")

plot_df = pd.DataFrame({"Price": prices}, index=prices.index)

# Buy points
buy_points = prices[raw_signals == 1]
sell_points = prices[raw_signals == -1]

st.line_chart(plot_df)

st.write("**Buy Signals (🔵 Blue)**")
st.write(buy_points)

st.write("**Sell Signals (🔴 Red)**")
st.write(sell_points)

# Show equity curve
st.subheader("Equity Curve")
st.line_chart(equity)

# Show metrics
st.subheader("Performance Metrics")
st.json(metrics)

# Show signals
st.subheader("Signals")
st.write(pd.DataFrame({
    "price": prices,
    "signal": positions
}))

st.subheader("📄 Trade Log")
st.dataframe(trades_df)

st.subheader("Portfolio Value Over Time")
st.line_chart(equity)
