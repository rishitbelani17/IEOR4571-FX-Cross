import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

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
        group_by='ticker',
        progress=False
    )
    df = df.dropna(axis=1, how="all")

    if isinstance(df.columns, pd.MultiIndex):
        df = df["EURJPY=X"]

    df = df[["Close"]].dropna()
    df.columns = ["price"]
    df["price"] = df["price"].astype(float)

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
# 3. SIGNAL GENERATION (general: α, β, x, exit_rule, θ)
# ========================================================
def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):

    d = es_beta - es_alpha
    raw = np.zeros(len(d))   # instantaneous entry/exit signals

    for t in range(1, len(d)):
        # LONG ENTRY with threshold x
        if (d[t-1] < 0) and (d[t] > x):
            raw[t] = 1

        # SHORT ENTRY with threshold x
        elif (d[t-1] > 0) and (d[t] < -x):
            raw[t] = -1

        # Optional deceleration-based exit (flatten position)
        if exit_rule == "deceleration":
            slope = d[t] - d[t-1]
            if slope < -theta:
                raw[t] = 0

    raw_signals = pd.Series(raw, index=index)

    # POSITIONS = held positions (ffill raw signals)
    positions = raw_signals.replace(0, np.nan).ffill().fillna(0)

    return raw_signals, positions


# ========================================================
# 4. BACKTEST (uses positions; trade log uses raw_signals)
# ========================================================
def backtest(prices, raw_signals, positions, cost=0.0001, starting_capital=10000):

    prices = prices.astype(float)
    returns = prices.pct_change().fillna(0)

    positions = positions.reindex(returns.index).astype(float)
    raw_signals = raw_signals.reindex(returns.index)

    # strategy daily return %
    strat_returns_pct = positions.shift(1) * returns

    # transaction cost
    turnover = positions.diff().abs().fillna(0)
    strat_returns_pct -= turnover * cost

    # equity curve
    equity = (1 + strat_returns_pct).cumprod() * starting_capital

    # ---- TRADE LOG ----
    trades = []
    position = 0
    entry_price = None
    entry_time = None

    for i in range(1, len(raw_signals)):
        sig = raw_signals.iloc[i]
        if sig != 0:  # new trading signal
            # close old position if any
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
                    "Direction": "Long" if position == 1 else "Short",
                    "PnL": pnl,
                    "Return %": ret * 100
                })

            # open/flip position
            position = sig
            entry_price = prices.iloc[i]
            entry_time = prices.index[i]

    trades_df = pd.DataFrame(trades)

    # ---- METRICS ----
    sr = np.sign(strat_returns_pct).fillna(0).to_numpy()
    sg = np.sign(positions.shift(1)).fillna(0).to_numpy()
    hit_rate = float(np.mean(sr == sg)) if len(sr) > 0 else 0.0

    total_return = equity.iloc[-1] - starting_capital
    total_return_pct = total_return / starting_capital * 100 if starting_capital > 0 else 0.0
    annualized_return = strat_returns_pct.mean() * 252
    vol = strat_returns_pct.std() * np.sqrt(252)
    sharpe = annualized_return / vol if vol > 0 else 0.0

    drawdown = equity.cummax() - equity
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

    metrics = {
        "Starting Capital": float(starting_capital),
        "Final Value": float(equity.iloc[-1]),
        "Net Profit": float(total_return),
        "Total Return %": float(total_return_pct),
        "Annualized Return": float(annualized_return),
        "Volatility": float(vol),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Hit Rate (daily direction)": float(hit_rate),
        "Number of Trades (signals)": int((raw_signals != 0).sum())
    }

    return strat_returns_pct, equity, metrics, trades_df


# ========================================================
# 5. STREAMLIT UI
# ========================================================
st.title("📈 EUR/JPY Exponential Smoothing Trading Strategy")

# ---- Session state defaults (for integration with optimization page) ----
if "alpha" not in st.session_state:
    st.session_state["alpha"] = 0.1
if "beta" not in st.session_state:
    st.session_state["beta"] = 0.3
if "x" not in st.session_state:
    st.session_state["x"] = 0.0
if "exit_rule" not in st.session_state:
    st.session_state["exit_rule"] = "cross"
if "theta" not in st.session_state:
    st.session_state["theta"] = 0.001

# Sidebar parameters
st.sidebar.header("Parameters")

st.sidebar.subheader("Backtest Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

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

alpha = st.sidebar.slider(
    "α (slow smoothing)", 0.01, 0.5, st.session_state["alpha"], 0.01, key="alpha"
)
beta = st.sidebar.slider(
    "β (fast smoothing)", st.session_state["alpha"] + 0.01, 0.9,
    st.session_state["beta"], 0.01, key="beta"
)

x_threshold = st.sidebar.number_input(
    "x threshold (ESβ − ESα)", min_value=0.0, max_value=5.0,
    value=float(st.session_state["x"]), step=0.01, key="x"
)

exit_rule = st.sidebar.radio(
    "Exit rule", ["cross", "deceleration"], index=0 if st.session_state["exit_rule"]=="cross" else 1, key="exit_rule"
)

theta = st.sidebar.slider(
    "θ (deceleration slope)", 0.0, 0.05, float(st.session_state["theta"]), 0.001, key="theta"
)

cost = st.sidebar.number_input("Transaction cost (per turnover)", value=0.0001)

# Load data
df = load_fx(timeframe, start_date, end_date)
prices = df["price"]

# Compute ES
es_alpha = exp_smooth(prices, alpha)
es_beta = exp_smooth(prices, beta)

# Signals
raw_signals, positions = generate_signals(
    es_alpha, es_beta, prices.index,
    x=x_threshold, exit_rule=exit_rule, theta=theta
)

# Backtest
strat_returns, equity, metrics, trades_df = backtest(
    prices, raw_signals, positions, cost, starting_capital
)

# ================== PLOTS ==================

# ES Plot
st.subheader("Exponential Smoothing")
es_df = pd.DataFrame({
    "Price": prices,
    f"ES(α={alpha:.2f})": es_alpha,
    f"ES(β={beta:.2f})": es_beta
}, index=prices.index)
st.line_chart(es_df)

# Price with Buy/Sell markers
st.subheader("Price with Trade Signals")

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color="white")))

buy_points = prices[raw_signals == 1]
sell_points = prices[raw_signals == -1]

fig.add_trace(go.Scatter(
    x=buy_points.index, y=buy_points,
    mode="markers", marker=dict(color="blue", size=10),
    name="BUY"
))

fig.add_trace(go.Scatter(
    x=sell_points.index, y=sell_points,
    mode="markers", marker=dict(color="red", size=10),
    name="SELL"
))

fig.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Equity curve
st.subheader("Equity Curve")
st.line_chart(equity)

# Metrics
st.subheader("Performance Metrics")
st.json(metrics)

# Trade log
st.subheader("📄 Trade Log")
st.dataframe(trades_df)
