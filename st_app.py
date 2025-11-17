import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
# 3. SIGNAL GENERATION
# ========================================================
def generate_signals(es_alpha, es_beta, index, x=0, exit_rule="cross", theta=0):

    d = es_beta - es_alpha
    raw = np.zeros(len(d))   # entry/exit signals only

    for t in range(1, len(d)):
        # LONG ENTRY
        if (d[t-1] < 0) and (d[t] > x):
            raw[t] = 1
        
        # SHORT ENTRY
        elif (d[t-1] > 0) and (d[t] < -x):
            raw[t] = -1

        # DECELERATION EXIT (flat)
        if exit_rule == "deceleration":
            slope = d[t] - d[t-1]
            if slope < -theta:
                raw[t] = 0

    raw_signals = pd.Series(raw, index=index)

    # POSITIONS = executed position state
    positions = raw_signals.replace(0, np.nan).ffill().fillna(0)

    return raw_signals, positions


# ========================================================
# 4. BACKTEST
# ========================================================
def backtest(prices, raw_signals, positions, cost=0.0001, starting_capital=10000):

    prices = prices.astype(float)
    returns = prices.pct_change().fillna(0)

    positions = positions.reindex(returns.index).astype(float)
    raw_signals = raw_signals.reindex(returns.index)

    # Daily P&L %
    strat_returns_pct = positions.shift(1) * returns

    # Cost
    turnover = positions.diff().abs().fillna(0)
    strat_returns_pct -= turnover * cost

    # Portfolio Curve
    equity = (1 + strat_returns_pct).cumprod() * starting_capital

    # TRADE LOG
    trades = []
    position = 0
    entry_price = None
    entry_time = None

    for i in range(1, len(raw_signals)):
        if raw_signals.iloc[i] != 0:  # new signal → trade event
            new_sig = raw_signals.iloc[i]

            # Close previous trade if any
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

            # Open new trade
            position = new_sig
            entry_price = prices.iloc[i]
            entry_time = prices.index[i]

    trades_df = pd.DataFrame(trades)

    # METRICS
    sr = np.sign(strat_returns_pct).fillna(0).to_numpy()
    sg = np.sign(positions.shift(1)).fillna(0).to_numpy()
    hit_rate = float(np.mean(sr == sg))

    total_return = equity.iloc[-1] - starting_capital
    annualized_return = strat_returns_pct.mean() * 252
    vol = strat_returns_pct.std() * np.sqrt(252)
    sharpe = annualized_return / vol if vol > 0 else 0

    drawdown = equity.cummax() - equity
    max_dd = drawdown.max()

    metrics = {
        "Starting Capital": starting_capital,
        "Final Value": float(equity.iloc[-1]),
        "Net Profit": float(total_return),
        "Total Return %": float(total_return / starting_capital * 100),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Hit Rate": hit_rate,
        "Number of Trades": len(trades_df)
    }

    return strat_returns_pct, equity, metrics, trades_df


# ========================================================
# STREAMLIT UI
# ========================================================
st.title("📈 EUR/JPY Exponential Smoothing Trading Strategy")

# Sidebar
st.sidebar.header("Parameters")

st.sidebar.subheader("Backtest Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020,1,1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

starting_capital = st.sidebar.number_input("Starting Capital", min_value=1000, max_value=10_000_000, value=10000)

timeframe = st.sidebar.selectbox("Select timeframe", ["Daily","4H","1H","Weekly"])

alpha = st.sidebar.slider("α (slow smoothing)", 0.01, 0.5, 0.1, 0.01)
beta = st.sidebar.slider("β (fast smoothing)", alpha+0.01, 0.9, 0.3, 0.01)

exit_rule = st.sidebar.radio("Exit rule", ["cross", "deceleration"])
theta = st.sidebar.slider("θ (deceleration)", 0.0, 0.01, 0.001)

cost = st.sidebar.number_input("Transaction cost", value=0.0001)

# Load
df = load_fx(timeframe, start_date, end_date)
prices = df["price"]

# Smoothing
es_alpha = exp_smooth(prices, alpha)
es_beta  = exp_smooth(prices, beta)

# Signals
raw_signals, positions = generate_signals(
    es_alpha, es_beta, prices.index, 
    x=0, exit_rule=exit_rule, theta=theta
)

# Backtest
strat_returns, equity, metrics, trades_df = backtest(
    prices, raw_signals, positions, cost, starting_capital
)

# ========================================================
# PLOTLY PRICE CHART WITH SIGNAL MARKERS
# ========================================================
st.subheader("Price Chart with Buy/Sell Signals")

fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color="white")))

# BUY markers
buy_points = prices[raw_signals == 1]
fig.add_trace(go.Scatter(
    x=buy_points.index, y=buy_points,
    mode="markers", marker=dict(color="blue", size=10),
    name="BUY"
))

# SELL markers
sell_points = prices[raw_signals == -1]
fig.add_trace(go.Scatter(
    x=sell_points.index, y=sell_points,
    mode="markers", marker=dict(color="red", size=10),
    name="SELL"
))

fig.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ========================================================
# EQUITY CURVE
# ========================================================
st.subheader("Equity Curve")
st.line_chart(equity)

# ========================================================
# METRICS
# ========================================================
st.subheader("Performance Metrics")
st.json(metrics)

# ========================================================
# TRADE LOG
# ========================================================
st.subheader("Trade Log")
st.dataframe(trades_df)
