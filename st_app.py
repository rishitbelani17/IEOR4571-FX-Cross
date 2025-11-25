# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# from datetime import datetime

# # ========================================================
# # 1. DATA LOADING
# # ========================================================
# @st.cache_data
# def load_fx(timeframe, start_date, end_date):
#     interval = {
#         "Daily": "1d",
#         "4H": "4h",
#         "1H": "1h",
#         "Weekly": "1wk"
#     }[timeframe]

#     df = yf.download(
#         "EURJPY=X",
#         interval=interval,
#         start=start_date,
#         end=end_date,
#         group_by='ticker',
#         progress=False
#     )
#     df = df.dropna(axis=1, how="all")

#     if isinstance(df.columns, pd.MultiIndex):
#         df = df["EURJPY=X"]

#     df = df[["Close"]].dropna()
#     df.columns = ["price"]
#     df["price"] = df["price"].astype(float)

#     return df


# # ========================================================
# # 2. EXPONENTIAL SMOOTHING
# # ========================================================
# def exp_smooth(series, lam):
#     es = np.zeros(len(series))
#     es[0] = series.iloc[0]
#     for t in range(1, len(series)):
#         es[t] = lam * series.iloc[t] + (1 - lam) * es[t-1]
#     return es


# # ========================================================
# # 3. SIGNAL GENERATION (general: α, β, x, exit_rule, θ)
# # ========================================================
# def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):

#     d = es_beta - es_alpha
#     raw = np.zeros(len(d))   # instantaneous entry/exit signals

#     for t in range(1, len(d)):
#         # LONG ENTRY with threshold x
#         if (d[t-1] < 0) and (d[t] > x):
#             raw[t] = 1

#         # SHORT ENTRY with threshold x
#         elif (d[t-1] > 0) and (d[t] < -x):
#             raw[t] = -1

#         # Optional deceleration-based exit (flatten position)
#         if exit_rule == "deceleration":
#             slope = d[t] - d[t-1]
#             if slope < -theta:
#                 raw[t] = 0

#     raw_signals = pd.Series(raw, index=index)

#     # POSITIONS = held positions (ffill raw signals)
#     positions = raw_signals.replace(0, np.nan).ffill().fillna(0)

#     return raw_signals, positions


# # ========================================================
# # 4. BACKTEST (uses positions; trade log uses raw_signals)
# # ========================================================
# def backtest(prices, raw_signals, positions, cost=0.0001, starting_capital=10000):

#     prices = prices.astype(float)
#     returns = prices.pct_change().fillna(0)

#     positions = positions.reindex(returns.index).astype(float)
#     raw_signals = raw_signals.reindex(returns.index)

#     # strategy daily return %
#     strat_returns_pct = positions.shift(1) * returns

#     # transaction cost
#     turnover = positions.diff().abs().fillna(0)
#     strat_returns_pct -= turnover * cost

#     # equity curve
#     equity = (1 + strat_returns_pct).cumprod() * starting_capital

#     # ---- TRADE LOG ----
#     trades = []
#     position = 0
#     entry_price = None
#     entry_time = None

#     for i in range(1, len(raw_signals)):
#         sig = raw_signals.iloc[i]
#         if sig != 0:  # new trading signal
#             # close old position if any
#             if position != 0:
#                 exit_price = prices.iloc[i]
#                 exit_time = prices.index[i]
#                 pnl = (exit_price - entry_price) * position
#                 ret = pnl / entry_price

#                 trades.append({
#                     "Entry Time": entry_time,
#                     "Entry Price": entry_price,
#                     "Exit Time": exit_time,
#                     "Exit Price": exit_price,
#                     "Direction": "Long" if position == 1 else "Short",
#                     "PnL": pnl,
#                     "Return %": ret * 100
#                 })

#             # open/flip position
#             position = sig
#             entry_price = prices.iloc[i]
#             entry_time = prices.index[i]

#     trades_df = pd.DataFrame(trades)

#     # ---- METRICS ----
#     sr = np.sign(strat_returns_pct).fillna(0).to_numpy()
#     sg = np.sign(positions.shift(1)).fillna(0).to_numpy()
#     hit_rate = float(np.mean(sr == sg)) if len(sr) > 0 else 0.0

#     total_return = equity.iloc[-1] - starting_capital
#     total_return_pct = total_return / starting_capital * 100 if starting_capital > 0 else 0.0
#     annualized_return = strat_returns_pct.mean() * 252
#     vol = strat_returns_pct.std() * np.sqrt(252)
#     sharpe = annualized_return / vol if vol > 0 else 0.0

#     drawdown = equity.cummax() - equity
#     max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

#     metrics = {
#         "Starting Capital": float(starting_capital),
#         "Final Value": float(equity.iloc[-1]),
#         "Net Profit": float(total_return),
#         "Total Return %": float(total_return_pct),
#         "Annualized Return": float(annualized_return),
#         "Volatility": float(vol),
#         "Sharpe Ratio": float(sharpe),
#         "Max Drawdown": float(max_dd),
#         "Hit Rate (daily direction)": float(hit_rate),
#         "Number of Trades (signals)": int((raw_signals != 0).sum())
#     }

#     return strat_returns_pct, equity, metrics, trades_df


# # ========================================================
# # 5. STREAMLIT UI
# # ========================================================
# st.title("📈 EUR/JPY Exponential Smoothing Trading Strategy")

# # ---- Session state defaults (for integration with optimization page) ----
# if "alpha" not in st.session_state:
#     st.session_state["alpha"] = 0.1
# if "beta" not in st.session_state:
#     st.session_state["beta"] = 0.3
# if "x" not in st.session_state:
#     st.session_state["x"] = 0.0
# if "exit_rule" not in st.session_state:
#     st.session_state["exit_rule"] = "cross"
# if "theta" not in st.session_state:
#     st.session_state["theta"] = 0.001

# # Sidebar parameters
# st.sidebar.header("Parameters")

# st.sidebar.subheader("Backtest Date Range")
# start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
# end_date = st.sidebar.date_input("End Date", value=datetime.now())

# if start_date >= end_date:
#     st.sidebar.error("End Date must be after Start Date.")
#     st.stop()

# starting_capital = st.sidebar.number_input(
#     "Starting Capital",
#     min_value=1000,
#     max_value=10_000_000,
#     value=10000
# )

# timeframe = st.sidebar.selectbox("Select timeframe", ["Daily", "4H", "1H", "Weekly"])

# alpha = st.sidebar.slider(
#     "α (slow smoothing)", 0.01, 0.5, st.session_state["alpha"], 0.01, key="alpha"
# )
# beta = st.sidebar.slider(
#     "β (fast smoothing)", st.session_state["alpha"] + 0.01, 0.9,
#     st.session_state["beta"], 0.01, key="beta"
# )

# x_threshold = st.sidebar.number_input(
#     "x threshold (ESβ − ESα)", min_value=0.0, max_value=5.0,
#     value=float(st.session_state["x"]), step=0.01, key="x"
# )

# exit_rule = st.sidebar.radio(
#     "Exit rule", ["cross", "deceleration"], index=0 if st.session_state["exit_rule"]=="cross" else 1, key="exit_rule"
# )

# theta = st.sidebar.slider(
#     "θ (deceleration slope)", 0.0, 0.05, float(st.session_state["theta"]), 0.001, key="theta"
# )

# cost = st.sidebar.number_input("Transaction cost (per turnover)", value=0.0001)

# # Load data
# df = load_fx(timeframe, start_date, end_date)
# prices = df["price"]

# # Compute ES
# es_alpha = exp_smooth(prices, alpha)
# es_beta = exp_smooth(prices, beta)

# # Signals
# raw_signals, positions = generate_signals(
#     es_alpha, es_beta, prices.index,
#     x=x_threshold, exit_rule=exit_rule, theta=theta
# )

# # Backtest
# strat_returns, equity, metrics, trades_df = backtest(
#     prices, raw_signals, positions, cost, starting_capital
# )

# # ================== PLOTS ==================

# # ES Plot
# st.subheader("Exponential Smoothing")
# es_df = pd.DataFrame({
#     "Price": prices,
#     f"ES(α={alpha:.2f})": es_alpha,
#     f"ES(β={beta:.2f})": es_beta
# }, index=prices.index)
# st.line_chart(es_df)

# # Price with Buy/Sell markers
# st.subheader("Price with Trade Signals")

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color="white")))

# buy_points = prices[raw_signals == 1]
# sell_points = prices[raw_signals == -1]

# fig.add_trace(go.Scatter(
#     x=buy_points.index, y=buy_points,
#     mode="markers", marker=dict(color="blue", size=10),
#     name="BUY"
# ))

# fig.add_trace(go.Scatter(
#     x=sell_points.index, y=sell_points,
#     mode="markers", marker=dict(color="red", size=10),
#     name="SELL"
# ))

# fig.update_layout(height=500, template="plotly_dark")
# st.plotly_chart(fig, use_container_width=True)

# # Equity curve
# st.subheader("Equity Curve")
# st.line_chart(equity)

# # Metrics
# st.subheader("Performance Metrics")
# st.json(metrics)

# # Trade log
# st.subheader("📄 Trade Log")
# st.dataframe(trades_df)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ========================================================
# STREAMLIT CONFIG + GLOBAL STYLES
# ========================================================
st.set_page_config(
    page_title="EUR/JPY Exponential Smoothing Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Major FX pairs for convenience
MAJOR_FX_PAIRS = [
    "EURJPY=X", "USDJPY=X", "EURUSD=X", "GBPUSD=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"
]

# Simple dark “terminal-like” styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .kpi-card {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background: #111827;
        border: 1px solid #1F2937;
        color: #F9FAFB;
        font-size: 0.95rem;
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #E5E7EB;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #6B7280;
    }
    .section-header {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .section-header h3 {
        color: #F9FAFB;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    .section-header p {
        color: #6B7280;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ========================================================
# 1. DATA LOADING
# ========================================================
@st.cache_data
def load_fx(symbol, timeframe, start_date, end_date):
    interval = {
        "Daily": "1d",
        "4H": "4h",
        "1H": "1h",
        "Weekly": "1wk"
    }[timeframe]

    df = yf.download(
        symbol,
        interval=interval,
        start=start_date,
        end=end_date,
        group_by="ticker",
        progress=False,
    )
    df = df.dropna(axis=1, how="all")

    if isinstance(df.columns, pd.MultiIndex):
        df = df[symbol]

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
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t - 1]
    return es


# ========================================================
# 3. SIGNAL GENERATION (α, β, x, exit_rule, θ)
# ========================================================
def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):
    d = es_beta - es_alpha
    raw = np.zeros(len(d))   # instantaneous entry/exit signals

    for t in range(1, len(d)):
        # LONG ENTRY with threshold x
        if (d[t - 1] < 0) and (d[t] > x):
            raw[t] = 1

        # SHORT ENTRY with threshold x
        elif (d[t - 1] > 0) and (d[t] < -x):
            raw[t] = -1

        # Optional deceleration-based exit (flatten position)
        if exit_rule == "deceleration":
            slope = d[t] - d[t - 1]
            if slope < -theta:
                raw[t] = 0

    raw_signals = pd.Series(raw, index=index)

    # POSITIONS = held positions (ffill raw signals)
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

    # strategy returns
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

    return strat_returns_pct, equity, metrics, trades_df, drawdown


# ========================================================
# 5. SESSION STATE DEFAULTS
# ========================================================
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
if "presets" not in st.session_state:
    st.session_state["presets"] = {}


# ========================================================
# 6. SIDEBAR – PARAMETERS + FX + PRESETS
# ========================================================
with st.sidebar:
    st.markdown("### ⚙️ Strategy Parameters")

    # ----- FX Pair Selection -----
    st.markdown("#### FX Pair Selection")

    selected_pair = st.selectbox(
        "Choose FX Cross (major pairs)",
        MAJOR_FX_PAIRS,
        index=0,
    )

    custom_pair = st.text_input(
        "Or enter custom Yahoo symbol",
        placeholder="e.g., BTC-USD, XAUUSD=X",
    )

    symbol = custom_pair.strip() if custom_pair.strip() != "" else selected_pair

    # Live FX pull
    if st.button("🔄 Live Price Pull"):
        try:
            live_df = yf.download(symbol, period="1d", interval="1m", progress=False)
            live_price = float(live_df["Close"].iloc[-1])
            st.success(f"Live {symbol} Price: {live_price}")
            st.session_state["latest_live_price"] = live_price
        except Exception as e:
            st.error(f"Live price fetch failed: {e}")

    if "latest_live_price" in st.session_state:
        st.metric("Latest Tick", st.session_state["latest_live_price"])

    st.markdown("#### Backtest Window")
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End Date", value=datetime.now())

    if start_date >= end_date:
        st.error("End Date must be after Start Date.")
        st.stop()

    timeframe = st.selectbox("Timeframe", ["Daily", "4H", "1H", "Weekly"])

    starting_capital = st.number_input(
        "Starting Capital",
        min_value=1000,
        max_value=10_000_000,
        value=10000,
        step=1000,
    )

    st.markdown("#### Smoothing Parameters")
    alpha = st.slider(
        "α (slow smoothing)", 0.01, 0.5, st.session_state["alpha"], 0.01, key="alpha"
    )
    beta = st.slider(
        "β (fast smoothing)",
        st.session_state["alpha"] + 0.01,
        0.9,
        st.session_state["beta"],
        0.01,
        key="beta",
    )

    st.markdown("#### Signal & Exit Logic")
    x_threshold = st.number_input(
        "x threshold (ESβ − ESα)",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state["x"]),
        step=0.01,
        key="x",
    )

    exit_rule = st.radio(
        "Exit rule",
        ["cross", "deceleration"],
        index=0 if st.session_state["exit_rule"] == "cross" else 1,
        key="exit_rule",
    )

    theta = st.slider(
        "θ (deceleration slope)",
        0.0,
        0.05,
        float(st.session_state["theta"]),
        0.001,
        key="theta",
    )

    cost = st.number_input(
        "Transaction cost (per turnover)",
        value=0.0001,
        step=0.0001,
    )

    # ----- PRESET MANAGER -----
    st.markdown("---")
    st.markdown("### 💾 Preset Manager")

    preset_name = st.text_input("Preset Name")

    # Save preset
    if st.button("Save Preset"):
        if preset_name.strip() == "":
            st.warning("Preset name cannot be empty.")
        else:
            st.session_state["presets"][preset_name] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "alpha": alpha,
                "beta": beta,
                "x": x_threshold,
                "exit_rule": exit_rule,
                "theta": theta,
                "cost": cost,
                "start_date": start_date,
                "end_date": end_date,
            }
            st.success(f"Saved preset '{preset_name}'")

    # Load preset
    if st.session_state["presets"]:
        preset_to_load = st.selectbox(
            "Load Preset",
            list(st.session_state["presets"].keys()),
        )

        if st.button("Load"):
            p = st.session_state["presets"][preset_to_load]
            st.session_state["symbol_override"] = p["symbol"]
            st.session_state["alpha"] = p["alpha"]
            st.session_state["beta"] = p["beta"]
            st.session_state["x"] = p["x"]
            st.session_state["exit_rule"] = p["exit_rule"]
            st.session_state["theta"] = p["theta"]
            st.session_state["preset_loaded"] = True
            st.success(
                f"Loaded preset '{preset_to_load}'. Parameters have been applied to the model."
            )

    # Delete preset
    if st.session_state["presets"]:
        delete_key = st.selectbox(
            "Delete Preset",
            list(st.session_state["presets"].keys()),
            key="delete_preset_selector",
        )

        if st.button("Delete"):
            del st.session_state["presets"][delete_key]
            st.success(f"Deleted preset '{delete_key}'")


# ========================================================
# 7. HEADER
# ========================================================
# If preset loaded, override symbol at compute-time
if "symbol_override" in st.session_state:
    symbol = st.session_state["symbol_override"]

st.markdown(
    f"""
    <div style="padding: 1rem 1.25rem; border-radius: 0.9rem;
                background: linear-gradient(90deg, #111827, #020617);
                border: 1px solid #1F2937; margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="color:#F9FAFB; margin-bottom:0.1rem;">Exponential Smoothing Trading Strategy</h2>
          <p style="color:#9CA3AF; font-size:0.85rem; margin-top:0.15rem;">
            Dual-smoothing momentum strategy with backtesting, trade log, and parameter optimization.
          </p>
        </div>
        <div style="text-align:right; color:#9CA3AF; font-size:0.80rem;">
          <div>Asset: <span style="color:#E5E7EB;">{symbol}</span></div>
          <div>Data Source: <span style="color:#E5E7EB;">Yahoo Finance</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================================================
# 8. DATA + STRATEGY COMPUTATION
# ========================================================
df = load_fx(symbol, timeframe, start_date, end_date)
prices = df["price"]

es_alpha = exp_smooth(prices, alpha)
es_beta = exp_smooth(prices, beta)

raw_signals, positions = generate_signals(
    es_alpha,
    es_beta,
    prices.index,
    x=x_threshold,
    exit_rule=exit_rule,
    theta=theta,
)

strat_returns, equity, metrics, trades_df, drawdown = backtest(
    prices,
    raw_signals,
    positions,
    cost,
    starting_capital,
)


# ========================================================
# 9. KPI ROWS
# ========================================================
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col5, kpi_col6, kpi_col7, _ = st.columns([1, 1, 1, 0.4])


def kpi_box(col, label, value, sub=None, fmt=None):
    with col:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-label'>{label}</div>", unsafe_allow_html=True)
        if fmt:
            v = fmt(value)
        else:
            v = value
        st.markdown(f"<div class='kpi-value'>{v}</div>", unsafe_allow_html=True)
        if sub is not None:
            st.markdown(f"<div class='kpi-sub'>{sub}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


kpi_box(
    kpi_col1,
    "Final Equity",
    metrics["Final Value"],
    sub="Account value at end of backtest",
    fmt=lambda v: f"${v:,.0f}",
)
kpi_box(
    kpi_col2,
    "Total Return",
    metrics["Total Return %"],
    sub="Net performance over window",
    fmt=lambda v: f"{v:.2f}%",
)
kpi_box(
    kpi_col3,
    "Sharpe Ratio",
    metrics["Sharpe Ratio"],
    sub="Annualized (252 trading days)",
    fmt=lambda v: f"{v:.2f}",
)
kpi_box(
    kpi_col4,
    "Max Drawdown",
    metrics["Max Drawdown"],
    sub="Peak-to-trough in $",
    fmt=lambda v: f"${v:,.0f}",
)

kpi_box(
    kpi_col5,
    "Hit Rate",
    metrics["Hit Rate (daily direction)"],
    sub="Daily direction accuracy",
    fmt=lambda v: f"{v * 100:.1f}%",
)
kpi_box(
    kpi_col6,
    "# Trades",
    metrics["Number of Trades (signals)"],
    sub="Unique entry signals",
    fmt=lambda v: f"{int(v)}",
)
kpi_box(
    kpi_col7,
    "Volatility",
    metrics["Volatility"],
    sub="Annualized std of strategy returns",
    fmt=lambda v: f"{v:.3f}",
)


# ========================================================
# 10. STRATEGY SUMMARY CARDS
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Strategy Configuration</h3>
      <p>Quick snapshot of the current backtest setup.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, _ = st.columns([1, 1, 1, 0.4])

c1.info(f"**Timeframe**\n{timeframe}")
c2.info(f"**Window**\n{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
c3.info(f"**α (slow)**\n{alpha:.2f}")
c4.info(f"**β (fast)**\n{beta:.2f}")

c5.info(f"**x threshold**\n{x_threshold:.3f}")
c6.info(f"**Exit rule**\n{exit_rule}")
c7.info(f"**θ (deceleration)**\n{theta:.4f}")


# ========================================================
# 11. CHARTS – TABS
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Charts & Diagnostics</h3>
      <p>Explore price, signals, equity curve, and smoothing behavior.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_price, tab_equity, tab_es, tab_returns = st.tabs(
    ["📈 Price & Trade Signals", "📉 Equity & Drawdown", "📊 Exponential Smoothing", "📦 Return Distribution"]
)

# --- Tab 1: Price & Trade Signals ---
with tab_price:
    fig = go.Figure()

    # Candlestick for price (using close as OHLC proxy)
    ohlc = pd.DataFrame(index=prices.index)
    ohlc["Open"] = prices
    ohlc["High"] = prices
    ohlc["Low"] = prices
    ohlc["Close"] = prices

    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["Open"],
            high=ohlc["High"],
            low=ohlc["Low"],
            close=ohlc["Close"],
            name=str(symbol),
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            showlegend=True,
        )
    )

    # ES overlays
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=es_alpha,
            mode="lines",
            name=f"ES α={alpha:.2f}",
            line=dict(width=1.5, color="#60A5FA"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=es_beta,
            mode="lines",
            name=f"ES β={beta:.2f}",
            line=dict(width=1.5, color="#FBBF24"),
        )
    )

    buy_points = prices[raw_signals == 1]
    sell_points = prices[raw_signals == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points,
            mode="markers",
            marker=dict(color="#3B82F6", size=10, symbol="triangle-up"),
            name="BUY",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points,
            mode="markers",
            marker=dict(color="#F97316", size=10, symbol="triangle-down"),
            name="SELL",
        )
    )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Equity & Drawdown ---
with tab_equity:
    equity_fig = go.Figure()
    equity_fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            mode="lines",
            name="Equity Curve",
            line=dict(width=2),
        )
    )

    equity_fig.update_layout(
        height=450,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Equity (USD)",
    )

    st.plotly_chart(equity_fig, use_container_width=True)

    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(width=1.5),
        )
    )
    dd_fig.update_layout(
        height=300,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Drawdown (USD)",
    )
    st.plotly_chart(dd_fig, use_container_width=True)

# --- Tab 3: Exponential Smoothing Curves ---
with tab_es:
    es_df = pd.DataFrame(
        {
            "Price": prices,
            f"ES(α={alpha:.2f})": es_alpha,
            f"ES(β={beta:.2f})": es_beta,
        },
        index=prices.index,
    )
    es_fig = px.line(
        es_df,
        labels={"value": "Value", "index": "Date", "variable": "Series"},
    )
    es_fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(es_fig, use_container_width=True)

# --- Tab 4: Return Distribution ---
with tab_returns:
    returns_df = pd.DataFrame(
        {
            "Strategy Returns %": strat_returns * 100,
        }
    ).dropna()

    if not returns_df.empty:
        hist_fig = px.histogram(
            returns_df,
            x="Strategy Returns %",
            nbins=50,
            marginal="box",
            opacity=0.8,
        )
        hist_fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.info("Not enough data to plot return distribution.")


# ========================================================
# 12. TRADE LOG & ANALYTICS
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Trade Log & Performance Breakdown</h3>
      <p>Inspect individual trades and PnL characteristics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if trades_df.empty:
    st.info("No completed trades for this configuration yet.")
else:
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        direction_filter = st.multiselect(
            "Direction filter", ["Long", "Short"], default=["Long", "Short"]
        )
    with filter_col2:
        min_return = st.number_input(
            "Min Return %",
            value=float(trades_df["Return %"].min() if not trades_df.empty else -100.0),
        )
    with filter_col3:
        max_return = st.number_input(
            "Max Return %",
            value=float(trades_df["Return %"].max() if not trades_df.empty else 100.0),
        )

    filtered_trades = trades_df[
        trades_df["Direction"].isin(direction_filter)
        & (trades_df["Return %"] >= min_return)
        & (trades_df["Return %"] <= max_return)
    ]

    st.write(f"Showing **{len(filtered_trades)}** trades (filtered).")

    st.dataframe(
        filtered_trades.sort_values("Entry Time"),
        use_container_width=True,
        height=350,
    )

    # PnL distribution
    pnl_tab1, pnl_tab2 = st.tabs(["📊 Trade Return Histogram", "⬇️ Download"])
    with pnl_tab1:
        pnl_fig = px.histogram(
            filtered_trades,
            x="Return %",
            color="Direction",
            nbins=30,
            barmode="overlay",
        )
        pnl_fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(pnl_fig, use_container_width=True)

    with pnl_tab2:
        csv = filtered_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download trade log as CSV",
            data=csv,
            file_name="eurjpy_expsmooth_trades.csv",
            mime="text/csv",
        )
