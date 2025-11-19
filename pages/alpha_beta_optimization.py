import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# 1. Data Loader (same as main)
# ============================================================
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


# ============================================================
# 2. Exponential smoothing
# ============================================================
def exp_smooth(series, lam):
    es = np.zeros(len(series))
    es[0] = series.iloc[0]
    for t in range(1, len(series)):
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t-1]
    return es


# ============================================================
# 3. Signal generation with x, exit_rule, theta
# ============================================================
def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):
    d = es_beta - es_alpha
    raw = np.zeros(len(d))

    for t in range(1, len(d)):
        if (d[t-1] < 0) and (d[t] > x):
            raw[t] = 1
        elif (d[t-1] > 0) and (d[t] < -x):
            raw[t] = -1

        if exit_rule == "deceleration":
            slope = d[t] - d[t-1]
            if slope < -theta:
                raw[t] = 0

    raw_signals = pd.Series(raw, index=index)
    positions = raw_signals.replace(0, np.nan).ffill().fillna(0)

    return raw_signals, positions


# ============================================================
# 4. Compute metrics (Sharpe, total return, trades)
# ============================================================
def compute_metrics(prices, raw_signals, positions):
    returns = prices.pct_change().fillna(0)
    positions = positions.reindex(returns.index)

    strat_returns = positions.shift(1) * returns

    if strat_returns.std() == 0:
        sharpe = -999.0
    else:
        sharpe = (strat_returns.mean() * 252) / (strat_returns.std() * np.sqrt(252))

    total_ret = (1 + strat_returns).prod() - 1
    n_trades = int((raw_signals != 0).sum())

    return float(sharpe), float(total_ret), n_trades


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🔍 α – β Optimization & Heatmaps")

st.sidebar.header("Optimization Settings")

# Date range & timeframe
start_date = st.sidebar.date_input("Start Date", value=datetime(2024,1,1))
end_date   = st.sidebar.date_input("End Date", value=datetime.now())

timeframe = st.sidebar.selectbox("Timeframe", ["Daily", "4H", "1H", "Weekly"])

# Grid size
alpha_values = st.sidebar.slider("Number of α (slow) values", 5, 30, 10)
beta_values  = st.sidebar.slider("Number of β (fast) values", 5, 30, 10)

# Threshold x grid
st.sidebar.subheader("Threshold x Grid")
x_min = st.sidebar.number_input("x min", 0.0, 5.0, 0.0, 0.01)
x_max = st.sidebar.number_input("x max", 0.0, 5.0, 0.5, 0.01)
x_count = st.sidebar.slider("Number of x values", 1, 10, 3)

# Deceleration θ grid
st.sidebar.subheader("Deceleration θ Grid")
theta_min = st.sidebar.number_input("θ min", 0.0, 0.1, 0.0, 0.001)
theta_max = st.sidebar.number_input("θ max", 0.0, 0.1, 0.01, 0.001)
theta_count = st.sidebar.slider("Number of θ values", 1, 10, 3)

st.write(f"Total α–β grid: **{alpha_values} × {beta_values}**")
st.write(f"x grid size: **{x_count}**, θ grid size: **{theta_count}**")
st.write("Optimization will pick, for each (α, β), the best (x, exit rule, θ) by Sharpe.")

# Load data
df = load_fx(timeframe, start_date, end_date)
prices = df["price"]

alpha_grid = np.linspace(0.01, 0.40, alpha_values)
beta_grid  = np.linspace(0.05, 0.90, beta_values)
x_grid     = np.linspace(x_min, x_max, x_count)
theta_grid = np.linspace(theta_min, theta_max, theta_count)

results = []

# ================== GRID SEARCH ==================
for alpha in alpha_grid:
    es_alpha = exp_smooth(prices, alpha)

    for beta in beta_grid:
        if beta <= alpha:
            continue

        es_beta = exp_smooth(prices, beta)

        best_sharpe = -9999.0
        best_total_ret = None
        best_trades = None
        best_x = None
        best_rule = None
        best_theta = None

        for x in x_grid:
            # Exit rule: cross
            raw, pos = generate_signals(es_alpha, es_beta, prices.index, x=x, exit_rule="cross", theta=0.0)
            sharpe, total_ret, trades = compute_metrics(prices, raw, pos)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_total_ret = total_ret
                best_trades = trades
                best_x = x
                best_rule = "cross"
                best_theta = 0.0

            # Exit rule: deceleration (loop θ)
            for theta in theta_grid:
                raw, pos = generate_signals(es_alpha, es_beta, prices.index, x=x, exit_rule="deceleration", theta=theta)
                sharpe, total_ret, trades = compute_metrics(prices, raw, pos)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_total_ret = total_ret
                    best_trades = trades
                    best_x = x
                    best_rule = "deceleration"
                    best_theta = theta

        results.append([
            alpha, beta,
            best_sharpe,
            best_total_ret,
            best_trades,
            best_x,
            best_rule,
            best_theta
        ])

results_df = pd.DataFrame(results, columns=[
    "alpha", "beta", "Sharpe", "TotalReturn", "Trades",
    "Best_x", "Best_exit_rule", "Best_theta"
])

results_df["TotalReturn %"] = results_df["TotalReturn"] * 100

# ================== HEATMAPS ==================
st.subheader("📊 Sharpe Ratio Heatmap")
pivot_sharpe = results_df.pivot(index="alpha", columns="beta", values="Sharpe")
fig_sharpe = px.imshow(
    pivot_sharpe,
    labels=dict(color="Sharpe"),
    color_continuous_scale="RdBu_r",
    aspect="auto",
)
fig_sharpe.update_layout(height=500)
st.plotly_chart(fig_sharpe, use_container_width=True)

st.subheader("📊 Total Return % Heatmap")
pivot_ret = results_df.pivot(index="alpha", columns="beta", values="TotalReturn %")
fig_ret = px.imshow(
    pivot_ret,
    labels=dict(color="Total Return %"),
    color_continuous_scale="Viridis",
    aspect="auto",
)
fig_ret.update_layout(height=500)
st.plotly_chart(fig_ret, use_container_width=True)

st.subheader("📊 Number of Trades Heatmap")
pivot_trades = results_df.pivot(index="alpha", columns="beta", values="Trades")
fig_trades = px.imshow(
    pivot_trades,
    labels=dict(color="#Trades"),
    color_continuous_scale="Plasma",
    aspect="auto",
)
fig_trades.update_layout(height=500)
st.plotly_chart(fig_trades, use_container_width=True)

# ================== 3D SURFACE PLOT (Sharpe) ==================
st.subheader("🌐 3D Surface: Sharpe vs α, β")

# ensure consistent grids
alphas_sorted = sorted(results_df["alpha"].unique())
betas_sorted = sorted(results_df["beta"].unique())
Z = np.full((len(alphas_sorted), len(betas_sorted)), np.nan)

for i, a in enumerate(alphas_sorted):
    for j, b in enumerate(betas_sorted):
        row = results_df[(results_df["alpha"] == a) & (results_df["beta"] == b)]
        if not row.empty:
            Z[i, j] = row["Sharpe"].values[0]

surface_fig = go.Figure(data=[go.Surface(
    x=np.array(betas_sorted),
    y=np.array(alphas_sorted),
    z=Z
)])
surface_fig.update_layout(
    scene=dict(
        xaxis_title="β (fast)",
        yaxis_title="α (slow)",
        zaxis_title="Sharpe"
    ),
    height=600
)
st.plotly_chart(surface_fig, use_container_width=True)

# ================== BEST PARAMS + INTEGRATION ==================
st.subheader("🏆 Best Parameter Combinations (by Sharpe)")

best = results_df.sort_values("Sharpe", ascending=False).head(15)
st.dataframe(best)

st.markdown("Select a row and click a button to apply parameters to the main page.")

for i, row in best.iterrows():
    col1, col2 = st.columns([3,1])
    with col1:
        st.write(
            f"α={row['alpha']:.3f}, β={row['beta']:.3f}, "
            f"x={row['Best_x']:.3f}, rule={row['Best_exit_rule']}, θ={row['Best_theta']:.4f}, "
            f"Sharpe={row['Sharpe']:.2f}, Ret={row['TotalReturn %']:.1f}%, Trades={int(row['Trades'])}"
        )
    with col2:
        if st.button("Use in main page", key=f"use_{i}"):
            st.session_state["alpha"] = float(row["alpha"])
            st.session_state["beta"] = float(row["beta"])
            st.session_state["x"] = float(row["Best_x"])
            st.session_state["exit_rule"] = row["Best_exit_rule"]
            st.session_state["theta"] = float(row["Best_theta"])
            st.success("Updated main page parameters. Go back to the main page to see them applied.")
