import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ============================================================
# STREAMLIT CONFIG + STYLES
# ============================================================
st.set_page_config(
    page_title="α–β Optimization: FX Exponential Smoothing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

MAJOR_FX_PAIRS = [
    "EURJPY=X", "USDJPY=X", "EURUSD=X", "GBPUSD=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"
]


# ============================================================
# 1. Data Loader
# ============================================================
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


# ============================================================
# 2. Exponential smoothing
# ============================================================
def exp_smooth(series, lam):
    es = np.zeros(len(series))
    es[0] = series.iloc[0]
    for t in range(1, len(series)):
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t - 1]
    return es


# ============================================================
# 3. Signal generation with x, exit_rule, theta
# ============================================================
def generate_signals(es_alpha, es_beta, index, x=0.0, exit_rule="cross", theta=0.0):
    d = es_beta - es_alpha
    raw = np.zeros(len(d))

    for t in range(1, len(d)):
        if (d[t - 1] < 0) and (d[t] > x):
            raw[t] = 1
        elif (d[t - 1] > 0) and (d[t] < -x):
            raw[t] = -1

        if exit_rule == "deceleration":
            slope = d[t] - d[t - 1]
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
# HEADER
# ============================================================
st.markdown(
    """
    <div style="padding: 1rem 1.25rem; border-radius: 0.9rem;
                background: linear-gradient(90deg, #111827, #020617);
                border: 1px solid #1F2937; margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="color:#F9FAFB; margin-bottom:0.1rem;">α–β Grid Search & Heatmaps</h2>
          <p style="color:#9CA3AF; font-size:0.85rem; margin-top:0.15rem;">
            Explore parameter surfaces for the exponential smoothing strategy and push best settings to the main dashboard.
          </p>
        </div>
        <div style="text-align:right; color:#9CA3AF; font-size:0.80rem;">
          <div>Objective: <span style="color:#E5E7EB;">Sharpe maximization</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================
with st.sidebar:
    st.markdown("### 🔧 Optimization Settings")

    # FX selection
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

    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End Date", value=datetime.now())

    if start_date >= end_date:
        st.error("End Date must be after Start Date.")
        st.stop()

    timeframe = st.selectbox("Timeframe", ["Daily", "4H", "1H", "Weekly"])

    st.markdown("#### α / β Grid Resolution")
    alpha_values = st.slider("Number of α (slow) values", 5, 30, 10)
    beta_values = st.slider("Number of β (fast) values", 5, 30, 10)

    st.markdown("#### Threshold x Grid")
    x_min = st.number_input("x min", 0.0, 5.0, 0.0, 0.01)
    x_max = st.number_input("x max", 0.0, 5.0, 0.5, 0.01)
    x_count = st.slider("Number of x values", 1, 10, 3)

    st.markdown("#### Deceleration θ Grid")
    theta_min = st.number_input("θ min", 0.0, 0.1, 0.0, 0.001)
    theta_max = st.number_input("θ max", 0.0, 0.1, 0.01, 0.001)
    theta_count = st.slider("Number of θ values", 1, 10, 3)

    st.markdown("---")
    st.caption(
        "For each (α, β) pair, the search scans (x, exit rule, θ) combinations "
        "and records the configuration with maximum Sharpe."
    )

st.write(
    f"**FX Symbol:** {symbol}"
)
st.write(
    f"**α–β grid:** {alpha_values} × {beta_values} = {alpha_values * beta_values} base pairs"
)
st.write(f"**x grid size:** {x_count}, **θ grid size:** {theta_count}")


# ============================================================
# DATA LOAD
# ============================================================
df = load_fx(symbol, timeframe, start_date, end_date)
prices = df["price"]

alpha_grid = np.linspace(0.01, 0.40, alpha_values)
beta_grid = np.linspace(0.05, 0.90, beta_values)
x_grid = np.linspace(x_min, x_max, x_count)
theta_grid = np.linspace(theta_min, theta_max, theta_count)

results = []


# ============================================================
# GRID SEARCH
# ============================================================
progress_text = "Running grid search over α–β–x–θ space..."
progress_bar = st.progress(0.0)
total_loops = len(alpha_grid) * len(beta_grid)
loop_count = 0

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
            raw, pos = generate_signals(
                es_alpha, es_beta, prices.index, x=x, exit_rule="cross", theta=0.0
            )
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
                raw, pos = generate_signals(
                    es_alpha,
                    es_beta,
                    prices.index,
                    x=x,
                    exit_rule="deceleration",
                    theta=theta,
                )
                sharpe, total_ret, trades = compute_metrics(prices, raw, pos)

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_total_ret = total_ret
                    best_trades = trades
                    best_x = x
                    best_rule = "deceleration"
                    best_theta = theta

        results.append(
            [
                alpha,
                beta,
                best_sharpe,
                best_total_ret,
                best_trades,
                best_x,
                best_rule,
                best_theta,
            ]
        )

        loop_count += 1
        progress_bar.progress(loop_count / max(total_loops, 1))

results_df = pd.DataFrame(
    results,
    columns=[
        "alpha",
        "beta",
        "Sharpe",
        "TotalReturn",
        "Trades",
        "Best_x",
        "Best_exit_rule",
        "Best_theta",
    ],
)

results_df["TotalReturn %"] = results_df["TotalReturn"] * 100


# ============================================================
# KPI SUMMARY
# ============================================================
if not results_df.empty:
    best_idx = results_df["Sharpe"].idxmax()
    best_row = results_df.loc[best_idx]

    col1, col2, col3, col4 = st.columns(4)

    def kpi_box(col, label, value, sub=None, fmt=None):
        with col:
            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-label'>{label}</div>", unsafe_allow_html=True)
            v = fmt(value) if fmt else value
            st.markdown(f"<div class='kpi-value'>{v}</div>", unsafe_allow_html=True)
            if sub is not None:
                st.markdown(f"<div class='kpi-sub'>{sub}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    kpi_box(
        col1,
        "Best Sharpe",
        best_row["Sharpe"],
        sub="Max across full grid",
        fmt=lambda v: f"{v:.2f}",
    )
    kpi_box(
        col2,
        "Best Total Return",
        best_row["TotalReturn %"],
        sub="Net % for best Sharpe config",
        fmt=lambda v: f"{v:.1f}%",
    )
    kpi_box(
        col3,
        "Best α / β",
        f"{best_row['alpha']:.3f} / {best_row['beta']:.3f}",
        sub="Slow / fast smoothing",
    )
    kpi_box(
        col4,
        "Best x, rule, θ",
        f"{best_row['Best_x']:.3f}, {best_row['Best_exit_rule']}, {best_row['Best_theta']:.4f}",
        sub="Threshold, exit rule, deceleration",
    )
else:
    st.warning("No optimization results computed. Check parameter settings.")


# ============================================================
# HEATMAPS & SURFACE – TABS
# ============================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Parameter Surfaces</h3>
      <p>Sharpe, total return, and trade count across the α–β grid.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_heat_sharpe, tab_heat_ret, tab_heat_trades, tab_surface = st.tabs(
    ["📊 Sharpe Heatmap", "📊 Total Return Heatmap", "📊 #Trades Heatmap", "🌐 3D Surface (Sharpe)"]
)

# Sharpe heatmap
with tab_heat_sharpe:
    pivot_sharpe = results_df.pivot(index="alpha", columns="beta", values="Sharpe")
    fig_sharpe = px.imshow(
        pivot_sharpe,
        labels=dict(color="Sharpe"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig_sharpe.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

# Total return heatmap
with tab_heat_ret:
    pivot_ret = results_df.pivot(index="alpha", columns="beta", values="TotalReturn %")
    fig_ret = px.imshow(
        pivot_ret,
        labels=dict(color="Total Return %"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_ret.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

# Trades heatmap
with tab_heat_trades:
    pivot_trades = results_df.pivot(index="alpha", columns="beta", values="Trades")
    fig_trades = px.imshow(
        pivot_trades,
        labels=dict(color="#Trades"),
        color_continuous_scale="Plasma",
        aspect="auto",
    )
    fig_trades.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fig_trades, use_container_width=True)

# 3D surface for Sharpe
with tab_surface:
    alphas_sorted = sorted(results_df["alpha"].unique())
    betas_sorted = sorted(results_df["beta"].unique())
    Z = np.full((len(alphas_sorted), len(betas_sorted)), np.nan)

    for i, a in enumerate(alphas_sorted):
        for j, b in enumerate(betas_sorted):
            row = results_df[(results_df["alpha"] == a) & (results_df["beta"] == b)]
            if not row.empty:
                Z[i, j] = row["Sharpe"].values[0]

    surface_fig = go.Figure(
        data=[
            go.Surface(
                x=np.array(betas_sorted),
                y=np.array(alphas_sorted),
                z=Z,
            )
        ]
    )
    surface_fig.update_layout(
        scene=dict(
            xaxis_title="β (fast)",
            yaxis_title="α (slow)",
            zaxis_title="Sharpe",
        ),
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(surface_fig, use_container_width=True)


# ============================================================
# BEST PARAMS TABLE + INTEGRATION
# ============================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Best Parameter Combinations</h3>
      <p>Top configurations sorted by Sharpe; push a row into the main strategy page.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not results_df.empty:
    top_n = st.slider("Show top N configs", 5, 50, 15)
    best = results_df.sort_values("Sharpe", ascending=False).head(top_n)
    st.dataframe(best, use_container_width=True, height=350)

    csv_res = best.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download top configs as CSV",
        data=csv_res,
        file_name="fx_expsmooth_alpha_beta_optimization.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### Push Parameters to Main Page")

    for i, row in best.iterrows():
        col1, col2 = st.columns([4, 1])
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
                st.success(
                    "Updated main page parameters. Go back to the main dashboard to see them applied."
                )
else:
    st.info("No results to display yet. Adjust grid parameters and re-run.")
