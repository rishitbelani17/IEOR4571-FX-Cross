# FILE: ml_fx_signals.py
# --------------------------------------------------
# ML-enhanced FX signal module for your ES strategy
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Optional XAI (SHAP) – guarded import
# try:
#     import shap
#     SHAP_AVAILABLE = True
# except ImportError:
#     SHAP_AVAILABLE = False

SHAP_AVAILABLE = False


# ========================================================
# STREAMLIT CONFIG + GLOBAL STYLES
# ========================================================
st.set_page_config(
    page_title="LSTM ES + ML Strategy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAJOR_FX_PAIRS = [
    "EURJPY=X", "USDJPY=X", "EURUSD=X", "GBPUSD=X",
    "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]

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
# 1. CORE HELPERS (same logic as main ES app)
# ========================================================
@st.cache_data
def load_fx(symbol, timeframe, start_date, end_date):
    interval = {
        "Daily": "1d",
        # "4H": "4h",
        # "1H": "1h",
        "Weekly": "1wk",
        "Monthly": "1mo",
        "Quarterly": "3mo"
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

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df = df.astype(float)    
    return df


def exp_smooth(series, lam):
    es = np.zeros(len(series))
    es[0] = series.iloc[0]
    for t in range(1, len(series)):
        es[t] = lam * series.iloc[t] + (1 - lam) * es[t - 1]
    return es


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


def backtest(prices, raw_signals, positions, cost=0.0001, starting_capital=10000):
    prices = prices.astype(float)
    returns = prices.pct_change().fillna(0)

    positions = positions.reindex(returns.index).astype(float)
    raw_signals = raw_signals.reindex(returns.index)

    # strategy returns (%)
    strat_returns_pct = positions.shift(1) * returns

    # transaction cost
    turnover = positions.diff().abs().fillna(0)
    strat_returns_pct -= turnover * cost

    # equity curve
    equity = (1 + strat_returns_pct).cumprod() * starting_capital

    # metrics
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
        "Number of Trades (signals)": int((raw_signals != 0).sum()),
    }

    return strat_returns_pct, equity, metrics, drawdown


# ========================================================
# 2. FEATURE ENGINEERING FOR ML
# ========================================================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_feature_matrix(df_ohlc, alpha, beta):
    """
    Build ML features + target:
    - Features: returns, volatility, MA diff, ES diff, RSI, etc.
    - Target: next-period direction (1 = up, 0 = down/flat)
    """
    df = pd.DataFrame(index=df_ohlc.index)

    close = df_ohlc["Close"]
    high  = df_ohlc["High"]
    low   = df_ohlc["Low"]

    df["price"] = close
    df["ret_1"] = close.pct_change()
    df["ret_2"] = close.pct_change(2)
    df["ret_5"] = close.pct_change(5)


    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["ma_fast"] = close.rolling(10).mean()
    df["ma_slow"] = close.rolling(30).mean()
    df["ma_diff"] = df["ma_fast"] - df["ma_slow"]

    df["rsi_14"] = compute_rsi(close, window=14)

    es_alpha = exp_smooth(close, alpha)
    es_beta = exp_smooth(close, beta)
    df["es_alpha"] = es_alpha
    df["es_beta"] = es_beta
    df["es_diff"] = df["es_beta"] - df["es_alpha"]

    # Lagged es_diff for dynamics
    df["es_diff_lag1"] = df["es_diff"].shift(1)
    df["es_diff_lag2"] = df["es_diff"].shift(2)

    # Target: sign of next-period return
    df["future_ret"] = df["ret_1"].shift(-1)
    df["y"] = (df["future_ret"] > 0).astype(int)

    df["log_ret_1"] = np.log(df["price"]).diff()
    df["mom_10"] = df["price"] / df["price"].shift(10) - 1
    df["mom_20"] = df["price"] / df["price"].shift(20) - 1
    df["mom_60"] = df["price"] / df["price"].shift(60) - 1

    def rolling_slope(series, window):
        return series.rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
            raw=False
        )
    
    df["trend_slope_20"] = rolling_slope(np.log(df["price"]), 20)
    df["trend_slope_60"] = rolling_slope(np.log(df["price"]), 60)

    df["dist_high_20"] = df["price"] / df["price"].rolling(20).max() - 1
    df["dist_low_20"]  = df["price"] / df["price"].rolling(20).min() - 1

    df["true_range"] = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        )
    )

    df["atr_14"] = df["true_range"].rolling(14).mean()

    df["parkinson_vol_20"] = (
        (1 / (4 * np.log(2))) * (np.log(high / low) ** 2)
    ).rolling(20).mean()

    vol10 = df["ret_1"].rolling(10).std()
    vol50 = df["ret_1"].rolling(50).std()

    df["vol_ratio_10_50"] = vol10 / (vol50.replace(0, np.nan))
    df["vol_ratio_10_50"] = df["vol_ratio_10_50"].fillna(1.0)

    ma20 = df["price"].rolling(20).mean()
    std20 = df["price"].rolling(20).std()

    df["z_price_20"] = (df["price"] - ma20) / (std20 + 1e-9)
    df["bb_pos_20"] = (df["price"] - ma20) / (2 * std20 + 1e-9)
    
    ema12 = df["price"].ewm(span=12).mean()
    ema26 = df["price"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["roc_10"] = df["price"].pct_change(10)

    df["skew_20"] = df["ret_1"].rolling(20).skew()
    df["kurt_20"] = df["ret_1"].rolling(20).kurt()

    df["var_5_20"] = df["ret_1"].rolling(20).quantile(0.05)
    df["cvar_5_20"] = df["ret_1"].rolling(20).apply(
        lambda x: x[x <= np.quantile(x, 0.05)].mean()
    )

    def safe_conditional_std(x, cond):
        vals = x[cond(x)]
        return vals.std() if len(vals) >= 2 else 0.0

    df["downside_vol_20"] = df["ret_1"].rolling(20).apply(
        lambda x: safe_conditional_std(x, lambda v: v < 0)
    )

    df["upside_vol_20"] = df["ret_1"].rolling(20).apply(
        lambda x: safe_conditional_std(x, lambda v: v > 0)
    )

    df["vol_asymmetry"] = df["downside_vol_20"] / (df["upside_vol_20"] + 1e-6)

    df["ret_autocorr_1"] = df["ret_1"].rolling(20).corr(df["ret_1"].shift(1))
    df["absret_autocorr"] = abs(df["ret_1"]).rolling(20).corr(abs(df["ret_1"]).shift(1))

    def rolling_entropy(x, bins=10):
        hist = np.histogram(x, bins=bins, density=True)[0]
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    df["entropy_ret_20"] = df["ret_1"].rolling(20).apply(rolling_entropy)
    df["entropy_sign_20"] = np.sign(df["ret_1"]).rolling(20).apply(lambda x: rolling_entropy(x, bins=3))

    cum = (1 + df["ret_1"]).cumprod()
    roll_max = cum.rolling(50).max()
    drawdown = cum / roll_max - 1

    df["max_dd_50"] = drawdown.rolling(50).min()
    df["max_dd_50"] = df["max_dd_50"].fillna(0)
    df["time_under_water"] = (drawdown < 0).rolling(50).sum()
    df["time_under_water"] = df["time_under_water"].fillna(0)

    df["es_slope"] = df["es_diff"].diff()
    df["es_accel"] = df["es_diff"].diff().diff()
    df["es_norm"] = df["es_diff"] / (df["price"].rolling(20).std() + 1e-9)
    df["es_strength"] = abs(df["es_diff"])

    cross = (df["es_diff"].shift(1) * df["es_diff"] < 0)
    df["bars_since_cross"] = cross.cumsum()
    df["bars_since_cross"] = df.groupby("bars_since_cross").cumcount()

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    na_counts = df.isna().sum().sort_values(ascending=False)
    print("Top NA features:")
    print(na_counts.head(10))

    # ---- controlled trimming based on max lookback ----
    MAX_LOOKBACK = 65  # must cover largest rolling window (60) + buffer
    df = df.iloc[MAX_LOOKBACK:-1]
    feature_cols = [col for col in df.columns if col not in ["price", "future_ret", "y"]]
    X = df[feature_cols].copy()
    y = df["y"].copy()

    return df, X, y, feature_cols

def make_lstm_sequences(X, y, window=20):
    X_seq, y_seq = [], []

    for i in range(window, len(X)):
        X_seq.append(X.iloc[i-window:i].values)
        y_seq.append(y.iloc[i])

    return np.array(X_seq), np.array(y_seq)

# ========================================================
# 3. SIDEBAR – CONTROLS
# ========================================================
with st.sidebar:
    st.markdown("### 🤖 ML FX Signal Engine")

    st.markdown("#### Live Refresh")
    refresh_now = st.button("🔄 Refresh Now (update latest data)")

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

    timeframe = st.selectbox(
        "Timeframe",
        ["1H", "4H", "Daily", "Weekly"]  # order: intraday → higher TF
    )

    start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    end_date = st.date_input("End Date", value=datetime.now())

    if start_date >= end_date:
        st.error("End Date must be after Start Date.")
        st.stop()
    
    # if timeframe == "1m":
    #     # Yahoo only supports ~30 days of 1m data
    #     if (end_date - start_date).days > 30:
    #         st.warning("1m data limited to ~30 days on Yahoo. Clamping start date.")
    #         start_date = end_date - pd.Timedelta(days=30)

    st.markdown("#### ES Parameters (for Features & Hybrid)")
    alpha = st.slider("α (slow smoothing)", 0.01, 0.5, 0.10, 0.01)
    beta = st.slider("β (fast smoothing)", alpha + 0.01, 0.9, 0.30, 0.01)

    st.markdown("#### Training Split")
    train_frac = st.slider(
        "Train fraction (chronological split)",
        0.4,
        0.9,
        0.7,
        0.05,
    )

    st.markdown("#### Hybrid Strategy Settings")
    x_threshold = st.number_input(
        "x threshold (ESβ − ESα)", min_value=0.0, max_value=5.0, value=0.0, step=0.01
    )
    exit_rule = st.radio("Exit rule", ["cross", "deceleration"], index=0)
    theta = st.slider("θ (deceleration slope)", 0.0, 0.05, 0.001, 0.001)

    st.markdown("#### ML Filter Thresholds")
    ml_long_cutoff = st.slider(
        "Min P(up) to allow LONG", 0.50, 0.80, 0.55, 0.01
    )
    ml_short_cutoff = st.slider(
        "Min P(down) to allow SHORT", 0.50, 0.80, 0.55, 0.01
    )

    starting_capital = st.number_input(
        "Starting Capital", 1000, 10_000_000, 10000, step=1000
    )
    cost = st.number_input(
        "Transaction cost (per turnover)", value=0.0001, step=0.0001
    )

    st.markdown("---")
    run_button = st.button("🚀 Run ML Training + Hybrid Backtest")


# ========================================================
# 4. HEADER
# ========================================================
st.markdown(
    f"""
    <div style="padding: 1rem 1.25rem; border-radius: 0.9rem;
                background: linear-gradient(90deg, #111827, #020617);
                border: 1px solid #1F2937; margin-bottom: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2>LSTM ES + ML Hybrid Strategy</h2>
            <p>
            Sequence-aware LSTM predicts next-period direction probability,
            used as a filter on the α–β Exponential Smoothing strategy.
            </p>
        </div>
        <div style="text-align:right; color:#9CA3AF; font-size:0.80rem;">
          <div>Asset: <span style="color:#E5E7EB;">{symbol}</span></div>
          <div>Data source: <span style="color:#E5E7EB;">Yahoo Finance</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================================================
# 5. MAIN LOGIC
# ========================================================
if not run_button and not refresh_now:
    st.info("Configure parameters and click **Run ML Training + Hybrid Backtest**, or use **Refresh Now**.")
    st.stop()

# ---- Load data ----
df_px = load_fx(symbol, timeframe, start_date, end_date)
prices = df_px["Close"]

if len(prices) < 200:
    st.error("Not enough data for meaningful ML training (need at least ~200 points).")
    st.stop()

# ---- Build features & target ----
feat_df, X, y, feature_cols = build_feature_matrix(df_px, alpha, beta)

n_samples = len(X)
train_size = int(n_samples * train_frac)
WINDOW = 20

X_seq, y_seq = make_lstm_sequences(X, y, window=WINDOW)

seq_train_size = int(len(X_seq) * train_frac)

X_train = X_seq[:seq_train_size]
X_test  = X_seq[seq_train_size:]

y_train = y_seq[:seq_train_size]
y_test  = y_seq[seq_train_size:]

# ---- Train RF model ----
model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=0
)

# ---- Predictions ----
proba_train = model.predict(X_train).flatten()
proba_test  = model.predict(X_test).flatten()

y_pred_train = (proba_train > 0.5).astype(int)
y_pred_test  = (proba_test > 0.5).astype(int)


# Attach probability to full index
X_full_seq, _ = make_lstm_sequences(X, y, window=WINDOW)
proba_full = model.predict(X_full_seq).flatten()

p_up = pd.Series(
    proba_full,
    index=X.index[WINDOW:],
    name="P(up)"
)

# ========================================================
# 6. ML PERFORMANCE KPIs
# ========================================================
def kpi_box(col, label, value, sub=None, fmt=None):
    with col:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-label'>{label}</div>", unsafe_allow_html=True)
        v = fmt(value) if fmt else value
        st.markdown(f"<div class='kpi-value'>{v}</div>", unsafe_allow_html=True)
        if sub is not None:
            st.markdown(f"<div class='kpi-sub'>{sub}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="section-header">
      <h3>ML Model Performance</h3>
      <p>RandomForest classifier predicting next-bar direction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

try:
    test_auc = roc_auc_score(y_test, proba_test)
except Exception:
    test_auc = np.nan

kpi_box(col1, "Train Accuracy", train_acc, sub="In-sample", fmt=lambda v: f"{v*100:.1f}%")
kpi_box(col2, "Test Accuracy", test_acc, sub="Out-of-sample", fmt=lambda v: f"{v*100:.1f}%")
kpi_box(col3, "Test AUC", test_auc, sub="ROC AUC (test)", fmt=lambda v: f"{v:.3f}")
kpi_box(col4, "# Training Samples", len(X_train), sub=f"Total samples: {len(X)}", fmt=lambda v: f"{v:,d}")

with st.expander("🔍 Classification report & confusion matrix (test set)"):
    st.text("Classification report (test):")
    st.text(classification_report(y_test, y_pred_test, digits=3))

    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Down/Flat (0)", "Actual Up (1)"],
        columns=["Predicted 0", "Predicted 1"],
    )
    st.write("Confusion matrix (test):")
    st.dataframe(cm_df)


# ========================================================
# 7. HYBRID ES + ML STRATEGY
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Hybrid ES + ML Strategy</h3>
      <p>ML model filters the α–β signals: only take trades consistent with P(up)/P(down).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ES signals on full price series
es_alpha_full = exp_smooth(prices, alpha)
es_beta_full = exp_smooth(prices, beta)
raw_es, pos_es = generate_signals(
    es_alpha_full, es_beta_full, prices.index, x=x_threshold, exit_rule=exit_rule, theta=theta
)

# Align P(up) with price index
p_up_aligned = p_up.reindex(prices.index).fillna(method="ffill")

# ML-filtered signals
raw_hybrid = raw_es.copy()

for t in range(len(raw_hybrid)):
    if raw_hybrid.iloc[t] == 1:
        if p_up_aligned.iloc[t] < ml_long_cutoff:
            raw_hybrid.iloc[t] = 0
    elif raw_hybrid.iloc[t] == -1:
        p_down = 1.0 - p_up_aligned.iloc[t]
        if p_down < ml_short_cutoff:
            raw_hybrid.iloc[t] = 0

pos_hybrid = raw_hybrid.replace(0, np.nan).ffill().fillna(0)

# Backtest both
strat_es, equity_es, metrics_es, dd_es = backtest(
    prices, raw_es, pos_es, cost=cost, starting_capital=starting_capital
)
strat_hybrid, equity_hybrid, metrics_hybrid, dd_hybrid = backtest(
    prices, raw_hybrid, pos_hybrid, cost=cost, starting_capital=starting_capital
)

# KPI comparison
c1, c2, c3, c4 = st.columns(4)
kpi_box(
    c1,
    "ES – Total Return",
    metrics_es["Total Return %"],
    sub="Baseline α–β strategy",
    fmt=lambda v: f"{v:.2f}%",
)
kpi_box(
    c2,
    "Hybrid – Total Return",
    metrics_hybrid["Total Return %"],
    sub="ML-filtered α–β",
    fmt=lambda v: f"{v:.2f}%",
)
kpi_box(
    c3,
    "ES Sharpe",
    metrics_es["Sharpe Ratio"],
    sub="Pure ES",
    fmt=lambda v: f"{v:.2f}",
)
kpi_box(
    c4,
    "Hybrid Sharpe",
    metrics_hybrid["Sharpe Ratio"],
    sub="ES + ML filter",
    fmt=lambda v: f"{v:.2f}",
)

c5, c6, c7, c8 = st.columns(4)
kpi_box(
    c5,
    "ES Max DD",
    metrics_es["Max Drawdown"],
    sub="Baseline",
    fmt=lambda v: f"${v:,.0f}",
)
kpi_box(
    c6,
    "Hybrid Max DD",
    metrics_hybrid["Max Drawdown"],
    sub="Filtered",
    fmt=lambda v: f"${v:,.0f}",
)
kpi_box(
    c7,
    "ES # Trades",
    metrics_es["Number of Trades (signals)"],
    sub="Baseline",
    fmt=lambda v: f"{int(v)}",
)
kpi_box(
    c8,
    "Hybrid # Trades",
    metrics_hybrid["Number of Trades (signals)"],
    sub="Filtered (ML-consistent)",
    fmt=lambda v: f"{int(v)}",
)

# ========================================================
# 7.5 LIVE SIGNAL SNAPSHOT (LATEST BAR)
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Live Signal Snapshot (Latest Bar)</h3>
      <p>Current recommended action based on the most recent candle.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def sig2text(sig):
    if sig == 1:
        return "📈 LONG"
    elif sig == -1:
        return "📉 SHORT"
    return "⏸️ FLAT / NO TRADE"

# Extract latest bar info
last_ts = prices.index[-1]
last_price = float(prices.iloc[-1])

last_es = float(raw_es.iloc[-1])
last_hybrid = float(raw_hybrid.iloc[-1])

prev_es = float(raw_es.iloc[-2]) if len(raw_es) > 1 else 0
prev_hybrid = float(raw_hybrid.iloc[-2]) if len(raw_hybrid) > 1 else 0

flip_es = (last_es != prev_es)
flip_hybrid = (last_hybrid != prev_hybrid)

p_last_up = float(p_up_aligned.iloc[-1])
p_last_down = 1.0 - p_last_up

col_es, col_h = st.columns(2)

# ----- PURE ES CARD -----
with col_es:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Pure ES α–β Strategy</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{sig2text(last_es)}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='kpi-sub'>Price: {last_price:.5f}<br>Bar time: {last_ts}</div>",
        unsafe_allow_html=True,
    )
    if flip_es:
        st.markdown("<div class='kpi-sub' style='color:#10B981;'>🔄 New trade signal triggered!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi-sub'>No change in signal</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- HYBRID CARD -----
with col_h:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-label'>Hybrid ES + ML Strategy</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{sig2text(last_hybrid)}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='kpi-sub'>P(up): {p_last_up:.3f} | P(down): {p_last_down:.3f}<br>Price: {last_price:.5f}<br>Bar: {last_ts}</div>",
        unsafe_allow_html=True,
    )
    if flip_hybrid:
        st.markdown("<div class='kpi-sub' style='color:#10B981;'>🔄 ML-filtered trade triggered!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kpi-sub'>No change in ML signal</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# 8. CHARTS – PRICE, EQUITY, P(UP)
# ========================================================
tab_price, tab_equity, tab_proba = st.tabs(
    ["📈 Price & Signals", "📉 Equity Curves", "📊 P(up) Path"]
)

with tab_price:
    fig = go.Figure()
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
        )
    )

    buy_es = prices[raw_es == 1]
    sell_es = prices[raw_es == -1]

    buy_h = prices[raw_hybrid == 1]
    sell_h = prices[raw_hybrid == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_es.index,
            y=buy_es,
            mode="markers",
            name="ES BUY",
            marker=dict(color="#3B82F6", size=9, symbol="triangle-up"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_es.index,
            y=sell_es,
            mode="markers",
            name="ES SELL",
            marker=dict(color="#F97316", size=9, symbol="triangle-down"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buy_h.index,
            y=buy_h,
            mode="markers",
            name="Hybrid BUY",
            marker=dict(color="#10B981", size=10, symbol="star"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_h.index,
            y=sell_h,
            mode="markers",
            name="Hybrid SELL",
            marker=dict(color="#FB7185", size=10, symbol="x"),
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

with tab_equity:
    eq_df = pd.DataFrame(
        {
            "ES": equity_es,
            "Hybrid": equity_hybrid,
        },
        index=equity_es.index,
    )
    eq_fig = px.line(eq_df)
    eq_fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Equity (USD)",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

with tab_proba:
    proba_fig = px.line(
        p_up_aligned,
        labels={"index": "Date", "value": "P(up)"},
    )
    proba_fig.add_hline(y=ml_long_cutoff, line_dash="dash", annotation_text="Long cutoff")
    proba_fig.add_hline(y=1 - ml_short_cutoff, line_dash="dash", annotation_text="Short cutoff (1 - c)")
    proba_fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(proba_fig, use_container_width=True)


# ========================================================
# 9. XAI VIEW – FEATURE IMPORTANCE + OPTIONAL SHAP
# ========================================================
st.markdown(
    """
    <div class="section-header">
      <h3>Explainability – What Drives P(up)?</h3>
      <p>Feature importances from RandomForest; optional SHAP values if installed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

fa = rf.feature_importances_
fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": fa})
fi_df = fi_df.sort_values("Importance", ascending=False)
fi_df_top = fi_df.head(10)

c1, c2 = st.columns([1.2, 1])

with c1:
    fi_fig = px.bar(
        fi_df_top,
        x="Importance",
        y="Feature",
        orientation="h",
        title="RandomForest Feature Importances",
    )
    fi_fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fi_fig, use_container_width=True)

with c2:
    st.write("Top features driving predictions:")
    st.dataframe(fi_df.head(10), use_container_width=True)

st.markdown("---")

with st.expander("🔬 SHAP Explainability (Plotly Version)", expanded=False):
    if SHAP_AVAILABLE:
        top_features = fi_df_top["Feature"].tolist()
        X_sample = X[top_features].sample(min(300, len(X)), random_state=42)

        explainer = shap.Explainer(rf, X_sample)
        shap_values = explainer(X_sample)

        # Handle classification: take class 1
        if shap_values.values.ndim == 3:
            shap_vals = shap_values.values[:, :, 1]  # (n_samples, n_features)
        else:
            shap_vals = shap_values.values
        st.write("### 📊 Global Feature Importance (Plotly)")

        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_abs
        }).sort_values("importance", ascending=False)

        fig_imp = px.bar(
            importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Mean |SHAP| Importance",
            color="importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        st.write("### 🌈 SHAP Beeswarm (Plotly Version)")

        # Flatten into long-format table
        shap_long = pd.DataFrame(shap_vals, columns=X_sample.columns).melt(
            var_name="feature", value_name="shap_value"
        )
        shap_long["abs"] = shap_long["shap_value"].abs()

        fig_bee = px.strip(
            shap_long,
            x="shap_value",
            y="feature",
            color="abs",
            # color_continuous_scale="RdBu_r",
            title="Beeswarm-like SHAP Plot (Plotly)"
        )
        st.plotly_chart(fig_bee, use_container_width=True)
        st.write("### 🎯 Local Explanation – Select a Sample")
        idx = st.number_input("Sample index", min_value=0, max_value=len(X_sample)-1, value=0)
        sample_features = X_sample.iloc[idx]
        sample_shap = shap_vals[idx]
        st.write("### 💧 Local Waterfall Plot (Plotly)")

        base_value = explainer.expected_value[1] if shap_values.values.ndim == 3 else explainer.expected_value

        df_local = pd.DataFrame({
            "feature": X_sample.columns,
            "shap": sample_shap,
            "direction": ["positive" if v > 0 else "negative" for v in sample_shap]
        }).sort_values("shap")

        fig_local = go.Figure()

        fig_local.add_trace(go.Bar(
            x=df_local["shap"],
            y=df_local["feature"],
            orientation="h",
            marker_color=df_local["shap"],
            marker_colorscale="RdBu",
        ))

        fig_local.update_layout(
            title=f"Local SHAP Values (Sample {idx})",
            xaxis_title="SHAP Contribution",
            yaxis_title="Feature",
            height=600
        )

        st.plotly_chart(fig_local, use_container_width=True)

    else:
        st.info("Install SHAP for explainability: `pip install shap`")
