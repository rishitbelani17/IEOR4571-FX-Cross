import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fredapi import Fred

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
# 1B. MACRO DATA LOADING (FRED + Yahoo Risk)
# ========================================================
@st.cache_data
def load_macro_fred(start_date, end_date):
    fred = Fred(api_key="bbe32ce50b3dd85ebfd546a30454b4c9")

    series_map = {
        "ecb_rate": "ECBDFR",
        "eur_3m_interbank": "IR3TIB01EZM156N",
        "jpn_interbank": "IRSTCI01JPM156N",
        "jgb_10y": "IRLTLT01JPM156N",
        "eur_cpi_yoy": "CP0000EZ19M086NEST",
        "eur_gdp_level": "CLVMEURSCAB1GQEA19",
        "jpn_gdp_level": "JPNRGDPEXP",
        "eur_ip": "IPN31152N",
        "jpn_ip": "JPNPROINDMISMEI",
    }

    df = pd.DataFrame()

    for col, sid in series_map.items():
        s = fred.get_series(sid)
        s = s.dropna()                      # 🔑 CRITICAL
        s.index = pd.to_datetime(s.index)
        df[col] = s

    df = df.sort_index()

    # buffer for pct_change
    df = df.loc[
        (pd.to_datetime(start_date) - pd.Timedelta(days=400)) :
        pd.to_datetime(end_date)
    ]

    return df

@st.cache_data
def load_macro_yahoo(start_date, end_date):

    def _download_flat(ticker, col_name):
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="column"  # IMPORTANT
        )

        # ---- FORCE FLATTEN ----
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df[["Close"]].rename(columns={"Close": col_name})
        return df

    vix = _download_flat("^VIX", "vix")
    spx = _download_flat("^GSPC", "spx")

    yahoo_df = vix.join(spx, how="outer")
    return yahoo_df.sort_index()

@st.cache_data
def load_fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    # st.write(f"Loaded FRED series {series_id} from CSV. Columns: {df.columns.tolist()}")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df.set_index("observation_date", inplace=True)
    df = df.replace(".", np.nan).astype(float)
    # st.dataframe(df.tail())
    return df.iloc[:, 0]

def build_macro_frame(prices_index, start_date, end_date):
    fred_df = load_macro_fred(start_date, end_date)
    yahoo_df = load_macro_yahoo(start_date, end_date)

    macro = fred_df.join(yahoo_df, how="outer").sort_index()

    # Align base macro frame
    macro = macro.reindex(prices_index).ffill()

    # ---- FIX EUR CPI ALIGNMENT ----
    eur_cpi = macro["eur_cpi_yoy"].dropna()
    eur_cpi = (
        eur_cpi
        .reindex(eur_cpi.index.union(prices_index))
        .sort_index()
        .ffill()
    )
    macro["eur_cpi_yoy"] = eur_cpi.reindex(prices_index)

    # ---- Japan CPI YoY (already YoY from FRED) ----
    jpn_cpi_yoy = load_fred_csv("CPALTT01JPM659N")

    # 1️⃣ Restrict CPI to backtest window
    jpn_cpi_yoy = jpn_cpi_yoy.loc[
        (jpn_cpi_yoy.index <= prices_index.max())
    ]

    # 2️⃣ Reindex using UNION, not overwrite
    jpn_cpi_yoy = (
        jpn_cpi_yoy
        .reindex(jpn_cpi_yoy.index.union(prices_index))
        .sort_index()
        .ffill()
    )

    # 3️⃣ Now align to trading index
    macro["jpn_cpi_yoy"] = jpn_cpi_yoy.reindex(prices_index)

    macro["eur_gdp_qoq"] = macro["eur_gdp_level"].pct_change()
    macro["jpn_gdp_qoq"] = macro["jpn_gdp_level"].pct_change()
    macro["gdp_diff_qoq"] = macro["eur_gdp_qoq"] - macro["jpn_gdp_qoq"]
    macro["spx_ret"] = macro["spx"].pct_change()

    macro["inflation_diff"] = (
        pd.to_numeric(macro["eur_cpi_yoy"], errors="coerce")
        - pd.to_numeric(macro["jpn_cpi_yoy"], errors="coerce")
    )
    macro["inflation_diff"] = macro["inflation_diff"].ffill()

    macro = macro.ffill()

    # st.write("JPN CPI YoY last valid:", macro["jpn_cpi_yoy"].last_valid_index())
    # st.write("Inflation diff last valid:", macro["inflation_diff"].last_valid_index())

    return macro

# def build_macro_frame(prices_index, start_date, end_date):
#     fred_df = load_macro_fred(start_date, end_date)
#     yahoo_df = load_macro_yahoo(start_date, end_date)

#     # ---- Base join ----
#     macro = fred_df.join(yahoo_df, how="outer").sort_index()

#     # ---- Japan CPI (monthly) ----
#     jpn_cpi_level = load_fred_csv("CPALTT01JPM659N")

#     # 🔑 ALIGN CPI TO TRADING INDEX BEFORE INSERTING
#     jpn_cpi_level = jpn_cpi_level.reindex(prices_index).ffill()

#     macro["jpn_cpi_level"] = jpn_cpi_level


#     # ---- Align EVERYTHING to trading index ----
#     macro = macro.reindex(prices_index)

#     # 🔑 THIS IS THE FIX
#     macro = macro.ffill()

#     # ---- Derived features (AFTER ffill) ----
#     macro["jpn_cpi_yoy"] = macro["jpn_cpi_level"].pct_change(12) * 100
#     macro["eur_gdp_qoq"] = macro["eur_gdp_level"].pct_change()
#     macro["jpn_gdp_qoq"] = macro["jpn_gdp_level"].pct_change()
#     macro["gdp_diff_qoq"] = macro["eur_gdp_qoq"] - macro["jpn_gdp_qoq"]
#     macro["spx_ret"] = macro["spx"].pct_change()

#     macro["inflation_diff"] = macro["eur_cpi_yoy"] - macro["jpn_cpi_yoy"]

#     # ---- Final fill for derived cols ----
#     macro = macro.ffill()

#     st.write("JPN CPI level last valid:", macro["jpn_cpi_level"].last_valid_index())
#     st.write("JPN CPI YoY last valid:", macro["jpn_cpi_yoy"].last_valid_index())
#     st.write("Inflation diff last valid:", macro["inflation_diff"].last_valid_index())


#     return macro

def macro_regime_filter(
    macro_df,
    vix_center=25.0,
    vix_width=5.0,
    infl_scale=4.0,
    trade_cutoff=0.25
):
    idx = macro_df.index

    # =========================
    # 1. VIX CONFIDENCE (logistic)
    # =========================
    if "vix" in macro_df.columns:
        vix = macro_df["vix"].fillna(vix_center)
        vix_conf = 1 / (1 + np.exp((vix - vix_center) / vix_width))
    else:
        vix_conf = pd.Series(1.0, index=idx)

    # =========================
    # 2. INFLATION CONFIDENCE (exponential)
    # =========================
    if "inflation_diff" in macro_df.columns:
        infl = macro_df["inflation_diff"].abs().fillna(0.0)
        infl_conf = np.exp(-infl / infl_scale)
    else:
        infl_conf = pd.Series(1.0, index=idx)

    # =========================
    # 3. COMBINE (NO MULTIPLICATIVE DECAY)
    # =========================
    macro_conf = 0.5 * vix_conf + 0.5 * infl_conf
    macro_conf = macro_conf.clip(0.0, 1.0)

    # =========================
    # 4. TRADE PERMISSION (binary)
    # =========================
    trade_allowed = (macro_conf >= trade_cutoff).astype(float)

    # =========================
    # DEBUG
    # =========================
    # st.write("Macro confidence stats:", macro_conf.describe())
    # st.write("Trade allowed %:", trade_allowed.mean())

    return macro_conf, trade_allowed

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

    # ========================================================
    # MACRO STRATEGY CONTROLS
    # ========================================================
    st.markdown("---")
    st.markdown("### 🌍 Macro Integration")

    enable_macro = st.checkbox("Enable ES + Macro Strategy", value=True)

    vix_center = st.slider("VIX stress center", 15.0, 40.0, 25.0, 1.0)
    vix_width = st.slider("VIX smoothness", 1.0, 10.0, 5.0, 0.5)

    infl_scale = st.slider(
        "Inflation tolerance (|EUR - JPN|)",
        1.0, 10.0, 4.0, 0.5
    )

    trade_cutoff = st.slider(
        "Macro confidence cutoff (flatten below)",
        0.0, 1.0, 0.25, 0.05
    )

    # gdp_diff_min = st.slider("Min GDP Diff QoQ (EUR - JPN)", -0.02, 0.02, -0.002, 0.001)

    strategy_view = st.radio(
        "Strategy View",
        ["Baseline ES", "ES + Macro", "Comparison"],
        horizontal=False
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
# 8. DATA + STRATEGY COMPUTATION (BASELINE + MACRO)
# ========================================================
df = load_fx(symbol, timeframe, start_date, end_date)
prices = df["price"]

# ---- BASELINE ES ----
es_alpha = exp_smooth(prices, alpha)
es_beta = exp_smooth(prices, beta)

raw_signals_base, positions_base = generate_signals(
    es_alpha, es_beta, prices.index, x=x_threshold, exit_rule=exit_rule, theta=theta
)

strat_returns_base, equity_base, metrics_base, trades_df_base, drawdown_base = backtest(
    prices, raw_signals_base, positions_base, cost, starting_capital
)

# ---- MACRO AUGMENTED ----
if enable_macro:
    macro_df = build_macro_frame(prices.index, start_date, end_date)
    macro_conf, trade_allowed = macro_regime_filter(
        macro_df,
        vix_center=vix_center,
        vix_width=vix_width,
        infl_scale=infl_scale,
        trade_cutoff=trade_cutoff
    )


    # 1️⃣ Keep original ES signals
    raw_signals_macro = raw_signals_base.copy() * trade_allowed

    # 2️⃣ Build ES positions normally
    positions_macro = raw_signals_macro.replace(0, np.nan).ffill().fillna(0.0)

    # 3️⃣ APPLY MACRO TO POSITIONS (NOT SIGNALS)
    positions_macro = positions_macro * macro_conf.reindex(prices.index).fillna(1.0)

    # st.write("Raw signals:", (raw_signals_base != 0).sum())
    # st.write("Avg macro confidence:", macro_conf.mean())
    # st.write("Avg |position|:", positions_macro.abs().mean())


    strat_returns_macro, equity_macro, metrics_macro, trades_df_macro, drawdown_macro = backtest(
        prices, raw_signals_macro, positions_macro, cost, starting_capital
    )
else:
    macro_df = None
    regime = pd.Series(1.0, index=prices.index)
    raw_signals_macro = raw_signals_base.copy()
    positions_macro = positions_base.copy()
    strat_returns_macro, equity_macro, metrics_macro, trades_df_macro, drawdown_macro = (
        strat_returns_base, equity_base, metrics_base, trades_df_base, drawdown_base
    )

# ---- ACTIVE VIEW SWITCH ----
if strategy_view == "Baseline ES":
    raw_signals = raw_signals_base
    positions = positions_base
    strat_returns = strat_returns_base
    equity = equity_base
    metrics = metrics_base
    trades_df = trades_df_base
    drawdown = drawdown_base

elif strategy_view == "ES + Macro":
    raw_signals = raw_signals_macro
    positions = positions_macro
    strat_returns = strat_returns_macro
    equity = equity_macro
    metrics = metrics_macro
    trades_df = trades_df_macro
    drawdown = drawdown_macro

else:
    # Comparison mode still uses charts based on both (handled below)
    raw_signals = raw_signals_base
    positions = positions_base
    strat_returns = strat_returns_base
    equity = equity_base
    metrics = metrics_base
    trades_df = trades_df_base
    drawdown = drawdown_base

# df = load_fx(symbol, timeframe, start_date, end_date)
# prices = df["price"]

# es_alpha = exp_smooth(prices, alpha)
# es_beta = exp_smooth(prices, beta)

# raw_signals, positions = generate_signals(
#     es_alpha,
#     es_beta,
#     prices.index,
#     x=x_threshold,
#     exit_rule=exit_rule,
#     theta=theta,
# )

# strat_returns, equity, metrics, trades_df, drawdown = backtest(
#     prices,
#     raw_signals,
#     positions,
#     cost,
#     starting_capital,
# )


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
# COMPARISON KPIs (only in Comparison view)
# ========================================================
if strategy_view == "Comparison":
    st.markdown(
        """
        <div class="section-header">
          <h3>Comparison KPIs (Macro − Baseline)</h3>
          <p>Positive is better for Sharpe/Return. Drawdown delta is Baseline − Macro (positive means Macro reduced DD).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cA, cB, cC, cD = st.columns(4)

    delta_sharpe = metrics_macro["Sharpe Ratio"] - metrics_base["Sharpe Ratio"]
    delta_ret = metrics_macro["Total Return %"] - metrics_base["Total Return %"]
    delta_dd = metrics_base["Max Drawdown"] - metrics_macro["Max Drawdown"]
    delta_trades = metrics_macro["Number of Trades (signals)"] - metrics_base["Number of Trades (signals)"]

    kpi_box(cA, "Δ Sharpe", delta_sharpe, sub="Macro − Baseline", fmt=lambda v: f"{v:+.2f}")
    kpi_box(cB, "Δ Total Return", delta_ret, sub="Macro − Baseline", fmt=lambda v: f"{v:+.2f}%")
    kpi_box(cC, "Δ Max Drawdown", delta_dd, sub="Baseline − Macro", fmt=lambda v: f"{v:+,.0f}")
    kpi_box(cD, "Δ Trades", delta_trades, sub="Macro − Baseline", fmt=lambda v: f"{int(v):+d}")


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

tab_price, tab_equity, tab_es, tab_returns, tab_compare = st.tabs(
    ["📈 Price & Trade Signals", "📉 Equity & Drawdown", "📊 Exponential Smoothing", "📦 Return Distribution", "🆚 Compare"]
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

with tab_compare:
    if enable_macro:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=equity_base.index, y=equity_base, mode="lines", name="Baseline ES"))
        fig_cmp.add_trace(go.Scatter(x=equity_macro.index, y=equity_macro, mode="lines", name="ES + Macro"))
        fig_cmp.update_layout(
            height=450,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Equity (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Optional: show regime line
        if macro_df is not None and "vix" in macro_df.columns:
            # st.markdown("#### Macro Regime (1=Trade Allowed, 0=Blocked)")
            # st.line_chart(regime)
            st.markdown("#### Macro Snapshot (latest)")
            st.dataframe(macro_df, use_container_width=True)
    else:
        st.info("Enable ES + Macro Strategy in the sidebar to view comparisons.")


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
