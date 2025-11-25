import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 0. CONFIG – UNIVERSE & ETF MAPPING (S&P 100, 2024)
# =========================================================

SP100_2024 = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","BRK-B","UNH",
    "XOM","JNJ","JPM","V","PG","MA","HD","CVX","ABBV","MRK","NFLX","PEP",
    "BAC","AVGO","KO","PFE","COST","TMO","MCD","CSCO","WMT","ABT","CRM",
    "ACN","ADBE","DHR","CMCSA","LIN","HON","PM","TXN","WFC","NEE","UPS",
    "MS","RTX","UNP","IBM","INTC","LOW","INTU","CAT","GS","AMGN","MDT",
    "QCOM","BLK","ISRG","SPGI","AMD","BKNG","GILD","ADP","NOW","SYK",
    "BDX","EL","DE","AMAT","MO","ZTS","CB","SO","FIS","CME","DUK","CI",
    "GE","LMT","PLD","MMC","ICE","PNC","EW","TGT","APD","USB","SCHW",
    "ADI","MU","SHW","REGN","AON","CL","MAR","ETN","ORLY","VRTX"
]

SECTOR_ETF_MAP = {
    # Tech / Comm / Growth
    "AAPL":"XLK","MSFT":"XLK","AMZN":"XLY","NVDA":"XLK",
    "GOOGL":"XLK","GOOG":"XLK","META":"XLC","ADBE":"XLK",
    "CRM":"XLK","CSCO":"XLK","INTC":"XLK","AMD":"XLK",
    "QCOM":"XLK","NOW":"XLK","ADI":"XLK","MU":"XLK",

    # Financials
    "JPM":"XLF","V":"XLF","MA":"XLF","BAC":"XLF","WFC":"XLF",
    "MS":"XLF","GS":"XLF","SCHW":"XLF","BLK":"XLF","SPGI":"XLF",
    "CME":"XLF","AON":"XLF","MMC":"XLF","PNC":"XLF","USB":"XLF",

    # Healthcare
    "UNH":"XLV","JNJ":"XLV","PFE":"XLV","ABBV":"XLV","MRK":"XLV",
    "ABT":"XLV","MDT":"XLV","AMGN":"XLV","GILD":"XLV","ISRG":"XLV",
    "SYK":"XLV","BDX":"XLV","ZTS":"XLV","REGN":"XLV","EW":"XLV","CI":"XLV",

    # Staples
    "PG":"XLP","KO":"XLP","PEP":"XLP","MO":"XLP","CL":"XLP",

    # Discretionary
    "HD":"XLY","MCD":"XLY","TGT":"XLY","BKNG":"XLY","MAR":"XLY",
    "ORLY":"XLY","NFLX":"XLY","TSLA":"XLY",

    # Industrials
    "HON":"XLI","CAT":"XLI","DE":"XLI","UNP":"XLI","UPS":"XLI",
    "RTX":"XLI","GE":"XLI","ETN":"XLI","LMT":"XLI",

    # Energy
    "XOM":"XLE","CVX":"XLE",

    # Utilities
    "NEE":"XLU","SO":"XLU","DUK":"XLU",

    # Materials
    "LIN":"XLB","APD":"XLB","SHW":"XLB",

    # Real Estate
    "PLD":"XLRE",
}

FACTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()) | {"SPY"})

# Fixed 2024 window
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)

TRAIN_END   = datetime(2024, 9, 30)   # training / parameter tuning period
TEST_START  = datetime(2024, 10, 1)   # out-of-sample backtest


# =========================================================
# 1. STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stat Arb – Avellaneda–Lee (S&P 100 x ETFs, 2024)",
    layout="wide"
)

st.title("📊 Statistical Arbitrage on S&P 100 (Avellaneda–Lee Style, 2024)")
st.markdown(
    """
This app implements an ETF-based **mean-reversion statistical arbitrage** strategy
inspired by **Avellaneda & Lee (2010)**, using **S&P 100** stocks and sector ETFs,
restricted to **2024 data**.

Core ideas:

- 60-day rolling **regressions** of stock returns on sector ETF returns  
- Residuals modeled as **OU / AR(1)** processes → **s-score** signals  
- Contrarian trades: short when residuals are extremely positive, long when extremely negative  
- Optional **train/test split** with **automatic threshold tuning** on the train set
    """
)

# =========================================================
# 2. SIDEBAR – PARAMETERS
# =========================================================

with st.sidebar:
    st.header("⚙️ Backtest Settings")

    st.info("Backtest Period: **2024-01-01 → 2024-12-31** (fixed)")

    initial_equity = st.number_input(
        "Initial Equity ($)", min_value=10_000, max_value=5_000_000,
        value=100_000, step=10_000
    )

    leverage = st.slider(
        "Leverage multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.5
    )

    tc_bps = st.slider(
        "Transaction Cost (per side, bps)", min_value=0, max_value=50,
        value=5, step=1
    )

    st.markdown("---")
    st.subheader("Signal Thresholds (s-score)")

    use_auto_tuning = st.checkbox(
        "🔍 Auto-tune thresholds on train period (Jan–Sep 2024)",
        value=True
    )

    entry_level = st.slider(
        "Manual: Entry threshold |s| ≥",
        min_value=0.5, max_value=3.0,
        value=1.25, step=0.05
    )
    exit_long_level = st.slider(
        "Manual: Exit long when s ≥",
        min_value=-0.5, max_value=1.0,
        value=-0.25, step=0.05
    )
    exit_short_level = st.slider(
        "Manual: Exit short when s ≤",
        min_value=-1.0, max_value=0.5,
        value=0.25, step=0.05
    )

    st.markdown("---")
    mean_reversion_days = st.slider(
        "Max OU half-life (days) to accept mean reversion",
        min_value=5, max_value=60, value=30, step=5
    )

    run_button = st.button("🚀 Run 2024 Backtest")


# =========================================================
# 3. DATA LOADING HELPERS
# =========================================================

@st.cache_data(show_spinner=True)
def download_panel(tickers, start, end):
    """
    Robust wrapper around yfinance.download.

    Returns:
        prices: DataFrame[dates x tickers]
        volumes: DataFrame[dates x tickers]
    """
    if len(tickers) == 0:
        return pd.DataFrame(), pd.DataFrame()

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or len(raw) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            prices = raw["Adj Close"].copy()
        elif "Close" in lvl0:
            prices = raw["Close"].copy()
        else:
            raise KeyError("No price field (Adj Close / Close) in downloaded data.")

        if "Volume" in lvl0:
            volumes = raw["Volume"].copy()
        else:
            volumes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        cols = list(raw.columns)
        if "Adj Close" in cols:
            prices = raw["Adj Close"].to_frame()
        elif "Close" in cols:
            prices = raw["Close"].to_frame()
        else:
            raise KeyError("No price field (Adj Close / Close) in downloaded data.")

        if "Volume" in cols:
            volumes = raw["Volume"].to_frame()
        else:
            volumes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    prices = prices.ffill().bfill()
    volumes = volumes.ffill().bfill()

    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].copy()

    return prices, volumes


@st.cache_data(show_spinner=True)
def load_price_data_2024(stocks, etfs, start, end):
    """
    Download stocks and ETFs separately, then outer-join to ensure
    all ETF columns (especially SPY) survive.
    """
    stock_prices, stock_volumes = download_panel(stocks, start, end)
    etf_prices, etf_volumes = download_panel(etfs, start, end)

    if stock_prices.empty:
        raise ValueError("No stock data returned for 2024.")
    if etf_prices.empty:
        raise ValueError("No ETF data returned for 2024.")

    prices = pd.concat([stock_prices, etf_prices], axis=1).sort_index()
    volumes = pd.concat([stock_volumes, etf_volumes], axis=1).sort_index()

    prices = prices.ffill().bfill()
    volumes = volumes.ffill().bfill()

    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].copy()

    return prices, volumes


# =========================================================
# 4. OU ESTIMATION & S-SCORES
# =========================================================

def estimate_ou_params(residuals: np.ndarray):
    """
    Estimate OU parameters via AR(1) mapping:
      X_{t+1} = a X_t + b + eps

    Returns:
        (kappa, mu, sigma_eq, half_life_days) or (None, ...).
    """
    if len(residuals) < 10:
        return None, None, None, None

    X = residuals
    X_lag = X[:-1]
    Y = X[1:]

    A = np.vstack([X_lag, np.ones(len(X_lag))]).T
    try:
        a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None, None, None

    # Stability and sanity checks
    if not np.isfinite(a) or a <= 0 or a >= 0.9999:
        return None, None, None, None

    eps = Y - (a * X_lag + b)
    var_eps = np.var(eps, ddof=1)
    if var_eps <= 0 or not np.isfinite(var_eps):
        return None, None, None, None

    kappa = -np.log(a)
    mu = b / (1.0 - a)

    try:
        sigma = np.sqrt(2 * kappa * var_eps / (1 - a**2))
    except ZeroDivisionError:
        return None, None, None, None

    if not np.isfinite(sigma) or sigma <= 0:
        return None, None, None, None

    sigma_eq = sigma / np.sqrt(2 * kappa)
    if not np.isfinite(sigma_eq) or sigma_eq <= 0:
        return None, None, None, None

    half_life = np.log(2) / kappa

    return kappa, mu, sigma_eq, half_life


def compute_signals_and_scores(
    prices: pd.DataFrame,
    etf_map: dict,
    window: int = 60,
    max_half_life: int = 30,
):
    """
    For each stock:
      - regress stock returns on its sector ETF returns (rolling window)
      - generate residual series
      - fit OU on rolling window of residuals to compute s-scores

    Returns:
        returns_stock, returns_factor, beta_df, s_score_df
    """
    all_cols = prices.columns.tolist()
    factor_tickers = sorted(set(etf_map.values()))
    stock_tickers = sorted(set(etf_map.keys()))

    factor_tickers = [e for e in factor_tickers if e in all_cols]
    stock_tickers = [t for t in stock_tickers if t in all_cols]

    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns data after pct_change().")

    returns_stock = rets[stock_tickers].copy()
    returns_factor = rets[factor_tickers].copy()

    idx = returns_stock.index
    beta_df = pd.DataFrame(index=idx, columns=stock_tickers, dtype=float)
    s_score_df = pd.DataFrame(index=idx, columns=stock_tickers, dtype=float)

    for ticker in stock_tickers:
        etf = etf_map.get(ticker)
        if etf not in returns_factor.columns:
            continue

        r_stock = returns_stock[ticker].dropna()
        r_factor = returns_factor[etf].dropna()

        # Construct a clean 2-column DataFrame
        df_pair = pd.concat(
            [r_stock.rename("stock"), r_factor.rename("factor")],
            axis=1
        ).dropna()

        if df_pair.shape[1] != 2:
            # Safety: if something weird happens, skip
            continue

        if len(df_pair) < window + 10:
            continue

        dates = df_pair.index
        stock_vals = df_pair["stock"].values
        factor_vals = df_pair["factor"].values

        residual_series = np.full_like(stock_vals, np.nan, dtype=float)

        # Rolling regression
        for i in range(window, len(df_pair)):
            ws = stock_vals[i-window:i]
            wf = factor_vals[i-window:i]
            A = np.vstack([wf, np.ones(window)]).T
            try:
                beta, alpha = np.linalg.lstsq(A, ws, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            fitted = alpha + beta * factor_vals[i]
            residual_series[i] = stock_vals[i] - fitted
            beta_df.loc[dates[i], ticker] = beta

        # OU s-scores
        for i in range(window * 2, len(df_pair)):
            res_window = residual_series[i-window+1:i+1]
            if np.isnan(res_window).any():
                continue

            kappa, mu, sigma_eq, half_life = estimate_ou_params(res_window)
            if kappa is None or sigma_eq is None:
                continue
            if half_life is None or half_life > max_half_life:
                continue

            x_t = residual_series[i]
            if not np.isfinite(x_t):
                continue

            s = (x_t - mu) / sigma_eq
            if not np.isfinite(s):
                continue

            s_score_df.loc[dates[i], ticker] = s

    return returns_stock, returns_factor, beta_df, s_score_df


# =========================================================
# 5. BACKTEST ENGINE
# =========================================================

def backtest_stat_arb(
    returns_stock: pd.DataFrame,
    returns_factor: pd.DataFrame,
    beta_df: pd.DataFrame,
    s_scores: pd.DataFrame,
    etf_map: dict,
    entry_level: float = 1.25,
    exit_long: float = -0.25,
    exit_short: float = 0.25,
    initial_equity: float = 100_000,
    leverage: float = 2.0,
    tc_bps: int = 5,
):
    """
    Simple dollar-neutral stat-arb:

    - For stock i with ETF f and beta β:
        residual return on day t: r_i,t - β r_f,t
    - Side = +1 for long (when s-score is very negative), -1 for short.
      P&L per trade per day is side * (r_i - β r_f).

    Positions:
      positions[ticker] = {"side": +1/-1, "etf": str, "beta": float}
    """
    dates = returns_stock.index
    stock_tickers = returns_stock.columns
    positions = {}  # ticker -> dict(side, etf, beta)

    equity_curve = []
    dates_list = []
    daily_pnl_list = []
    trades_list = []
    trade_records = []

    equity = float(initial_equity)
    tc = tc_bps / 10_000.0  # transaction cost in fraction

    for date in dates:
        if date not in s_scores.index:
            continue

        s_today = s_scores.loc[date]
        beta_today = beta_df.loc[date]

        pnl_day = 0.0
        trades_today = 0

        # 1) PnL from existing positions
        for ticker, pos in list(positions.items()):
            if ticker not in returns_stock.columns:
                continue
            etf = pos["etf"]
            side = pos["side"]
            beta = pos["beta"]

            if etf not in returns_factor.columns:
                continue

            r_s = returns_stock.loc[date, ticker]
            r_f = returns_factor.loc[date, etf]

            if pd.isna(r_s) or pd.isna(r_f):
                continue

            residual_ret = r_s - beta * r_f
            pnl_trade = side * residual_ret
            pnl_day += pnl_trade

        # 2) Signals & trades (entry/exit)
        for ticker in stock_tickers:
            s = s_today.get(ticker, np.nan)
            if pd.isna(s):
                continue

            etf = etf_map.get(ticker)
            if etf not in returns_factor.columns:
                continue

            beta = beta_today.get(ticker, np.nan)

            # Beta safety check
            if not np.isfinite(beta) or beta <= 0 or beta > 5:
                continue

            in_pos = ticker in positions

            # Exit rules
            if in_pos:
                side = positions[ticker]["side"]

                # Close long when s has reverted upwards
                if side == +1 and s >= exit_long:
                    pnl_day -= 2 * tc
                    trades_today += 1
                    trade_records.append(
                        {"date": date, "ticker": ticker, "action": "CLOSE_LONG", "s_score": s}
                    )
                    del positions[ticker]
                    continue

                # Close short when s has reverted downwards
                if side == -1 and s <= exit_short:
                    pnl_day -= 2 * tc
                    trades_today += 1
                    trade_records.append(
                        {"date": date, "ticker": ticker, "action": "CLOSE_SHORT", "s_score": s}
                    )
                    del positions[ticker]
                    continue

            # Entry rules if flat
            if not in_pos:
                # Short signal: s very positive
                if s >= entry_level:
                    positions[ticker] = {"side": -1, "etf": etf, "beta": beta}
                    pnl_day -= 2 * tc
                    trades_today += 1
                    trade_records.append(
                        {"date": date, "ticker": ticker, "action": "OPEN_SHORT", "s_score": s}
                    )

                # Long signal: s very negative
                elif s <= -entry_level:
                    positions[ticker] = {"side": +1, "etf": etf, "beta": beta}
                    pnl_day -= 2 * tc
                    trades_today += 1
                    trade_records.append(
                        {"date": date, "ticker": ticker, "action": "OPEN_LONG", "s_score": s}
                    )

        # 3) Update equity (levered)
        equity *= (1.0 + leverage * pnl_day)

        dates_list.append(date)
        equity_curve.append(equity)
        daily_pnl_list.append(pnl_day)
        trades_list.append(trades_today)

    equity_series = pd.Series(equity_curve, index=dates_list, name="equity")
    pnl_series = pd.Series(daily_pnl_list, index=dates_list, name="pnl")
    trades_series = pd.Series(trades_list, index=dates_list, name="num_trades")
    trades_df = pd.DataFrame(trade_records)

    return equity_series, pnl_series, trades_series, trades_df


# =========================================================
# 6. METRICS & TUNING
# =========================================================

def compute_performance_metrics(equity_series: pd.Series):
    if len(equity_series) < 5:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Vol": np.nan, "MaxDD": np.nan}

    rets_daily = equity_series.pct_change().dropna()
    if rets_daily.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Vol": np.nan, "MaxDD": np.nan}

    mean_ret = rets_daily.mean()
    vol = rets_daily.std()

    sharpe = (mean_ret / vol) * np.sqrt(252) if vol > 0 else np.nan

    total_days = len(rets_daily)
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / total_days) - 1

    roll_max = equity_series.cummax()
    dd = (equity_series / roll_max) - 1.0
    max_dd = dd.min()

    return {"CAGR": cagr, "Sharpe": sharpe, "Vol": vol * np.sqrt(252), "MaxDD": max_dd}


def format_pct(x):
    if pd.isna(x):
        return "–"
    return f"{x*100:,.2f}%"


def auto_tune_thresholds(
    returns_stock_train,
    returns_factor_train,
    beta_train,
    s_train,
    etf_map,
    initial_equity,
    leverage,
    tc_bps,
):
    """
    Simple grid search on train period to select thresholds
    that give the **highest Sharpe** (ties broken by higher CAGR).

    The search space is intentionally small to keep it fast.
    """
    # Parameter grid (small but reasonable)
    entry_grid = np.arange(0.9, 1.8, 0.2)   # |s| entry
    exit_long_grid = np.arange(-0.5, 0.1, 0.2)
    exit_short_grid = np.arange(0.0, 0.6, 0.2)

    best_params = None
    best_sharpe = -np.inf
    best_cagr = -np.inf

    for entry in entry_grid:
        for e_long in exit_long_grid:
            for e_short in exit_short_grid:
                eq, _, _, _ = backtest_stat_arb(
                    returns_stock_train,
                    returns_factor_train,
                    beta_train,
                    s_train,
                    etf_map,
                    entry_level=entry,
                    exit_long=e_long,
                    exit_short=e_short,
                    initial_equity=initial_equity,
                    leverage=leverage,
                    tc_bps=tc_bps,
                )
                if eq.empty:
                    continue
                perf = compute_performance_metrics(eq)
                sharpe = perf["Sharpe"]
                cagr = perf["CAGR"]

                if pd.isna(sharpe):
                    continue

                if (sharpe > best_sharpe) or (
                    np.isclose(sharpe, best_sharpe) and cagr > best_cagr
                ):
                    best_sharpe = sharpe
                    best_cagr = cagr
                    best_params = (entry, e_long, e_short)

    return best_params, best_sharpe, best_cagr


# =========================================================
# 7. MAIN EXECUTION
# =========================================================

if not run_button:
    st.info("Configure parameters in the sidebar and click **Run 2024 Backtest**.")
    st.stop()

# ---------- DATA LOAD ----------
with st.spinner("Loading 2024 market data from Yahoo Finance (S&P 100 + Sector ETFs + SPY)..."):
    prices, volumes = load_price_data_2024(SP100_2024, FACTOR_ETFS, START_DATE, END_DATE)

all_cols = prices.columns.tolist()

usable_stocks = []
effective_map = {}
for t in SP100_2024:
    etf = SECTOR_ETF_MAP.get(t)
    if (t in all_cols) and (etf in all_cols):
        usable_stocks.append(t)
        effective_map[t] = etf

if len(usable_stocks) == 0:
    st.error("No usable stock–ETF pairs found for 2024. Check data or mapping.")
    st.stop()

st.success(f"Using {len(usable_stocks)} S&P 100 names with valid ETF mapping in 2024.")

# ---------- SIGNALS / S-SCORES ----------
with st.spinner("Computing returns, regressions, OU parameters, and s-scores..."):
    returns_stock, returns_factor, beta_df, s_scores = compute_signals_and_scores(
        prices, effective_map, window=60, max_half_life=mean_reversion_days
    )

if s_scores.isna().all().all():
    st.warning("No valid s-scores were computed. Try relaxing OU half-life or using a longer window.")
    st.stop()

# Align all to common index
common_idx = returns_stock.index.intersection(beta_df.index).intersection(s_scores.index)
returns_stock = returns_stock.loc[common_idx]
returns_factor = returns_factor.loc[common_idx]
beta_df = beta_df.loc[common_idx]
s_scores = s_scores.loc[common_idx]

# ---------- TRAIN / TEST SPLIT ----------
mask_train = common_idx <= TRAIN_END
mask_test = common_idx >= TEST_START

if mask_test.sum() < 10:
    st.warning("Not enough test-period data (Oct–Dec 2024). Using full 2024 as test.")
    mask_train = common_idx <= TRAIN_END
    mask_test = common_idx > TRAIN_END

# train slices (for tuning)
rs_train = returns_stock.loc[mask_train]
rf_train = returns_factor.loc[mask_train]
beta_train = beta_df.loc[mask_train]
s_train = s_scores.loc[mask_train]

# test slices (final backtest)
rs_test = returns_stock.loc[mask_test]
rf_test = returns_factor.loc[mask_test]
beta_test = beta_df.loc[mask_test]
s_test = s_scores.loc[mask_test]

# ---------- THRESHOLD TUNING ----------
if use_auto_tuning and len(rs_train) > 50:
    with st.spinner("Auto-tuning thresholds on train period (Jan–Sep 2024)..."):
        best_params, best_sharpe, best_cagr = auto_tune_thresholds(
            rs_train,
            rf_train,
            beta_train,
            s_train,
            effective_map,
            initial_equity,
            leverage,
            tc_bps,
        )

    if best_params is not None:
        entry_level, exit_long_level, exit_short_level = best_params
        st.success(
            f"Auto-tuned thresholds on train set:\n"
            f"- Entry |s| ≥ **{entry_level:.2f}**\n"
            f"- Exit long when s ≥ **{exit_long_level:.2f}**\n"
            f"- Exit short when s ≤ **{exit_short_level:.2f}**\n"
            f"Train Sharpe ≈ **{best_sharpe:.2f}**, Train CAGR ≈ **{best_cagr*100:.2f}%**"
        )
    else:
        st.warning("Auto-tuning failed to find good parameters. Using manual thresholds.")
else:
    if not use_auto_tuning:
        st.info("Using manual thresholds from the sidebar (no auto-tuning).")
    else:
        st.warning("Not enough train data to auto-tune. Using manual thresholds.")

# ---------- BACKTEST: TRAIN & TEST ----------
with st.spinner("Running stat-arb backtest on train & test periods..."):
    # Train (for info, using tuned thresholds)
    equity_train, pnl_train, trades_train, trades_df_train = backtest_stat_arb(
        rs_train,
        rf_train,
        beta_train,
        s_train,
        effective_map,
        entry_level=entry_level,
        exit_long=exit_long_level,
        exit_short=exit_short_level,
        initial_equity=initial_equity,
        leverage=leverage,
        tc_bps=tc_bps,
    )

    # Test (main performance)
    equity_test, pnl_test, trades_test, trades_df_test = backtest_stat_arb(
        rs_test,
        rf_test,
        beta_test,
        s_test,
        effective_map,
        entry_level=entry_level,
        exit_long=exit_long_level,
        exit_short=exit_short_level,
        initial_equity=initial_equity,
        leverage=leverage,
        tc_bps=tc_bps,
    )

if equity_test.empty:
    st.warning("Test equity series is empty – likely no trades placed. Try relaxing thresholds.")
    st.stop()

perf_train = compute_performance_metrics(equity_train) if not equity_train.empty else {
    "CAGR": np.nan, "Sharpe": np.nan, "Vol": np.nan, "MaxDD": np.nan
}
perf_test = compute_performance_metrics(equity_test)

# =========================================================
# 8. KPI CARDS
# =========================================================

st.subheader("📌 Out-of-Sample Test Performance (Oct–Dec 2024)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final Equity (Test)", f"${equity_test.iloc[-1]:,.0f}")
with col2:
    st.metric("CAGR (ann., Test)", format_pct(perf_test["CAGR"]))
with col3:
    st.metric("Sharpe (ann., Test)", f"{perf_test['Sharpe']:.2f}" if not pd.isna(perf_test["Sharpe"]) else "–")
with col4:
    st.metric("Max Drawdown (Test)", format_pct(perf_test["MaxDD"]))

st.markdown("---")

st.subheader("🧪 In-Sample Train Performance (Jan–Sep 2024, for info)")
colt1, colt2, colt3, colt4 = st.columns(4)
with colt1:
    if not equity_train.empty:
        st.metric("Final Equity (Train)", f"${equity_train.iloc[-1]:,.0f}")
    else:
        st.metric("Final Equity (Train)", "–")
with colt2:
    st.metric("CAGR (ann., Train)", format_pct(perf_train["CAGR"]))
with colt3:
    st.metric("Sharpe (ann., Train)", f"{perf_train['Sharpe']:.2f}" if not pd.isna(perf_train["Sharpe"]) else "–")
with colt4:
    st.metric("Max Drawdown (Train)", format_pct(perf_train["MaxDD"]))

st.markdown("---")

# =========================================================
# 9. TABS – PERFORMANCE, TRADES, STOCK EXPLORER
# =========================================================

tab_perf, tab_trades, tab_stock = st.tabs(
    ["📈 Performance", "🧾 Trades & Turnover", "🔍 Stock Explorer"]
)

# ----- Performance Tab -----
with tab_perf:
    st.subheader("Equity Curve & Drawdown – Test Period (Oct–Dec 2024)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_test.index,
        y=equity_test.values,
        mode="lines",
        name="Equity (Test)",
    ))

    roll_max = equity_test.cummax()
    dd = (equity_test / roll_max - 1.0) * 100
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name="Drawdown (Test, %)",
        yaxis="y2"
    ))

    fig.update_layout(
        xaxis_title="Date (2024)",
        yaxis=dict(title="Equity"),
        yaxis2=dict(
            title="Drawdown (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of Daily Returns – Test Period")
    rets_daily_test = equity_test.pct_change().dropna() * 100
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=rets_daily_test, nbinsx=40, name="Daily returns (%)"))
    hist_fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(hist_fig, use_container_width=True)

# ----- Trades Tab -----
with tab_trades:
    st.subheader("Trade Count Over Time – Test Period")
    trades_series_test = trades_test.reindex(equity_test.index).fillna(0)
    fig_trades = go.Figure()
    fig_trades.add_trace(go.Bar(
        x=trades_series_test.index,
        y=trades_series_test.values,
        name="# Trades (Test)"
    ))
    fig_trades.update_layout(
        xaxis_title="Date (2024)",
        yaxis_title="Number of Trades",
        height=400
    )
    st.plotly_chart(fig_trades, use_container_width=True)

    st.subheader("Trade Log – Test Period (most recent first)")
    if trades_df_test.empty:
        st.info("No trades executed in the test period.")
    else:
        trades_df_sorted = trades_df_test.sort_values("date", ascending=False)
        st.dataframe(trades_df_sorted, use_container_width=True, height=400)

# ----- Stock Explorer Tab -----
with tab_stock:
    st.subheader("Per-Stock s-score & Price Explorer (Full 2024)")

    all_tickers = sorted(returns_stock.columns.tolist())
    sel_ticker = st.selectbox("Select a stock", all_tickers)

    s_ticker = s_scores[sel_ticker].dropna()
    if s_ticker.empty:
        st.warning("No s-score history for this ticker – it may have failed OU criteria.")
    else:
        st.markdown(f"**s-score history for {sel_ticker} (2024)**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=s_ticker.index,
            y=s_ticker.values,
            mode="lines",
            name="s-score"
        ))
        fig_s.add_hline(y=entry_level, line_dash="dash", annotation_text="Short entry")
        fig_s.add_hline(y=-entry_level, line_dash="dash", annotation_text="Long entry")
        fig_s.update_layout(
            xaxis_title="Date (2024)",
            yaxis_title="s-score",
            height=400
        )
        st.plotly_chart(fig_s, use_container_width=True)

    etf = effective_map.get(sel_ticker)
    if etf in prices.columns:
        st.markdown(f"**Normalized Price: {sel_ticker} vs {etf} (2024)**")
        df_price_pair = prices[[sel_ticker, etf]].dropna()
        norm = df_price_pair / df_price_pair.iloc[0]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=norm.index, y=norm[sel_ticker], mode="lines", name=sel_ticker
        ))
        fig_p.add_trace(go.Scatter(
            x=norm.index, y=norm[etf], mode="lines", name=etf
        ))
        fig_p.update_layout(
            xaxis_title="Date (2024)",
            yaxis_title="Normalized Price",
            height=400
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("**Test-Period Trades for Selected Ticker (Oct–Dec 2024)**")
    if not trades_df_test.empty:
        td = trades_df_test[trades_df_test["ticker"] == sel_ticker]
        if td.empty:
            st.info("No trades for this ticker in the test period.")
        else:
            st.dataframe(
                td.sort_values("date", ascending=False),
                use_container_width=True,
                height=300
            )
    else:
        st.info("No trades in the test period.")
