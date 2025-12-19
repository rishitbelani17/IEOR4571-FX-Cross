# FX Exponential Smoothing Dashboard

Streamlit workspace for researching and stress-testing a dual-exponential-smoothing FX trading strategy. The app ingests Yahoo Finance prices, applies configurable α/β smoothing, produces trade signals, backtests with transaction costs, and offers parameter optimization plus ML-based signal experiments.

## Collaborators

- Tanish Patel <tp2899@columbia.edu>
- Aditya Pendyala <ap4839@columbia.edu>
- Rishit Belani <rb3881@columbia.edu>
- Arnav Ahuja <aa5790@columbia.edu>
- Aryamaan Srivastava <as7845@columbia.edu>

---

## Table of Contents
- [Concept in Brief](#concept-in-brief)
- [Key Features](#key-features)
- [Screens / Pages](#screens--pages)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Strategy Logic](#strategy-logic)
- [Backtest & Metrics](#backtest--metrics)
- [Presets](#presets)
- [Data & Symbols](#data--symbols)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)

---

## Concept in Brief
- Two exponential smoothers: **ES(α)** (slow, trend) and **ES(β)** (fast, momentum).
- Signal driver: **spread = ES(β) – ES(α)**. Crosses and thresholds generate longs/shorts; optional **deceleration exit** flattens on momentum loss.
- Simple enough for intuition, flexible enough to adapt across assets/timeframes.

## Key Features
- Dual-smoothing with tunable **α (slow)** and **β (fast)**, threshold **x**, and **θ** deceleration exit logic.
- Rich visuals: KPI cards, equity and drawdown curves, price with trade markers, return distributions, ES overlays.
- Trade log with filters and CSV export; turnover-aware transaction costs.
- Preset manager to save/load parameter sets across sessions.
- Grid-search page for **α/β/x/θ** surfaces with best-config selection.
- ML experiments page that fuses ES signals with a Random Forest classifier, feature importance, and optional SHAP interpretability.
- Works with major FX pairs by default; accepts any Yahoo Finance symbol (FX, crypto, metals, indices).
- Quick live tick pull to sanity check the latest price.

## Screens / Pages
- `fx-cross.py` — **Main Dashboard**: parameter inputs, live pull, backtest, KPIs, charts, trade log, presets.
- `pages/01_About_Strategy_Notes.py` — **Strategy Overview**: quick description and contact.
- `pages/02_Alpha_Beta_Optimization.py` — **Grid Search**: sweep α/β with x/θ combinations, view heatmaps, and push best settings.
- `pages/03_ML_FX_Signals.py` — **ML Hybrid Signals**: Random Forest on engineered features + ES signals, performance diagnostics, SHAP (if installed).

## Project Structure
- `fx-cross.py` — main Streamlit app (dashboard/backtest/presets).
- `pages/01_About_Strategy_Notes.py` — overview page.
- `pages/02_Alpha_Beta_Optimization.py` — grid-search UI and plots.
- `pages/03_ML_FX_Signals.py` — ML hybrid signals and explainability.
- `utils/presets.py` / `utils/presets.json` — preset I/O helpers and stored presets.
- `requirements.txt` — Python dependencies.

## Setup
1) Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) (Optional) Install SHAP if you want explainability plots on the ML page and it fails to import automatically:
```bash
pip install shap
```

## Running the App
Start the main dashboard:
```bash
streamlit run fx-cross.py
```
Streamlit will open in your browser. Use the left sidebar to switch pages or adjust parameters.

## Strategy Logic
- **Smoothing:**  
  - ES(α) = slow smoother; ES(β) = fast smoother.  
  - Parameters α, β ∈ (0,1); β > α typically.
- **Signal generation (`generate_signals`):**  
  - Long entry when spread crosses above +x from negative.  
  - Short entry when spread crosses below –x from positive.  
  - Optional deceleration exit: if Δspread < –θ, flatten position.
  - Signals are converted to **positions** via forward-fill of non-zero signals.
- **Backtest (`backtest`):**  
  - Position-timed returns: `positions.shift(1) * returns`.  
  - Transaction costs applied to turnover: `positions.diff().abs() * cost`.  
  - Equity curve: cumulative product of (1 + strategy returns) scaled by starting capital.

## Backtest & Metrics
Computed in `backtest` and surfaced in KPIs:
- Final Equity and Net Profit
- Total Return % and Annualized Return
- Volatility (annualized) and Sharpe Ratio
- Max Drawdown (absolute)
- Hit Rate (daily direction agreement with position sign)
- Number of Trades (unique entry signals)
- Trade log with Entry/Exit time, price, direction, PnL, Return %

## Presets
- Stored locally in `utils/presets.json`; saved/loaded via the sidebar preset manager.
- A preset captures: symbol, timeframe, α, β, x, exit_rule, θ, cost, start/end dates.
- Safe to version-control but be mindful of personal preferences or experimental settings.

## Data & Symbols
- Data source: Yahoo Finance via `yfinance`.
- Supported timeframes:  
  - Main app: 1H, 4H, Daily, Weekly.  
  - ML page: adds 1m, 15m, Monthly, Quarterly (subject to Yahoo data availability).
- FX examples: `EURJPY=X`, `USDJPY=X`, `EURUSD=X`, `GBPUSD=X`, `AUDUSD=X`, `USDCAD=X`, `USDCHF=X`, `NZDUSD=X`.
- Other assets: metals (`XAUUSD=X`), crypto (`BTC-USD`), indices/ETFs (any valid Yahoo symbol).
- Network access is required to fetch historical and live prices.

## Troubleshooting
- **No data returned / empty charts:**  
  - Check symbol spelling and timeframe support on Yahoo Finance.  
  - Ensure start/end dates span at least a few bars.  
  - Confirm network access.
- **SHAP errors on ML page:** install `shap` (see Setup step 3) or toggle off SHAP features.
- **Beta slider disabled at low values:** β lower bound is set to α + 0.01 to keep the “fast” smoother faster than the “slow” one.
- **Preset not saving/loading:** verify write access to `utils/presets.json`.

## Roadmap Ideas
- Add walk-forward or rolling-window evaluation.
- Include position sizing and risk overlays (e.g., volatility scaling).
- Add portfolio view to combine multiple symbols.
- Expand ML page with alternative models and richer feature sets.
- Expose API endpoints for programmatic backtests.
