# Pairs Trading Research Project

This repository contains a full research pipeline for building and testing a **pairs trading** system. The implementation is organized so that a reader can:

1. download or cache market data,
2. build a tradable stock universe,
3. clean and align historical prices,
4. select candidate pairs using clustering and statistical filters,
5. test multiple trading strategies,
6. backtest results with transaction costs,
7. compare strategy performance against **SPY**,
8. save tables and charts for analysis.

The project is designed as a reproducible research repo rather than a production trading system. It is intentionally modular so that every stage can be inspected, modified, and rerun independently.

---

## Table of contents

- [Project overview](#project-overview)
- [Pipeline summary](#pipeline-summary)
- [Repository structure](#repository-structure)
- [Core methodology](#core-methodology)
- [Data sources](#data-sources)
- [Setup](#setup)
- [Installing the package](#installing-the-package)
- [Running the notebook workflow](#running-the-notebook-workflow)
- [Running the command-line interface](#running-the-command-line-interface)
- [Caching behavior](#caching-behavior)
- [Charts and outputs](#charts-and-outputs)
- [How to benchmark against SPY](#how-to-benchmark-against-spy)
- [Implementation notes](#implementation-notes)
- [Troubleshooting](#troubleshooting)
- [What still needs improvement](#what-still-needs-improvement)
- [Suggested next steps](#suggested-next-steps)

---

## Project overview

Pairs trading is a market-neutral strategy that looks for two assets whose prices usually move together. When one asset becomes unusually expensive relative to the other, the strategy opens a long/short position expecting the relationship to revert.

This repository implements a workflow for the pairs-trading algorithm development. The workflow uses:

- **daily prices** for pair selection,
- **clustering** to find similar assets,
- **cointegration / Hurst / half-life / mean-cross filters** to validate pairs,
- **Bollinger**, **OU**, **Copula**, and **Cointegration** trading strategies,
- a backtest engine that uses the **hedge ratio** for sizing both legs,
- optional benchmarking against **SPY**.

The project is currently organized as a research scaffold with working implementations for the main data, selection, strategy, and backtest components.

---

## Pipeline summary

The end-to-end flow is:

1. **Load the stock universe**
   - Read S&P 500 constituents from cache or a CSV snapshot.

2. **Download historical OHLCV data**
   - Download from Stooq in the current implementation.
   - Save cached raw price files in `data/raw/`.

3. **Clean and align prices**
   - Standardize dates, column names, and numeric fields.
   - Combine individual assets into one aligned price matrix.

4. **Compute returns and cluster assets**
   - Returns are computed from daily close prices.
   - PCA reduces the dimensionality.
   - OPTICS groups similar assets into clusters.

5. **Generate candidate pairs**
   - All pairs inside each cluster are considered.

6. **Filter pairs statistically**
   - Cointegration p-value
   - Hurst exponent
   - Half-life
   - Mean-cross count per year

7. **Fit strategies**
   - Bollinger-band strategy
   - Ornstein-Uhlenbeck forecast strategy
   - Copula-based strategy
   - Cointegration strategy

8. **Backtest strategies**
   - Use hedge-ratio-based leg sizing.
   - Include transaction costs and slippage.
   - Track equity curve and returns.

9. **Benchmark against SPY**
   - Compare strategy performance against a buy-and-hold SPY equity curve.

10. **Save charts and tables**
    - Strategy metrics
    - Equity curves
    - Drawdowns
    - Excess return versus SPY

---

## Repository structure

A typical layout looks like this:

```text
BTP2 - Pairs Trading/
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── universe/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── tables/
├── scripts/
├── src/
│   └── pairs_trading/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       │   ├── universe.py
│       │   ├── stooq_loader.py
│       │   ├── alphavantage_loader.py
│       │   ├── yahoo_loader.py
│       │   └── cleaning.py
│       ├── features/
│       │   ├── returns.py
│       │   ├── scaling.py
│       │   ├── dimensionality.py
│       │   └── spreads.py
│       ├── clustering/
│       │   └── optics_cluster.py
│       ├── selection/
│       │   ├── statistics.py
│       │   ├── filters.py
│       │   └── pair_selector.py
│       ├── strategies/
│       │   ├── base.py
│       │   ├── bollinger.py
│       │   ├── ou.py
│       │   ├── copula.py
│       │   └── cointegration.py
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── portfolio.py
│       │   ├── execution.py
│       │   ├── costs.py
│       │   └── metrics.py
│       └── risk/
│           ├── stops.py
│           ├── refresh.py
│           └── exposure.py
└── tests/
```

---

## Core methodology

### Universe selection

The universe loader returns S&P 500 constituents. The current setup uses a cached CSV snapshot if available, and only falls back to a web scrape if the file is missing. This keeps the project reproducible.

### Data download

The primary free data source in the current implementation is **Stooq**. The loader is written to download OHLCV bars and save them locally so the data does not need to be downloaded repeatedly.

### Pair selection logic

The selection pipeline is deliberately split into two stages:

#### 1) Clustering
The idea is to group assets that look similar over time before testing pairs individually. The pipeline:

- computes daily returns,
- scales features,
- runs PCA until the target explained variance is reached,
- clusters the assets using OPTICS.

#### 2) Statistical filtering
Within each cluster, every candidate pair is checked using:

- cointegration p-value,
- Hurst exponent,
- estimated half-life,
- mean-cross count per year.

Only pairs that pass the filters move on to the strategy stage.

### Trading strategies

The repo includes four strategy modules:

- **BollingerStrategy**: mean-reversion on a rolling z-score.
- **OUForecastStrategy**: Ornstein-Uhlenbeck style forecasting with confidence bands.
- **CopulaStrategy**: rank-based conditional-probability logic.
- **CointegrationStrategy**: spread and z-score style cointegration signal.

### Backtesting

The backtest engine does not treat the spread as a directly tradable asset. Instead, it:

- interprets a strategy signal as long / short / flat,
- uses the hedge ratio to size both legs,
- computes PnL from the two asset legs,
- charges transaction costs and slippage,
- creates an equity curve.

This makes the backtest more realistic than a toy spread-only implementation.

### SPY benchmark

To compare results against the market, the notebook workflow can download or reuse SPY prices, turn them into a buy-and-hold equity curve, and compare:

- cumulative return,
- drawdown,
- Sharpe ratio,
- excess return over time.

---

## Data sources

### Current default source

- **Stooq** for OHLCV downloads.

### Optional / fallback sources

- **Alpha Vantage**
- **Yahoo Finance / yfinance**

### Important note on intraday data

The first working version of this repository is built around **daily data** because it is the most reliable free data path for repeated experiments. Intraday support can be added later if you have a stable source and enough historical coverage.

---

## Setup

### 1) Clone or copy the repo

Place the repository anywhere on your machine. The folder name can include spaces, but a simpler folder name makes command-line work easier.

### 2) Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

If you use the included `pyproject.toml`, install the project in editable mode:

```bash
pip install -e .
```

If you run into issues, install the basic libraries manually:

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib requests pyarrow hurst lxml
```

### 4) Confirm the package imports

From a Python shell or notebook:

```python
from pairs_trading.data.universe import get_sp500_constituents
from pairs_trading.data.stooq_loader import fetch_stooq_ohlcv
from pairs_trading.selection.pair_selector import PairSelector
```

---

## Installing the package

The repository uses a `src/` layout. That means the package lives under:

```text
src/pairs_trading/
```

To make imports work from notebooks and scripts, run:

```bash
pip install -e .
```

This links the source tree into the active environment so changes in the `.py` files are reflected immediately after a kernel restart.

---

## Running the notebook workflow

The notebook workflow is the easiest way to reproduce the full pipeline.

### Recommended notebook order

1. **Universe and data download notebook**
2. **Pair selection notebook**
3. **Strategy and backtest notebook**
4. **Benchmark and plotting notebook**

### Important notebook rule

Always define the project root first and use it consistently for all file paths. Do **not** use `Path.cwd()` blindly inside a notebook unless you are sure the notebook is launched from the repository root.

A safe pattern is:

```python
from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root")

PROJECT_ROOT = find_project_root()
```

Then build all other paths from `PROJECT_ROOT`:

```python
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_DIR = DATA_DIR / "universe"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
```

### Recommended notebook flow

#### Step 1: Load the universe

```python
from pairs_trading.data.universe import get_sp500_constituents
universe = get_sp500_constituents("2023-03-31")
```

#### Step 2: Download or reuse price data

Use a helper that checks local files first before downloading again. The repository is designed so the same raw CSV is reused when it already exists.

#### Step 3: Build the processed price matrix

Use `align_prices()` to combine raw files into one wide DataFrame of close prices.

#### Step 4: Select pairs

```python
from pairs_trading.selection.pair_selector import PairSelector
selector = PairSelector(target_variance=0.8, min_samples=3)
selector.fit(prices)
selected_pairs = selector.select_pairs()
```

#### Step 5: Fit strategies and backtest

Pick one selected pair, then run each strategy through the backtest engine.

#### Step 6: Compare against SPY

Download or reuse SPY, compute the benchmark equity curve, and plot the comparison.

#### Step 7: Save charts and tables

Use the `outputs/figures/` and `outputs/tables/` folders.

---

## Running the command-line interface

The repository includes a CLI entrypoint in `src/pairs_trading/cli.py`.

### Show configuration

```bash
python -m pairs_trading.cli show-config
```

### Create directories

```bash
python -m pairs_trading.cli init-dirs
```

### Download the universe snapshot

```bash
python -m pairs_trading.cli download-universe --as-of 2023-03-31 --out data/universe/sp500_constituents.csv
```

### Download raw prices

```bash
python -m pairs_trading.cli download-prices --universe-csv data/universe/sp500_constituents.csv --start 2022-03-01 --end 2023-03-31 --interval 1d --max-symbols 100 --out-dir data/raw
```

### Build the processed dataset

```bash
python -m pairs_trading.cli build-dataset --raw-dir data/raw --out data/processed/prices.parquet
```

### Select candidate pairs

```bash
python -m pairs_trading.cli select-pairs --prices data/processed/prices.parquet --pca-variance 0.8 --min-samples 3 --out data/processed/selected_pairs.csv
```

---

## Caching behavior

The repository is designed to avoid redownloading data repeatedly.

### Universe cache

If a universe CSV exists in `data/universe/`, it is used first.

### Raw price cache

Each symbol is stored as its own CSV under `data/raw/`. If a file already exists, the notebook or helper can reuse it instead of downloading again.

### Processed price cache

The aligned wide price matrix can be saved in `data/processed/prices.parquet`. If that file exists and you do not force a rebuild, the notebook can load it directly.

### Forcing a rebuild

When you want to change the number of symbols, date range, or source behavior, delete the processed file or use a `force_rebuild=True` option in your notebook helper.

---

## Charts and outputs

The notebook workflow should save outputs in these folders:

```text
outputs/figures/
outputs/tables/
```

### Suggested charts

#### 1) Equity curve comparison
Plots normalized equity curves for all strategies.

#### 2) Drawdown comparison
Shows how each strategy performs during downturns.

#### 3) Spread and signal chart
Plots the spread and trading signal for the best strategy on the selected pair.

#### 4) Cumulative return bar chart
Compares cumulative returns across strategies.

#### 5) Strategy comparison against SPY
Shows strategy equity, drawdown, and excess return relative to the benchmark.

### Suggested tables

#### 1) Selected pairs
Contains the selected asset pairs and their quality metrics.

#### 2) Strategy metrics
Contains final equity, cumulative return, annualized return, volatility, max drawdown, Sharpe ratio, and trade count.

#### 3) Strategy metrics with SPY
Includes the benchmark row so the strategies can be compared side by side.

---

## How to benchmark against SPY

The recommended comparison uses SPY as a buy-and-hold benchmark.

### Steps

1. Download SPY using the same data loader helper you use for other tickers.
2. Extract the close series.
3. Reindex SPY to the same date index as your pair strategy results.
4. Compute a buy-and-hold equity curve starting from the same capital.
5. Add the SPY metrics row to the summary table.
6. Plot strategy equity curves against SPY.
7. Plot strategy drawdowns against SPY.
8. Plot excess returns over SPY.

### Why SPY is useful

SPY gives you a broad-market benchmark so you can see whether the strategy is doing anything more useful than passive long exposure to the market.

---

## Implementation notes

### 1) Spread is not the traded asset

The spread is a signal derived from the two price series. The actual trade is executed in the two underlying legs.

### 2) Hedge ratio matters

If the spread is defined as:

```python
spread = right - beta * left
```

then a long-spread position should be sized as:

- long `right`,
- short `beta * left`.

The backtest engine should therefore use the hedge ratio for sizing.

### 3) The backtest engine is pair-aware

The corrected backtest engine computes positions using both legs rather than treating the spread itself as a tradable asset.

### 4) Cache invalidation matters

If you edit code and the notebook seems to use old behavior, restart the notebook kernel. Python caches imported modules.

### 5) Empty pair selection usually means one filter is too strict

If `selected_pairs.csv` is empty, the likely cause is not the download step. It usually means one of the statistical filters is removing every candidate pair.

---

## Troubleshooting

### Problem: `pip install -e .` fails

Check that `pyproject.toml` is placed in the project root, not inside `src/` or `notebooks/`.

### Problem: notebook creates a nested `notebooks/data/` folder

Use a root-finding helper and set `PROJECT_ROOT` from the repository root rather than from the notebook directory.

### Problem: `ImportError: lxml failed`

Install the `lxml` package:

```bash
pip install lxml
```

### Problem: `selected_pairs.csv` is empty

Possible causes:
- filters too strict,
- not enough overlapping data,
- old processed cache still in use,
- statistics function bug,
- notebook kernel is still using an old imported module.

### Problem: strategies produce no trades

Check the signal logic, the z-score thresholds, and the date coverage of the selected pair.

---

## What still needs improvement

The current implementation is a very good research scaffold, but it can still be improved.

### Good future upgrades

- better OU calibration,
- more rigorous copula fitting,
- dynamic hedge ratio estimation,
- realistic commission and slippage models,
- intraday support,
- walk-forward evaluation,
- portfolio-level allocation across multiple pairs,
- more complete reporting charts,
- proper experiment tracking.

### Research quality upgrades

- out-of-sample validation,
- rolling parameter estimation,
- sector-controlled universe selection,
- volatility-adjusted position sizing,
- pair refresh logic tied to realized performance,
- portfolio risk limits.

---

## Suggested next steps

If you want to extend the project further, the best order is:

1. improve the statistical filters,
2. make hedge ratio estimation rolling,
3. improve the OU strategy,
4. add richer benchmark reporting,
5. build a multi-pair portfolio backtester,
6. add experiment logging and parameter sweeps.