# Pairs Trading FLXAI

This repository is a scaffold for a pairs-trading research pipeline.

## Pipeline overview

1. Build the stock universe
2. Download price data
3. Clean and align data
4. Compute returns and spreads
5. Reduce dimensions with PCA
6. Cluster assets with OPTICS
7. Select pairs using statistical filters
8. Fit trading strategies
9. Backtest and evaluate

## Generated structure

- `data/` raw and processed datasets
- `notebooks/` exploratory notebooks
- `src/pairs_trading/` Python package
- `scripts/` command-line entry scripts
- `tests/` unit tests

## How to use

### 1. Create the scaffold

```bash
bash pairs_trading_scaffold.sh
```

You can also pass a custom folder name:

```bash
bash pairs_trading_scaffold.sh my_repo
```

### 2. Create a virtual environment

```bash
cd pairs-trading-flxai
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install pandas numpy scipy statsmodels scikit-learn matplotlib requests pyarrow typer pytest hurst
```

### 4. Implement in this order

Start with:

- `src/pairs_trading/data/universe.py`
- `src/pairs_trading/data/stooq_loader.py`
- `src/pairs_trading/data/cleaning.py`

Then continue with:

- `src/pairs_trading/features/returns.py`
- `src/pairs_trading/features/scaling.py`
- `src/pairs_trading/features/dimensionality.py`
- `src/pairs_trading/clustering/optics_cluster.py`
- `src/pairs_trading/selection/statistics.py`
- `src/pairs_trading/selection/filters.py`
- `src/pairs_trading/selection/pair_selector.py`

Then implement strategies:

- `src/pairs_trading/strategies/base.py`
- `src/pairs_trading/strategies/bollinger.py`
- `src/pairs_trading/strategies/ou.py`
- `src/pairs_trading/strategies/copula.py`
- `src/pairs_trading/strategies/cointegration.py`

Then implement backtesting:

- `src/pairs_trading/backtest/engine.py`
- `src/pairs_trading/backtest/portfolio.py`
- `src/pairs_trading/backtest/execution.py`
- `src/pairs_trading/backtest/costs.py`
- `src/pairs_trading/backtest/metrics.py`

Then add risk controls:

- `src/pairs_trading/risk/stops.py`
- `src/pairs_trading/risk/refresh.py`
- `src/pairs_trading/risk/exposure.py`

## Expected outputs

When implemented, the repo should produce:

- downloaded price data
- cleaned datasets
- selected pairs
- trading signals
- backtest results
- summary metrics
