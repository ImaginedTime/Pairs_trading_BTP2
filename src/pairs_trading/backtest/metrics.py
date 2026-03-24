from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_return(equity_curve: pd.Series) -> float:
    """
    Compute total return from an equity curve.
    """
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0

    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized return from an equity curve.
    """
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0

    total_return = s.iloc[-1] / s.iloc[0]
    n_periods = len(s) - 1
    if n_periods <= 0:
        return 0.0

    return float(total_return ** (periods_per_year / n_periods) - 1.0)


def volatility(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized volatility from equity curve returns.
    """
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0

    rets = s.pct_change().dropna()
    if rets.empty:
        return 0.0

    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown.
    """
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0

    running_max = s.cummax()
    drawdowns = s / running_max - 1.0
    return float(drawdowns.min())


def sharpe_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute Sharpe ratio using equity curve returns.
    """
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0

    rets = s.pct_change().dropna()
    if rets.empty:
        return 0.0

    std = rets.std(ddof=1)
    if std == 0 or pd.isna(std):
        return 0.0

    return float((rets.mean() / std) * np.sqrt(periods_per_year))


def trade_count(trades: pd.DataFrame) -> int:
    """
    Count trades from a trade log DataFrame.
    """
    if trades is None or trades.empty:
        return 0

    if "event" in trades.columns:
        return int(len(trades))

    if "position_change" in trades.columns:
        return int((trades["position_change"].fillna(0) > 0).sum())

    return int(len(trades))