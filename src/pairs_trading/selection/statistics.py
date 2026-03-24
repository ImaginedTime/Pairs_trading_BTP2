from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint


def cointegration_pvalue(x: pd.Series, y: pd.Series) -> float:
    """
    Return the Engle-Granger cointegration p-value.
    """
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must be non-empty")

    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0

    x1 = aligned.iloc[:, 0].astype(float)
    y1 = aligned.iloc[:, 1].astype(float)

    try:
        _, pvalue, _ = coint(y1, x1)
        return float(pvalue)
    except Exception:
        return 1.0


def estimate_spread_beta(x: pd.Series, y: pd.Series) -> float:
    """
    Estimate hedge ratio beta from y ~ alpha + beta * x.

    Returns beta.
    """
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must be non-empty")

    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("Not enough data to estimate beta")

    x1 = aligned.iloc[:, 0].astype(float)
    y1 = aligned.iloc[:, 1].astype(float)

    X = add_constant(x1.values)
    model = OLS(y1.values, X).fit()
    return float(model.params[1])


def hurst_exponent(series: pd.Series) -> float:
    """
    Estimate the Hurst exponent using the log-log variance method.
    """
    if series is None or len(series) < 20:
        raise ValueError("series must contain at least 20 observations")

    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 20:
        raise ValueError("series must contain at least 20 valid observations")

    lags = range(2, min(20, len(s) // 2))
    tau = []
    for lag in lags:
        diff = s.diff(lag).dropna()
        tau.append(np.sqrt(np.std(diff)))

    if len(tau) < 2:
        return 0.5

    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    hurst = 2.0 * poly[0]
    return float(hurst)


def half_life(series: pd.Series) -> float:
    """
    Estimate the mean reversion half-life from an AR(1) model on the series.
    """
    if series is None or len(series) < 20:
        raise ValueError("series must contain at least 20 observations")

    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 20:
        raise ValueError("series must contain at least 20 valid observations")

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()

    aligned = pd.concat([delta, lagged], axis=1).dropna()
    aligned.columns = ["delta", "lagged"]

    X = add_constant(aligned["lagged"].values)
    model = OLS(aligned["delta"].values, X).fit()
    beta = model.params[1]

    if beta >= 0:
        return float("inf")

    hl = -np.log(2) / beta
    return float(hl)


def mean_crosses_per_year(series: pd.Series, freq: str = "D") -> float:
    """
    Count how often the series crosses its mean, scaled to a yearly rate.
    """
    if series is None or len(series) < 2:
        raise ValueError("series must contain at least 2 observations")

    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 2:
        raise ValueError("series must contain at least 2 valid observations")

    mean_val = s.mean()
    above = s > mean_val
    crosses = (above.astype(int).diff().abs() == 2).sum()

    # Convert observed crossings to annual rate.
    n = len(s)
    if freq.lower() in {"d", "1d", "day", "daily"}:
        periods_per_year = 252
    elif freq.lower() in {"h", "1h", "hour", "hourly"}:
        periods_per_year = 252 * 6.5
    else:
        periods_per_year = 252

    return float(crosses / n * periods_per_year)