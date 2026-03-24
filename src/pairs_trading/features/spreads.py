from __future__ import annotations

import pandas as pd


def build_spread(x: pd.Series, y: pd.Series, beta: float = 1.0) -> pd.Series:
    """
    Build a spread from two aligned price series.

    Spread definition:
        spread = y - beta * x

    Args:
        x: Left asset price series.
        y: Right asset price series.
        beta: Hedge ratio.

    Returns:
        Spread series indexed like the aligned input.
    """
    if x is None or y is None:
        raise ValueError("x and y must not be None")

    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No overlapping data between x and y")

    x1 = pd.to_numeric(aligned.iloc[:, 0], errors="coerce")
    y1 = pd.to_numeric(aligned.iloc[:, 1], errors="coerce")
    spread = y1 - float(beta) * x1
    spread.name = "spread"
    return spread.dropna()


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling z-score of a series.

    Z_t = (X_t - rolling_mean) / rolling_std
    """
    if series is None:
        raise ValueError("series must not be None")
    if window <= 1:
        raise ValueError("window must be greater than 1")

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("series is empty after cleaning")

    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=0)

    z = (s - mean) / std
    z = z.replace([float("inf"), float("-inf")], pd.NA)
    z.name = f"zscore_{window}"
    return z