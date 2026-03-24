from __future__ import annotations

import pandas as pd

from pairs_trading.constants import (
    DEFAULT_COINTEGRATION_ALPHA,
    DEFAULT_HURST_THRESHOLD,
    DEFAULT_MAX_HALF_LIFE_DAYS,
    DEFAULT_MIN_HALF_LIFE_DAYS,
    DEFAULT_MIN_MEAN_CROSSES_PER_YEAR,
)
from .statistics import (
    cointegration_pvalue,
    estimate_spread_beta,
    half_life,
    hurst_exponent,
    mean_crosses_per_year,
)


def passes_pair_filters(x: pd.Series, y: pd.Series) -> bool:
    """
    Check whether a pair passes all selection filters.
    """
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 20:
        return False

    x1 = aligned.iloc[:, 0]
    y1 = aligned.iloc[:, 1]

    pval = cointegration_pvalue(x1, y1)
    if pval >= DEFAULT_COINTEGRATION_ALPHA:
        return False

    beta = estimate_spread_beta(x1, y1)
    spread = y1 - beta * x1

    h = hurst_exponent(spread)
    if h >= DEFAULT_HURST_THRESHOLD:
        return False

    hl = half_life(spread)
    if not (DEFAULT_MIN_HALF_LIFE_DAYS <= hl <= DEFAULT_MAX_HALF_LIFE_DAYS):
        return False

    crosses = mean_crosses_per_year(spread)
    if crosses <= DEFAULT_MIN_MEAN_CROSSES_PER_YEAR:
        return False

    return True


def pair_quality_report(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """
    Return a dictionary of quality statistics for a pair.
    """
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 20:
        return {
            "cointegration_pvalue": 1.0,
            "beta": float("nan"),
            "hurst": float("nan"),
            "half_life": float("nan"),
            "mean_crosses_per_year": 0.0,
        }

    x1 = aligned.iloc[:, 0]
    y1 = aligned.iloc[:, 1]

    beta = estimate_spread_beta(x1, y1)
    spread = y1 - beta * x1

    return {
        "cointegration_pvalue": cointegration_pvalue(x1, y1),
        "beta": float(beta),
        "hurst": float(hurst_exponent(spread)),
        "half_life": float(half_life(spread)),
        "mean_crosses_per_year": float(mean_crosses_per_year(spread)),
    }