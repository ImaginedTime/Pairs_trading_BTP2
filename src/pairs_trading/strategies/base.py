from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import pandas as pd


def extract_pair_series(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Extract two aligned price series from a DataFrame.

    Accepts:
    - a DataFrame with exactly two columns
    - a DataFrame with any two numeric columns

    Returns:
        (left_series, right_series)
    """
    if data is None or data.empty:
        raise ValueError("pair data is empty")

    work = data.copy().sort_index()
    numeric = work.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        raise ValueError("pair data must contain at least two numeric columns")

    left = numeric.iloc[:, 0].dropna()
    right = numeric.iloc[:, 1].dropna()

    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.shape[1] < 2 or len(aligned) < 2:
        raise ValueError("could not align the pair series")

    s1 = aligned.iloc[:, 0].astype(float)
    s2 = aligned.iloc[:, 1].astype(float)
    s1.name = left.name or "asset_left"
    s2.name = right.name or "asset_right"
    return s1, s2


@dataclass
class StrategyState:
    pair_left: str | None = None
    pair_right: str | None = None
    hedge_ratio: float = 1.0
    intercept: float = 0.0
    window: int = 20


class BasePairStrategy(ABC):
    """
    Base class for pair-trading strategies.
    """

    def __init__(self) -> None:
        self.state = StrategyState()
        self.is_fitted = False

    @abstractmethod
    def fit(self, pair_data: pd.DataFrame) -> None:
        """Fit the strategy to a pair's historical data."""
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate position signals for the provided data."""
        raise NotImplementedError

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Default simple backtest:
        - generate signals
        - compute daily spread changes as proxy returns
        - multiply by lagged position

        This is intentionally lightweight; the full backtest engine can replace it later.
        """
        signals = self.generate_signals(data)
        left, right = extract_pair_series(data)

        spread = right - self.state.hedge_ratio * left - self.state.intercept
        spread_ret = spread.diff().fillna(0.0)

        position = signals.reindex(spread.index).fillna(0.0).astype(float)
        strategy_return = position.shift(1).fillna(0.0) * (-spread_ret)

        result = pd.DataFrame(
            {
                "spread": spread,
                "signal": position,
                "spread_change": spread_ret,
                "strategy_return": strategy_return,
                "equity_curve": (1.0 + strategy_return).cumprod(),
            },
            index=spread.index,
        )
        return result