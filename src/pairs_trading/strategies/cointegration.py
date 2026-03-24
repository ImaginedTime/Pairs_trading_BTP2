from __future__ import annotations

import pandas as pd

from pairs_trading.features.spreads import zscore
from pairs_trading.selection.statistics import estimate_spread_beta
from .base import BasePairStrategy, extract_pair_series


def fit_hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """
    Estimate hedge ratio beta from y ~ alpha + beta * x.
    """
    return estimate_spread_beta(x, y)


def build_spread(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Construct a cointegration spread using an estimated hedge ratio.
    """
    beta = fit_hedge_ratio(x, y)
    aligned = pd.concat([x, y], axis=1).dropna()
    left = aligned.iloc[:, 0]
    right = aligned.iloc[:, 1]
    spread = right - beta * left
    spread.name = "spread"
    return spread


class CointegrationStrategy(BasePairStrategy):
    def __init__(self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5) -> None:
        super().__init__()
        self.state.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.spread_: pd.Series | None = None
        self.zscore_: pd.Series | None = None

    def fit(self, pair_data: pd.DataFrame) -> None:
        left, right = extract_pair_series(pair_data)
        beta = fit_hedge_ratio(left, right)
        spread = right - beta * left

        self.state.hedge_ratio = beta
        self.state.pair_left = left.name
        self.state.pair_right = right.name
        self.spread_ = spread
        self.zscore_ = zscore(spread, window=self.state.window)
        self.is_fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            self.fit(data)

        left, right = extract_pair_series(data)
        spread = right - self.state.hedge_ratio * left
        z = zscore(spread, window=self.state.window)

        signals = pd.Series(0.0, index=spread.index, name="signal")
        position = 0.0

        for i in range(len(z)):
            zi = z.iloc[i]
            if pd.isna(zi):
                signals.iloc[i] = position
                continue

            if position == 0.0:
                if zi >= self.entry_z:
                    position = -1.0
                elif zi <= -self.entry_z:
                    position = 1.0
            else:
                if abs(zi) <= self.exit_z:
                    position = 0.0

            signals.iloc[i] = position

        return signals

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().backtest(data)