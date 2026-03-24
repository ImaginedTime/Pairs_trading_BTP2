from __future__ import annotations

import pandas as pd

from pairs_trading.features.spreads import build_spread, zscore
from pairs_trading.selection.statistics import estimate_spread_beta
from .base import BasePairStrategy, extract_pair_series


class BollingerStrategy(BasePairStrategy):
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        exit_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.state.window = window
        self.num_std = num_std
        self.exit_std = exit_std
        self.spread_: pd.Series | None = None
        self.zscore_: pd.Series | None = None

    def fit(self, pair_data: pd.DataFrame) -> None:
        left, right = extract_pair_series(pair_data)
        beta = estimate_spread_beta(left, right)
        spread = build_spread(left, right, beta=beta)

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
        spread = build_spread(left, right, beta=self.state.hedge_ratio)
        z = zscore(spread, window=self.state.window)

        signals = pd.Series(0.0, index=spread.index, name="signal")
        position = 0.0

        for i in range(len(z)):
            zi = z.iloc[i]
            if pd.isna(zi):
                signals.iloc[i] = position
                continue

            if position == 0.0:
                if zi >= self.num_std:
                    position = -1.0  # short spread
                elif zi <= -self.num_std:
                    position = 1.0   # long spread
            else:
                if abs(zi) <= self.exit_std:
                    position = 0.0

            signals.iloc[i] = position

        return signals

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().backtest(data)