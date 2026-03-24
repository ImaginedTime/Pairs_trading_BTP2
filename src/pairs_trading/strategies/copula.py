from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.features.spreads import build_spread
from pairs_trading.selection.statistics import estimate_spread_beta
from .base import BasePairStrategy, extract_pair_series


def _to_uniform_ranks(s: pd.Series) -> pd.Series:
    """
    Convert a return series into pseudo-observations on (0, 1).
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        raise ValueError("series too short for rank transform")
    ranks = x.rank(method="average")
    u = (ranks - 0.5) / len(x)
    return u


def fit_copula(u: pd.Series, v: pd.Series) -> dict:
    """
    Lightweight copula fit placeholder.

    This returns rank-correlation and dependence stats that can later be
    replaced by a full Archimedean / Gaussian copula implementation.
    """
    uu = pd.to_numeric(u, errors="coerce").dropna().astype(float)
    vv = pd.to_numeric(v, errors="coerce").dropna().astype(float)

    aligned = pd.concat([uu, vv], axis=1).dropna()
    if len(aligned) < 10:
        raise ValueError("not enough data to fit copula")

    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    rho = x.corr(y, method="spearman")
    tau = x.corr(y, method="kendall")

    return {
        "spearman_rho": float(rho),
        "kendall_tau": float(tau),
        "n": int(len(aligned)),
    }


def conditional_probabilities(*args, **kwargs) -> pd.DataFrame:
    """
    Placeholder conditional probability table.

    For the initial implementation we use:
    - empirical joint rank buckets
    - conditional probability estimates by quartile
    """
    if len(args) < 2:
        raise ValueError("expected two series arguments")

    u = pd.to_numeric(args[0], errors="coerce").dropna().astype(float)
    v = pd.to_numeric(args[1], errors="coerce").dropna().astype(float)

    aligned = pd.concat([u, v], axis=1).dropna()
    if len(aligned) < 20:
        raise ValueError("not enough data")

    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    x_bucket = pd.qcut(x, 4, labels=False, duplicates="drop")
    y_bucket = pd.qcut(y, 4, labels=False, duplicates="drop")

    table = pd.crosstab(x_bucket, y_bucket, normalize="index")
    return table


class CopulaStrategy(BasePairStrategy):
    def __init__(self, entry_quantile: float = 0.15, exit_quantile: float = 0.5) -> None:
        super().__init__()
        self.entry_quantile = entry_quantile
        self.exit_quantile = exit_quantile
        self.copula_params_: dict | None = None
        self.left_ranks_: pd.Series | None = None
        self.right_ranks_: pd.Series | None = None
        self.spread_: pd.Series | None = None

    def fit(self, pair_data: pd.DataFrame) -> None:
        left, right = extract_pair_series(pair_data)
        beta = estimate_spread_beta(left, right)
        spread = build_spread(left, right, beta=beta)

        self.state.hedge_ratio = beta
        self.state.pair_left = left.name
        self.state.pair_right = right.name
        self.spread_ = spread

        self.left_ranks_ = _to_uniform_ranks(left)
        self.right_ranks_ = _to_uniform_ranks(right)

        aligned = pd.concat([self.left_ranks_, self.right_ranks_], axis=1).dropna()
        self.copula_params_ = fit_copula(aligned.iloc[:, 0], aligned.iloc[:, 1])
        self.is_fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            self.fit(data)

        left, right = extract_pair_series(data)
        spread = build_spread(left, right, beta=self.state.hedge_ratio)

        left_u = _to_uniform_ranks(left)
        right_u = _to_uniform_ranks(right)
        aligned = pd.concat([left_u, right_u, spread], axis=1).dropna()
        aligned.columns = ["u", "v", "spread"]

        signals = pd.Series(0.0, index=aligned.index, name="signal")
        position = 0.0

        # Use marginal percentiles as an easy first-pass proxy for copula tail risk.
        u_q_low = aligned["u"].quantile(self.entry_quantile)
        u_q_high = aligned["u"].quantile(1.0 - self.entry_quantile)
        v_q_low = aligned["v"].quantile(self.entry_quantile)
        v_q_high = aligned["v"].quantile(1.0 - self.entry_quantile)
        spread_mid = aligned["spread"].quantile(self.exit_quantile)

        for idx, row in aligned.iterrows():
            u = row["u"]
            v = row["v"]
            sp = row["spread"]

            if position == 0.0:
                if (u <= u_q_low and v >= v_q_high) or sp < spread_mid:
                    position = 1.0
                elif (u >= u_q_high and v <= v_q_low) or sp > spread_mid:
                    position = -1.0
            else:
                if abs(sp - spread_mid) <= abs(spread_mid) * 0.05:
                    position = 0.0

            signals.loc[idx] = position

        return signals.reindex(spread.index).fillna(0.0)

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().backtest(data)