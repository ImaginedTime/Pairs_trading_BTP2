from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pairs_trading.clustering.optics_cluster import cluster_assets, get_clusters
from pairs_trading.features.dimensionality import fit_pca_to_variance
from pairs_trading.features.scaling import scale_features
from pairs_trading.features.returns import compute_daily_returns
from pairs_trading.selection.filters import pair_quality_report, passes_pair_filters


@dataclass
class PairSelector:
    target_variance: float = 0.8
    min_samples: int = 3
    prices_df: pd.DataFrame | None = field(default=None, init=False)
    returns_df: pd.DataFrame | None = field(default=None, init=False)
    scaled_df: pd.DataFrame | None = field(default=None, init=False)
    pca_df: pd.DataFrame | None = field(default=None, init=False)
    pca_model: object | None = field(default=None, init=False)
    labels: pd.Series | None = field(default=None, init=False)
    clusters: dict[int, list[str]] = field(default_factory=dict, init=False)
    selected_pairs_: pd.DataFrame | None = field(default=None, init=False)

    def fit(self, prices_df: pd.DataFrame) -> "PairSelector":
        """
        Fit the full selection pipeline on a wide price matrix.
        """
        if prices_df is None or prices_df.empty:
            raise ValueError("prices_df is empty")

        prices = prices_df.copy().sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

        self.prices_df = prices
        self.returns_df = compute_daily_returns(prices).dropna(axis=0, how="any")

        self.scaled_df = scale_features(self.returns_df)
        self.pca_model, self.pca_df = fit_pca_to_variance(
            self.scaled_df, target_variance=self.target_variance
        )
        self.labels = cluster_assets(self.scaled_df, min_samples=self.min_samples)
        self.clusters = get_clusters(self.labels)
        self.selected_pairs_ = None
        return self

    def generate_candidates(self) -> list[tuple[str, str]]:
        if not self.clusters:
            return []

        candidates: list[tuple[str, str]] = []
        for _, members in self.clusters.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.append((members[i], members[j]))
        return candidates

    def select_pairs(self) -> pd.DataFrame:
        if self.prices_df is None:
            raise ValueError("Call fit() before select_pairs().")

        candidates = self.generate_candidates()
        rows: list[dict[str, object]] = []

        for left, right in candidates:
            if left not in self.prices_df.columns or right not in self.prices_df.columns:
                continue

            x = self.prices_df[left]
            y = self.prices_df[right]

            if not passes_pair_filters(x, y):
                continue

            metrics = pair_quality_report(x, y)
            rows.append(
                {
                    "asset_left": left,
                    "asset_right": right,
                    **metrics,
                }
            )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(
                by=["cointegration_pvalue", "hurst", "half_life"],
                ascending=[True, True, True],
            ).reset_index(drop=True)

        self.selected_pairs_ = result
        return result