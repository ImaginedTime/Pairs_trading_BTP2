from __future__ import annotations

import pandas as pd
from sklearn.cluster import OPTICS


def cluster_assets(X: pd.DataFrame, min_samples: int = 3) -> pd.Series:
    """
    Cluster assets using OPTICS.

    Input:
        X rows = dates, columns = asset features

    Output:
        Series indexed by asset/column name with cluster labels.
    """
    if X is None or X.empty:
        raise ValueError("X is empty")

    work = X.copy().sort_index()
    work = work.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    if work.shape[1] < 2:
        raise ValueError("Need at least two assets to cluster")

    model = OPTICS(min_samples=min_samples)
    labels = model.fit_predict(work.T.values)

    return pd.Series(labels, index=work.columns, name="cluster")


def get_clusters(labels: pd.Series) -> dict[int, list[str]]:
    """
    Group assets by cluster label.
    Noise points have label -1 and are excluded.
    """
    if labels is None or labels.empty:
        raise ValueError("labels is empty")

    clusters: dict[int, list[str]] = {}
    for asset, label in labels.items():
        if int(label) == -1:
            continue
        clusters.setdefault(int(label), []).append(str(asset))
    return clusters