from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA


def fit_pca_to_variance(X: pd.DataFrame, target_variance: float = 0.8) -> tuple[PCA, pd.DataFrame]:
    """
    Fit PCA until the cumulative explained variance reaches target_variance.

    Returns:
        fitted PCA model, transformed DataFrame
    """
    if X is None or X.empty:
        raise ValueError("X is empty")

    work = X.copy().sort_index()
    work = work.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    n_features = work.shape[1]
    if n_features < 1:
        raise ValueError("Need at least one feature for PCA")

    full_pca = PCA(n_components=min(work.shape[0], n_features))
    full_pca.fit(work.values)

    cum_var = full_pca.explained_variance_ratio_.cumsum()
    k = int((cum_var < target_variance).sum() + 1)
    k = max(1, min(k, n_features))

    pca = PCA(n_components=k)
    transformed = pca.fit_transform(work.values)

    cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    transformed_df = pd.DataFrame(transformed, index=work.index, columns=cols)
    return pca, transformed_df


def transform_pca(model: PCA, X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a matrix with a fitted PCA model.
    """
    if model is None:
        raise ValueError("model is None")
    if X is None or X.empty:
        raise ValueError("X is empty")

    work = X.copy().sort_index()
    work = work.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    transformed = model.transform(work.values)
    cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    return pd.DataFrame(transformed, index=work.index, columns=cols)