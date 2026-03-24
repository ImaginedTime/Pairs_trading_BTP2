from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a feature matrix column-wise.
    """
    if X is None or X.empty:
        raise ValueError("X is empty")

    work = X.copy().sort_index()
    work = work.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(work.values)

    return pd.DataFrame(scaled, index=work.index, columns=work.columns)


def standardize_per_asset(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each column independently to mean 0 and std 1.
    """
    if X is None or X.empty:
        raise ValueError("X is empty")

    work = X.copy().sort_index()
    work = work.apply(pd.to_numeric, errors="coerce")

    out = work.copy()
    for col in out.columns:
        s = out[col]
        mu = s.mean()
        sigma = s.std(ddof=0)
        if pd.isna(sigma) or sigma == 0:
            out[col] = 0.0
        else:
            out[col] = (s - mu) / sigma

    return out.dropna(axis=0, how="any")