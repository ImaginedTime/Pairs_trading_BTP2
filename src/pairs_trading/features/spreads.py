"""Spread construction helpers."""
from __future__ import annotations
import pandas as pd

def build_spread(x: pd.Series, y: pd.Series, beta: float = 1.0) -> pd.Series:
    """Build a spread from two price series."""
    raise NotImplementedError

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score."""
    raise NotImplementedError
