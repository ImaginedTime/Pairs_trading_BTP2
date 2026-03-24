"""Yahoo Finance / yfinance downloader utilities."""
from __future__ import annotations
import pandas as pd

def fetch_yfinance(symbols: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch data from yfinance."""
    raise NotImplementedError
