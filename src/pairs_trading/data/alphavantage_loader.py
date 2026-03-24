"""Alpha Vantage downloader utilities."""
from __future__ import annotations
import pandas as pd

def fetch_alpha_vantage_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily bars from Alpha Vantage."""
    raise NotImplementedError

def fetch_alpha_vantage_intraday(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch intraday bars from Alpha Vantage."""
    raise NotImplementedError
