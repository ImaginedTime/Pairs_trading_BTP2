from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the DataFrame is indexed by datetime.
    """
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        return out

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        return out

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
        out.index.name = "date"
        return out

    raise ValueError("DataFrame must have a DatetimeIndex or a date/Date column.")


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an OHLCV frame.

    Steps:
    - normalize the index
    - standardize column names
    - coerce numeric fields
    - drop duplicate timestamps
    - drop rows without close prices
    """
    out = _ensure_datetime_index(df)

    rename_map = {}
    for col in out.columns:
        c = str(col).strip().lower()
        if c in {"open", "high", "low", "close", "adj close", "adj_close", "volume"}:
            rename_map[col] = c.replace("adj close", "adj_close")
        elif c == "date":
            rename_map[col] = "date"

    out = out.rename(columns=rename_map)

    numeric_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[~out.index.duplicated(keep="last")]
    if "close" in out.columns:
        out = out.dropna(subset=["close"])
    else:
        raise ValueError("OHLCV data must contain a close column.")

    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0)

    return out.sort_index()


def align_prices(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple symbol DataFrames into one wide close-price matrix.

    Input:
        {
            "AAPL": df_aapl,
            "MSFT": df_msft,
            ...
        }

    Output:
        index = dates
        columns = symbols
        values = close prices
    """
    if not data:
        raise ValueError("Input data dictionary is empty.")

    series_list: list[pd.Series] = []

    for symbol, df in data.items():
        if df is None or len(df) == 0:
            continue

        cleaned = clean_ohlcv(df)

        if "close" not in cleaned.columns:
            raise ValueError(f"Missing close column for symbol {symbol}")

        s = cleaned["close"].copy()
        s.name = symbol
        series_list.append(s)

    if not series_list:
        raise ValueError("No valid price series found.")

    aligned = pd.concat(series_list, axis=1, join="inner").sort_index()
    aligned = aligned.dropna(how="any")
    return aligned


def _interval_to_pandas_rule(interval: str) -> str:
    """
    Convert a human-friendly interval to a pandas resample rule.
    """
    normalized = str(interval).strip().lower()

    mapping = {
        "1d": "1D",
        "d": "1D",
        "day": "1D",
        "daily": "1D",
        "1h": "1H",
        "h": "1H",
        "hour": "1H",
        "hourly": "1H",
        "5m": "5T",
        "5min": "5T",
        "5minute": "5T",
        "5minutes": "5T",
    }

    if normalized not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")

    return mapping[normalized]


def resample_prices(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a target interval.

    The input should be an OHLCV DataFrame indexed by datetime.
    """
    out = clean_ohlcv(df)
    rule = _interval_to_pandas_rule(interval)

    agg_map: dict[str, Any] = {}
    for col in out.columns:
        if col == "open":
            agg_map[col] = "first"
        elif col == "high":
            agg_map[col] = "max"
        elif col == "low":
            agg_map[col] = "min"
        elif col == "close":
            agg_map[col] = "last"
        elif col == "volume":
            agg_map[col] = "sum"
        else:
            agg_map[col] = "last"

    resampled = out.resample(rule).agg(agg_map)
    resampled = resampled.dropna(subset=["close"])
    return resampled