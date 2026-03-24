from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


STOOQ_BASE_URL = "https://stooq.com/q/d/l/"


def _to_stooq_symbol(symbol: str) -> str:
    """
    Convert a ticker to the lowercase Stooq style used in download URLs.

    Examples:
    - AAPL -> aapl.us
    - BRK.B -> brk.b.us
    """
    if symbol is None:
        raise ValueError("symbol cannot be None")

    s = str(symbol).strip().lower()
    s = s.replace(" ", "")
    if not s.endswith(".us"):
        s = f"{s}.us"
    return s


def _interval_to_stooq_code(interval: str) -> str:
    """
    Map a human-friendly interval to a Stooq download code.

    For the initial version, daily data is supported directly.
    """
    normalized = str(interval).strip().lower()

    if normalized in {"1d", "d", "day", "daily"}:
        return "d"

    # If you want to extend this later, add the correct Stooq code here.
    # Keep the error explicit for now so the pipeline stays predictable.
    raise NotImplementedError(
        f"Interval '{interval}' is not implemented yet for Stooq downloads. "
        "Start with daily data first."
    )


def fetch_stooq_ohlcv(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch OHLCV data from Stooq.

    Returns a DataFrame indexed by datetime with columns:
    open, high, low, close, volume

    Notes:
    - This initial implementation supports daily data.
    - It filters rows to the requested date range.
    """
    stooq_symbol = _to_stooq_symbol(symbol)
    stooq_interval = _interval_to_stooq_code(interval)

    params = {
        "s": stooq_symbol,
        "i": stooq_interval,
    }

    response = requests.get(STOOQ_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    raw_text = response.text.strip()
    if not raw_text or "No data" in raw_text:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.read_csv(StringIO(raw_text))

    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Unexpected Stooq response columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).set_index("date")

    df = df.sort_index()
    return df


def save_raw_prices(df: pd.DataFrame, path: str) -> None:
    """
    Save raw price data to disk.

    Supported formats:
    - .csv
    - .parquet
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=True)
    elif suffix == ".parquet":
        df.to_parquet(out_path, index=True)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")


def batch_download(
    symbols: list[str],
    interval: str,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Download multiple symbols and return a symbol -> DataFrame mapping.
    """
    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            result[symbol] = fetch_stooq_ohlcv(symbol, interval, start, end)
        except Exception as exc:
            # Keep the pipeline moving; you can log these later if needed.
            result[symbol] = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
            print(f"[WARN] Failed to download {symbol}: {exc}")

    return result