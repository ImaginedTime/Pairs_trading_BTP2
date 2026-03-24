from __future__ import annotations

import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple percentage returns from a wide price matrix.

    Input:
        rows = dates
        columns = symbols
        values = close prices

    Output:
        same shape, percentage returns
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty")

    out = prices.sort_index().copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    returns = out.pct_change().replace([float("inf"), float("-inf")], pd.NA)
    return returns.dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from a wide price matrix.
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty")

    out = prices.sort_index().copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    log_prices = out.where(out > 0).applymap(lambda x: pd.NA if pd.isna(x) else x)
    log_returns = (log_prices.apply(lambda col: pd.Series(col).astype(float).apply(lambda x: x)).applymap(lambda x: x))
    log_returns = pd.DataFrame(index=out.index, columns=out.columns, dtype=float)

    for col in out.columns:
        series = pd.to_numeric(out[col], errors="coerce")
        series = series.where(series > 0)
        log_returns[col] = series.apply(lambda x: pd.NA if pd.isna(x) else float(x)).astype(float)
        log_returns[col] = log_returns[col].apply(lambda x: pd.NA if pd.isna(x) else x)

    log_returns = pd.DataFrame(
        {col: pd.Series(pd.to_numeric(out[col], errors="coerce")).apply(
            lambda x: pd.NA if pd.isna(x) or x <= 0 else x
        ) for col in out.columns},
        index=out.index,
    )

    # Simpler and correct computation:
    log_prices = out.where(out > 0)
    result = (log_prices.apply(lambda s: pd.Series(s).apply(lambda x: pd.NA if pd.isna(x) else float(x))).applymap(lambda x: x))
    result = pd.DataFrame(index=out.index, columns=out.columns, dtype=float)
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        s = s.where(s > 0)
        result[col] = s.apply(lambda x: pd.NA if pd.isna(x) else x)

    result = pd.DataFrame(index=out.index, columns=out.columns, dtype=float)
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        s = s.where(s > 0)
        result[col] = s.pipe(lambda x: x.apply(lambda v: pd.NA if pd.isna(v) else float(v)))
        result[col] = result[col].astype(float)

    return np_log_returns(out)


def np_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper for clean log return calculation.
    """
    import numpy as np

    if prices is None or prices.empty:
        raise ValueError("prices is empty")

    out = prices.sort_index().copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.where(out > 0)
    return np.log(out).diff().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")


def pivot_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-form returns DataFrame into a wide matrix.

    Expected long-form columns:
        - date (or datetime-like index)
        - symbol
        - return

    Output:
        rows = dates
        columns = symbols
        values = returns
    """
    if df is None or df.empty:
        raise ValueError("df is empty")

    work = df.copy()

    if isinstance(work.index, pd.DatetimeIndex) and {"symbol", "return"}.issubset(work.columns):
        work = work.reset_index().rename(columns={work.index.name or "index": "date"})

    if "date" not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("Expected a date column or DatetimeIndex.")

    required = {"date", "symbol", "return"}
    if not required.issubset(work.columns):
        raise ValueError(f"df must contain columns: {required}")

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "symbol"])

    wide = work.pivot(index="date", columns="symbol", values="return").sort_index()
    return wide