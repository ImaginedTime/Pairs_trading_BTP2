from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE_DIR = PROJECT_ROOT / "data" / "universe"


def normalize_ticker(symbol: str) -> str:
    """
    Normalize ticker symbols for internal use.

    This keeps the symbol readable and consistent across providers.
    It does not try to convert into a provider-specific format.
    """
    if symbol is None:
        raise ValueError("symbol cannot be None")

    s = str(symbol).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("–", "-")
    s = s.replace("/", "-")
    return s


def _normalize_constituents_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a constituents DataFrame so it always has a Symbol column.
    """
    out = df.copy()

    rename_map = {}
    for col in out.columns:
        if str(col).strip().lower() == "symbol":
            rename_map[col] = "Symbol"
        elif str(col).strip().lower() == "security":
            rename_map[col] = "Security"
        elif str(col).strip().lower() == "gics sector":
            rename_map[col] = "GICS Sector"
        elif str(col).strip().lower() == "gics sub-industry":
            rename_map[col] = "GICS Sub-Industry"
        elif str(col).strip().lower() == "headquarters location":
            rename_map[col] = "Headquarters Location"
        elif str(col).strip().lower() == "date added":
            rename_map[col] = "Date added"
        elif str(col).strip().lower() == "cik":
            rename_map[col] = "CIK"
        elif str(col).strip().lower() == "founded":
            rename_map[col] = "Founded"

    out = out.rename(columns=rename_map)

    if "Symbol" not in out.columns:
        # If the source is a one-column CSV, treat the first column as symbol.
        if len(out.columns) == 0:
            raise ValueError("No columns found in constituents data.")
        out = out.rename(columns={out.columns[0]: "Symbol"})

    out["Symbol"] = out["Symbol"].astype(str).map(normalize_ticker)

    # Keep only common useful columns if present, in a stable order.
    preferred_cols = [
        "Symbol",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "Headquarters Location",
        "Date added",
        "CIK",
        "Founded",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [
        c for c in out.columns if c not in preferred_cols
    ]
    out = out[cols]

    return out.reset_index(drop=True)


def load_constituents_from_csv(path: str) -> pd.DataFrame:
    """
    Load S&P 500 constituents from a CSV file.

    Expected formats:
    - a CSV with a Symbol column
    - a CSV with the first column as the symbol list
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(csv_path)
    return _normalize_constituents_frame(df)


def _load_current_sp500_from_wikipedia() -> pd.DataFrame:
    """
    Fetch the current S&P 500 constituents from Wikipedia.

    This is a practical free starting point for initial development.
    """
    tables = pd.read_html(WIKIPEDIA_SP500_URL)

    candidates: list[pd.DataFrame] = []
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if "symbol" in cols and ("security" in cols or "company" in cols):
            candidates.append(table)

    if not candidates:
        raise RuntimeError("Could not find the S&P 500 constituents table on Wikipedia.")

    df = candidates[0].copy()
    return _normalize_constituents_frame(df)


def get_sp500_constituents(as_of: str) -> pd.DataFrame:
    """
    Return S&P 500 constituents for a given as-of date.

    Initial-version behavior:
    - If a cached snapshot exists at data/universe/sp500_constituents_{as_of}.csv,
      load that file.
    - Otherwise, fall back to the current constituents from Wikipedia.

    The historical as-of reconstruction can be added later using a curated
    snapshot file or corporate-action history.
    """
    snapshot_path = DEFAULT_UNIVERSE_DIR / f"sp500_constituents_{as_of}.csv"
    if snapshot_path.exists():
        return load_constituents_from_csv(str(snapshot_path))

    cache_path = DEFAULT_UNIVERSE_DIR / "sp500_constituents.csv"
    if cache_path.exists():
        return load_constituents_from_csv(str(cache_path))

    return _load_current_sp500_from_wikipedia()