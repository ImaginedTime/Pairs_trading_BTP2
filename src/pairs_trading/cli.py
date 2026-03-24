from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import sys

from .config import ProjectConfig


def _cmd_show_config(args: argparse.Namespace) -> int:
    cfg = ProjectConfig()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


def _cmd_init_dirs(args: argparse.Namespace) -> int:
    from .constants import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, UNIVERSE_DATA_DIR, OUTPUTS_DIR

    for d in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, UNIVERSE_DATA_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"created: {d}")
    return 0


def _cmd_download_universe(args: argparse.Namespace) -> int:
    from .data.universe import get_sp500_constituents

    df = get_sp500_constituents(args.as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved universe to {out_path}")
    print(f"rows: {len(df)}")
    return 0


def _cmd_download_prices(args: argparse.Namespace) -> int:
    from .data.universe import load_constituents_from_csv
    from .data.stooq_loader import batch_download, save_raw_prices

    universe = load_constituents_from_csv(args.universe_csv)
    symbols = universe["Symbol"].dropna().astype(str).tolist()

    import datetime as dt
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    data = batch_download(symbols=symbols[: args.max_symbols], interval=args.interval, start=start, end=end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym, df in data.items():
        if df is None or df.empty:
            continue
        save_raw_prices(df, str(out_dir / f"{sym}.{args.interval}.csv"))
        print(f"saved {sym}")

    return 0


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    from .data.cleaning import align_prices
    import pandas as pd

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        print("no raw csv files found", file=sys.stderr)
        return 1

    data = {}
    for f in files:
        sym = f.stem.split(".")[0]
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        data[sym] = df

    aligned = align_prices(data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(out_path)
    print(f"saved processed dataset to {out_path}")
    return 0


def _cmd_select_pairs(args: argparse.Namespace) -> int:
    import pandas as pd
    from .features.returns import compute_daily_returns
    from .selection.pair_selector import PairSelector

    df = pd.read_parquet(args.prices)
    returns = compute_daily_returns(df)

    selector = PairSelector(
        target_variance=args.pca_variance,
        min_samples=args.min_samples,
    )
    selector.fit(returns)
    selected = selector.select_pairs()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)
    print(f"saved selected pairs to {out_path}")
    print(f"pairs: {len(selected)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pairs_trading")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show-config").set_defaults(func=_cmd_show_config)
    sub.add_parser("init-dirs").set_defaults(func=_cmd_init_dirs)

    p = sub.add_parser("download-universe")
    p.add_argument("--as-of", dest="as_of", default="2023-03-31")
    p.add_argument("--out", default="data/universe/sp500_constituents.csv")
    p.set_defaults(func=_cmd_download_universe)

    p = sub.add_parser("download-prices")
    p.add_argument("--universe-csv", required=True)
    p.add_argument("--start", default="2022-03-01")
    p.add_argument("--end", default="2023-03-31")
    p.add_argument("--interval", default="1d")
    p.add_argument("--max-symbols", type=int, default=50)
    p.add_argument("--out-dir", default="data/raw")
    p.set_defaults(func=_cmd_download_prices)

    p = sub.add_parser("build-dataset")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/processed/prices.parquet")
    p.set_defaults(func=_cmd_build_dataset)

    p = sub.add_parser("select-pairs")
    p.add_argument("--prices", required=True)
    p.add_argument("--pca-variance", type=float, default=0.8)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--out", default="data/processed/selected_pairs.csv")
    p.set_defaults(func=_cmd_select_pairs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()