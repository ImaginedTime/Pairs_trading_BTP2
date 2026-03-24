from __future__ import annotations

import pandas as pd

from pairs_trading.backtest.costs import apply_transaction_costs
from pairs_trading.backtest.execution import (
    enter_pair_trade,
    exit_pair_trade,
    rebalance_pair_trade,
)
from pairs_trading.backtest.portfolio import Portfolio
from pairs_trading.strategies.base import extract_pair_series


class BacktestEngine:
    """
    Simple pair-trading backtest engine.

    The engine treats strategy signals as:
        +1 = long spread
        -1 = short spread
         0 = flat
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        notional_per_trade: float = 1.0,
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 1.0,
        periods_per_year: int = 252,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.notional_per_trade = float(notional_per_trade)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.slippage_bps = float(slippage_bps)
        self.periods_per_year = int(periods_per_year)

    def _build_spread(self, strategy, data: pd.DataFrame) -> pd.Series:
        left, right = extract_pair_series(data)
        beta = float(getattr(strategy.state, "hedge_ratio", 1.0))
        intercept = float(getattr(strategy.state, "intercept", 0.0))
        spread = right - beta * left - intercept
        spread.name = "spread"
        return spread

    def run(self, strategy, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a backtest for the given strategy and pair data.
        """
        if data is None or data.empty:
            raise ValueError("data is empty")

        if not getattr(strategy, "is_fitted", False):
            strategy.fit(data)

        spread = self._build_spread(strategy, data)
        signals = strategy.generate_signals(data).reindex(spread.index).fillna(0.0).astype(float)

        aligned = pd.DataFrame(
            {
                "spread": spread,
                "signal": signals.clip(-1.0, 1.0),
            }
        ).dropna()

        if aligned.empty:
            raise ValueError("No overlapping data available for backtest")

        portfolio = Portfolio(cash=self.initial_capital, equity=self.initial_capital)

        position = aligned["signal"].shift(1).fillna(0.0)
        spread_change = aligned["spread"].diff().fillna(0.0)

        gross_pnl = position * spread_change * self.notional_per_trade

        turnover = (aligned["signal"] - position).abs() * self.notional_per_trade
        costs = turnover.apply(
            lambda x: apply_transaction_costs(
                x, bps=self.transaction_cost_bps + self.slippage_bps
            )
        )

        net_pnl = gross_pnl - costs
        equity_curve = self.initial_capital + net_pnl.cumsum()
        returns = equity_curve.pct_change().fillna(0.0)

        trade_events = []
        prev_signal = 0.0

        for ts, row in aligned.iterrows():
            curr_signal = float(row["signal"])
            sp = float(row["spread"])

            if prev_signal == 0.0 and curr_signal != 0.0:
                trade_events.append(
                    enter_pair_trade(
                        timestamp=ts,
                        position=curr_signal,
                        spread_price=sp,
                        notional=self.notional_per_trade,
                    )
                )
            elif prev_signal != 0.0 and curr_signal == 0.0:
                trade_events.append(
                    exit_pair_trade(
                        timestamp=ts,
                        current_position=prev_signal,
                        spread_price=sp,
                        notional=self.notional_per_trade,
                    )
                )
            elif prev_signal != curr_signal:
                trade_events.append(
                    rebalance_pair_trade(
                        timestamp=ts,
                        current_position=prev_signal,
                        target_position=curr_signal,
                        spread_price=sp,
                        notional=self.notional_per_trade,
                    )
                )

            prev_signal = curr_signal

        trades_df = pd.DataFrame(trade_events)

        result = pd.DataFrame(
            {
                "spread": aligned["spread"],
                "signal": aligned["signal"],
                "position": position,
                "spread_change": spread_change,
                "gross_pnl": gross_pnl,
                "costs": costs,
                "net_pnl": net_pnl,
                "equity_curve": equity_curve,
                "returns": returns,
            },
            index=aligned.index,
        )

        result.attrs["portfolio"] = portfolio
        result.attrs["trades"] = trades_df
        result.attrs["summary"] = {
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return": float(equity_curve.iloc[-1] / self.initial_capital - 1.0),
            "num_trades": int(len(trades_df)),
        }

        return result