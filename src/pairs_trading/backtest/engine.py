from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pairs_trading.backtest.costs import apply_transaction_costs
from pairs_trading.backtest.execution import (
    enter_pair_trade,
    exit_pair_trade,
    rebalance_pair_trade,
)
from pairs_trading.backtest.portfolio import Portfolio
from pairs_trading.strategies.base import extract_pair_series
from pairs_trading.constants import DEFAULT_NOTIONAL_PER_TRADE



@dataclass
class LegPosition:
    """
    Share positions for the two legs of a pair trade.

    left  -> x
    right -> y
    """
    left_shares: float = 0.0
    right_shares: float = 0.0


class BacktestEngine:
    """
    Pair-trading backtest engine with hedge-ratio-based leg sizing.

    Signal convention:
        +1  => long spread
        -1  => short spread
         0  => flat

    Spread definition used by the strategies:
        spread = right - beta * left

    Trade mapping:
        long spread  => long right, short beta * left
        short spread => short right, long beta * left

    Notes:
    - This engine uses close-to-close prices.
    - PnL is computed from the two legs directly.
    - Transaction costs are charged on turnover at each rebalance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        notional_per_trade: float = DEFAULT_NOTIONAL_PER_TRADE,
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 1.0,
        periods_per_year: int = 252,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.notional_per_trade = float(notional_per_trade)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.slippage_bps = float(slippage_bps)
        self.periods_per_year = int(periods_per_year)

    def _get_beta(self, strategy: Any) -> float:
        """
        Pull hedge ratio from the strategy.
        Falls back to 1.0 if unavailable.
        """
        beta = 1.0
        if hasattr(strategy, "state") and hasattr(strategy.state, "hedge_ratio"):
            try:
                beta = float(strategy.state.hedge_ratio)
            except Exception:
                beta = 1.0
        if beta == 0 or np.isnan(beta):
            beta = 1.0
        return beta

    def _shares_for_signal(
        self,
        signal: float,
        left_price: float,
        right_price: float,
        beta: float,
    ) -> LegPosition:
        """
        Convert a signal into actual leg shares.

        For spread = right - beta * left:

        long spread:
            right_shares = +q
            left_shares  = -beta * q

        short spread:
            right_shares = -q
            left_shares  = +beta * q
        """
        signal = float(signal)

        if signal == 0.0:
            return LegPosition(0.0, 0.0)

        if left_price <= 0 or right_price <= 0:
            raise ValueError("Prices must be positive for sizing.")

        denom = right_price + abs(beta) * left_price
        if denom <= 0:
            raise ValueError("Invalid prices or hedge ratio for sizing.")

        q = self.notional_per_trade / denom

        if signal > 0:
            # Long spread
            right_shares = +q
            left_shares = -beta * q
        else:
            # Short spread
            right_shares = -q
            left_shares = +beta * q

        return LegPosition(left_shares=left_shares, right_shares=right_shares)

    def _turnover_notional(
        self,
        prev_pos: LegPosition,
        new_pos: LegPosition,
        left_price: float,
        right_price: float,
    ) -> float:
        """
        Approximate dollar turnover for a rebalance.
        """
        left_turnover = abs(new_pos.left_shares - prev_pos.left_shares) * left_price
        right_turnover = abs(new_pos.right_shares - prev_pos.right_shares) * right_price
        return float(left_turnover + right_turnover)

    def _pnl_from_positions(
        self,
        pos: LegPosition,
        left_now: float,
        left_next: float,
        right_now: float,
        right_next: float,
    ) -> float:
        """
        Close-to-close PnL from held share positions.
        """
        delta_left = left_next - left_now
        delta_right = right_next - right_now
        return float(pos.left_shares * delta_left + pos.right_shares * delta_right)

    def run(self, strategy: Any, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a backtest for one pair.

        Expected input:
            data with two numeric columns representing the pair prices.

        Output columns:
            left_price, right_price, signal, left_shares, right_shares,
            gross_pnl, costs, net_pnl, equity_curve, returns
        """
        if data is None or data.empty:
            raise ValueError("data is empty")

        if not getattr(strategy, "is_fitted", False):
            strategy.fit(data)

        left, right = extract_pair_series(data)
        beta = self._get_beta(strategy)

        left = pd.to_numeric(left, errors="coerce")
        right = pd.to_numeric(right, errors="coerce")

        prices = pd.concat([left, right], axis=1).dropna()
        prices.columns = ["left_price", "right_price"]

        if len(prices) < 3:
            raise ValueError("Not enough data points for backtesting")

        # Strategy decision signal. This is the desired position at each timestamp.
        decision_signal = strategy.generate_signals(data).reindex(prices.index).fillna(0.0)
        decision_signal = decision_signal.astype(float).clip(-1.0, 1.0)

        # Positions held over the NEXT interval are based on the current decision.
        held_signal = decision_signal.copy()

        rows: list[dict[str, Any]] = []
        portfolio = Portfolio(cash=self.initial_capital, equity=self.initial_capital)

        prev_signal = 0.0
        equity = self.initial_capital

        idx = prices.index
        for i in range(len(idx)):
            ts = idx[i]
            left_now = float(prices.iloc[i]["left_price"])
            right_now = float(prices.iloc[i]["right_price"])
            sig = float(held_signal.iloc[i])

            # Current leg positions from the signal at this timestamp.
            pos = self._shares_for_signal(sig, left_now, right_now, beta)

            # Transaction cost when the decision changes from previous signal to current.
            prev_pos = self._shares_for_signal(prev_signal, left_now, right_now, beta)
            turnover_notional = self._turnover_notional(prev_pos, pos, left_now, right_now)
            costs = apply_transaction_costs(
                turnover_notional,
                bps=self.transaction_cost_bps + self.slippage_bps,
            )

            # PnL from t -> t+1. Last row cannot produce a next-day PnL.
            if i < len(idx) - 1:
                left_next = float(prices.iloc[i + 1]["left_price"])
                right_next = float(prices.iloc[i + 1]["right_price"])
                gross_pnl = self._pnl_from_positions(
                    pos, left_now, left_next, right_now, right_next
                )
            else:
                gross_pnl = 0.0

            net_pnl = gross_pnl - costs
            equity = equity + net_pnl
            ret = 0.0 if i == 0 else (equity / rows[-1]["equity_curve"] - 1.0 if rows[-1]["equity_curve"] != 0 else 0.0)
            spread_now = right_now - beta * left_now

            rows.append(
                {
                    "date": ts,
                    "left_price": left_now,
                    "right_price": right_now,
                    "spread": spread_now,
                    "signal": sig,
                    "prev_signal": prev_signal,
                    "beta": beta,
                    "left_shares": pos.left_shares,
                    "right_shares": pos.right_shares,
                    "gross_pnl": gross_pnl,
                    "costs": costs,
                    "net_pnl": net_pnl,
                    "equity_curve": equity,
                }
            )

            # Log trade events whenever the signal changes.
            if sig != prev_signal:
                if prev_signal == 0.0 and sig != 0.0:
                    enter_pair_trade(
                        timestamp=ts,
                        position=sig,
                        spread_price=right_now - beta * left_now,
                        notional=self.notional_per_trade,
                    )
                elif sig == 0.0 and prev_signal != 0.0:
                    exit_pair_trade(
                        timestamp=ts,
                        current_position=prev_signal,
                        spread_price=right_now - beta * left_now,
                        notional=self.notional_per_trade,
                    )
                else:
                    rebalance_pair_trade(
                        timestamp=ts,
                        current_position=prev_signal,
                        target_position=sig,
                        spread_price=right_now - beta * left_now,
                        notional=self.notional_per_trade,
                    )

            prev_signal = sig

        result = pd.DataFrame(rows).set_index("date")
        result["returns"] = result["equity_curve"].pct_change().fillna(0.0)

        # Attach useful metadata
        trade_changes = result["signal"].diff().fillna(result["signal"].iloc[0]).ne(0)
        trades_df = result.loc[trade_changes, ["signal", "left_price", "right_price", "beta"]].copy()

        result.attrs["portfolio"] = portfolio
        result.attrs["trades"] = trades_df
        result.attrs["summary"] = {
            "final_equity": float(result["equity_curve"].iloc[-1]),
            "total_return": float(result["equity_curve"].iloc[-1] / self.initial_capital - 1.0),
            "num_trades": int(len(trades_df)),
            "beta": float(beta),
        }

        return result