from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def pair_stop_loss(
    unrealized_pnl: float,
    entry_notional: float,
    max_loss_pct: float = 0.05,
) -> bool:
    """
    Return True if the pair should be force-closed due to loss.

    Args:
        unrealized_pnl: Current unrealized PnL of the pair.
        entry_notional: Capital deployed in the pair.
        max_loss_pct: Maximum allowed loss as a fraction of notional.

    Example:
        entry_notional = 10_000
        max_loss_pct = 0.05
        stop triggers when pnl <= -500
    """
    if entry_notional <= 0:
        raise ValueError("entry_notional must be positive")
    if max_loss_pct <= 0:
        raise ValueError("max_loss_pct must be positive")

    loss_limit = -abs(entry_notional) * max_loss_pct
    return float(unrealized_pnl) <= loss_limit


def portfolio_stop_loss(
    equity_curve_value: float,
    initial_capital: float,
    max_drawdown_pct: float = 0.15,
) -> bool:
    """
    Return True if the overall portfolio should be stopped.

    This is a simple capital-protection rule based on total drawdown from the
    initial capital.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if max_drawdown_pct <= 0:
        raise ValueError("max_drawdown_pct must be positive")

    floor_value = initial_capital * (1.0 - max_drawdown_pct)
    return float(equity_curve_value) <= floor_value