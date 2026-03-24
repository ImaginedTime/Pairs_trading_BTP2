from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    """
    Lightweight portfolio state for pair-trading backtests.

    position:
        +1 = long spread
        -1 = short spread
         0 = flat
    """
    cash: float = 0.0
    equity: float = 0.0
    position: float = 0.0
    entry_spread: float | None = None
    last_spread: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    history: list[dict] = field(default_factory=list)

    def mark_to_market(self, spread_price: float, notional: float = 1.0) -> float:
        """
        Update unrealized PnL using the latest spread price.
        """
        self.last_spread = float(spread_price)

        if self.position == 0.0 or self.entry_spread is None:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = self.position * (float(spread_price) - self.entry_spread) * float(notional)

        self.equity = self.cash + self.realized_pnl + self.unrealized_pnl
        return self.equity

    def open_position(self, target_position: float, spread_price: float, notional: float = 1.0) -> None:
        """
        Open a new position from flat.
        """
        if self.position != 0.0:
            raise ValueError("Cannot open a new position when one is already open.")

        self.position = float(target_position)
        self.entry_spread = float(spread_price)
        self.last_spread = float(spread_price)
        self.trade_count += 1
        self.mark_to_market(spread_price, notional=notional)

    def close_position(self, spread_price: float, notional: float = 1.0) -> float:
        """
        Close the current position and realize PnL.
        """
        if self.position == 0.0 or self.entry_spread is None:
            return self.equity

        pnl = self.position * (float(spread_price) - self.entry_spread) * float(notional)
        self.realized_pnl += pnl
        self.cash += pnl

        self.position = 0.0
        self.entry_spread = None
        self.unrealized_pnl = 0.0
        self.trade_count += 1
        self.last_spread = float(spread_price)
        self.equity = self.cash + self.realized_pnl
        return self.equity

    def update(self, spread_price: float, target_position: float, notional: float = 1.0) -> float:
        """
        Move portfolio state to a target position at the latest spread price.

        Logic:
        - If flat -> open target
        - If target is flat -> close current
        - If target changes sign -> close then open
        - If same sign -> just mark-to-market
        """
        spread_price = float(spread_price)
        target_position = float(target_position)

        if self.position == target_position:
            return self.mark_to_market(spread_price, notional=notional)

        if target_position == 0.0:
            return self.close_position(spread_price, notional=notional)

        if self.position == 0.0:
            self.open_position(target_position, spread_price, notional=notional)
            return self.equity

        # Sign flip or partial change
        self.close_position(spread_price, notional=notional)
        self.open_position(target_position, spread_price, notional=notional)
        return self.equity