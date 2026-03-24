from __future__ import annotations

def apply_transaction_costs(notional: float, bps: float = 1.0) -> float:
    """
    Compute transaction costs in absolute currency terms.

    Args:
        notional: Trade notional value.
        bps: Cost in basis points.

    Returns:
        Absolute cost.
    """
    if notional < 0:
        notional = abs(notional)
    return float(notional) * float(bps) / 10000.0


def apply_slippage(price: float, side: str, bps: float = 1.0) -> float:
    """
    Apply slippage to an execution price.

    Args:
        price: Raw market price.
        side: 'buy' or 'sell'.
        bps: Slippage in basis points.

    Returns:
        Adjusted execution price.
    """
    side = side.lower().strip()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    adj = float(price) * float(bps) / 10000.0
    if side == "buy":
        return float(price) + adj
    return float(price) - adj