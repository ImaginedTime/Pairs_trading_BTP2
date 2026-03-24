from __future__ import annotations

import math


def position_size(
    capital: float,
    price_left: float,
    price_right: float,
    hedge_ratio: float = 1.0,
    risk_fraction: float = 0.02,
) -> float:
    """
    Compute a dollar-normalized position size for one pair.

    This returns the dollar amount to allocate to the pair, not the number of shares.
    You can convert this into shares later using the hedge ratio.

    Args:
        capital: Total account capital.
        price_left: Price of the left asset.
        price_right: Price of the right asset.
        hedge_ratio: Estimated hedge ratio between the assets.
        risk_fraction: Fraction of capital to allocate.

    Returns:
        Dollar notional for the pair.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    if price_left <= 0 or price_right <= 0:
        raise ValueError("prices must be positive")
    if hedge_ratio == 0:
        raise ValueError("hedge_ratio cannot be zero")
    if not (0 < risk_fraction <= 1):
        raise ValueError("risk_fraction must be in (0, 1]")

    # Basic allocation scaled by hedge ratio magnitude.
    base_notional = capital * risk_fraction
    hedge_scale = 1.0 / (1.0 + abs(float(hedge_ratio) - 1.0))
    return float(base_notional * hedge_scale)


def gross_exposure(
    long_notional: float,
    short_notional: float,
) -> float:
    """
    Compute gross exposure of a pair position.
    """
    if long_notional < 0 or short_notional < 0:
        raise ValueError("notionals must be non-negative")
    return float(abs(long_notional) + abs(short_notional))


def shares_from_notional(
    notional: float,
    price: float,
) -> int:
    """
    Convert a dollar notional into whole shares.
    """
    if notional < 0:
        raise ValueError("notional must be non-negative")
    if price <= 0:
        raise ValueError("price must be positive")
    return int(math.floor(notional / price))