from __future__ import annotations

import pandas as pd


def enter_pair_trade(*args, **kwargs) -> dict:
    """
    Create a trade event dictionary for opening a position.
    """
    timestamp = kwargs.get("timestamp")
    position = float(kwargs.get("position", 0.0))
    spread_price = float(kwargs.get("spread_price", 0.0))
    notional = float(kwargs.get("notional", 1.0))
    side = "long_spread" if position > 0 else "short_spread"

    return {
        "timestamp": timestamp,
        "event": "enter",
        "side": side,
        "position": position,
        "spread_price": spread_price,
        "notional": notional,
        "position_change": abs(position),
    }


def exit_pair_trade(*args, **kwargs) -> dict:
    """
    Create a trade event dictionary for closing a position.
    """
    timestamp = kwargs.get("timestamp")
    current_position = float(kwargs.get("current_position", 0.0))
    spread_price = float(kwargs.get("spread_price", 0.0))
    notional = float(kwargs.get("notional", 1.0))

    return {
        "timestamp": timestamp,
        "event": "exit",
        "side": "flat",
        "position": 0.0,
        "current_position": current_position,
        "spread_price": spread_price,
        "notional": notional,
        "position_change": abs(current_position),
    }


def rebalance_pair_trade(*args, **kwargs) -> dict:
    """
    Create a trade event dictionary for changing a live position.
    """
    timestamp = kwargs.get("timestamp")
    current_position = float(kwargs.get("current_position", 0.0))
    target_position = float(kwargs.get("target_position", 0.0))
    spread_price = float(kwargs.get("spread_price", 0.0))
    notional = float(kwargs.get("notional", 1.0))

    return {
        "timestamp": timestamp,
        "event": "rebalance",
        "side": "long_spread" if target_position > 0 else "short_spread" if target_position < 0 else "flat",
        "current_position": current_position,
        "target_position": target_position,
        "spread_price": spread_price,
        "notional": notional,
        "position_change": abs(target_position - current_position),
    }