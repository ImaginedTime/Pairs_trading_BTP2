from __future__ import annotations

from datetime import datetime, date
from typing import Optional


def should_refresh_pairs(
    current_date: date | datetime,
    last_refresh_date: date | datetime | None,
    refresh_period_days: int = 180,
) -> bool:
    """
    Return True when pair selection should be recomputed.

    Default is 180 days, which is roughly a 6-month refresh cycle.
    """
    if refresh_period_days <= 0:
        raise ValueError("refresh_period_days must be positive")

    if last_refresh_date is None:
        return True

    if isinstance(current_date, datetime):
        current_date = current_date.date()
    if isinstance(last_refresh_date, datetime):
        last_refresh_date = last_refresh_date.date()

    delta_days = (current_date - last_refresh_date).days
    return delta_days >= refresh_period_days


def should_refresh_forecast(
    current_date: date | datetime,
    last_forecast_date: date | datetime | None,
    refresh_period_days: int = 5,
    prediction_error: float | None = None,
    max_allowed_error: float = 0.02,
) -> bool:
    """
    Return True when OU / forecast-based predictions should be rebuilt.

    This supports two triggers:
    1. Time-based refresh after refresh_period_days
    2. Quality-based refresh when prediction_error exceeds max_allowed_error
    """
    if refresh_period_days <= 0:
        raise ValueError("refresh_period_days must be positive")
    if max_allowed_error <= 0:
        raise ValueError("max_allowed_error must be positive")

    if prediction_error is not None and float(prediction_error) >= max_allowed_error:
        return True

    if last_forecast_date is None:
        return True

    if isinstance(current_date, datetime):
        current_date = current_date.date()
    if isinstance(last_forecast_date, datetime):
        last_forecast_date = last_forecast_date.date()

    delta_days = (current_date - last_forecast_date).days
    return delta_days >= refresh_period_days