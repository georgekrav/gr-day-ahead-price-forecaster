"""Forecasting models: naive baselines now, LightGBM in Phase 4.

Availability note: when bidding for day D (gate closure on D-1, 12:00 CET),
all day-ahead prices through the end of D-1 are public — they were set in
the D-2 auction. Both baselines therefore use only information available at
issue time.
"""

from __future__ import annotations

import pandas as pd


def naive_forecast(prices: pd.Series, lag_hours: int) -> pd.Series:
    """Forecast each hour with the price lag_hours earlier.

    The lag is applied through the index, not by row position, so gaps in
    the data cannot silently misalign the forecast.
    """
    shifted = prices.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=lag_hours)
    return shifted.reindex(prices.index)


def naive_24h(prices: pd.Series) -> pd.Series:
    """Same hour yesterday."""
    return naive_forecast(prices, 24)


def seasonal_naive_168h(prices: pd.Series) -> pd.Series:
    """Same hour and weekday, one week back."""
    return naive_forecast(prices, 168)
