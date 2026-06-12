"""Baseline forecast tests, including alignment and no-leakage checks."""

import numpy as np
import pandas as pd

from gr_epf import models


def make_prices(hours: int) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    return pd.Series(np.arange(hours, dtype=float), index=idx)


class TestNaiveForecast:
    def test_24h_maps_same_hour_yesterday(self):
        prices = make_prices(72)
        fc = models.naive_24h(prices)
        assert fc.iloc[:24].isna().all()
        pd.testing.assert_series_equal(
            fc.iloc[24:], prices.iloc[:-24].set_axis(fc.index[24:]), check_names=False
        )

    def test_168h_maps_one_week_back(self):
        prices = make_prices(24 * 8)
        fc = models.seasonal_naive_168h(prices)
        assert fc.iloc[:168].isna().all()
        assert fc.iloc[168] == prices.iloc[0]

    def test_gap_does_not_misalign(self):
        prices = make_prices(96).drop(make_prices(96).index[30:36])
        fc = models.naive_24h(prices)
        t = prices.index[-1]
        assert fc[t] == prices[t - pd.Timedelta(hours=24)]
        gap_plus_24h = pd.Timestamp("2025-01-03 08:00", tz="UTC")
        assert pd.isna(fc[gap_plus_24h])

    def test_no_leakage_future_change_leaves_past_forecasts_intact(self):
        prices = make_prices(72)
        perturbed = prices.copy()
        perturbed.iloc[-10:] += 1000.0
        fc_a = models.naive_24h(prices)
        fc_b = models.naive_24h(perturbed)
        pd.testing.assert_series_equal(fc_a.iloc[:-10], fc_b.iloc[:-10])
