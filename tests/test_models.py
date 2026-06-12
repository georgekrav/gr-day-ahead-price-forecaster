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


class TestLightGBM:
    def make_features(self, rows: int = 400) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, rows)
        b = rng.normal(0, 1, rows)
        return pd.DataFrame(
            {"a": a, "b": b, "target": 3 * a - 2 * b + rng.normal(0, 0.01, rows)},
            index=idx,
        )

    def test_learns_simple_relation(self):
        feats = self.make_features()
        model = models.train_lightgbm(feats, params={"n_estimators": 200})
        pred = models.predict_lightgbm(model, feats)
        assert (feats["target"] - pred).abs().mean() < 0.5
        assert (pred.index == feats.index).all()

    def test_nan_target_rows_excluded_from_fit(self):
        feats = self.make_features()
        feats.loc[feats.index[:50], "target"] = np.nan
        model = models.train_lightgbm(feats, params={"n_estimators": 50})
        assert model.feature_columns == ["a", "b"]

    def test_save_load_round_trip(self, tmp_path):
        feats = self.make_features()
        model = models.train_lightgbm(feats, params={"n_estimators": 50})
        path = models.save_model(model, tmp_path / "m.txt")
        restored = models.load_model(path)
        pd.testing.assert_series_equal(
            models.predict_lightgbm(model, feats),
            models.predict_lightgbm(restored, feats),
        )

    def test_importances_named_and_sorted(self):
        feats = self.make_features()
        model = models.train_lightgbm(feats, params={"n_estimators": 50})
        imp = models.feature_importances(model)
        assert set(imp.index) == {"a", "b"}
        assert imp.iloc[0] >= imp.iloc[1]
