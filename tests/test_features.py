"""Feature generation tests: alignment, calendar, and no-leakage guarantees."""

import numpy as np
import pandas as pd

from gr_epf import features


def make_dataset(hours: int = 24 * 10, start: str = "2025-03-20") -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price_eur_mwh": rng.normal(100, 20, hours),
            "load_actual_mw": rng.normal(5500, 800, hours),
            "load_forecast_mw": rng.normal(5500, 800, hours),
            "gen_solar_mw": rng.uniform(0, 4000, hours),
            "gen_wind_onshore_mw": rng.uniform(0, 3000, hours),
            "gen_fossil_gas_mw": rng.uniform(500, 5000, hours),
            "gen_hydro_mw": rng.uniform(0, 2000, hours),
        },
        index=idx,
    )


class TestStructure:
    def test_columns_match_declared_list(self):
        out = features.build_features(make_dataset())
        assert list(out.columns) == features.FEATURE_COLUMNS + ["target"]
        assert (out.index == make_dataset().index).all()

    def test_price_lags_align_by_timestamp(self):
        df = make_dataset()
        out = features.build_features(df)
        t = df.index[200]
        for h in features.PRICE_LAGS_H:
            source = t - pd.Timedelta(hours=h)
            assert out.loc[t, f"price_lag_{h}h"] == df.loc[source, "price_eur_mwh"]

    def test_generation_lag_aligns_by_timestamp(self):
        df = make_dataset()
        out = features.build_features(df)
        t = df.index[200]
        source = t - pd.Timedelta(hours=features.GEN_LAG_H)
        assert out.loc[t, "gen_solar_mw_lag_48h"] == df.loc[source, "gen_solar_mw"]


class TestCalendar:
    def test_hour_and_weekday_are_local_time(self):
        # 2025-01-15 22:00 UTC is 2025-01-16 00:00 in Athens (EET, UTC+2)
        df = make_dataset(hours=3, start="2025-01-15 22:00")
        out = features.build_features(df)
        assert out["hour"].iloc[0] == 0
        assert out["weekday"].iloc[0] == 3

    def test_greek_national_holiday_flagged(self):
        df = make_dataset(hours=48, start="2025-03-24 22:00")
        out = features.build_features(df)
        local = out.index.tz_convert("Europe/Athens")
        on_holiday = out.loc[local.date == pd.Timestamp("2025-03-25").date(), "is_holiday"]
        day_after = out.loc[local.date == pd.Timestamp("2025-03-26").date(), "is_holiday"]
        assert (on_holiday == 1).all()
        assert (day_after == 0).all()


class TestWeather:
    def _weather(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        return pd.DataFrame(
            {
                "wx_solar_rad": rng.uniform(0, 900, len(idx)),
                "wx_wind_100m": rng.uniform(0, 40, len(idx)),
            },
            index=idx,
        )

    def test_weather_columns_appended_after_base(self):
        df = make_dataset()
        out = features.build_features(df, weather=self._weather(df.index))
        assert list(out.columns) == features.FEATURE_COLUMNS + [
            "wx_solar_rad", "wx_wind_100m", "target",
        ]

    def test_weather_is_same_hour_not_lagged(self):
        # weather is a day-ahead forecast, so same-hour use is legitimate
        df = make_dataset()
        wx = self._weather(df.index)
        out = features.build_features(df, weather=wx)
        t = df.index[100]
        assert out.loc[t, "wx_solar_rad"] == wx.loc[t, "wx_solar_rad"]

    def test_weather_future_does_not_leak_into_past(self):
        df = make_dataset()
        wx = self._weather(df.index)
        cutoff = df.index[120]
        perturbed = wx.copy()
        perturbed.loc[perturbed.index > cutoff] += 10_000
        a = features.build_features(df, weather=wx)
        b = features.build_features(df, weather=perturbed)
        before = a.index <= cutoff
        for col in ("wx_solar_rad", "wx_wind_100m"):
            pd.testing.assert_series_equal(a.loc[before, col], b.loc[before, col])


class TestNoLeakage:
    """Perturbing data after a cutoff must not change features before it."""

    def test_price_lags_blind_to_the_future(self):
        df = make_dataset()
        cutoff = df.index[120]
        perturbed = df.copy()
        perturbed.loc[perturbed.index > cutoff, "price_eur_mwh"] += 10_000
        a = features.build_features(df)
        b = features.build_features(perturbed)
        before = a.index <= cutoff
        for h in features.PRICE_LAGS_H:
            col = f"price_lag_{h}h"
            pd.testing.assert_series_equal(a.loc[before, col], b.loc[before, col])

    def test_generation_lags_blind_to_the_future(self):
        df = make_dataset()
        cutoff = df.index[120]
        perturbed = df.copy()
        gen_cols = list(features.GEN_COLUMNS)
        perturbed.loc[perturbed.index > cutoff, gen_cols] += 10_000
        a = features.build_features(df)
        b = features.build_features(perturbed)
        before = a.index <= cutoff
        for col in features.FEATURE_COLUMNS:
            if col.startswith("gen_"):
                pd.testing.assert_series_equal(a.loc[before, col], b.loc[before, col])

    def test_generation_lag_is_at_least_48h(self):
        # actual generation for D-1 afternoon is unpublished at gate closure,
        # so any lag below 48h would leak; guard the constant itself
        assert features.GEN_LAG_H >= 48

    def test_no_feature_reads_same_hour_actuals(self):
        df = make_dataset()
        t = df.index[200]
        perturbed = df.copy()
        actual_cols = ["price_eur_mwh", "load_actual_mw", *features.GEN_COLUMNS]
        perturbed.loc[t, actual_cols] = 99_999.0
        a = features.build_features(df)
        b = features.build_features(perturbed)
        row_a = a.loc[t].drop("target")
        row_b = b.loc[t].drop("target")
        pd.testing.assert_series_equal(row_a, row_b)
