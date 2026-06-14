"""Conformal interval tests: quantile rule, grouping, coverage."""

import numpy as np
import pandas as pd
import pytest

from gr_epf import conformal


def hourly_index(periods: int, start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


class TestQuantile:
    def test_order_statistic_rule(self):
        errors = pd.Series(np.arange(1.0, 100.0))  # 1..99, n=99
        assert conformal.conformal_quantile(errors, 0.95) == 95.0
        assert conformal.conformal_quantile(errors, 0.80) == 80.0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="calibration"):
            conformal.conformal_quantile(pd.Series([1.0, 2.0]), 0.95)


class TestHourlyQuantiles:
    def test_widths_differ_by_hour(self):
        idx = hourly_index(24 * 60)
        rng = np.random.default_rng(3)
        forecast = pd.Series(100.0, index=idx)
        noise = rng.normal(0, 1, len(idx))
        noise[idx.tz_convert("Europe/Athens").hour == 20] *= 10
        actual = forecast + noise
        q = conformal.hourly_quantiles(forecast, actual)
        assert list(q.columns) == ["q80", "q95"]
        assert q.loc[20, "q95"] > 3 * q.loc[4, "q95"]

    def test_apply_intervals_uses_hour_specific_width(self):
        idx = hourly_index(48)
        forecast = pd.Series(100.0, index=idx)
        quantiles = pd.DataFrame(
            {"q80": 5.0, "q95": np.where(np.arange(24) == 20, 50.0, 10.0)},
            index=pd.RangeIndex(24, name="hour"),
        )
        out = conformal.apply_intervals(forecast, quantiles)
        local_hours = out.index.tz_convert("Europe/Athens").hour
        at_20 = out[local_hours == 20]
        elsewhere = out[local_hours == 4]
        assert (at_20["hi_95"] - at_20["lo_95"]).iloc[0] == 100.0
        assert (elsewhere["hi_95"] - elsewhere["lo_95"]).iloc[0] == 20.0


class TestAdaptiveConformal:
    def test_coverage_near_target_under_drift(self):
        # residual scale doubles partway through — static calibration would
        # under-cover; ACI should track close to nominal
        idx = hourly_index(24 * 200)
        rng = np.random.default_rng(8)
        forecast = pd.Series(100.0, index=idx)
        scale = np.where(np.arange(len(idx)) < len(idx) // 2, 5.0, 15.0)
        actual = forecast + rng.normal(0, 1, len(idx)) * scale
        iv = conformal.adaptive_conformal(
            forecast, actual, level=0.90, warmup=24 * 10
        )
        scored = iv.dropna()
        cov = conformal.coverage(actual[scored.index], scored["lo"], scored["hi"])
        assert 0.85 <= cov <= 0.95

    def test_warmup_rows_are_nan(self):
        idx = hourly_index(24 * 60)
        forecast = pd.Series(50.0, index=idx)
        actual = forecast + 1.0
        iv = conformal.adaptive_conformal(forecast, actual, level=0.8, warmup=24 * 30)
        assert iv["lo"].iloc[: 24 * 30].isna().all()


class TestCoverageSummary:
    def test_returns_static_and_adaptive_levels(self):
        idx = hourly_index(24 * 200)
        rng = np.random.default_rng(7)
        forecast = pd.Series(100.0, index=idx)
        actual = forecast + rng.normal(0, 5, len(idx))
        summary = conformal.coverage_summary(
            forecast, actual, warmup_months=1, calibration_window_days=60
        )
        assert summary["calibration_window_days"] == 60
        for key in ("80", "95", "80_adaptive", "95_adaptive"):
            assert 0.0 <= summary[key]["coverage"] <= 1.0
            assert summary[key]["mean_width"] > 0
        # the wider nominal level must cover at least as much as the narrower
        assert summary["95"]["coverage"] >= summary["80"]["coverage"]


class TestCoverage:
    def test_exact_fraction(self):
        idx = hourly_index(10)
        actual = pd.Series(np.arange(10.0), index=idx)
        lo = pd.Series(0.0, index=idx)
        hi = pd.Series(6.5, index=idx)
        assert conformal.coverage(actual, lo, hi) == 0.7

    def test_calibrated_intervals_cover_on_fresh_data(self):
        idx = hourly_index(24 * 200)
        rng = np.random.default_rng(4)
        forecast = pd.Series(100.0, index=idx)
        actual = forecast + rng.normal(0, 5, len(idx))
        half = len(idx) // 2
        q = conformal.hourly_quantiles(forecast.iloc[:half], actual.iloc[:half])
        out = conformal.apply_intervals(forecast.iloc[half:], q)
        cov95 = conformal.coverage(actual.iloc[half:], out["lo_95"], out["hi_95"])
        cov80 = conformal.coverage(actual.iloc[half:], out["lo_80"], out["hi_80"])
        assert 0.93 <= cov95 <= 0.97
        assert 0.77 <= cov80 <= 0.83
