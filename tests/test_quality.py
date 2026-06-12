"""Tests for gap/resolution detection and price diagnostics."""

import pandas as pd

from gr_epf import quality


def hourly(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def quarters(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="15min", tz="UTC")


class TestSegments:
    def test_single_regular_segment(self):
        segs = quality.resolution_segments(hourly("2025-01-01", 10))
        assert len(segs) == 1
        assert segs[0].step == pd.Timedelta("1h")
        assert segs[0].n_steps == 9

    def test_contiguous_regime_change_is_not_a_gap(self):
        idx = hourly("2025-09-30 20:00", 3).append(quarters("2025-09-30 23:00", 8))
        segs = quality.resolution_segments(idx)
        steps = {s.step for s in segs if s.step in quality.DATA_STEPS}
        assert steps == {pd.Timedelta("1h"), pd.Timedelta("15min")}
        assert quality.find_gaps(segs) == []


class TestGaps:
    def test_single_missing_hour(self):
        idx = hourly("2025-01-01", 10).delete(4)
        gaps = quality.find_gaps(quality.resolution_segments(idx))
        assert len(gaps) == 1
        assert gaps[0].missing_hours == 1.0

    def test_multi_hour_hole(self):
        idx = hourly("2025-01-01", 24).delete([10, 11, 12])
        gaps = quality.find_gaps(quality.resolution_segments(idx))
        assert len(gaps) == 1
        assert gaps[0].missing_hours == 3.0
        assert gaps[0].after == pd.Timestamp("2025-01-01 09:00", tz="UTC")
        assert gaps[0].until == pd.Timestamp("2025-01-01 13:00", tz="UTC")

    def test_gap_in_quarter_data(self):
        idx = quarters("2025-10-01", 16).delete([6, 7])
        gaps = quality.find_gaps(quality.resolution_segments(idx))
        assert len(gaps) == 1
        assert gaps[0].missing_hours == 0.5

    def test_gap_at_regime_change(self):
        idx = hourly("2025-09-30 20:00", 3).append(quarters("2025-10-01 00:00", 8))
        gaps = quality.find_gaps(quality.resolution_segments(idx))
        assert len(gaps) == 1
        assert gaps[0].missing_hours == 1.0


class TestIrregular:
    def test_alternating_missing_flagged(self):
        idx = pd.date_range("2025-01-01", periods=6, freq="2h", tz="UTC")
        regions = quality.irregular_regions(quality.resolution_segments(idx))
        assert len(regions) == 1
        assert regions[0].step == pd.Timedelta("2h")


class TestNegativePrices:
    def test_episode_grouping(self):
        idx = hourly("2025-04-01", 12)
        price = pd.Series(
            [50, -1, -5, -2, 60, 70, -10, 80, 90, 100, 110, 120], index=idx, dtype=float
        )
        ep = quality.negative_price_episodes(price)
        assert len(ep) == 2
        assert list(ep["hours"]) == [3, 1]
        assert ep["min_price"].min() == -10.0

    def test_no_negatives(self):
        price = pd.Series(10.0, index=hourly("2025-04-01", 5))
        assert quality.negative_price_episodes(price).empty


class TestOutliers:
    def test_hard_bounds_counted(self):
        price = pd.Series([50.0] * 100 + [-200.0, 1500.0], index=hourly("2025-01-01", 102))
        s = quality.outlier_summary(price, hard_low=-150, hard_high=1000)
        assert s["n_below_hard"] == 1
        assert s["n_above_hard"] == 1
