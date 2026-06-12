"""DST handling: Europe/Athens spring (23h) and autumn (25h) local days.

The pipeline stores UTC indices, so DST days must produce no gaps and no
duplicates in UTC, and resampling must stay aligned across transitions.
"""

import pandas as pd

from gr_epf import data

ATHENS = "Europe/Athens"
SPRING_DAY = "2025-03-30"  # 03:00 -> 04:00, local day has 23 hours
AUTUMN_DAY = "2025-10-26"  # 04:00 -> 03:00, local day has 25 hours


def local_day_index(day: str, freq: str) -> pd.DatetimeIndex:
    # build in UTC: tz-aware date_range steps in wall-clock time and would
    # mangle the skipped/repeated hour on DST days
    start = pd.Timestamp(day, tz=ATHENS)
    end = start + pd.DateOffset(days=1)
    rng = pd.date_range(
        start.tz_convert("UTC"), end.tz_convert("UTC"), freq=freq, inclusive="left"
    )
    return rng.tz_convert(ATHENS)


class TestSpringForward:
    def test_local_day_has_23_hours(self):
        idx = local_day_index(SPRING_DAY, "h")
        assert len(idx) == 23
        assert 3 not in idx.hour

    def test_utc_index_is_continuous(self):
        idx = local_day_index(SPRING_DAY, "h").tz_convert("UTC")
        steps = idx[1:] - idx[:-1]
        assert (steps == pd.Timedelta("1h")).all()

    def test_resample_over_transition(self):
        idx = local_day_index(SPRING_DAY, "15min")
        assert len(idx) == 23 * 4
        s = pd.Series(1.0, index=idx.tz_convert("UTC"))
        out = data.resample_to_hourly(s)
        assert len(out) == 23
        assert out.notna().all()


class TestFallBack:
    def test_local_day_has_25_hours(self):
        idx = local_day_index(AUTUMN_DAY, "h")
        assert len(idx) == 25
        assert (idx.hour == 3).sum() == 2

    def test_utc_index_has_no_duplicates(self):
        idx = local_day_index(AUTUMN_DAY, "h").tz_convert("UTC")
        assert not idx.duplicated().any()
        steps = idx[1:] - idx[:-1]
        assert (steps == pd.Timedelta("1h")).all()

    def test_resample_over_transition(self):
        idx = local_day_index(AUTUMN_DAY, "15min")
        assert len(idx) == 25 * 4
        s = pd.Series(2.0, index=idx.tz_convert("UTC"))
        out = data.resample_to_hourly(s)
        assert len(out) == 25
        assert (out == 2.0).all()
