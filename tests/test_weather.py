"""Weather parsing tests (no network): multi-point averaging, column mapping."""


from gr_epf import weather


def _loc(times, **series):
    return {"hourly": {"time": list(times), **{k: list(v) for k, v in series.items()}}}


class TestParsePoints:
    def test_single_location_passthrough(self):
        payload = _loc(
            ["2025-01-01T00:00", "2025-01-01T01:00"],
            shortwave_radiation=[0.0, 100.0],
            wind_speed_100m=[10.0, 20.0],
        )
        df = weather._parse_points(payload, ("shortwave_radiation", "wind_speed_100m"))
        assert list(df.columns) == ["shortwave_radiation", "wind_speed_100m"]
        assert df["wind_speed_100m"].tolist() == [10.0, 20.0]
        assert df.index.tz is not None

    def test_multi_location_averaged(self):
        times = ["2025-01-01T00:00", "2025-01-01T01:00"]
        payload = [
            _loc(times, shortwave_radiation=[0.0, 100.0]),
            _loc(times, shortwave_radiation=[200.0, 300.0]),
        ]
        df = weather._parse_points(payload, ("shortwave_radiation",))
        assert df["shortwave_radiation"].tolist() == [100.0, 200.0]
        assert len(df) == 2

    def test_index_is_utc_and_sorted(self):
        payload = _loc(
            ["2025-01-01T01:00", "2025-01-01T00:00"],
            temperature_2m=[5.0, 4.0],
        )
        df = weather._parse_points(payload, ("temperature_2m",))
        assert df.index.is_monotonic_increasing
        assert str(df.index.tz) == "UTC"


def test_feature_map_covers_all_variables():
    assert set(weather.FEATURE_MAP) == set(weather.VARIABLES)
    assert all(v.startswith("wx_") for v in weather.FEATURE_MAP.values())
