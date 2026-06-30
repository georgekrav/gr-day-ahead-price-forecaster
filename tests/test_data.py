"""Tests for chunking, resampling, cache loading and dataset assembly."""

import numpy as np
import pandas as pd
import pytest
import requests
from entsoe.exceptions import NoMatchingDataError

from gr_epf import data

TZ = data.LOCAL_TZ


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


class TestMonthChunks:
    def test_full_and_partial_months(self):
        chunks = data.month_chunks(ts("2023-06-01"), ts("2023-09-15"))
        assert chunks == [
            (ts("2023-06-01"), ts("2023-07-01")),
            (ts("2023-07-01"), ts("2023-08-01")),
            (ts("2023-08-01"), ts("2023-09-01")),
            (ts("2023-09-01"), ts("2023-09-15")),
        ]

    def test_chunks_are_contiguous_and_cover_range(self):
        chunks = data.month_chunks(ts("2023-06-01"), ts("2026-06-12"))
        assert chunks[0][0] == ts("2023-06-01")
        assert chunks[-1][1] == ts("2026-06-12")
        for (_, a_end), (b_start, _) in zip(chunks, chunks[1:], strict=False):
            assert a_end == b_start

    def test_naive_timestamps_rejected(self):
        with pytest.raises(ValueError):
            data.month_chunks(pd.Timestamp("2023-06-01"), ts("2023-07-01"))


class TestResample:
    def test_quarter_hourly_means_to_hourly(self):
        idx = pd.date_range("2025-10-01 00:00", periods=8, freq="15min", tz="UTC")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 20.0, 20.0], index=idx)
        out = data.resample_to_hourly(s)
        assert len(out) == 2
        assert out.iloc[0] == 2.5
        assert out.iloc[1] == 15.0

    def test_hourly_is_identity(self):
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        s = pd.Series(np.arange(24.0), index=idx)
        out = data.resample_to_hourly(s)
        pd.testing.assert_series_equal(out, s, check_freq=False)

    def test_mixed_resolution_like_sdac_golive(self):
        hourly = pd.date_range("2025-09-30 20:00", periods=2, freq="h", tz="UTC")
        quarters = pd.date_range("2025-09-30 22:00", periods=8, freq="15min", tz="UTC")
        s = pd.concat(
            [
                pd.Series([100.0, 200.0], index=hourly),
                pd.Series([4.0] * 4 + [8.0] * 4, index=quarters),
            ]
        )
        out = data.resample_to_hourly(s)
        assert list(out.values) == [100.0, 200.0, 4.0, 8.0]

    def test_internal_gap_becomes_nan_not_filled(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        s = pd.Series(1.0, index=idx.delete(2))
        out = data.resample_to_hourly(s)
        assert len(out) == 5
        assert out.isna().sum() == 1
        assert pd.isna(out.iloc[2])


class TestLoadRaw:
    def _write_chunk(self, raw_dir, series, fname, start, values):
        idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
        df = pd.DataFrame({"price_eur_mwh": values}, index=idx)
        path = raw_dir / series / f"{fname}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def test_concat_sorted(self, tmp_path):
        self._write_chunk(tmp_path, "prices", "b", "2025-02-01", [3.0, 4.0])
        self._write_chunk(tmp_path, "prices", "a", "2025-01-01", [1.0, 2.0])
        df = data.load_raw("prices", tmp_path)
        assert df.index.is_monotonic_increasing
        assert list(df["price_eur_mwh"]) == [1.0, 2.0, 3.0, 4.0]

    def test_identical_duplicates_dropped(self, tmp_path):
        self._write_chunk(tmp_path, "prices", "a", "2025-01-01 00:00", [1.0, 2.0])
        self._write_chunk(tmp_path, "prices", "b", "2025-01-01 01:00", [2.0, 3.0])
        df = data.load_raw("prices", tmp_path)
        assert len(df) == 3

    def test_conflicting_duplicates_raise(self, tmp_path):
        self._write_chunk(tmp_path, "prices", "a", "2025-01-01 00:00", [1.0, 2.0])
        self._write_chunk(tmp_path, "prices", "b", "2025-01-01 01:00", [99.0, 3.0])
        with pytest.raises(ValueError, match="conflicting"):
            data.load_raw("prices", tmp_path)

    def test_empty_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            data.load_raw("prices", tmp_path)


class TestBuildHourlyDataset:
    @pytest.fixture
    def raw_dir(self, tmp_path):
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")

        def write(series, df):
            path = data.chunk_path(tmp_path, series, idx[0])
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)

        write("prices", pd.DataFrame({"price_eur_mwh": 50.0}, index=idx))
        write("load_actual", pd.DataFrame({"load_actual_mw": 5000.0}, index=idx))
        write("load_forecast", pd.DataFrame({"load_forecast_mw": 5100.0}, index=idx))
        gen = pd.DataFrame(
            {
                "Solar": 100.0,
                "Wind Onshore": 200.0,
                "Fossil Gas": 300.0,
                "Hydro Water Reservoir": 40.0,
                "Hydro Run-of-river and poundage": 10.0,
            },
            index=idx,
        )
        gen.loc[idx[5], "Hydro Run-of-river and poundage"] = np.nan
        write("generation", gen)
        return tmp_path

    def test_columns_and_hydro_sum(self, raw_dir):
        df = data.build_hourly_dataset(raw_dir)
        assert list(df.columns) == [
            "price_eur_mwh",
            "load_actual_mw",
            "load_forecast_mw",
            "gen_solar_mw",
            "gen_wind_onshore_mw",
            "gen_fossil_gas_mw",
            "gen_hydro_mw",
        ]
        assert df["gen_hydro_mw"].iloc[0] == 50.0

    def test_partial_hydro_stays_nan(self, raw_dir):
        df = data.build_hourly_dataset(raw_dir)
        assert pd.isna(df["gen_hydro_mw"].iloc[5])

    def test_range_trim(self, raw_dir):
        start = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        end = pd.Timestamp("2025-01-01 20:00", tz="UTC")
        df = data.build_hourly_dataset(raw_dir, start=start, end=end)
        assert df.index.min() == start
        assert df.index.max() == end - pd.Timedelta("1h")

    def test_missing_generation_type_raises(self, tmp_path):
        idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
        gen = pd.DataFrame({"Solar": 1.0}, index=idx)
        path = data.chunk_path(tmp_path, "generation", idx[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        gen.to_parquet(path)
        with pytest.raises(ValueError, match="generation types missing"):
            data.build_hourly_dataset(tmp_path)


class TestDownloadRetry:
    """download_series rides out transient ENTSO-E 5xx, surfaces the rest."""

    def _http_error(self, status: int) -> requests.HTTPError:
        resp = requests.Response()
        resp.status_code = status
        return requests.HTTPError(f"{status} error", response=resp)

    def _patch(self, monkeypatch, side_effects):
        calls = {"n": 0}
        good = pd.DataFrame(
            {"price_eur_mwh": [1.0, 2.0]},
            index=pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        )

        def fake_fetch(client, series, start, end):
            effect = side_effects[min(calls["n"], len(side_effects) - 1)]
            calls["n"] += 1
            if effect is not None:
                raise effect
            return good

        monkeypatch.setattr(data, "fetch_chunk", fake_fetch)
        monkeypatch.setattr(data.time, "sleep", lambda _s: None)
        return calls

    def _run(self, tmp_path):
        return data.download_series(
            None, "prices", ts("2025-01-01"), ts("2025-02-01"), raw_dir=tmp_path
        )

    def test_transient_then_success(self, monkeypatch, tmp_path):
        calls = self._patch(
            monkeypatch, [self._http_error(503), self._http_error(503), None]
        )
        failures = self._run(tmp_path)
        assert failures == []
        assert calls["n"] == 3
        assert (tmp_path / "prices" / "2025-01-01.parquet").exists()

    def test_persistent_5xx_surfaces(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, [self._http_error(503)])
        failures = self._run(tmp_path)
        assert len(failures) == 1
        assert calls["n"] == data.FETCH_RETRIES
        assert not (tmp_path / "prices" / "2025-01-01.parquet").exists()

    def test_non_transient_not_retried(self, monkeypatch, tmp_path):
        calls = self._patch(monkeypatch, [self._http_error(400)])
        failures = self._run(tmp_path)
        assert len(failures) == 1
        assert calls["n"] == 1

    def test_past_no_data_surfaces(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, [NoMatchingDataError()])
        failures = self._run(tmp_path)
        assert len(failures) == 1

    def test_future_no_data_skipped(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, [NoMatchingDataError()])
        now = pd.Timestamp.now(tz=data.LOCAL_TZ)
        start = (now + pd.DateOffset(months=2)).normalize()
        end = start + pd.DateOffset(days=10)
        failures = data.download_series(None, "prices", start, end, raw_dir=tmp_path)
        assert failures == []
