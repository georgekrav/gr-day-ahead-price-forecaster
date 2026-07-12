"""Forecast artifact helpers: history maintenance and payload format."""

from pathlib import Path

import numpy as np
import pandas as pd

from gr_epf import forecast


def make_intervals(start: str, hours: int, value: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "forecast": value,
            "lo_80": value - 10,
            "hi_80": value + 10,
            "lo_95": value - 20,
            "hi_95": value + 20,
        },
        index=idx,
    )


class TestUpdateHistory:
    def test_first_run_creates_history_with_nan_actuals(self):
        intervals = make_intervals("2026-06-13", 24)
        prices = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        out = forecast.update_history(None, intervals, prices)
        assert list(out.columns) == forecast.HISTORY_COLUMNS
        assert out["actual"].isna().all()

    def test_actuals_filled_once_published(self):
        intervals = make_intervals("2026-06-13", 24)
        history = forecast.update_history(None, intervals, pd.Series(dtype=float))
        prices = pd.Series(80.0, index=intervals.index[:12])
        out = forecast.update_history(history, make_intervals("2026-06-14", 24), prices)
        assert len(out) == 48
        assert out["actual"].notna().sum() == 12
        assert (out["actual"].dropna() == 80.0).all()

    def test_rerun_overwrites_same_day_forecast(self):
        first = make_intervals("2026-06-13", 24, value=100.0)
        second = make_intervals("2026-06-13", 24, value=110.0)
        history = forecast.update_history(None, first, pd.Series(dtype=float))
        out = forecast.update_history(history, second, pd.Series(dtype=float))
        assert len(out) == 24
        assert (out["forecast"] == 110.0).all()

    def test_old_forecasts_preserved(self):
        day1 = make_intervals("2026-06-13", 24, value=100.0)
        day2 = make_intervals("2026-06-14", 24, value=120.0)
        history = forecast.update_history(None, day1, pd.Series(dtype=float))
        out = forecast.update_history(history, day2, pd.Series(dtype=float))
        assert len(out) == 48
        assert (out.loc[day1.index, "forecast"] == 100.0).all()

    def test_refresh_actuals_fills_without_changing_forecast(self):
        intervals = make_intervals("2026-06-13", 24, value=100.0)
        history = forecast.update_history(None, intervals, pd.Series(dtype=float))
        prices = pd.Series(80.0, index=intervals.index[:12])
        out = forecast.refresh_actuals(history, prices)
        assert list(out.columns) == forecast.HISTORY_COLUMNS
        assert (out["forecast"] == 100.0).all()
        assert out["actual"].notna().sum() == 12
        assert (out["actual"].dropna() == 80.0).all()

    def test_history_json_shape_window_and_nulls(self):
        intervals = make_intervals("2026-06-13", 24, value=100.0)
        history = forecast.update_history(None, intervals, pd.Series(dtype=float))
        history = forecast.refresh_actuals(history, pd.Series(80.0, index=intervals.index[:12]))
        payload = forecast.history_json(history, days=90)
        assert len(payload["rows"]) == 24
        assert set(payload["rows"][0]) == {
            "time_local", "forecast", "lo_80", "hi_80", "lo_95", "hi_95", "actual",
        }
        assert payload["rows"][0]["actual"] == 80.0
        assert payload["rows"][-1]["actual"] is None

    def test_history_json_trims_to_recent_days(self):
        old = make_intervals("2026-01-01", 24, value=50.0)
        new = make_intervals("2026-06-13", 24, value=100.0)
        history = forecast.update_history(None, old, pd.Series(dtype=float))
        history = forecast.update_history(history, new, pd.Series(dtype=float))
        payload = forecast.history_json(history, days=90)
        assert len(payload["rows"]) == 24  # only the recent block survives the window


class TestLatestPayload:
    def test_payload_shape_and_rounding(self):
        intervals = make_intervals("2026-06-13", 24, value=100.123456)
        payload = forecast.latest_payload(
            intervals, generated_at=pd.Timestamp("2026-06-12 08:00", tz="UTC"),
            target_day="2026-06-13",
        )
        assert payload["zone"] == "GR"
        assert payload["target_day"] == "2026-06-13"
        assert len(payload["rows"]) == 24
        row = payload["rows"][0]
        assert row["forecast"] == 100.12
        assert "+03:00" in row["time_local"] or "+02:00" in row["time_local"]

    def test_json_section_merge(self, tmp_path):
        path = tmp_path / "summary.json"
        forecast.update_json_section(path, "metrics", {"a": 1})
        forecast.update_json_section(path, "conformal", {"b": 2})
        import json

        merged = json.loads(path.read_text())
        assert merged == {"metrics": {"a": 1}, "conformal": {"b": 2}}


class TestAppSmoke:
    def test_app_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
        at = AppTest.from_file(str(app_path)).run(timeout=30)
        assert not at.exception


def test_update_history_keeps_intervals(tmp_path):
    intervals = make_intervals("2026-06-13", 24)
    out = forecast.update_history(None, intervals, pd.Series(dtype=float))
    assert np.allclose(out["hi_95"] - out["lo_95"], 40.0)


class TestSufficientPriceAnchor:
    def _fold(self, present_hours: int) -> pd.DataFrame:
        idx = pd.date_range("2026-07-02", periods=24, freq="h", tz="UTC")
        lag = pd.Series(100.0, index=idx)
        lag.iloc[present_hours:] = np.nan
        return pd.DataFrame({"price_lag_24h": lag})

    def test_full_anchor_passes(self):
        assert forecast.sufficient_price_anchor(self._fold(24))

    def test_threshold_boundary(self):
        assert forecast.sufficient_price_anchor(self._fold(forecast.MIN_ANCHOR_HOURS))
        assert not forecast.sufficient_price_anchor(
            self._fold(forecast.MIN_ANCHOR_HOURS - 1)
        )

    def test_single_hour_rejected(self):
        # the 2026-07-01 case: a 1-of-24-hours anchor produced MAE 62
        assert not forecast.sufficient_price_anchor(self._fold(1))

    def test_empty_rejected(self):
        assert not forecast.sufficient_price_anchor(self._fold(0))
