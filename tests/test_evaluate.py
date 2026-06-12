"""Metric correctness on hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from gr_epf import evaluate


def series(values: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


class TestMetrics:
    def test_mae_rmse_hand_case(self):
        y = series([10.0, 20.0])
        yhat = series([11.0, 17.0])
        assert evaluate.mae(y, yhat) == 2.0
        assert evaluate.rmse(y, yhat) == pytest.approx(np.sqrt(5))

    def test_perfect_forecast_scores_zero(self):
        y = series([5.0, -3.0, 100.0])
        assert evaluate.mae(y, y) == 0.0
        assert evaluate.rmse(y, y) == 0.0
        assert evaluate.smape(y, y) == 0.0

    def test_smape_hand_case(self):
        y = series([100.0])
        yhat = series([50.0])
        assert evaluate.smape(y, yhat) == pytest.approx(200 * 50 / 150)

    def test_smape_skips_zero_denominator(self):
        y = series([0.0, 100.0])
        yhat = series([0.0, 100.0])
        assert evaluate.smape(y, yhat) == 0.0

    def test_nan_rows_excluded(self):
        y = series([10.0, np.nan, 30.0])
        yhat = series([12.0, 20.0, np.nan])
        assert evaluate.mae(y, yhat) == 2.0

    def test_no_overlap_raises(self):
        y = series([np.nan, np.nan])
        with pytest.raises(ValueError):
            evaluate.mae(y, y)


class TestTables:
    def test_metrics_table_rows(self):
        y = series([10.0, 20.0, 30.0])
        table = evaluate.metrics_table(y, {"a": y, "b": y + 1})
        assert list(table.index) == ["a", "b"]
        assert table.loc["a", "MAE"] == 0.0
        assert table.loc["b", "MAE"] == 1.0

    def test_metrics_by_hour(self):
        y = series([10.0] * 48)
        yhat = y.copy()
        yhat[yhat.index.hour == 7] += 5.0
        by_hour = evaluate.metrics_by_hour(y, yhat)
        assert by_hour.loc[7, "MAE"] == 5.0
        assert by_hour.loc[8, "MAE"] == 0.0
        assert len(by_hour) == 24
