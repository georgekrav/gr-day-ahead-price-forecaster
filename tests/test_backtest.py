"""Walk-forward backtest tests on synthetic data."""

import numpy as np
import pandas as pd

from gr_epf import evaluate

FAST_PARAMS = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}


def make_features(days: int, regime_break: pd.Timestamp | None = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, len(idx))
    target = 5 * a + rng.normal(0, 0.1, len(idx))
    feats = pd.DataFrame({"a": a, "target": target}, index=idx)
    if regime_break is not None:
        feats.loc[feats.index >= regime_break, "target"] = 100.0
    return feats


class TestWalkForward:
    def test_covers_test_span_exactly_once(self):
        feats = make_features(120)
        test_start = feats.index.max() - pd.DateOffset(months=2) + pd.Timedelta("1h")
        pred = evaluate.walk_forward_predictions(feats, test_start, params=FAST_PARAMS)
        expected = feats.index[feats.index >= test_start]
        assert (pred.index == expected).all()
        assert not pred.index.duplicated().any()

    def test_folds_trained_only_on_past(self):
        # target jumps to a constant 100 exactly at the test start; a model
        # trained without future data cannot know that and must keep
        # predicting from the old regime in the first fold
        feats = make_features(120)
        test_start = feats.index.max() - pd.DateOffset(months=2) + pd.Timedelta("1h")
        feats.loc[feats.index >= test_start, "target"] = 100.0
        pred = evaluate.walk_forward_predictions(feats, test_start, params=FAST_PARAMS)
        first_fold = pred[pred.index < test_start + pd.DateOffset(months=1)]
        assert first_fold.abs().max() < 50.0
        assert (first_fold - 100.0).abs().mean() > 50.0

    def test_rolling_window_restricts_training(self):
        feats = make_features(120)
        test_start = feats.index.max() - pd.DateOffset(months=1) + pd.Timedelta("1h")
        pred = evaluate.walk_forward_predictions(
            feats, test_start, train_window=pd.Timedelta(days=30), params=FAST_PARAMS
        )
        assert len(pred) == len(feats.index[feats.index >= test_start])
