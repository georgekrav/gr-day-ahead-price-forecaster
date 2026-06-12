"""Forecast evaluation: metrics, comparison tables, walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gr_epf import models


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    if df.empty:
        raise ValueError("no overlapping non-NaN observations to score")
    return df


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = _aligned(y_true, y_pred)
    return float((df["y"] - df["yhat"]).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = _aligned(y_true, y_pred)
    return float(np.sqrt(((df["y"] - df["yhat"]) ** 2).mean()))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric MAPE in percent.

    Hours where |y| + |yhat| == 0 are skipped. With prices crossing zero
    sMAPE is unstable near zero, so it is reported for comparability with
    the EPF literature while MAE stays the headline metric.
    """
    df = _aligned(y_true, y_pred)
    denom = df["y"].abs() + df["yhat"].abs()
    err = (df["y"] - df["yhat"]).abs()[denom > 0]
    return float((200 * err / denom[denom > 0]).mean())


def metrics_table(y_true: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {
        name: {
            "MAE": mae(y_true, f),
            "RMSE": rmse(y_true, f),
            "sMAPE": smape(y_true, f),
        }
        for name, f in forecasts.items()
    }
    return pd.DataFrame(rows).T


def walk_forward_predictions(
    features_df: pd.DataFrame,
    test_start: pd.Timestamp,
    train_window: pd.Timedelta | None = None,
    params: dict | None = None,
) -> pd.Series:
    """Out-of-sample predictions for [test_start, end], retraining monthly.

    Each fold is predicted by a model trained only on rows strictly before
    the fold start — expanding window by default, or the last train_window
    of data when given. This mimics live operation: the model that prices
    a given day has never seen that day or anything after it.
    """
    folds = []
    cur = test_start
    end = features_df.index.max()
    while cur <= end:
        nxt = cur + pd.DateOffset(months=1)
        train_mask = features_df.index < cur
        if train_window is not None:
            train_mask &= features_df.index >= cur - train_window
        model = models.train_lightgbm(features_df[train_mask], params)
        fold = features_df[(features_df.index >= cur) & (features_df.index < nxt)]
        folds.append(models.predict_lightgbm(model, fold))
        cur = nxt
    return pd.concat(folds)


def metrics_by_hour(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """MAE and RMSE per hour of day, in the timezone of the input index."""
    df = _aligned(y_true, y_pred)
    err = df["y"] - df["yhat"]
    grouped = err.groupby(df.index.hour)
    return pd.DataFrame(
        {
            "MAE": grouped.apply(lambda e: e.abs().mean()),
            "RMSE": grouped.apply(lambda e: float(np.sqrt((e**2).mean()))),
        }
    ).rename_axis("hour")
