"""Forecast evaluation: MAE, RMSE, sMAPE, comparison tables, per-hour errors.

The walk-forward backtest itself arrives in Phase 5; these metrics are the
shared vocabulary for it and for the baseline bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
