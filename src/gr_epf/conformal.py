"""Split-conformal prediction intervals with per-hour calibration.

Hand-rolled rather than MAPIE: the method is a few dozen lines, and the
implementation is part of the portfolio story. Quantiles are calibrated
per hour of day (group-conditional conformal): errors differ structurally
between calm nights and volatile evening ramps, so a single global
quantile would over-cover one and under-cover the other.

Calibration must use out-of-sample residuals only — here, the walk-forward
backtest predictions. The finite-sample (n+1) quantile rule keeps the
coverage guarantee valid for finite calibration sets.
"""

from __future__ import annotations

import math

import pandas as pd

from gr_epf.data import LOCAL_TZ

LEVELS = (0.80, 0.95)


def conformal_quantile(abs_errors: pd.Series, level: float) -> float:
    """The ceil((n+1) * level)-th order statistic of the absolute errors."""
    clean = abs_errors.dropna()
    n = len(clean)
    rank = math.ceil((n + 1) * level)
    if n == 0 or rank > n:
        raise ValueError(f"need at least {rank} calibration errors, have {n}")
    return float(clean.sort_values().iloc[rank - 1])


def _col(level: float) -> str:
    return f"q{round(level * 100)}"


def hourly_quantiles(
    forecast: pd.Series,
    actual: pd.Series,
    levels: tuple[float, ...] = LEVELS,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Conformal interval half-widths per local hour of day."""
    err = (actual - forecast).abs().dropna()
    hours = err.index.tz_convert(tz).hour
    out = {
        _col(level): err.groupby(hours).apply(
            lambda e, lv=level: conformal_quantile(e, lv)
        )
        for level in levels
    }
    table = pd.DataFrame(out)
    table.index.name = "hour"
    return table


def apply_intervals(
    forecast: pd.Series, quantiles: pd.DataFrame, tz: str = LOCAL_TZ
) -> pd.DataFrame:
    """Symmetric intervals around the forecast, width chosen by local hour."""
    hours = forecast.index.tz_convert(tz).hour
    out = pd.DataFrame({"forecast": forecast})
    for col in quantiles.columns:
        width = quantiles[col].reindex(hours).to_numpy()
        out[f"lo_{col[1:]}"] = forecast - width
        out[f"hi_{col[1:]}"] = forecast + width
    return out


def coverage(actual: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """Share of hours whose actual price falls inside [lo, hi]."""
    df = pd.DataFrame({"y": actual, "lo": lo, "hi": hi}).dropna()
    if df.empty:
        raise ValueError("no overlapping observations")
    return float(((df["y"] >= df["lo"]) & (df["y"] <= df["hi"])).mean())
