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

import numpy as np
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


def adaptive_conformal(
    forecast: pd.Series,
    actual: pd.Series,
    level: float,
    gamma: float = 0.02,
    lookback: int = 24 * 90,
    warmup: int = 24 * 30,
) -> pd.DataFrame:
    """Adaptive Conformal Inference (Gibbs & Candes 2021), per-hour widths.

    Static conformal assumes the error distribution is stationary; under
    drift the empirical coverage slips below nominal. ACI tracks an
    effective miscoverage rate alpha_t and nudges it after every hour: a
    miss widens the next interval, a hit narrows it, keeping realized
    coverage close to the target regardless of drift. Width still comes
    from the per-hour-of-day pool of recent absolute residuals, so the
    hour-to-hour structure is preserved.
    """
    s = pd.DataFrame({"f": forecast, "y": actual}).dropna().sort_index()
    err = (s["y"] - s["f"]).abs().to_numpy()
    fv, yv = s["f"].to_numpy(), s["y"].to_numpy()
    hour = s.index.tz_convert(LOCAL_TZ).hour.to_numpy()
    target = 1 - level
    alpha = target
    lo, hi = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for i in range(warmup, len(s)):
        lo_idx = max(0, i - lookback)
        mask = hour[lo_idx:i] == hour[i]
        pool = err[lo_idx:i][mask]
        if len(pool) == 0:
            continue
        width = float(np.quantile(pool, min(max(1 - alpha, 0.0), 1.0)))
        lo[i], hi[i] = fv[i] - width, fv[i] + width
        covered = lo[i] <= yv[i] <= hi[i]
        alpha += gamma * (target - (0.0 if covered else 1.0))
    return pd.DataFrame({"forecast": s["f"], "lo": lo, "hi": hi}, index=s.index)


def adaptive_alpha(
    forecast: pd.Series,
    actual: pd.Series,
    level: float,
    gamma: float = 0.02,
    lookback: int = 24 * 90,
    warmup: int = 24 * 30,
) -> float:
    """Run ACI over (forecast, actual) and return the converged miscoverage.

    The returned alpha is what the next interval should target; applying
    its (1 - alpha) quantile to fresh forecasts carries the drift
    correction forward without keeping per-hour online state.
    """
    s = pd.DataFrame({"f": forecast, "y": actual}).dropna().sort_index()
    err = (s["y"] - s["f"]).abs().to_numpy()
    fv, yv = s["f"].to_numpy(), s["y"].to_numpy()
    hour = s.index.tz_convert(LOCAL_TZ).hour.to_numpy()
    target = 1 - level
    alpha = target
    for i in range(warmup, len(s)):
        lo_idx = max(0, i - lookback)
        pool = err[lo_idx:i][hour[lo_idx:i] == hour[i]]
        if len(pool) == 0:
            continue
        width = float(np.quantile(pool, min(max(1 - alpha, 0.0), 1.0)))
        covered = fv[i] - width <= yv[i] <= fv[i] + width
        alpha += gamma * (target - (0.0 if covered else 1.0))
    return float(min(max(alpha, 0.0), 1.0))


def adaptive_hourly_quantiles(
    forecast: pd.Series,
    actual: pd.Series,
    levels: tuple[float, ...] = LEVELS,
    lookback: int = 24 * 90,
) -> pd.DataFrame:
    """Per-hour interval half-widths at the ACI-converged effective level.

    Drop-in replacement for hourly_quantiles that absorbs drift: instead of
    the nominal (1 - level) quantile it uses (1 - adaptive_alpha), then reads
    that quantile from the most recent `lookback` residuals per hour of day.
    """
    err = (actual - forecast).abs().dropna()
    recent = err.iloc[-lookback:]
    hours = recent.index.tz_convert(LOCAL_TZ).hour
    out = {}
    for level in levels:
        eff = 1 - adaptive_alpha(forecast, actual, level, lookback=lookback)
        out[_col(level)] = recent.groupby(hours).apply(
            lambda e, q=eff: float(np.quantile(e, q))
        )
    table = pd.DataFrame(out)
    table.index.name = "hour"
    return table
