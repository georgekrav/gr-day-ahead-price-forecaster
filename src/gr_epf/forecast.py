"""Daily forecast artifacts: history maintenance and app payloads.

The app and the daily job communicate only through small committed files
under forecasts/: latest.json (tomorrow), history.parquet (track record),
calibration.parquet (conformal widths), backtest_summary.json (headline
results). The app never trains or downloads anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gr_epf.data import LOCAL_TZ

HISTORY_COLUMNS = ["forecast", "lo_80", "hi_80", "lo_95", "hi_95", "actual"]


def update_history(
    history: pd.DataFrame | None, intervals: pd.DataFrame, prices: pd.Series
) -> pd.DataFrame:
    """Append forecast rows (newest run wins) and refresh actuals.

    Actuals are recomputed from the price series on every update: a
    day-ahead price is final once published, and NaN simply marks hours
    whose auction has not run yet.
    """
    frames = [history, intervals] if history is not None else [intervals]
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["actual"] = prices.reindex(out.index)
    return out[HISTORY_COLUMNS]


def refresh_actuals(history: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Refill the actual column from published prices, leaving forecasts as is.

    Used by the afternoon scoring job: once the day-ahead auction publishes,
    the track record can be updated the same day without re-issuing (and
    overwriting) the morning forecast.
    """
    out = history.copy()
    out["actual"] = prices.reindex(out.index)
    return out[HISTORY_COLUMNS]


def latest_payload(
    intervals: pd.DataFrame, generated_at: pd.Timestamp, target_day: str
) -> dict:
    local_times = intervals.index.tz_convert(LOCAL_TZ)
    rows = [
        {
            "time_local": ts.isoformat(),
            "forecast": round(float(r["forecast"]), 2),
            "lo_80": round(float(r["lo_80"]), 2),
            "hi_80": round(float(r["hi_80"]), 2),
            "lo_95": round(float(r["lo_95"]), 2),
            "hi_95": round(float(r["hi_95"]), 2),
        }
        for ts, (_, r) in zip(local_times, intervals.iterrows(), strict=True)
    ]
    return {
        "zone": "GR",
        "target_day": target_day,
        "generated_at_utc": generated_at.isoformat(),
        "rows": rows,
    }


def update_json_section(path: Path, section: str, payload: dict) -> None:
    """Merge one section into a JSON file shared by several scripts."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")
