"""Weather features from Open-Meteo (free, no API key).

Motivation: residual load (demand minus solar and wind) is the dominant
price driver, but the model only sees generation lagged 48 h. Weather is
forecast and available before gate closure, so it is the one forward-looking
signal for tomorrow's RES output that the lagged generation cannot provide.

Leakage note: the backtest uses ERA5 reanalysis from Open-Meteo's archive,
i.e. "actual" realized weather, as a PROXY for the day-ahead weather
forecast that would be available live. Day-ahead forecasts of irradiance,
wind and temperature are very accurate (a far easier problem than prices),
so the proxy is close to operational reality; it is mildly optimistic and
documented as such. The live job (make_forecast) uses the forecast API
instead, which is a genuine D-1 forecast with no leakage.

National aggregates are the mean over four load/RES centres (Athens,
Thessaloniki, Patras, Heraklion). All series are stored on a UTC index.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from gr_epf.data import DATASET_START, LOCAL_TZ, REPO_ROOT

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_DIR = REPO_ROOT / "data" / "weather"

# Four centres covering the bulk of Greek demand and RES capacity.
POINTS = {
    "athens": (37.98, 23.73),
    "thessaloniki": (40.64, 22.94),
    "patras": (38.25, 21.73),
    "heraklion": (35.34, 25.13),
}
VARIABLES = ("shortwave_radiation", "wind_speed_100m", "temperature_2m", "cloud_cover")
# Column names in the feature matrix; same-hour values (weather is forecast,
# so unlike generation it needs no lag).
FEATURE_MAP = {
    "shortwave_radiation": "wx_solar_rad",
    "wind_speed_100m": "wx_wind_100m",
    "temperature_2m": "wx_temp_2m",
    "cloud_cover": "wx_cloud",
}


def _parse_points(payload, variables: tuple[str, ...]) -> pd.DataFrame:
    """Mean over locations; Open-Meteo returns a list when multi-point."""
    locations = payload if isinstance(payload, list) else [payload]
    frames = []
    for loc in locations:
        h = loc["hourly"]
        idx = pd.to_datetime(h["time"], utc=True)
        frames.append(pd.DataFrame({v: h[v] for v in variables}, index=idx))
    stacked = pd.concat(frames)
    return stacked.groupby(level=0).mean().sort_index()


def fetch_archive(
    start: pd.Timestamp, end: pd.Timestamp, variables: tuple[str, ...] = VARIABLES
) -> pd.DataFrame:
    """ERA5 reanalysis over [start, end] for the national point set."""
    params = {
        "latitude": ",".join(str(lat) for lat, _ in POINTS.values()),
        "longitude": ",".join(str(lon) for _, lon in POINTS.values()),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    return _parse_points(resp.json(), variables)


def fetch_forecast(variables: tuple[str, ...] = VARIABLES) -> pd.DataFrame:
    """Genuine day-ahead forecast (today + next days) for live use."""
    params = {
        "latitude": ",".join(str(lat) for lat, _ in POINTS.values()),
        "longitude": ",".join(str(lon) for _, lon in POINTS.values()),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "forecast_days": 3,
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=120)
    resp.raise_for_status()
    return _parse_points(resp.json(), variables)


def download_archive(
    start: str = DATASET_START,
    end: pd.Timestamp | None = None,
    path: Path | None = None,
) -> Path:
    """Fetch the full archive history and cache it as one parquet file."""
    path = path or WEATHER_DIR / "archive.parquet"
    end = end or pd.Timestamp.now(tz=LOCAL_TZ).normalize() - pd.Timedelta(days=1)
    df = fetch_archive(pd.Timestamp(start, tz="UTC"), end.tz_convert("UTC"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info("weather archive: %d rows -> %s", len(df), path)
    return path


def load_weather(path: Path | None = None) -> pd.DataFrame:
    """Cached archive renamed to wx_* feature columns, UTC index.

    Used for backtest/training, where realized weather stands in for the
    day-ahead forecast (see module docstring).
    """
    path = path or WEATHER_DIR / "archive.parquet"
    df = pd.read_parquet(path)
    return df.rename(columns=FEATURE_MAP)


def build_live_weather() -> pd.DataFrame:
    """Archive history + genuine forecast for the upcoming days, wx_* columns.

    The forecast covers today and the next days (the target). The archive
    fills the history. On overlap the forecast wins. This is the only
    leakage-free path: the target day uses a real day-ahead forecast, never
    realized weather.
    """
    archive = fetch_archive(
        pd.Timestamp(DATASET_START, tz="UTC"),
        pd.Timestamp.now(tz="UTC").normalize(),
    )
    forecast = fetch_forecast()
    combined = pd.concat([archive, forecast])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined.rename(columns=FEATURE_MAP)
