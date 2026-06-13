"""Feature matrix for day-ahead price forecasting.

Forecast issue time: day D-1, before the SDAC gate closure (12:00 CET).
Every feature below documents when it becomes available relative to that
moment; tests/test_features.py asserts the alignment.

Availability per feature:
- price_lag_24h/48h/168h: the day-ahead price for any hour of D-1 or
  earlier was set in an auction at least one day before issue time.
- hour/weekday/month/is_holiday: deterministic from the calendar
  (Europe/Athens local time, Greek public holidays).
- load_forecast_mw: ENTSO-E publishes the day-ahead total load forecast
  for day D by ~09:00 CET on D-1, before gate closure.
- gen_*_lag_48h: actual generation is published with about an hour of
  delay, so at issue time on D-1 only day D-2 is fully complete. A 24h
  generation lag would leak: afternoon hours of D-1 are not yet published
  at 12:00 CET. Hence the minimum generation lag is 48h.
- wx_* (optional): weather is a day-ahead forecast available before gate
  closure, so it is used same-hour (no lag) — the one forward-looking
  signal for tomorrow's solar and wind output. See gr_epf.weather for the
  archive-as-proxy / live-forecast leakage handling.
"""

from __future__ import annotations

import holidays
import pandas as pd

from gr_epf.data import LOCAL_TZ
from gr_epf.models import naive_forecast

GR_HOLIDAYS = holidays.country_holidays("GR")
PRICE_LAGS_H = (24, 48, 168)
GEN_LAG_H = 48  # minimum non-leaking lag for published actual generation
GEN_COLUMNS = (
    "gen_solar_mw",
    "gen_wind_onshore_mw",
    "gen_hydro_mw",
    "gen_fossil_gas_mw",
)
FEATURE_COLUMNS = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "hour",
    "weekday",
    "month",
    "is_holiday",
    "load_forecast_mw",
    "gen_solar_mw_lag_48h",
    "gen_wind_onshore_mw_lag_48h",
    "gen_hydro_mw_lag_48h",
    "gen_fossil_gas_mw_lag_48h",
]


def build_features(
    df: pd.DataFrame, weather: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Feature matrix + 'target' column, on the same UTC index as df.

    Rows with NaN features are kept: LightGBM handles missing values
    natively and the known data gaps total ~30 hours in three years.

    If weather is given, its columns are joined same-hour (weather is a
    day-ahead forecast, so unlike generation it needs no lag) and appended
    after the base features. See gr_epf.weather for the leakage rationale.
    """
    local = df.index.tz_convert(LOCAL_TZ)
    price = df["price_eur_mwh"]
    out = pd.DataFrame(index=df.index)
    for h in PRICE_LAGS_H:
        out[f"price_lag_{h}h"] = naive_forecast(price, h)
    out["hour"] = local.hour
    out["weekday"] = local.weekday
    out["month"] = local.month
    out["is_holiday"] = pd.Series(
        [d in GR_HOLIDAYS for d in local.date], index=df.index, dtype=int
    )
    out["load_forecast_mw"] = df["load_forecast_mw"]
    for col in GEN_COLUMNS:
        out[f"{col}_lag_{GEN_LAG_H}h"] = naive_forecast(df[col], GEN_LAG_H)
    assert list(out.columns) == FEATURE_COLUMNS
    if weather is not None:
        wx = weather.reindex(df.index)
        for col in wx.columns:
            out[col] = wx[col]
    out["target"] = price
    return out
