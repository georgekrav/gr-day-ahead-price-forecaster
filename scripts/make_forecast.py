"""Daily forecast job: refresh data, retrain, write forecasts/ artifacts.

Usage:
    python scripts/make_forecast.py

Designed to run on day D in the morning, after the day-ahead load forecast
for D+1 is published (~10:00 CET) and before the SDAC gate closure
(12:00 CET), forecasting the 24 hours of day D+1. Steps: refetch the
current and previous month for every series, rebuild the hourly dataset,
retrain LightGBM on all available history, predict D+1, apply conformal
widths from forecasts/calibration.parquet, update forecasts/history.parquet
and forecasts/latest.json.
"""

import json
import sys

import pandas as pd

from gr_epf import conformal, data, features, forecast, models, weather


def main() -> None:
    now = pd.Timestamp.now(tz=data.LOCAL_TZ)
    target_start = now.normalize() + pd.DateOffset(days=1)
    target_end = target_start + pd.DateOffset(days=1)
    # full-history start: cached complete months are skipped, so a warm
    # cache refetches only the current month while a cold CI runner
    # rebuilds the whole dataset and trains on identical data
    fetch_start = pd.Timestamp(data.DATASET_START, tz=data.LOCAL_TZ)

    client = data.get_client()
    failures = []
    for series in data.SERIES:
        failures += data.download_series(client, series, fetch_start, target_end)
    # A failed refetch of the current chunk (a transient ENTSO-E error, or the
    # source not having published the latest day yet) is not fatal: the run
    # falls back to whatever the cache holds, and the data-sufficiency check
    # below decides whether a forecast can honestly be issued.
    for chunk_start, err in failures:
        print(f"warning: fetch {chunk_start:%Y-%m} failed: {err}", file=sys.stderr)

    df = data.build_hourly_dataset()
    target_index = pd.date_range(
        target_start.tz_convert("UTC"),
        target_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    df = df.reindex(df.index.union(target_index))
    try:
        wx = weather.build_live_weather()
    except Exception as exc:
        print(f"warning: weather unavailable ({exc!r}); predicting without it",
              file=sys.stderr)
        wx = None
    # res_forecast is downloaded above as part of SERIES and already covers
    # the target day (it is a day-ahead forecast)
    res = data.load_res_forecast()
    feats = features.build_features(df, weather=wx, res_forecast=res)
    fold = feats.loc[target_index]
    if fold["res_fc_solar_mw"].isna().all() and fold["res_fc_wind_mw"].isna().all():
        print(
            "warning: RES forecast for the target day not yet published;"
            " predicting through LightGBM NaN routing",
            file=sys.stderr,
        )
    if fold["price_lag_24h"].isna().all():
        # Without today's day-ahead prices the 24h lag feature is empty, so
        # there is no honest forecast to make. This happens when the source
        # has not published the latest day yet (a delay or a gap), not because
        # anything broke -- skip cleanly and let the next scheduled run pick it
        # up once the data lands, rather than failing the workflow.
        print(
            "skipping: today's day-ahead prices are not published yet;"
            " the next scheduled run will retry once they are available",
            file=sys.stderr,
        )
        return
    if fold["load_forecast_mw"].isna().all():
        print(
            "warning: load forecast for the target day not yet published;"
            " predicting through LightGBM NaN routing",
            file=sys.stderr,
        )

    model = models.train_lightgbm(feats)
    prediction = models.predict_lightgbm(model, fold)
    quantiles = pd.read_parquet(data.REPO_ROOT / "forecasts" / "calibration.parquet")
    intervals = conformal.apply_intervals(prediction, quantiles)

    history_path = data.REPO_ROOT / "forecasts" / "history.parquet"
    history = pd.read_parquet(history_path) if history_path.exists() else None
    updated = forecast.update_history(history, intervals, df["price_eur_mwh"])
    updated.to_parquet(history_path)
    forecast.write_history_json(updated, data.REPO_ROOT / "forecasts" / "history.json")

    payload = forecast.latest_payload(
        intervals,
        generated_at=pd.Timestamp.now(tz="UTC"),
        target_day=str(target_start.date()),
    )
    latest_path = data.REPO_ROOT / "forecasts" / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"forecast for {target_start.date()}: mean {prediction.mean():.1f},"
        f" min {prediction.min():.1f}, max {prediction.max():.1f} EUR/MWh"
    )
    print(f"history: {len(updated)} rows, latest written to {latest_path}")


if __name__ == "__main__":
    main()
