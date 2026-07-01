"""Monthly recalibration and release gate.

Run once a month (GitHub Actions, 1st of the month). The model itself is
retrained from scratch every morning by scripts/make_forecast.py, so this
job does NOT touch the live model weights. It refreshes the two artifacts
the daily job never recomputes:

  1. The conformal calibration widths (forecasts/calibration.parquet) that
     the daily forecast applies live -- recomputed from a fresh walk-forward
     backtest so the prediction intervals keep tracking the drifting error
     distribution.
  2. The published accuracy metrics (forecasts/backtest_summary.json) shown
     in the app.

Both are promoted only if the refreshed ("challenger") backtest clears the
release gate (gr_epf.governance): the model must still clearly beat the
naive baseline and stay within sanity bounds on the most recent
out-of-sample window. If the gate fails, the live artifacts are left
untouched and the job exits non-zero, so the daily forecast keeps using the
last-good calibration and a human is alerted via the failed CI run. Every
accepted run is appended to forecasts/model_health.json as an audit trail.

Usage:
    python scripts/monthly_revalidation.py [--months 12] [--dry-run]
"""

import argparse
import json
import sys

import pandas as pd

from gr_epf import (
    conformal,
    data,
    evaluate,
    features,
    forecast,
    governance,
    models,
    weather,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the gate and print the decision without writing artifacts",
    )
    args = parser.parse_args()

    # Refresh the raw cache exactly like the daily job: a warm cache refetches
    # only the current month, a cold CI runner rebuilds the full history.
    fetch_start = pd.Timestamp(data.DATASET_START, tz=data.LOCAL_TZ)
    now = pd.Timestamp.now(tz=data.LOCAL_TZ)
    client = data.get_client()
    failures = []
    for series in data.SERIES:
        failures += data.download_series(client, series, fetch_start, now)
    # A failed current-month refetch (a transient ENTSO-E error) is not fatal:
    # recalibration runs on the cached history and the release gate still
    # guards what gets promoted, so a stale-by-a-day window is harmless. Surface
    # it as a warning rather than failing the job.
    for chunk_start, err in failures:
        print(f"warning: fetch {chunk_start:%Y-%m} failed: {err}", file=sys.stderr)

    df = data.build_hourly_dataset()
    # The backtest uses the ERA5 weather archive (realized weather as a proxy
    # for the day-ahead forecast). It is cached under data/ (gitignored), so a
    # cold CI runner has to fetch it -- Open-Meteo, free, no key.
    if not (weather.WEATHER_DIR / "archive.parquet").exists():
        weather.download_archive()
    feats = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )

    # Challenger: a fresh walk-forward backtest over the last --months, the
    # same expanding-window monthly-retrain protocol as scripts/backtest.py.
    test_start = (
        feats.index.max() - pd.DateOffset(months=args.months) + pd.Timedelta("1h")
    )
    prediction = evaluate.walk_forward_predictions(feats, test_start)
    y = feats.loc[prediction.index, "target"]
    prices = df["price_eur_mwh"]
    naive = models.naive_24h(prices)[prediction.index]
    seasonal = models.seasonal_naive_168h(prices)[prediction.index]
    table = evaluate.metrics_table(
        y,
        {"lightgbm": prediction, "naive_24h": naive, "seasonal_naive_168h": seasonal},
    )
    model_mae = float(table.loc["lightgbm", "MAE"])
    naive_mae = float(table.loc["naive_24h", "MAE"])

    health_path = data.REPO_ROOT / "forecasts" / "model_health.json"
    baseline_mae = governance.load_baseline_mae(health_path)
    result = governance.evaluate_gate(model_mae, naive_mae, baseline_mae)

    print(f"challenger backtest: {prediction.index.min()} -> {prediction.index.max()}")
    print(table.round(2).to_string())
    print()
    baseline_str = f"{baseline_mae:.2f}" if baseline_mae is not None else "none (first run)"
    print(f"baseline MAE: {baseline_str}")
    print(f"gate: {'PASS' if result.passed else 'FAIL'}")
    for reason in result.reasons:
        print(f"  - {reason}")

    entry = {
        "evaluated_at": now.isoformat(),
        "window_start": str(prediction.index.min()),
        "window_end": str(prediction.index.max()),
        "mae": round(model_mae, 2),
        "rmse": round(float(table.loc["lightgbm", "RMSE"]), 2),
        "naive_mae": round(naive_mae, 2),
        "baseline_mae": round(baseline_mae, 2) if baseline_mae is not None else None,
        "status": "accepted" if result.passed else "rejected",
        "reasons": result.reasons,
    }

    if args.dry_run:
        print("\ndry-run: no artifacts written")
        print(json.dumps(entry, indent=2))
        sys.exit(0 if result.passed else 1)

    if not result.passed:
        # Keep the live artifacts; the failed CI run is the alert. The reject
        # is not committed -- the GitHub Actions log carries the reasons.
        print(
            "\ngate failed: live calibration and metrics left unchanged",
            file=sys.stderr,
        )
        sys.exit(1)

    # Promote the challenger: refresh the live conformal calibration, the
    # published metrics + coverage, and the health audit trail.
    live_quantiles = conformal.adaptive_hourly_quantiles(prediction, y)
    calib_path = data.REPO_ROOT / "forecasts" / "calibration.parquet"
    live_quantiles.to_parquet(calib_path)

    summary_path = data.REPO_ROOT / "forecasts" / "backtest_summary.json"
    forecast.update_json_section(
        summary_path,
        "metrics",
        {
            "window_start": str(prediction.index.min()),
            "window_end": str(prediction.index.max()),
            "training": "expanding, monthly retrain",
            "table": json.loads(table.round(2).to_json(orient="index")),
        },
    )
    forecast.update_json_section(
        summary_path, "conformal", conformal.coverage_summary(prediction, y)
    )
    governance.append_health_record(health_path, entry)
    print(f"\ngate passed: calibration -> {calib_path}; metrics + health updated")


if __name__ == "__main__":
    main()
