"""Walk-forward backtest: final 12 months, monthly retraining.

Usage:
    python scripts/backtest.py [--months 12] [--train-window-days N]

Expanding training window by default; pass --train-window-days for a fixed
rolling window. Saves out-of-sample predictions (reused by the Phase 6
conformal calibration) and the error-by-hour figure under data/reports/.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gr_epf import data, evaluate, features, forecast, models, weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--train-window-days", type=int, default=None)
    args = parser.parse_args()

    df = data.load_processed()
    feats = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )
    test_start = (
        feats.index.max() - pd.DateOffset(months=args.months) + pd.Timedelta("1h")
    )
    window = (
        pd.Timedelta(days=args.train_window_days) if args.train_window_days else None
    )
    prediction = evaluate.walk_forward_predictions(feats, test_start, train_window=window)

    y = feats.loc[prediction.index, "target"]
    prices = df["price_eur_mwh"]
    naive = models.naive_24h(prices)[prediction.index]
    seasonal = models.seasonal_naive_168h(prices)[prediction.index]
    label = f"rolling {args.train_window_days}d" if window else "expanding"
    print(f"backtest window: {prediction.index.min()} -> {prediction.index.max()}")
    print(f"training window: {label}, monthly retrain, {len(prediction)} hours")
    print()
    table = evaluate.metrics_table(
        y, {"lightgbm": prediction, "naive_24h": naive, "seasonal_naive_168h": seasonal}
    )
    print(table.round(2).to_string())

    y_local = y.tz_convert(data.LOCAL_TZ)
    by_hour_model = evaluate.metrics_by_hour(y_local, prediction.tz_convert(data.LOCAL_TZ))
    by_hour_naive = evaluate.metrics_by_hour(y_local, naive.tz_convert(data.LOCAL_TZ))
    print()
    print("MAE by hour of day (local):")
    comparison = pd.DataFrame(
        {"lightgbm": by_hour_model["MAE"], "naive_24h": by_hour_naive["MAE"]}
    )
    print(comparison.round(2).to_string())

    reports = data.REPO_ROOT / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    comparison.plot(ax=ax, lw=2)
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("MAE EUR/MWh")
    ax.set_title(f"Walk-forward MAE by hour ({label} window)")
    ax.grid(True)
    fig.savefig(reports / "backtest_error_by_hour.png", dpi=150, bbox_inches="tight")

    out = pd.DataFrame({"forecast": prediction, "actual": y})
    out.to_parquet(reports / "backtest_predictions.parquet")

    forecast.update_json_section(
        data.REPO_ROOT / "forecasts" / "backtest_summary.json",
        "metrics",
        {
            "window_start": str(prediction.index.min()),
            "window_end": str(prediction.index.max()),
            "training": f"{label}, monthly retrain",
            "table": json.loads(table.round(2).to_json(orient="index")),
        },
    )
    print(f"\npredictions and figure saved under {reports}")


if __name__ == "__main__":
    main()
