"""Empirical coverage of conformal intervals on the walk-forward backtest.

Usage:
    python scripts/conformal_report.py [--warmup-months 3]

Walk-forward calibration: intervals for month M use only residuals from
before M; the first --warmup-months are calibration-only. Calibration is
restricted to the last --calibration-window-days of residuals because the
error distribution drifts (the duck curve keeps deepening) and stale
residuals under-estimate current uncertainty. Also writes
forecasts/calibration.parquet — per-hour quantiles over the chosen window
at the end of the backtest — which the daily forecast job applies live.
"""

import argparse

import pandas as pd

from gr_epf import conformal, data, forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-months", type=int, default=3)
    parser.add_argument("--calibration-window-days", type=int, default=90)
    parser.add_argument(
        "--recalibrate-days",
        type=int,
        default=1,
        help="recalibration cadence; 1 mirrors the daily live job",
    )
    args = parser.parse_args()

    bt = pd.read_parquet(data.REPO_ROOT / "data" / "reports" / "backtest_predictions.parquet")
    predicted, actual = bt["forecast"], bt["actual"]
    window = (
        pd.Timedelta(days=args.calibration_window_days)
        if args.calibration_window_days
        else None
    )

    folds = []
    cur = bt.index.min() + pd.DateOffset(months=args.warmup_months)
    end = bt.index.max()
    step = pd.Timedelta(days=args.recalibrate_days)
    while cur <= end:
        nxt = cur + step
        past = bt.index < cur
        if window is not None:
            past &= bt.index >= cur - window
        quantiles = conformal.hourly_quantiles(predicted[past], actual[past])
        fold = predicted[(bt.index >= cur) & (bt.index < nxt)]
        folds.append(conformal.apply_intervals(fold, quantiles))
        cur = nxt
    intervals = pd.concat(folds)
    y = actual[intervals.index]

    print(f"evaluation window: {intervals.index.min()} -> {intervals.index.max()}")
    print(f"{len(intervals)} hours, calibration walk-forward, warmup {args.warmup_months} months")
    print()
    summary = {}
    for level in (80, 95):
        cov = conformal.coverage(y, intervals[f"lo_{level}"], intervals[f"hi_{level}"])
        width = (intervals[f"hi_{level}"] - intervals[f"lo_{level}"]).mean()
        summary[str(level)] = {"coverage": round(cov, 3), "mean_width": round(width, 1)}
        print(f"{level}% interval: empirical coverage {cov:.1%}, mean width {width:.1f} EUR/MWh")
    summary["calibration_window_days"] = args.calibration_window_days
    print("adaptive conformal (ACI), single pass over the backtest:")
    for level in (0.80, 0.95):
        iv = conformal.adaptive_conformal(predicted, actual, level=level)
        scored = iv.dropna()
        cov = conformal.coverage(actual[scored.index], scored["lo"], scored["hi"])
        width = (scored["hi"] - scored["lo"]).mean()
        summary[f"{round(level * 100)}_adaptive"] = {
            "coverage": round(cov, 3), "mean_width": round(width, 1)
        }
        print(f"{level:.0%} interval: empirical coverage {cov:.1%}, mean width {width:.1f}")
    forecast.update_json_section(
        data.REPO_ROOT / "forecasts" / "backtest_summary.json", "conformal", summary
    )

    # Live calibration uses adaptive conformal: the ACI-converged effective
    # level absorbs drift, so live coverage tracks nominal more closely than
    # the static quantile (measured above).
    live_quantiles = conformal.adaptive_hourly_quantiles(predicted, actual)
    out = data.REPO_ROOT / "forecasts" / "calibration.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    live_quantiles.to_parquet(out)
    print(f"\nadaptive live calibration written to {out}")


if __name__ == "__main__":
    main()
