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

    summary = conformal.coverage_summary(
        predicted,
        actual,
        warmup_months=args.warmup_months,
        calibration_window_days=args.calibration_window_days,
        recalibrate_days=args.recalibrate_days,
    )
    print(f"backtest window: {bt.index.min()} -> {bt.index.max()}")
    print(f"{len(bt)} hours, calibration walk-forward, warmup {args.warmup_months} months")
    print()
    for level in (80, 95):
        s = summary[str(level)]
        print(
            f"{level}% interval: empirical coverage {s['coverage']:.1%},"
            f" mean width {s['mean_width']:.1f} EUR/MWh"
        )
    print("adaptive conformal (ACI), single pass over the backtest:")
    for level in (80, 95):
        s = summary[f"{level}_adaptive"]
        print(
            f"{level}% interval: empirical coverage {s['coverage']:.1%},"
            f" mean width {s['mean_width']:.1f}"
        )
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
