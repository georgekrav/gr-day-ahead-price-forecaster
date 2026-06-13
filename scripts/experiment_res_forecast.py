"""Does the ENTSO-E day-ahead RES generation forecast help? Walk-forward A/B.

Same 12-month walk-forward as the headline result. Conclusion (2026-06-13):
the RES forecast is the strongest single feature added (-8% MAE, stable
across seeds) and largely subsumes the weather signal, since the operator
has already converted weather into expected MW. Weather is kept alongside
it as cheap redundancy.

Usage:
    python scripts/experiment_res_forecast.py
"""

import pandas as pd

from gr_epf import data, evaluate, features, weather


def main() -> None:
    df = data.load_processed()
    wx = weather.load_weather()
    res = data.load_res_forecast()
    test_start = df.index.max() - pd.DateOffset(months=12) + pd.Timedelta("1h")
    configs = {
        "lags only": {},
        "+ weather": {"weather": wx},
        "+ RES forecast": {"res_forecast": res},
        "+ weather + RES": {"weather": wx, "res_forecast": res},
    }
    print(f"walk-forward, 12 months ({'MAE':>8}{'RMSE':>8})")
    base = None
    for name, kw in configs.items():
        feats = features.build_features(df, **kw)
        pred = evaluate.walk_forward_predictions(feats, test_start, params={"n_jobs": 4})
        y = feats.loc[pred.index, "target"]
        mae, rmse = evaluate.mae(y, pred), evaluate.rmse(y, pred)
        base = base or mae
        print(f"{name:<20}{mae:>8.2f}{rmse:>8.2f}  ({(mae / base - 1) * 100:+.1f}% MAE)")


if __name__ == "__main__":
    main()
