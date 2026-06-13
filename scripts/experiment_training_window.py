"""Does a short rolling training window beat the expanding window?

arXiv 2506.10536 found 45-60 day windows best for Greece in 2023. We test
that on the harder 2025-26 window with the final feature set. Conclusion
(2026-06-13): the opposite holds for us — MAE falls monotonically as the
window grows, and the expanding window wins (16.85 vs 19.06 at 45d). With a
richer feature set and a more volatile market, more history is pure signal;
monthly retraining already handles the drift.

Usage:
    python scripts/experiment_training_window.py
"""

import pandas as pd

from gr_epf import data, evaluate, features, weather


def main() -> None:
    df = data.load_processed()
    feats = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )
    test_start = df.index.max() - pd.DateOffset(months=12) + pd.Timedelta("1h")
    print(f"{'window':>10}{'MAE':>9}{'RMSE':>9}")
    for w in (45, 60, 90, 120, 180, 365, None):
        tw = pd.Timedelta(days=w) if w else None
        pred = evaluate.walk_forward_predictions(
            feats, test_start, train_window=tw, params={"n_jobs": 4}
        )
        y = feats.loc[pred.index, "target"]
        label = f"{w}d" if w else "expanding"
        print(f"{label:>10}{evaluate.mae(y, pred):>9.2f}{evaluate.rmse(y, pred):>9.2f}")


if __name__ == "__main__":
    main()
