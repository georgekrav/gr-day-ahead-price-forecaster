"""Train LightGBM on the processed dataset and compare against baselines.

Usage:
    python scripts/train_model.py [--holdout-days 90] [--out data/models/lgbm.txt]

Chronological split: the last --holdout-days are never seen in training.
This gives a quick honest read and a saved model artifact; the definitive
evaluation is the Phase 5 walk-forward backtest.
"""

import argparse
from pathlib import Path

import pandas as pd

from gr_epf import data, evaluate, features, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument(
        "--out", default=str(data.REPO_ROOT / "data" / "models" / "lgbm.txt")
    )
    args = parser.parse_args()

    df = data.load_processed()
    feats = features.build_features(df)
    cutoff = feats.index.max() - pd.Timedelta(days=args.holdout_days)
    train = feats[feats.index <= cutoff]
    holdout = feats[feats.index > cutoff]

    model = models.train_lightgbm(train)
    prediction = models.predict_lightgbm(model, holdout)

    prices = df["price_eur_mwh"]
    table = evaluate.metrics_table(
        holdout["target"],
        {
            "lightgbm": prediction,
            "naive_24h": models.naive_24h(prices)[holdout.index],
            "seasonal_naive_168h": models.seasonal_naive_168h(prices)[holdout.index],
        },
    )
    print(f"train:   {train.index.min()} -> {train.index.max()} ({len(train)} rows)")
    print(f"holdout: {holdout.index.min()} -> {holdout.index.max()} ({len(holdout)} rows)")
    print()
    print(table.round(2).to_string())
    print()
    print("feature importances (gain):")
    print(models.feature_importances(model).round(0).to_string())

    path = models.save_model(model, Path(args.out))
    print(f"\nmodel saved to {path}")


if __name__ == "__main__":
    main()
