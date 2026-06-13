"""Does weather improve the forecast? Walk-forward A/B over feature sets.

Each configuration is evaluated with the exact same 12-month walk-forward
protocol as the headline result (monthly retrain, expanding window), so the
MAE numbers are directly comparable to the no-weather baseline.

Usage:
    python scripts/experiment_weather.py
"""

import pandas as pd

from gr_epf import data, evaluate, features, models, weather

WX = {
    "solar": ["wx_solar_rad"],
    "wind": ["wx_wind_100m"],
    "temp": ["wx_temp_2m"],
    "cloud": ["wx_cloud"],
}
CONFIGS = {
    "baseline (no weather)": [],
    "+ solar": WX["solar"],
    "+ wind": WX["wind"],
    "+ temp": WX["temp"],
    "+ solar + wind": WX["solar"] + WX["wind"],
    "+ solar + wind + temp": WX["solar"] + WX["wind"] + WX["temp"],
    "+ all weather": WX["solar"] + WX["wind"] + WX["temp"] + WX["cloud"],
}


def main() -> None:
    df = data.load_processed()
    wx = weather.load_weather()
    prices = df["price_eur_mwh"]
    test_start = df.index.max() - pd.DateOffset(months=12) + pd.Timedelta("1h")

    naive = models.naive_24h(prices)
    base_idx = None
    rows = {}
    for name, cols in CONFIGS.items():
        feats = features.build_features(df, weather=wx[cols] if cols else None)
        pred = evaluate.walk_forward_predictions(feats, test_start)
        base_idx = pred.index if base_idx is None else base_idx
        y = feats.loc[pred.index, "target"]
        rows[name] = {
            "MAE": evaluate.mae(y, pred),
            "RMSE": evaluate.rmse(y, pred),
            "sMAPE": evaluate.smape(y, pred),
        }

    table = pd.DataFrame(rows).T
    base_mae = table.loc["baseline (no weather)", "MAE"]
    table["MAE_vs_base_%"] = (table["MAE"] / base_mae - 1) * 100
    naive_mae = evaluate.mae(prices[base_idx], naive[base_idx])

    print(f"walk-forward, 12 months, monthly retrain ({len(base_idx)} hours)")
    print(f"naive-24h MAE on this window: {naive_mae:.2f}\n")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
