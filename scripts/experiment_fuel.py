"""A/B test: do fuel signals (natural gas price, lignite generation) help?

Two candidate features, both leakage-checked against the D-1 12:00 CET gate:

  - gas_ttf_eur_mwh: the Dutch TTF gas front-month settlement (Yahoo Finance
    "TTF=F"), the marginal-fuel cost for the gas plants that usually set the
    Greek price. Daily series, lagged 2 calendar days: the settlement for day
    X publishes at the close of X, so the last value known at the D-1 morning
    issue time is X = D-2. Forward-filled over weekends/holidays.
  - gen_lignite_mw_lag_48h: lignite (brown coal) generation, lagged 48 h like
    every other generation feature (actuals publish too late for a 24 h lag).

Walk-forward MAE over the final 12 months for base / +gas / +lignite / +both.
Gas history is cached under data/gas/ (gitignored); pass --refresh to refetch.

Result (2026-06-18): neither helps. +gas is +0.65% MAE (worse), +lignite is
-0.06% (noise). The autoregressive price lags already encode the fuel-cost
level (corr price_lag_24h~price = 0.78 vs gas~price = 0.18), so the slow gas
series adds noise, not signal, at the day-ahead horizon. Not integrated.

Requires yfinance (pip install yfinance), used only by this experiment.

Usage:
    python scripts/experiment_fuel.py [--months 12] [--refresh]
"""

import argparse

import pandas as pd

from gr_epf import data, evaluate, features, models, weather
from gr_epf.data import LOCAL_TZ

GAS_PATH = data.REPO_ROOT / "data" / "gas" / "ttf.parquet"
LIGNITE_RAW = "Fossil Brown coal/Lignite"


def load_ttf(start: str, end: pd.Timestamp, refresh: bool = False) -> pd.Series:
    if GAS_PATH.exists() and not refresh:
        return pd.read_parquet(GAS_PATH)["gas_ttf_eur_mwh"]
    import yfinance as yf

    raw = yf.download("TTF=F", start=start, end=end.strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = "gas_ttf_eur_mwh"
    GAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    close.to_frame().to_parquet(GAS_PATH)
    return close


def gas_feature(index: pd.DatetimeIndex, ttf: pd.Series) -> pd.Series:
    """Daily TTF mapped to the hourly UTC index, lagged 2 days, forward-filled.

    Lagging the settlement date by 2 days guarantees the value attributed to
    electricity delivery day D was known by the D-1 morning forecast.
    """
    shifted = ttf.copy()
    shifted.index = shifted.index + pd.Timedelta(days=2)
    cal = pd.date_range(shifted.index.min(), shifted.index.max(), freq="D")
    daily = shifted[~shifted.index.duplicated()].reindex(cal).ffill()
    local_dates = pd.DatetimeIndex(index.tz_convert(LOCAL_TZ).normalize().tz_localize(None))
    return pd.Series(daily.reindex(local_dates).to_numpy(), index=index,
                     name="gas_ttf_eur_mwh")


def lignite_feature(index: pd.DatetimeIndex) -> pd.Series:
    gen = data.load_raw("generation")
    hourly = data.resample_to_hourly(gen[LIGNITE_RAW])
    lagged = models.naive_forecast(hourly, features.GEN_LAG_H)
    return lagged.reindex(index).rename(f"gen_lignite_mw_lag_{features.GEN_LAG_H}h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    df = data.load_processed()
    base = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )
    ttf = load_ttf(data.DATASET_START, df.index.max(), refresh=args.refresh)
    gas = gas_feature(df.index, ttf)
    lignite = lignite_feature(df.index)

    sets = {
        "base (current)": base,
        "+ gas price": base.assign(gas_ttf_eur_mwh=gas),
        "+ lignite gen": base.assign(**{lignite.name: lignite}),
        "+ both": base.assign(gas_ttf_eur_mwh=gas, **{lignite.name: lignite}),
    }
    test_start = df.index.max() - pd.DateOffset(months=args.months) + pd.Timedelta("1h")
    print(f"walk-forward MAE/RMSE, last {args.months} months "
          f"({test_start.date()} -> {df.index.max().date()})\n")
    base_mae = None
    for name, feats in sets.items():
        pred = evaluate.walk_forward_predictions(feats, test_start)
        y = feats.loc[pred.index, "target"]
        mae = evaluate.mae(y, pred)
        rmse = evaluate.rmse(y, pred)
        if base_mae is None:
            base_mae = mae
        delta = f"{(mae - base_mae) / base_mae:+.2%}" if base_mae else ""
        print(f"  {name:18s}  MAE {mae:6.3f}  RMSE {rmse:6.3f}  ({delta} vs base)")


if __name__ == "__main__":
    main()
