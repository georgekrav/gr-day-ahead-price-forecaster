"""A/B test: explicit residual-load feature, ramp features, and LightGBM tuning.

Three candidate improvements, all leakage-safe (built only from the day-ahead
load and RES forecasts already used as features, both published before the
D-1 12:00 CET gate):

  - residual_load_fc = load_forecast - (res_fc_solar + res_fc_wind): the
    dominant price driver (EDA corr ~0.77). The pieces are already features,
    but a tree model cannot form the linear combination itself, so handing it
    the physical quantity directly may help.
  - residual_ramp_1h / _3h: the hourly / 3-hourly change of residual load,
    i.e. the steepness of the morning solar dive and the evening ramp -- the
    hours where the model misses most (RMSE >> MAE).
  - Optuna tuning: the model uses LightGBM defaults; tune depth/learning-rate/
    regularization on a pre-test validation split, then score the tuned params
    on the same walk-forward window. The test window is never seen in tuning.

Walk-forward MAE/RMSE over the final 12 months. Nothing is written or
committed -- this only reports whether the changes help.

Requires optuna (pip install optuna), used only by this experiment.

Usage:
    python scripts/experiment_residual_tuning.py [--months 12] [--trials 40]
"""

import argparse

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from gr_epf import data, evaluate, features, weather


def add_residual(feats: pd.DataFrame) -> pd.Series:
    return feats["load_forecast_mw"] - (
        feats["res_fc_solar_mw"] + feats["res_fc_wind_mw"]
    )


def tune(feats: pd.DataFrame, test_start: pd.Timestamp, n_trials: int) -> dict:
    """Tune on data strictly before test_start (chronological train/valid)."""
    pre = feats[feats.index < test_start].dropna(subset=["target"])
    cut = pre.index.max() - pd.DateOffset(months=12)
    train, valid = pre[pre.index <= cut], pre[pre.index > cut]
    x_tr, y_tr = train.drop(columns="target"), train["target"]
    x_v, y_v = valid.drop(columns="target"), valid["target"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "l1",
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params).fit(x_tr, y_tr)
        return float(np.abs(y_v.to_numpy() - model.predict(x_v)).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def score(feats: pd.DataFrame, test_start: pd.Timestamp, params=None) -> tuple:
    pred = evaluate.walk_forward_predictions(feats, test_start, params=params)
    y = feats.loc[pred.index, "target"]
    return evaluate.mae(y, pred), evaluate.rmse(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()

    df = data.load_processed()
    base = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )
    resid = add_residual(base)
    base_resid = base.assign(residual_load_fc=resid)
    base_ramp = base_resid.assign(
        residual_ramp_1h=resid.diff(1), residual_ramp_3h=resid.diff(3)
    )
    test_start = df.index.max() - pd.DateOffset(months=args.months) + pd.Timedelta("1h")

    print(f"walk-forward, last {args.months} months "
          f"({test_start.date()} -> {df.index.max().date()})\n")
    variants = {
        "base (current)": base,
        "+ residual_load": base_resid,
        "+ residual + ramp": base_ramp,
    }
    base_mae = None
    best_name, best_feats = "base (current)", base
    best_mae = None
    for name, feats in variants.items():
        mae, rmse = score(feats, test_start)
        if base_mae is None:
            base_mae, best_mae = mae, mae
        if mae < best_mae:
            best_mae, best_name, best_feats = mae, name, feats
        print(f"  {name:20s}  MAE {mae:6.3f}  RMSE {rmse:6.3f}  "
              f"({(mae - base_mae) / base_mae:+.2%} vs base)")

    print(f"\ntuning LightGBM ({args.trials} Optuna trials) on '{best_name}' "
          "features, validated pre-test...")
    best_params = tune(best_feats, test_start, args.trials)
    mae_t, rmse_t = score(best_feats, test_start, params=best_params)
    print(f"  + tuned params       MAE {mae_t:6.3f}  RMSE {rmse_t:6.3f}  "
          f"({(mae_t - base_mae) / base_mae:+.2%} vs base)")
    print("\nbest tuned params:")
    for k, v in best_params.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
