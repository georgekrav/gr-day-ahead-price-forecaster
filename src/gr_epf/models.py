"""Forecasting models: naive baselines and LightGBM.

Availability note: when bidding for day D (gate closure on D-1, 12:00 CET),
all day-ahead prices through the end of D-1 are public — they were set in
the D-2 auction. The baselines therefore use only information available at
issue time; feature availability for LightGBM is documented in features.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# l1 objective: MAE is the headline metric and price spikes (up to 942
# EUR/MWh) would dominate an l2 fit. Fixed seed for reproducibility.
DEFAULT_PARAMS = {
    "objective": "l1",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "random_state": 42,
    "verbosity": -1,
}


def naive_forecast(prices: pd.Series, lag_hours: int) -> pd.Series:
    """Forecast each hour with the price lag_hours earlier.

    The lag is applied through the index, not by row position, so gaps in
    the data cannot silently misalign the forecast.
    """
    shifted = prices.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=lag_hours)
    return shifted.reindex(prices.index)


def naive_24h(prices: pd.Series) -> pd.Series:
    """Same hour yesterday."""
    return naive_forecast(prices, 24)


def seasonal_naive_168h(prices: pd.Series) -> pd.Series:
    """Same hour and weekday, one week back."""
    return naive_forecast(prices, 168)


@dataclass
class PriceModel:
    booster: lgb.Booster

    @property
    def feature_columns(self) -> list[str]:
        return self.booster.feature_name()


def train_lightgbm(features_df: pd.DataFrame, params: dict | None = None) -> PriceModel:
    """Fit LightGBM on all rows with a known target.

    Rows with NaN features are kept — LightGBM routes missing values
    natively, so the handful of gap hours need no imputation.
    """
    train = features_df.dropna(subset=["target"])
    X = train.drop(columns=["target"])
    y = train["target"]
    regressor = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    regressor.fit(X, y)
    return PriceModel(regressor.booster_)


def predict_lightgbm(model: PriceModel, features_df: pd.DataFrame) -> pd.Series:
    X = features_df.drop(columns=["target"], errors="ignore")[model.feature_columns]
    return pd.Series(model.booster.predict(X), index=features_df.index, name="forecast")


def feature_importances(model: PriceModel) -> pd.Series:
    gains = model.booster.feature_importance(importance_type="gain")
    return pd.Series(gains, index=model.feature_columns).sort_values(ascending=False)


def save_model(model: PriceModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(path))
    return path


def load_model(path: Path) -> PriceModel:
    return PriceModel(lgb.Booster(model_file=str(path)))
