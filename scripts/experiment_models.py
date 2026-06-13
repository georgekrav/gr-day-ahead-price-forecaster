"""Which model family wins? Walk-forward A/B across model types.

Same 12-month walk-forward as the headline result, with weather features.
Compares LightGBM (native NaN handling) against a LEAR-style linear model
and a small neural net (both need imputation + scaling), plus simple
ensembles. Conclusion (2026-06-13): LightGBM alone has the best MAE; the
neural net is worst; ensembles trade a little MAE for a little RMSE. The
single LightGBM is kept — added complexity is not justified for the
MAE headline.

Usage:
    python scripts/experiment_models.py
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gr_epf import data, evaluate, features, models, weather


def sk_pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def walk_forward(feats, test_start, x_cols, fit_predict):
    folds = []
    cur = test_start
    end = feats.index.max()
    while cur <= end:
        nxt = cur + pd.DateOffset(months=1)
        train = feats[feats.index < cur].dropna(subset=["target"])
        fold = feats[(feats.index >= cur) & (feats.index < nxt)]
        folds.append(pd.Series(fit_predict(train, fold, x_cols), index=fold.index))
        cur = nxt
    return pd.concat(folds)


def lgbm_fp(train, fold, x_cols):
    model = models.train_lightgbm(train, params={"n_jobs": 4})
    return models.predict_lightgbm(model, fold).to_numpy()


def sk_fp(estimator):
    def fp(train, fold, x_cols):
        pipe = sk_pipeline(estimator)
        pipe.fit(train[x_cols], train["target"])
        return pipe.predict(fold[x_cols])
    return fp


def main() -> None:
    df = data.load_processed()
    feats = features.build_features(df, weather=weather.load_weather())
    test_start = df.index.max() - pd.DateOffset(months=12) + pd.Timedelta("1h")
    x_cols = [c for c in feats.columns if c != "target"]

    def wf(fp):
        return walk_forward(feats, test_start, x_cols, fp)

    preds = {
        "lightgbm": wf(lgbm_fp),
        "ridge (LEAR-style)": wf(sk_fp(Ridge(alpha=10.0))),
        "mlp (neural net)": wf(
            sk_fp(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                               early_stopping=True, random_state=42))
        ),
    }
    preds["ensemble lgbm+ridge"] = (preds["lightgbm"] + preds["ridge (LEAR-style)"]) / 2
    preds["ensemble lgbm+ridge+mlp"] = sum(
        preds[k] for k in ("lightgbm", "ridge (LEAR-style)", "mlp (neural net)")
    ) / 3

    y = feats.loc[preds["lightgbm"].index, "target"]
    table = pd.DataFrame(
        {name: {"MAE": evaluate.mae(y, p), "RMSE": evaluate.rmse(y, p)}
         for name, p in preds.items()}
    ).T.sort_values("MAE")
    print(f"walk-forward, 12 months, weather features ({len(y)} hours)\n")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
