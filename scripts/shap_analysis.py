"""SHAP explainability for the final LightGBM model.

Trains on the full history with the production feature set, then uses a
TreeExplainer to attribute each prediction to its features. Produces two
figures under assets/:
  - shap_bar.png: global feature importance (mean absolute SHAP value)
  - shap_beeswarm.png: per-feature effect and direction

SHAP values are in the units of the target (EUR/MWh): a feature's SHAP
value is how many EUR/MWh it pushed that hour's prediction up or down
relative to the average prediction.

Usage:
    python scripts/shap_analysis.py [--sample 3000]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from gr_epf import data, features, models, weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args()

    df = data.load_processed()
    feats = features.build_features(
        df, weather=weather.load_weather(), res_forecast=data.load_res_forecast()
    )
    model = models.train_lightgbm(feats)
    X = feats.drop(columns=["target"])[model.feature_columns]
    # explain the most recent rows: the regime the live model operates in
    sample = X.dropna().tail(args.sample)

    explainer = shap.TreeExplainer(model.booster)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("Global feature importance (mean |SHAP|, EUR/MWh)")
    plt.gcf().set_size_inches(7, 4.5)
    plt.tight_layout()
    bar_path = data.REPO_ROOT / "assets" / "shap_bar.png"
    plt.savefig(bar_path, dpi=110, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, sample, show=False)
    plt.title("Feature effects (SHAP beeswarm)")
    plt.tight_layout()
    bee_path = data.REPO_ROOT / "assets" / "shap_beeswarm.png"
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=sample.columns
    ).sort_values(ascending=False)
    print(f"explained {len(sample)} recent hours\n")
    print("mean |SHAP| (EUR/MWh) per feature:")
    print(importance.round(2).to_string())
    print(f"\nfigures: {bar_path}\n         {bee_path}")


if __name__ == "__main__":
    main()
