"""Score the naive baselines on the last 12 months of the processed dataset.

This is the bar the Phase 4 model must beat; the definitive comparison is
the Phase 5 walk-forward backtest over the same window.
"""

import pandas as pd

from gr_epf import data, evaluate, models


def main() -> None:
    prices = data.load_processed()["price_eur_mwh"]
    cutoff = prices.index.max() - pd.DateOffset(months=12)
    window = prices[prices.index > cutoff]
    forecasts = {
        "naive_24h": models.naive_24h(prices)[window.index],
        "seasonal_naive_168h": models.seasonal_naive_168h(prices)[window.index],
    }
    print(f"window: {window.index.min()} -> {window.index.max()} ({len(window)} hours)")
    print(evaluate.metrics_table(window, forecasts).round(2).to_string())


if __name__ == "__main__":
    main()
