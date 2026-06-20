"""Fill in actual prices for already-issued forecasts (same-day scoring).

Runs in the early afternoon, after the day-ahead auction for tomorrow has
published. It refetches the day-ahead prices and refreshes the actual column
of forecasts/history.parquet so the track record shows how today's forecast
did, without waiting for the next morning's run. It never touches the forecast
itself (latest.json) or the model — only the actual column of the track record.

Usage:
    python scripts/update_actuals.py
"""

import sys

import pandas as pd

from gr_epf import data, forecast


def main() -> None:
    history_path = data.REPO_ROOT / "forecasts" / "history.parquet"
    if not history_path.exists():
        print("no history yet; nothing to score", file=sys.stderr)
        return

    fetch_start = pd.Timestamp(data.DATASET_START, tz=data.LOCAL_TZ)
    # reach past today: the day-ahead prices for tomorrow are already published
    # (that is the whole point of scoring same-day), so the fetch must extend
    # beyond the wall clock to include them
    fetch_end = pd.Timestamp.now(tz=data.LOCAL_TZ).normalize() + pd.Timedelta(days=2)
    client = data.get_client()
    failures = data.download_series(client, "prices", fetch_start, fetch_end)
    if failures:
        for chunk_start, err in failures:
            print(f"FAILED {chunk_start:%Y-%m}: {err}", file=sys.stderr)
        sys.exit(1)

    prices = data.resample_to_hourly(data.load_raw("prices")["price_eur_mwh"])
    history = pd.read_parquet(history_path)
    updated = forecast.refresh_actuals(history, prices)
    updated.to_parquet(history_path)
    forecast.write_history_json(updated, data.REPO_ROOT / "forecasts" / "history.json")
    scored = int(updated["actual"].notna().sum())
    print(f"history: {len(updated)} rows, {scored} with actual prices")


if __name__ == "__main__":
    main()
