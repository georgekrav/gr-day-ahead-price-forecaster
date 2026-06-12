"""Download GR series from ENTSO-E and build the processed hourly dataset.

Usage:
    python scripts/download_data.py [--start 2023-06-01] [--end YYYY-MM-DD]
                                    [--series prices generation] [--force]

End is exclusive and defaults to today 00:00 Europe/Athens (full days only).
On any failed chunk the dataset is NOT built and the script exits 1; cached
chunks are kept, so re-running retries only what is missing.
"""

import argparse
import logging
import sys

import pandas as pd

from gr_epf import data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=data.DATASET_START)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--series", nargs="+", default=list(data.SERIES), choices=data.SERIES
    )
    parser.add_argument("--force", action="store_true", help="refetch cached chunks")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start = pd.Timestamp(args.start, tz=data.LOCAL_TZ)
    end = (
        pd.Timestamp(args.end, tz=data.LOCAL_TZ)
        if args.end
        else pd.Timestamp.now(tz=data.LOCAL_TZ).normalize()
    )
    client = data.get_client()

    failures = {}
    for series in args.series:
        fails = data.download_series(client, series, start, end, force=args.force)
        if fails:
            failures[series] = fails
    if failures:
        for series, fails in failures.items():
            for chunk_start, err in fails:
                print(f"FAILED {series} {chunk_start:%Y-%m}: {err}", file=sys.stderr)
        print(
            "Some chunks failed; dataset NOT built. Cached chunks are kept —"
            " re-run to retry only the missing ones.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.skip_build:
        df = data.build_hourly_dataset(start=start, end=end)
        path = data.save_processed(df)
        print(f"hourly dataset: {len(df)} rows,"
              f" {df.index.min()} -> {df.index.max()} -> {path}")
        print("coverage per column:")
        print(df.notna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
