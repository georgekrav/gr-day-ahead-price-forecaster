"""Generate the data-quality report from cached raw chunks + processed dataset.

Usage:
    python scripts/data_quality_report.py [--out data/reports/quality_report.md]
"""

import argparse
from pathlib import Path

from gr_epf import data, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(data.REPO_ROOT / "data" / "reports" / "quality_report.md")
    )
    args = parser.parse_args()

    raw = {series: data.load_raw(series) for series in data.SERIES}
    hourly = data.load_processed()
    report = quality.render_report(raw, hourly)
    path = quality.write_report(report, Path(args.out))
    print(report)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
