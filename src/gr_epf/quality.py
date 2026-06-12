"""Data-quality analysis: resolution segments, gaps, outliers, negative prices.

Everything here reports; nothing repairs. Gaps and outliers are surfaced
for a human decision, per project rule "never silently impute".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Market time units that count as regular data; any other spacing is either
# a gap or an irregular region and gets reported as such.
DATA_STEPS = (pd.Timedelta("15min"), pd.Timedelta("30min"), pd.Timedelta("60min"))
HOUR = pd.Timedelta("1h")


@dataclass
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp  # last timestamp covered, inclusive
    step: pd.Timedelta
    n_steps: int


@dataclass
class Gap:
    after: pd.Timestamp  # last good timestamp before the gap
    until: pd.Timestamp  # first good timestamp after the gap
    missing_hours: float


def resolution_segments(index: pd.DatetimeIndex) -> list[Segment]:
    """Contiguous runs of constant sampling step.

    A run boundary is either a gap (jump between equal-step runs) or a
    resolution regime change (e.g. SDAC 60-min -> 15-min on 2025-10-01).
    """
    if len(index) < 2:
        return []
    steps = index[1:] - index[:-1]
    out = []
    run_start = 0
    for i in range(1, len(steps) + 1):
        if i == len(steps) or steps[i] != steps[run_start]:
            out.append(
                Segment(
                    start=index[run_start],
                    end=index[i],
                    step=steps[run_start],
                    n_steps=i - run_start,
                )
            )
            run_start = i
    return out


def find_gaps(segments: list[Segment]) -> list[Gap]:
    """Gaps between consecutive regular-step segments, in wall-clock hours.

    The expected next point after segment a is a.end + a.step; anything
    later means missing periods. A contiguous regime change (next segment
    starts exactly where the previous ends) is not a gap.
    """
    data_segs = [s for s in segments if s.step in DATA_STEPS]
    gaps = []
    for a, b in zip(data_segs, data_segs[1:], strict=False):
        missing = b.start - a.end - a.step
        if missing > pd.Timedelta(0):
            gaps.append(Gap(after=a.end, until=b.start, missing_hours=missing / HOUR))
    return gaps


def irregular_regions(segments: list[Segment]) -> list[Segment]:
    """Multi-point runs at a non-standard step (e.g. alternating missing values)."""
    return [s for s in segments if s.step not in DATA_STEPS and s.n_steps > 1]


def negative_price_episodes(price: pd.Series) -> pd.DataFrame:
    """Consecutive runs of negative hourly prices."""
    neg = price < 0
    run_id = (neg != neg.shift()).cumsum()
    rows = [
        {
            "start": g.index[0],
            "end": g.index[-1],
            "hours": len(g),
            "min_price": g.min(),
            "mean_price": g.mean(),
        }
        for _, g in price[neg].groupby(run_id[neg])
    ]
    return pd.DataFrame(rows)


def outlier_summary(
    s: pd.Series, hard_low: float | None = None, hard_high: float | None = None
) -> dict:
    """Robust-bound and hard-bound outlier counts; flags only, no removal."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 5 * iqr, q3 + 5 * iqr
    out = {
        "iqr_bounds": (round(lo, 2), round(hi, 2)),
        "n_below_iqr": int((s < lo).sum()),
        "n_above_iqr": int((s > hi).sum()),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
    }
    if hard_low is not None:
        out["n_below_hard"] = int((s < hard_low).sum())
    if hard_high is not None:
        out["n_above_hard"] = int((s > hard_high).sum())
    return out


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


def _series_section(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## Raw series: {name}", ""]
    idx = df.index
    lines.append(f"- rows: {len(df)}, span: {_fmt_ts(idx.min())} -> {_fmt_ts(idx.max())} UTC")
    segments = resolution_segments(idx)
    regimes = [s for s in segments if s.step in DATA_STEPS]
    steps_seen = sorted({s.step for s in regimes})
    lines.append(f"- resolutions present: {', '.join(str(s) for s in steps_seen)}")
    changes = [
        (a, b) for a, b in zip(regimes, regimes[1:], strict=False) if a.step != b.step
    ]
    for a, b in changes:
        lines.append(
            f"- resolution change {a.step} -> {b.step} at {_fmt_ts(b.start)} UTC"
        )
    gaps = find_gaps(segments)
    total_missing = sum(g.missing_hours for g in gaps)
    lines.append(f"- gaps: {len(gaps)} ({total_missing:.2f} hours missing total)")
    for g in gaps[:20]:
        lines.append(
            f"  - {_fmt_ts(g.after)} -> {_fmt_ts(g.until)}: {g.missing_hours:.2f} h"
        )
    if len(gaps) > 20:
        lines.append(f"  - ... and {len(gaps) - 20} more")
    for s in irregular_regions(segments):
        lines.append(
            f"- IRREGULAR sampling {_fmt_ts(s.start)} -> {_fmt_ts(s.end)}"
            f" (step {s.step}, {s.n_steps} steps) — inspect manually"
        )
    lines.append("")
    return lines


def render_report(raw: dict[str, pd.DataFrame], hourly: pd.DataFrame) -> str:
    """Markdown data-quality report for raw series + processed hourly dataset."""
    lines = [
        "# Data quality report",
        "",
        f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC."
        " All timestamps UTC. Report only — no values were modified.",
        "",
        "## Processed hourly dataset",
        "",
        f"- rows: {len(hourly)}, span: {_fmt_ts(hourly.index.min())} ->"
        f" {_fmt_ts(hourly.index.max())}",
        "",
        "| column | coverage | missing hours |",
        "|---|---|---|",
    ]
    for col in hourly.columns:
        n_missing = int(hourly[col].isna().sum())
        cov = 1 - n_missing / len(hourly)
        lines.append(f"| {col} | {cov:.4%} | {n_missing} |")
    lines.append("")

    price = hourly["price_eur_mwh"].dropna()
    episodes = negative_price_episodes(price)
    lines += ["## Negative price episodes (hourly)", ""]
    if episodes.empty:
        lines.append("None.")
    else:
        total = int(episodes["hours"].sum())
        lines.append(
            f"- {len(episodes)} episodes, {total} hours total,"
            f" min price {episodes['min_price'].min():.2f} EUR/MWh"
        )
        top = episodes.sort_values("hours", ascending=False).head(10)
        lines += ["", "| start | hours | min | mean |", "|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(
                f"| {_fmt_ts(r['start'])} | {r['hours']} |"
                f" {r['min_price']:.2f} | {r['mean_price']:.2f} |"
            )
    lines.append("")

    lines += ["## Outlier flags", ""]
    summaries = {
        "price_eur_mwh": outlier_summary(price, hard_low=-150, hard_high=1000),
    }
    for col in hourly.columns:
        if col != "price_eur_mwh":
            summaries[col] = outlier_summary(hourly[col].dropna(), hard_low=0)
    for col, s in summaries.items():
        lines.append(f"- **{col}**: {s}")
    lines.append("")

    for name, df in raw.items():
        lines += _series_section(name, df)
    return "\n".join(lines)


def write_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
