"""ENTSO-E Transparency Platform download + local parquet cache.

Raw series are cached per calendar month (Europe/Athens boundaries) as
parquet under data/raw/<series>/YYYY-MM-DD.parquet with tz-aware UTC
indices. Downstream code reads only the processed hourly dataset built by
build_hourly_dataset().

Resampling rule (tested in tests/test_data.py): every quantity here is
either average power (MW) or a price (EUR/MWh), so sub-hourly periods
aggregate to hourly by MEAN, never sum. Hours with no observations stay
NaN — gaps are reported by gr_epf.quality, never imputed.

Resolution note: SDAC switched from 60-min to 15-min products on
2025-10-01; entsoe-py 0.8 already returns day-ahead prices hourly before
and quarter-hourly after the switch. Load and generation resolution is
whatever ENTSO-E publishes per period — detected, not assumed.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

log = logging.getLogger(__name__)

ZONE = "GR"  # bidding zone EIC 10YGR-HTSO-----Y
LOCAL_TZ = "Europe/Athens"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "hourly.parquet"

SERIES = ("prices", "load_actual", "load_forecast", "generation")
# Generation types kept in the processed dataset. Hydro = reservoir +
# run-of-river; pumped storage is excluded: marginal in GR and it is a
# storage asset (consumes to refill), not free inflow generation.
GEN_SIMPLE = {
    "Solar": "gen_solar_mw",
    "Wind Onshore": "gen_wind_onshore_mw",
    "Fossil Gas": "gen_fossil_gas_mw",
}
GEN_HYDRO = ("Hydro Water Reservoir", "Hydro Run-of-river and poundage")
REQUEST_PAUSE_S = 0.6


def get_client() -> EntsoePandasClient:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("ENTSOE_API_KEY not set; copy .env.example to .env")
    return EntsoePandasClient(api_key=key, retry_count=3, retry_delay=10)


def month_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into calendar-month chunks in local time."""
    if start.tz is None or end.tz is None:
        raise ValueError("start and end must be tz-aware")
    chunks = []
    cur = start
    while cur < end:
        nxt = min((cur + pd.offsets.MonthBegin(1)).normalize(), end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def chunk_path(raw_dir: Path, series: str, chunk_start: pd.Timestamp) -> Path:
    return raw_dir / series / f"{chunk_start.strftime('%Y-%m-%d')}.parquet"


def fetch_chunk(
    client: EntsoePandasClient, series: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch one chunk and normalize: UTC index, plain string columns."""
    if series == "prices":
        df = client.query_day_ahead_prices(ZONE, start=start, end=end).to_frame(
            "price_eur_mwh"
        )
    elif series == "load_actual":
        df = client.query_load(ZONE, start=start, end=end)
        df = df.rename(columns={"Actual Load": "load_actual_mw"})
    elif series == "load_forecast":
        df = client.query_load_forecast(ZONE, start=start, end=end)
        df = df.rename(columns={"Forecasted Load": "load_forecast_mw"})
    elif series == "generation":
        df = client.query_generation(ZONE, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            # keep production only; "Actual Consumption" exists for storage types
            keep = [c for c in df.columns if c[-1] == "Actual Aggregated"]
            df = df[keep]
            df.columns = [c[0] for c in keep]
    else:
        raise ValueError(f"unknown series {series!r}")
    df = df.tz_convert("UTC").sort_index()
    # entsoe-py truncates inclusively at end; enforce half-open [start, end)
    return df[(df.index >= start) & (df.index < end)]


def download_series(
    client: EntsoePandasClient,
    series: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> list[tuple[pd.Timestamp, str]]:
    """Download missing month chunks; returns (chunk_start, error) failures.

    The chunk containing "now" is always refetched because the month is
    still growing; completed cached months are skipped unless force=True.
    """
    failures = []
    now = pd.Timestamp.now(tz=LOCAL_TZ)
    for cs, ce in month_chunks(start, end):
        path = chunk_path(raw_dir, series, cs)
        if path.exists() and not force and ce <= now:
            continue
        try:
            df = fetch_chunk(client, series, cs, ce)
        except Exception as exc:
            log.error("%s %s failed: %r", series, cs.date(), exc)
            failures.append((cs, repr(exc)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        log.info("%s %s: %d rows", series, cs.date(), len(df))
        time.sleep(REQUEST_PAUSE_S)
    return failures


def load_raw(series: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate all cached chunks of a series, UTC-indexed and sorted.

    Identical duplicate timestamps (chunk-boundary artifacts) are dropped;
    duplicates with conflicting values are a data error and raise.
    """
    paths = sorted((raw_dir / series).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no cached chunks for {series!r} in {raw_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    dup = df.index.duplicated(keep="first")
    if dup.any():
        if df.groupby(level=0).nunique(dropna=False).gt(1).any(axis=None):
            raise ValueError(f"{series}: conflicting values at duplicate timestamps")
        df = df[~dup]
    return df


def detect_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("need at least two timestamps")
    return index.to_series().diff().median()


def resample_to_hourly(s: pd.Series) -> pd.Series:
    """Hourly mean. Identity for hourly input; mean of quarters for 15-min.

    Mean (not sum) because MW is average power and EUR/MWh is a unit price.
    Internal gaps become NaN rows; partial hours average the values present
    and are flagged separately by gr_epf.quality.
    """
    return s.resample("h").mean()


def build_hourly_dataset(
    raw_dir: Path = RAW_DIR,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join all series on a complete hourly UTC index; NaN where data is missing."""
    gen = load_raw("generation", raw_dir)
    missing_types = [c for c in GEN_SIMPLE if c not in gen.columns]
    hydro_cols = [c for c in GEN_HYDRO if c in gen.columns]
    if missing_types or not hydro_cols:
        raise ValueError(
            f"generation types missing: {missing_types or GEN_HYDRO};"
            f" available: {list(gen.columns)}"
        )
    cols = {
        "price_eur_mwh": load_raw("prices", raw_dir)["price_eur_mwh"],
        "load_actual_mw": load_raw("load_actual", raw_dir)["load_actual_mw"],
        "load_forecast_mw": load_raw("load_forecast", raw_dir)["load_forecast_mw"],
    }
    for src, name in GEN_SIMPLE.items():
        cols[name] = gen[src]
    # min_count=len: an hour missing either hydro component stays NaN instead
    # of silently counting the absent one as zero
    cols["gen_hydro_mw"] = gen[hydro_cols].sum(axis=1, min_count=len(hydro_cols))
    out = pd.DataFrame({name: resample_to_hourly(s) for name, s in cols.items()})
    full = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC")
    out = out.reindex(full)
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index < end]
    out.index.name = "time_utc"
    return out


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
