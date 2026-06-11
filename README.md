# GR Day-Ahead Electricity Price Forecaster

Hourly day-ahead electricity price forecasting for the Greek bidding zone
(EIC: `10YGR-HTSO-----Y`), built on ENTSO-E Transparency Platform data.
LightGBM vs naive baselines, walk-forward backtesting, conformal prediction
intervals, and a Streamlit app with daily auto-updating forecasts.

> Work in progress — phase status lives in [CLAUDE.md](CLAUDE.md).
> This README will be completed in Phase 9 (problem, data, method, leakage
> analysis, results vs baselines, limitations).

## Setup

```bash
conda create -n gr-epf python=3.11 -y
conda activate gr-epf
pip install -e ".[dev]"
cp .env.example .env   # then put your ENTSOE_API_KEY in .env
```

## Repo layout

```
src/gr_epf/        # library code (data, features, models, evaluate, conformal)
scripts/           # CLI entry points: download, train, forecast
app/               # Streamlit app
notebooks/         # EDA only — nothing is imported from here
tests/             # pytest (incl. no-leakage and DST/resolution tests)
data/              # local parquet cache (gitignored)
.github/workflows/ # daily forecast job
```

## Checks

```bash
pytest
ruff check .
```
