"""Streamlit app: tomorrow's GR day-ahead price forecast with intervals.

Pure presentation: reads the small committed artifacts under forecasts/
written by the daily job and the backtest scripts. No training, no API
calls, no business logic here.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FORECASTS = ROOT / "forecasts"
BAND_95 = "rgba(31, 119, 180, 0.15)"
BAND_80 = "rgba(31, 119, 180, 0.30)"
LINE = "#1f77b4"
ACTUAL = "#d62728"


def load_json(name: str) -> dict | None:
    path = FORECASTS / name
    return json.loads(path.read_text()) if path.exists() else None


def load_history() -> pd.DataFrame | None:
    path = FORECASTS / "history.parquet"
    return pd.read_parquet(path) if path.exists() else None


def band_figure(rows: pd.DataFrame) -> go.Figure:
    x = rows["time_local"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rows["hi_95"], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_95"], fill="tonexty", fillcolor=BAND_95,
            line={"width": 0}, name="95% interval",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=rows["hi_80"], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_80"], fill="tonexty", fillcolor=BAND_80,
            line={"width": 0}, name="80% interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["forecast"], line={"color": LINE, "width": 3}, name="forecast"
        )
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", xaxis_title="hour (Europe/Athens)",
        margin={"t": 30, "b": 40}, legend={"orientation": "h"}, height=420,
    )
    return fig


def track_record_figure(recent: pd.DataFrame) -> go.Figure:
    x = recent.index.tz_convert("Europe/Athens")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=recent["forecast"], name="forecast", line={"color": LINE})
    )
    fig.add_trace(
        go.Scatter(x=x, y=recent["actual"], name="actual", line={"color": ACTUAL})
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", margin={"t": 30, "b": 40},
        legend={"orientation": "h"}, height=380,
    )
    return fig


st.set_page_config(page_title="GR Day-Ahead Price Forecast", layout="wide")
st.title("Greek Day-Ahead Electricity Price Forecast")
st.caption(
    "Bidding zone GR (10YGR-HTSO-----Y). ENTSO-E data, LightGBM,"
    " split-conformal intervals. Educational project — not trading advice."
)

latest = load_json("latest.json")
if latest is None:
    st.info("No forecast generated yet. Run scripts/make_forecast.py first.")
    st.stop()

rows = pd.DataFrame(latest["rows"])
rows["time_local"] = pd.to_datetime(rows["time_local"], utc=True).dt.tz_convert(
    "Europe/Athens"
)

st.subheader(f"Forecast for {latest['target_day']}")
peak = rows.loc[rows["forecast"].idxmax()]
trough = rows.loc[rows["forecast"].idxmin()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Daily mean", f"{rows['forecast'].mean():.1f} €/MWh")
c2.metric("Peak", f"{peak['forecast']:.1f} €/MWh", f"at {peak['time_local']:%H:%M}")
c3.metric(
    "Trough", f"{trough['forecast']:.1f} €/MWh", f"at {trough['time_local']:%H:%M}"
)
c4.metric("Generated", f"{pd.Timestamp(latest['generated_at_utc']):%d %b %H:%M} UTC")
st.plotly_chart(band_figure(rows), use_container_width=True)

history = load_history()
if history is not None:
    scored = history.dropna(subset=["actual", "forecast"])
    if not scored.empty:
        st.subheader("Track record")
        recent = scored.tail(14 * 24)
        live_mae = (recent["actual"] - recent["forecast"]).abs().mean()
        inside_80 = (
            (recent["actual"] >= recent["lo_80"]) & (recent["actual"] <= recent["hi_80"])
        ).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Live MAE (14d)", f"{live_mae:.1f} €/MWh")
        c2.metric("Hours scored", f"{len(recent)}")
        c3.metric("Inside 80% band", f"{inside_80:.0%}")
        st.plotly_chart(track_record_figure(recent), use_container_width=True)

summary = load_json("backtest_summary.json")
if summary and "metrics" in summary:
    st.subheader("Backtest (12 months walk-forward, monthly retrain)")
    table = pd.DataFrame(summary["metrics"]["table"]).T
    st.dataframe(table, use_container_width=True)
    if "conformal" in summary:
        conf = summary["conformal"]
        st.caption(
            f"Conformal intervals, daily recalibration on the last"
            f" {conf['calibration_window_days']} days of residuals — empirical"
            f" coverage {conf['80']['coverage']:.1%} (80% nominal) and"
            f" {conf['95']['coverage']:.1%} (95% nominal) on the backtest."
        )

with st.expander("How it works"):
    st.markdown(
        """
**Data.** Three years of hourly observations for the Greek bidding zone from
the ENTSO-E Transparency Platform: day-ahead prices, actual load, the
day-ahead load forecast, and generation by type (solar, wind onshore,
hydro, fossil gas). Everything is stored on a UTC index; 15-minute periods
(after the SDAC switch of October 2025) are averaged to hourly.

**Model.** LightGBM with an l1 objective on 12 features: price lags
(24/48/168 h), local calendar (hour, weekday, month, Greek holidays), the
day-ahead load forecast, and 48-hour generation lags. Every feature is
available before the market gate closure at 12:00 CET on D-1 — generation
lags start at 48 h because actuals for the afternoon of D-1 are not yet
published when the forecast is issued.

**Evaluation.** Walk-forward over the final 12 months with monthly
retraining: every shown metric comes from a model that had never seen the
period it was scored on. Baselines: same hour yesterday (naive-24h) and
same hour last week (seasonal-naive-168h).

**Uncertainty.** Split-conformal intervals calibrated per hour of day on
the most recent 90 days of out-of-sample errors, recalibrated daily.

Source: [github.com/georgekrav/gr-day-ahead-price-forecaster](https://github.com/georgekrav/gr-day-ahead-price-forecaster)
"""
    )
