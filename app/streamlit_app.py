"""Streamlit app: tomorrow's GR day-ahead price forecast with intervals.

Pure presentation: reads the small committed artifacts under forecasts/
written by the daily job and the backtest scripts. No training, no API
calls, no business logic here. Bilingual UI (English / Greek).
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

STRINGS = {
    "title": {
        "en": "Greek Day-Ahead Electricity Price Forecast",
        "el": "Πρόβλεψη Τιμής Ρεύματος Επόμενης Ημέρας — Ελλάδα",
    },
    "caption": {
        "en": "Bidding zone GR (10YGR-HTSO-----Y). ENTSO-E data, LightGBM,"
        " split-conformal intervals. Educational project — not trading advice.",
        "el": "Ζώνη προσφορών GR (10YGR-HTSO-----Y). Δεδομένα ENTSO-E, LightGBM,"
        " conformal διαστήματα. Εκπαιδευτικό project — όχι συμβουλή συναλλαγών.",
    },
    "no_forecast": {
        "en": "No forecast generated yet. Run scripts/make_forecast.py first.",
        "el": "Δεν υπάρχει ακόμη πρόβλεψη. Τρέξε πρώτα το scripts/make_forecast.py.",
    },
    "forecast_for": {"en": "Forecast for", "el": "Πρόβλεψη για"},
    "daily_mean": {"en": "Daily mean", "el": "Μέση τιμή ημέρας"},
    "peak": {"en": "Peak", "el": "Μέγιστο"},
    "trough": {"en": "Trough", "el": "Ελάχιστο"},
    "at": {"en": "at", "el": "στις"},
    "generated": {"en": "Generated", "el": "Δημιουργήθηκε"},
    "hour_axis": {"en": "hour (Europe/Athens)", "el": "ώρα (Ελλάδας)"},
    "forecast_line": {"en": "forecast", "el": "πρόβλεψη"},
    "actual_line": {"en": "actual", "el": "πραγματική"},
    "band_80": {"en": "80% interval", "el": "ζώνη 80%"},
    "band_95": {"en": "95% interval", "el": "ζώνη 95%"},
    "track_record": {"en": "Track record", "el": "Ιστορικό επιδόσεων"},
    "live_mae": {"en": "Live MAE (14d)", "el": "Ζωντανό MAE (14 ημ.)"},
    "hours_scored": {"en": "Hours scored", "el": "Ώρες με αποτέλεσμα"},
    "inside_80": {"en": "Inside 80% band", "el": "Εντός ζώνης 80%"},
    "backtest_title": {
        "en": "Backtest (12 months walk-forward, monthly retrain)",
        "el": "Backtest (12 μήνες walk-forward, μηνιαία επανεκπαίδευση)",
    },
    "conformal_caption": {
        "en": "Conformal intervals, daily recalibration on the last {days} days of"
        " residuals — empirical coverage {c80:.1%} (80% nominal) and {c95:.1%}"
        " (95% nominal) on the backtest.",
        "el": "Conformal διαστήματα με ημερήσια επαναβαθμονόμηση στα σφάλματα των"
        " τελευταίων {days} ημερών — εμπειρική κάλυψη {c80:.1%} (στόχος 80%) και"
        " {c95:.1%} (στόχος 95%) στο backtest.",
    },
    "how_title": {"en": "How it works", "el": "Πώς λειτουργεί"},
}

HOW_EN = """
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
"""

HOW_EL = """
**Δεδομένα.** Τρία χρόνια ωριαίων παρατηρήσεων για την ελληνική ζώνη από την
πλατφόρμα διαφάνειας ENTSO-E: τιμές επόμενης ημέρας, πραγματικό φορτίο,
πρόβλεψη φορτίου επόμενης ημέρας και παραγωγή ανά τεχνολογία (ηλιακά,
αιολικά, υδροηλεκτρικά, φυσικό αέριο). Όλα αποθηκεύονται σε ώρα UTC ώστε η
αλλαγή ώρας να μην δημιουργεί ποτέ διπλές ή χαμένες ώρες· τα 15λεπτα (μετά
την αλλαγή της αγοράς τον Οκτώβριο 2025) γίνονται ωριαία με μέσο όρο.

**Μοντέλο.** LightGBM με στόχο l1 πάνω σε 12 χαρακτηριστικά: τιμές με
υστέρηση 24/48/168 ωρών, ημερολόγιο (ώρα, ημέρα εβδομάδας, μήνας, ελληνικές
αργίες), την πρόβλεψη φορτίου της επόμενης ημέρας και παραγωγή με υστέρηση
48 ωρών. Κάθε χαρακτηριστικό είναι διαθέσιμο πριν από το κλείσιμο της αγοράς
(12:00 CET της προηγούμενης μέρας) — η παραγωγή ξεκινά από 48 ώρες πίσω
γιατί τα απογευματινά δεδομένα της χθεσινής μέρας δεν έχουν δημοσιευτεί
ακόμη όταν εκδίδεται η πρόβλεψη.

**Αξιολόγηση.** Walk-forward στους τελευταίους 12 μήνες με μηνιαία
επανεκπαίδευση: κάθε μετρική προέρχεται από μοντέλο που δεν είχε δει ποτέ
την περίοδο στην οποία βαθμολογήθηκε. Μέτρα σύγκρισης: ίδια ώρα χθες
(naive-24h) και ίδια ώρα πριν από μία εβδομάδα (seasonal-naive-168h).

**Αβεβαιότητα.** Conformal διαστήματα βαθμονομημένα ανά ώρα της ημέρας στα
σφάλματα των τελευταίων 90 ημερών, με ημερήσια επαναβαθμονόμηση.
"""

SOURCE_LINK = (
    "Source: [github.com/georgekrav/gr-day-ahead-price-forecaster]"
    "(https://github.com/georgekrav/gr-day-ahead-price-forecaster)"
)


def load_json(name: str) -> dict | None:
    path = FORECASTS / name
    return json.loads(path.read_text()) if path.exists() else None


def load_history() -> pd.DataFrame | None:
    path = FORECASTS / "history.parquet"
    return pd.read_parquet(path) if path.exists() else None


def band_figure(rows: pd.DataFrame, t) -> go.Figure:
    x = rows["time_local"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rows["hi_95"], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_95"], fill="tonexty", fillcolor=BAND_95,
            line={"width": 0}, name=t("band_95"),
        )
    )
    fig.add_trace(go.Scatter(x=x, y=rows["hi_80"], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_80"], fill="tonexty", fillcolor=BAND_80,
            line={"width": 0}, name=t("band_80"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["forecast"], line={"color": LINE, "width": 3},
            name=t("forecast_line"),
        )
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", xaxis_title=t("hour_axis"),
        margin={"t": 30, "b": 40}, legend={"orientation": "h"}, height=420,
    )
    return fig


def track_record_figure(recent: pd.DataFrame, t) -> go.Figure:
    x = recent.index.tz_convert("Europe/Athens")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=recent["forecast"], name=t("forecast_line"), line={"color": LINE})
    )
    fig.add_trace(
        go.Scatter(x=x, y=recent["actual"], name=t("actual_line"), line={"color": ACTUAL})
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", margin={"t": 30, "b": 40},
        legend={"orientation": "h"}, height=380,
    )
    return fig


st.set_page_config(page_title="GR Day-Ahead Price Forecast", layout="wide")

head_left, head_right = st.columns([5, 1])
with head_right:
    lang = st.radio(
        "Language / Γλώσσα", ["English", "Ελληνικά"],
        horizontal=True, label_visibility="collapsed",
    )
code = "el" if lang == "Ελληνικά" else "en"


def t(key: str) -> str:
    return STRINGS[key][code]


with head_left:
    st.title(t("title"))
    st.caption(t("caption"))

latest = load_json("latest.json")
if latest is None:
    st.info(t("no_forecast"))
    st.stop()

rows = pd.DataFrame(latest["rows"])
rows["time_local"] = pd.to_datetime(rows["time_local"], utc=True).dt.tz_convert(
    "Europe/Athens"
)

st.subheader(f"{t('forecast_for')} {latest['target_day']}")
peak = rows.loc[rows["forecast"].idxmax()]
trough = rows.loc[rows["forecast"].idxmin()]
c1, c2, c3, c4 = st.columns(4)
c1.metric(t("daily_mean"), f"{rows['forecast'].mean():.1f} €/MWh")
c2.metric(t("peak"), f"{peak['forecast']:.1f} €/MWh", f"{t('at')} {peak['time_local']:%H:%M}")
c3.metric(
    t("trough"), f"{trough['forecast']:.1f} €/MWh", f"{t('at')} {trough['time_local']:%H:%M}"
)
c4.metric(t("generated"), f"{pd.Timestamp(latest['generated_at_utc']):%d %b %H:%M} UTC")
st.plotly_chart(band_figure(rows, t), use_container_width=True)

history = load_history()
if history is not None:
    scored = history.dropna(subset=["actual", "forecast"])
    if not scored.empty:
        st.subheader(t("track_record"))
        recent = scored.tail(14 * 24)
        live_mae = (recent["actual"] - recent["forecast"]).abs().mean()
        inside_80 = (
            (recent["actual"] >= recent["lo_80"]) & (recent["actual"] <= recent["hi_80"])
        ).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric(t("live_mae"), f"{live_mae:.1f} €/MWh")
        c2.metric(t("hours_scored"), f"{len(recent)}")
        c3.metric(t("inside_80"), f"{inside_80:.0%}")
        st.plotly_chart(track_record_figure(recent, t), use_container_width=True)

summary = load_json("backtest_summary.json")
if summary and "metrics" in summary:
    st.subheader(t("backtest_title"))
    table = pd.DataFrame(summary["metrics"]["table"]).T
    st.dataframe(table, use_container_width=True)
    if "conformal" in summary:
        conf = summary["conformal"]
        st.caption(
            t("conformal_caption").format(
                days=conf["calibration_window_days"],
                c80=conf["80"]["coverage"],
                c95=conf["95"]["coverage"],
            )
        )

with st.expander(t("how_title")):
    first, second = (HOW_EL, HOW_EN) if code == "el" else (HOW_EN, HOW_EL)
    st.markdown(first)
    st.divider()
    st.markdown(second)
    st.markdown(SOURCE_LINK)
