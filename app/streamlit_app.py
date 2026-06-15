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
PEAK_COLOR = "#ff4b4b"
TROUGH_COLOR = "#21c55d"

STRINGS = {
    "title": {
        "en": "Greek Day-Ahead Electricity Price Forecast",
        "el": "Πρόβλεψη Τιμής Ρεύματος Επόμενης Ημέρας — Ελλάδα",
    },
    "caption": {
        "en": "This app forecasts tomorrow's 24 hourly wholesale electricity"
        " prices for Greece (bidding zone 10YGR-HTSO-----Y). Every morning,"
        " before the day-ahead auction closes at noon, a machine-learning model"
        " (LightGBM) trained on three years of ENTSO-E data predicts each hour's"
        " price together with how uncertain that prediction is, and tracks how"
        " well past forecasts turned out. Educational project — not trading"
        " advice.",
        "el": "Η εφαρμογή προβλέπει τις 24 ωριαίες χονδρεμπορικές τιμές ρεύματος"
        " της αυριανής μέρας για την Ελλάδα (ζώνη 10YGR-HTSO-----Y). Κάθε πρωί,"
        " πριν κλείσει το μεσημέρι η δημοπρασία της επόμενης ημέρας, ένα μοντέλο"
        " μηχανικής μάθησης (LightGBM) εκπαιδευμένο σε τρία χρόνια δεδομένων"
        " ENTSO-E προβλέπει την τιμή κάθε ώρας μαζί με το πόσο αβέβαιη είναι, και"
        " παρακολουθεί πόσο καλά βγήκαν οι παλιές προβλέψεις. Εκπαιδευτικό project"
        " — όχι συμβουλή συναλλαγών.",
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
    "overview_caption": {
        "en": "Overview — the last 14 days at a glance: how each forecast lined"
        " up with the actual price over time. Pick a single day below to zoom in.",
        "el": "Επισκόπηση — οι τελευταίες 14 ημέρες με μια ματιά: πώς ταίριαξε"
        " κάθε πρόβλεψη με την πραγματική τιμή. Διάλεξε μία μέρα πιο κάτω για zoom.",
    },
    "live_mae": {"en": "Live MAE (14d)", "el": "Ζωντανό MAE (14 ημ.)"},
    "hours_scored": {"en": "Hours scored", "el": "Ώρες με αποτέλεσμα"},
    "inside_80": {"en": "Inside 80% band", "el": "Εντός ζώνης 80%"},
    "select_day": {
        "en": "Compare a specific day (forecast vs. what actually happened)",
        "el": "Σύγκριση συγκεκριμένης ημέρας (πρόβλεψη vs. τι έγινε τελικά)",
    },
    "day_mae": {"en": "MAE that day", "el": "MAE εκείνης της ημέρας"},
    "day_rmse": {"en": "RMSE that day", "el": "RMSE εκείνης της ημέρας"},
    "day_inside_95": {"en": "Inside 95% band", "el": "Εντός ζώνης 95%"},
    "backtest_title": {
        "en": "Backtest (12 months walk-forward, monthly retrain)",
        "el": "Backtest (12 μήνες walk-forward, μηνιαία επανεκπαίδευση)",
    },
    "backtest_caption": {
        "en": "Each row is a forecasting method, scored over the last 12 months on"
        " data it never saw while training. Lower MAE and RMSE is better (they are"
        " the average miss in EUR/MWh). The two “naive” rows are simple"
        " rules of thumb — same hour yesterday, or same hour last week — and the"
        " model is worth using only if it clearly beats them, which it does.",
        "el": "Κάθε γραμμή είναι μια μέθοδος πρόβλεψης, βαθμολογημένη στους"
        " τελευταίους 12 μήνες σε δεδομένα που δεν είδε ποτέ στην εκπαίδευση. Όσο"
        " χαμηλότερα τα MAE και RMSE, τόσο καλύτερα (είναι το μέσο σφάλμα σε"
        " EUR/MWh). Οι δύο “naive” γραμμές είναι απλοί εμπειρικοί κανόνες"
        " — ίδια ώρα χθες ή ίδια ώρα πριν μία εβδομάδα — και το μοντέλο αξίζει μόνο"
        " αν τις ξεπερνά καθαρά, κάτι που κάνει.",
    },
    "how_title": {"en": "How it works", "el": "Πώς λειτουργεί"},
    "shap_title": {"en": "How the model decides", "el": "Πώς αποφασίζει το μοντέλο"},
    "shap_features_title": {
        "en": "What each feature means",
        "el": "Τι σημαίνει κάθε χαρακτηριστικό",
    },
    "shap_features": {
        "en": "- **price_lag_24/48/168h** — the price 1, 2 or 7 days ago at the"
        " same hour (the past is the strongest clue to the future).\n"
        "- **res_fc_solar_mw / res_fc_wind_mw** — the grid operator's own"
        " day-ahead forecast of tomorrow's solar / wind output, in MW.\n"
        "- **load_forecast_mw** — the day-ahead forecast of electricity demand.\n"
        "- **gen_*_lag_48h** — actual generation by source (solar, wind, gas,"
        " hydro) two days ago.\n"
        "- **wx_\\*** — weather: solar radiation, wind, temperature, cloud.\n"
        "- **hour / weekday / month / is_holiday** — position in the calendar.",
        "el": "- **price_lag_24/48/168h** — η τιμή 1, 2 ή 7 μέρες πριν, την ίδια"
        " ώρα (το παρελθόν είναι η ισχυρότερη ένδειξη για το μέλλον).\n"
        "- **res_fc_solar_mw / res_fc_wind_mw** — η πρόβλεψη του διαχειριστή για"
        " την αυριανή ηλιακή / αιολική παραγωγή, σε MW.\n"
        "- **load_forecast_mw** — η πρόβλεψη ζήτησης ρεύματος για αύριο.\n"
        "- **gen_*_lag_48h** — πραγματική παραγωγή ανά πηγή (ήλιος, άνεμος, αέριο,"
        " νερό) πριν δύο μέρες.\n"
        "- **wx_\\*** — καιρός: ηλιακή ακτινοβολία, άνεμος, θερμοκρασία, νέφωση.\n"
        "- **hour / weekday / month / is_holiday** — θέση στο ημερολόγιο.",
    },
    "help_shap": {
        "en": "SHAP attributes each prediction to its features. The bars show"
        " the average impact (in EUR/MWh) of each feature on the forecast —"
        " which inputs the model leans on most.",
        "el": "Το SHAP αποδίδει κάθε πρόβλεψη στα χαρακτηριστικά της. Οι μπάρες"
        " δείχνουν τη μέση επίδραση (σε EUR/MWh) κάθε χαρακτηριστικού στην"
        " πρόβλεψη — σε ποιες εισόδους στηρίζεται περισσότερο το μοντέλο.",
    },
    "shap_caption": {
        "en": "Yesterday's price dominates, followed by the day-ahead RES"
        " generation forecast: more forecast solar/wind pushes the price down"
        " (the duck curve), which is the physically correct relationship.",
        "el": "Κυριαρχεί η χθεσινή τιμή, ακολουθεί η πρόβλεψη παραγωγής ΑΠΕ"
        " επόμενης ημέρας: περισσότερη προβλεπόμενη ηλιακή/αιολική ρίχνει την"
        " τιμή (καμπύλη πάπιας) — η φυσικά σωστή σχέση.",
    },
    "help_title": {
        "en": "Every morning, before the day-ahead auction closes (12:00 CET),"
        " this app forecasts all 24 hourly wholesale electricity prices for"
        " tomorrow in the Greek bidding zone.",
        "el": "Κάθε πρωί, πριν κλείσει η δημοπρασία επόμενης ημέρας (12:00 CET),"
        " η εφαρμογή προβλέπει και τις 24 ωριαίες χονδρεμπορικές τιμές ρεύματος"
        " της αυριανής μέρας για την ελληνική ζώνη.",
    },
    "help_forecast_for": {
        "en": "The blue line is the model's forecast for each hour of the target"
        " day. The shaded bands are conformal uncertainty intervals: the true"
        " price should fall inside the dark band 80% of the time and inside the"
        " light band 95% of the time. Hover over the line to read each value.",
        "el": "Η μπλε γραμμή είναι η πρόβλεψη του μοντέλου για κάθε ώρα της"
        " ημέρας-στόχου. Οι σκιασμένες ζώνες είναι διαστήματα αβεβαιότητας"
        " (conformal): η πραγματική τιμή αναμένεται μέσα στη σκούρα ζώνη το 80%"
        " των ωρών και μέσα στην ανοιχτόχρωμη το 95%. Πέρνα τον κέρσορα πάνω στη"
        " γραμμή για να δεις κάθε τιμή.",
    },
    "help_daily_mean": {
        "en": "Average of the 24 hourly forecasts — a quick sense of how cheap"
        " or expensive tomorrow is overall.",
        "el": "Ο μέσος όρος των 24 ωριαίων προβλέψεων — γρήγορη αίσθηση του"
        " πόσο φτηνή ή ακριβή είναι συνολικά η αυριανή μέρα.",
    },
    "help_peak": {
        "en": "The most expensive forecast hour — typically the evening ramp,"
        " when solar fades while demand peaks and gas plants set the price.",
        "el": "Η ακριβότερη προβλεπόμενη ώρα — συνήθως το βράδυ, όταν ο ήλιος"
        " χάνεται ενώ η ζήτηση κορυφώνεται και την τιμή ορίζουν οι μονάδες"
        " φυσικού αερίου.",
    },
    "help_trough": {
        "en": "The cheapest forecast hour — typically midday, when solar floods"
        " the market; on sunny low-demand days it can go negative.",
        "el": "Η φθηνότερη προβλεπόμενη ώρα — συνήθως το μεσημέρι, όταν τα"
        " ηλιακά πλημμυρίζουν την αγορά· σε ηλιόλουστες μέρες χαμηλής ζήτησης"
        " μπορεί να βγει και αρνητική.",
    },
    "help_generated": {
        "en": "When the daily job produced this forecast (Greek time). It runs"
        " on GitHub's servers every morning before the market gate closure.",
        "el": "Πότε παρήχθη αυτή η πρόβλεψη από το καθημερινό αυτόματο τρέξιμο"
        " (ώρα Ελλάδας). Εκτελείται σε servers του GitHub κάθε πρωί, πριν από το"
        " κλείσιμο της αγοράς.",
    },
    "help_track_record": {
        "en": "Past forecasts frozen at issue time, compared against the actual"
        " auction prices once published. Forecasts are never rewritten after"
        " the fact.",
        "el": "Παλιές προβλέψεις «παγωμένες» όπως εκδόθηκαν, σε σύγκριση με τις"
        " πραγματικές τιμές της δημοπρασίας μόλις δημοσιευτούν. Καμία πρόβλεψη"
        " δεν ξαναγράφεται εκ των υστέρων.",
    },
    "help_live_mae": {
        "en": "Mean Absolute Error over the scored hours of the last 14 days:"
        " by how many EUR/MWh the forecast missed on average. The 12-month"
        " backtest average is 16.85.",
        "el": "Μέσο απόλυτο σφάλμα στις βαθμολογημένες ώρες των τελευταίων 14"
        " ημερών: πόσα EUR/MWh έπεσε έξω η πρόβλεψη κατά μέσο όρο. Ο μέσος όρος"
        " του 12μηνου backtest είναι 16,85.",
    },
    "help_hours_scored": {
        "en": "How many forecast hours have a published actual price to compare"
        " against so far.",
        "el": "Πόσες προβλεπόμενες ώρες έχουν ήδη δημοσιευμένη πραγματική τιμή"
        " για σύγκριση.",
    },
    "help_inside_80": {
        "en": "Share of actual prices that landed inside the 80% band. If the"
        " intervals are honest, this should hover around 80% as data"
        " accumulates.",
        "el": "Ποσοστό των πραγματικών τιμών που έπεσαν μέσα στη ζώνη 80%. Αν"
        " οι ζώνες είναι σωστά βαθμονομημένες, με τον καιρό θα κινείται γύρω"
        " στο 80%.",
    },
    "help_backtest": {
        "en": "Each forecast was made by a model trained only on the past, then"
        " checked against what actually happened — repeated month by month, just"
        " like the live system. MAE is the average miss in EUR/MWh; lower is"
        " better.",
        "el": "Κάθε πρόβλεψη έγινε από μοντέλο εκπαιδευμένο μόνο στο παρελθόν και"
        " μετά συγκρίθηκε με το τι έγινε στην πραγματικότητα — επαναλαμβανόμενα"
        " κάθε μήνα, ακριβώς όπως το ζωντανό σύστημα. Το MAE είναι το μέσο σφάλμα"
        " σε EUR/MWh· όσο χαμηλότερα τόσο καλύτερα.",
    },
}

HOW_EN = """
**The problem.** Greek wholesale electricity prices for every hour of a day
are set the day before, in a single auction that closes at 12:00 CET. The
task is to predict all 24 of tomorrow's hourly prices using only information
that already exists before that deadline. It is a genuinely hard problem:
prices are spiky, can turn negative at sunny midday, and the market keeps
shifting as solar grows.

**Data.** Three years of hourly observations for the Greek bidding zone from
the ENTSO-E Transparency Platform: day-ahead prices, actual load, the
day-ahead load forecast, generation by type (solar, wind onshore, hydro,
fossil gas), and the operator's day-ahead wind/solar forecast. Open-Meteo
adds weather. Everything is stored on a UTC index so daylight-saving never
creates a duplicate or missing hour; 15-minute periods (after the SDAC switch
of October 2025) are averaged to hourly.

**Model.** LightGBM (gradient-boosted trees) with an l1 objective, so price
spikes do not dominate the fit. It reads price lags (24/48/168 h), the local
calendar, the day-ahead load forecast, 48-hour generation lags, and the
day-ahead RES (wind/solar) forecast — which turned out to be the single most
useful feature, because more expected solar pushes the midday price down.
Every input is available before the 12:00 CET gate closure: generation lags
start at 48 h because yesterday afternoon's actuals are not published yet.

**Honest evaluation.** A 12-month walk-forward backtest with monthly
retraining: every reported number comes from a model that had never seen the
period it was scored on — the only fair way to estimate live accuracy. The
model beats both naive baselines (same hour yesterday, same hour last week)
by about 25% on MAE.

**Uncertainty.** Split-conformal intervals, calibrated separately for each
hour of the day on the most recent 90 days of out-of-sample errors and
recalibrated every day, with an adaptive correction that keeps coverage on
target as the market drifts.

**Automation.** A GitHub Actions cron refreshes the data, retrains, and
publishes a new forecast every morning before the auction; a second monthly
job re-checks accuracy and refreshes the uncertainty bands only if the model
still clearly beats the baseline. The app you are reading is pure
presentation on top of those committed files.
"""

HOW_EL = """
**Το πρόβλημα.** Οι ελληνικές χονδρεμπορικές τιμές ρεύματος για κάθε ώρα μιας
μέρας καθορίζονται την προηγούμενη, σε μία δημοπρασία που κλείνει στις 12:00
CET. Ζητούμενο είναι να προβλέψουμε και τις 24 αυριανές ωριαίες τιμές
χρησιμοποιώντας μόνο πληροφορία που υπάρχει ήδη πριν από αυτή την προθεσμία.
Είναι πραγματικά δύσκολο: οι τιμές είναι αιχμηρές, μπορούν να γίνουν αρνητικές
το ηλιόλουστο μεσημέρι, και η αγορά αλλάζει συνεχώς καθώς μεγαλώνουν τα ηλιακά.

**Δεδομένα.** Τρία χρόνια ωριαίων παρατηρήσεων για την ελληνική ζώνη από την
πλατφόρμα διαφάνειας ENTSO-E: τιμές επόμενης ημέρας, πραγματικό φορτίο,
πρόβλεψη φορτίου, παραγωγή ανά τεχνολογία (ηλιακά, αιολικά, υδροηλεκτρικά,
φυσικό αέριο) και την πρόβλεψη ΑΠΕ του διαχειριστή. Ο καιρός έρχεται από το
Open-Meteo. Όλα αποθηκεύονται σε ώρα UTC ώστε η αλλαγή ώρας να μη δημιουργεί
ποτέ διπλές ή χαμένες ώρες· τα 15λεπτα (μετά την αλλαγή της αγοράς τον
Οκτώβριο 2025) γίνονται ωριαία με μέσο όρο.

**Μοντέλο.** LightGBM (gradient-boosted δέντρα) με στόχο l1, ώστε οι ακραίες
τιμές να μην κυριαρχούν. Διαβάζει τιμές με υστέρηση (24/48/168 ωρών), το
ημερολόγιο, την πρόβλεψη φορτίου, παραγωγή με υστέρηση 48 ωρών, και την
πρόβλεψη ΑΠΕ (αιολικά/ηλιακά) επόμενης ημέρας — που αποδείχθηκε το πιο χρήσιμο
χαρακτηριστικό, αφού περισσότερη αναμενόμενη ηλιακή ρίχνει τη μεσημεριανή τιμή.
Κάθε είσοδος είναι διαθέσιμη πριν το κλείσιμο στις 12:00 CET· η παραγωγή
ξεκινά από 48 ώρες πίσω γιατί τα χθεσινά απογευματινά δεδομένα δεν έχουν
δημοσιευτεί ακόμη.

**Τίμια αξιολόγηση.** 12μηνο walk-forward backtest με μηνιαία επανεκπαίδευση:
κάθε νούμερο προέρχεται από μοντέλο που δεν είχε δει ποτέ την περίοδο στην
οποία βαθμολογήθηκε — ο μόνος δίκαιος τρόπος να εκτιμηθεί η πραγματική
ακρίβεια. Το μοντέλο νικά και τα δύο naive μέτρα σύγκρισης (ίδια ώρα χθες,
ίδια ώρα πριν μία εβδομάδα) κατά περίπου 25% στο MAE.

**Αβεβαιότητα.** Conformal διαστήματα, βαθμονομημένα ξεχωριστά για κάθε ώρα
της ημέρας στα σφάλματα των τελευταίων 90 ημερών και επαναβαθμονομημένα κάθε
μέρα, με προσαρμοστική διόρθωση που κρατά την κάλυψη στον στόχο καθώς αλλάζει
η αγορά.

**Αυτοματισμός.** Ένα cron του GitHub Actions ανανεώνει τα δεδομένα,
επανεκπαιδεύει και δημοσιεύει νέα πρόβλεψη κάθε πρωί πριν τη δημοπρασία· ένα
δεύτερο μηνιαίο job ξαναελέγχει την ακρίβεια και ανανεώνει τις ζώνες
αβεβαιότητας μόνο αν το μοντέλο εξακολουθεί να νικά καθαρά το baseline. Η
εφαρμογή που διαβάζεις είναι καθαρή παρουσίαση πάνω σε αυτά τα αρχεία.
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


def _price_hover(name: str) -> str:
    return "%{x|%H:%M} — %{y:.1f} €/MWh<extra>" + name + "</extra>"


def band_figure(rows: pd.DataFrame, t) -> go.Figure:
    x = rows["time_local"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rows["hi_95"], line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_95"], fill="tonexty", fillcolor=BAND_95,
            line={"width": 0}, name=t("band_95"), hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=rows["hi_80"], line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["lo_80"], fill="tonexty", fillcolor=BAND_80,
            line={"width": 0}, name=t("band_80"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=rows["forecast"], line={"color": LINE, "width": 3},
            name=t("forecast_line"), hovertemplate=_price_hover(t("forecast_line")),
        )
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", xaxis_title=t("hour_axis"),
        margin={"t": 30, "b": 70}, height=420, hovermode="x",
        legend={"orientation": "h", "yanchor": "top", "y": -0.22},
    )
    return fig


def track_record_figure(recent: pd.DataFrame, t) -> go.Figure:
    x = recent.index.tz_convert("Europe/Athens")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=recent["forecast"], name=t("forecast_line"),
                   line={"color": LINE}, hovertemplate=_price_hover(t("forecast_line")))
    )
    fig.add_trace(
        go.Scatter(x=x, y=recent["actual"], name=t("actual_line"),
                   line={"color": ACTUAL}, hovertemplate=_price_hover(t("actual_line")))
    )
    fig.update_layout(
        yaxis_title="EUR/MWh", margin={"t": 30, "b": 40},
        legend={"orientation": "h"}, height=380, hovermode="x",
    )
    return fig


def day_figure(day: pd.DataFrame, t) -> go.Figure:
    x = day.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=day["hi_95"], line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=day["lo_95"], fill="tonexty", fillcolor=BAND_95,
                             line={"width": 0}, name=t("band_95"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=day["hi_80"], line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=day["lo_80"], fill="tonexty", fillcolor=BAND_80,
                             line={"width": 0}, name=t("band_80"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=day["forecast"], name=t("forecast_line"),
                             line={"color": LINE, "width": 3},
                             hovertemplate=_price_hover(t("forecast_line"))))
    fig.add_trace(go.Scatter(x=x, y=day["actual"], name=t("actual_line"),
                             line={"color": ACTUAL, "width": 2},
                             hovertemplate=_price_hover(t("actual_line"))))
    fig.update_layout(
        yaxis_title="EUR/MWh", xaxis_title=t("hour_axis"),
        margin={"t": 30, "b": 70}, height=380, hovermode="x",
        legend={"orientation": "h", "yanchor": "top", "y": -0.22},
    )
    return fig


st.set_page_config(page_title="GR Day-Ahead Price Forecast", layout="wide")
# Force the vertical scrollbar to always show: otherwise it appears/hides as
# content height crosses the viewport, resizing the full-width charts in a
# loop — the "flicker" seen on the Space.
st.markdown(
    "<style>html { overflow-y: scroll; }"
    # hide the anchor-link (chain) icon Streamlit adds next to every heading
    '[data-testid="stHeaderActionElements"] a { display: none; }'
    "</style>",
    unsafe_allow_html=True,
)
# Color the peak value red and the trough value green. st.metric cannot color
# its value, so an invisible marker is placed in each column and :has() reaches
# the metric value inside it (keeps the native "?" help tooltip intact).
st.markdown(
    "<style>"
    '[data-testid="stColumn"]:has(.gr-peak) [data-testid="stMetricValue"],'
    '[data-testid="column"]:has(.gr-peak) [data-testid="stMetricValue"]'
    "{color:" + PEAK_COLOR + " !important;}"
    '[data-testid="stColumn"]:has(.gr-trough) [data-testid="stMetricValue"],'
    '[data-testid="column"]:has(.gr-trough) [data-testid="stMetricValue"]'
    "{color:" + TROUGH_COLOR + " !important;}"
    "</style>",
    unsafe_allow_html=True,
)

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
    st.title(t("title"), help=t("help_title"))
    st.caption(t("caption"))

latest = load_json("latest.json")
if latest is None:
    st.info(t("no_forecast"))
    st.stop()

rows = pd.DataFrame(latest["rows"])
rows["time_local"] = pd.to_datetime(rows["time_local"], utc=True).dt.tz_convert(
    "Europe/Athens"
)

target_day_fmt = pd.Timestamp(latest["target_day"]).strftime("%d/%m/%Y")
st.subheader(
    f"{t('forecast_for')} {target_day_fmt}", help=t("help_forecast_for")
)
peak = rows.loc[rows["forecast"].idxmax()]
trough = rows.loc[rows["forecast"].idxmin()]
generated = pd.Timestamp(latest["generated_at_utc"])
if generated.tzinfo is None:
    generated = generated.tz_localize("UTC")
generated = generated.tz_convert("Europe/Athens")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    t("daily_mean"), f"{rows['forecast'].mean():.1f} €/MWh", help=t("help_daily_mean")
)
c2.markdown('<span class="gr-peak"></span>', unsafe_allow_html=True)
c2.metric(
    t("peak"), f"{peak['forecast']:.1f} €/MWh",
    f"{t('at')} {peak['time_local']:%H:%M}", delta_color="off", help=t("help_peak"),
)
c3.markdown('<span class="gr-trough"></span>', unsafe_allow_html=True)
c3.metric(
    t("trough"), f"{trough['forecast']:.1f} €/MWh",
    f"{t('at')} {trough['time_local']:%H:%M}", delta_color="off", help=t("help_trough"),
)
c4.metric(
    t("generated"),
    f"{generated:%d/%m/%Y %H:%M}",
    help=t("help_generated"),
)
st.plotly_chart(band_figure(rows, t), width="stretch")

history = load_history()
if history is not None:
    scored = history.dropna(subset=["actual", "forecast"])
    if not scored.empty:
        st.subheader(t("track_record"), help=t("help_track_record"))
        recent = scored.tail(14 * 24)
        live_mae = (recent["actual"] - recent["forecast"]).abs().mean()
        inside_80 = (
            (recent["actual"] >= recent["lo_80"]) & (recent["actual"] <= recent["hi_80"])
        ).mean()
        m1, m2, m3 = st.columns(3)
        m1.metric(t("live_mae"), f"{live_mae:.1f} €/MWh", help=t("help_live_mae"))
        m2.metric(t("hours_scored"), f"{len(recent)}", help=t("help_hours_scored"))
        m3.metric(t("inside_80"), f"{inside_80:.0%}", help=t("help_inside_80"))
        st.caption(t("overview_caption"))
        st.plotly_chart(track_record_figure(recent, t), width="stretch")

        local = scored.copy()
        local.index = local.index.tz_convert("Europe/Athens")
        days = sorted({ts.date() for ts in local.index})
        labels = [d.strftime("%d/%m/%Y") for d in days]
        picked = st.selectbox(t("select_day"), labels, index=len(labels) - 1)
        day = local[local.index.date == days[labels.index(picked)]]
        err = day["actual"] - day["forecast"]
        d_mae = err.abs().mean()
        d_rmse = (err**2).mean() ** 0.5
        d_in95 = (
            (day["actual"] >= day["lo_95"]) & (day["actual"] <= day["hi_95"])
        ).mean()
        d1, d2, d3 = st.columns(3)
        d1.metric(t("day_mae"), f"{d_mae:.1f} €/MWh")
        d2.metric(t("day_rmse"), f"{d_rmse:.1f} €/MWh")
        d3.metric(t("day_inside_95"), f"{d_in95:.0%}")
        st.plotly_chart(day_figure(day, t), width="stretch")

summary = load_json("backtest_summary.json")
if summary and "metrics" in summary:
    st.subheader(t("backtest_title"), help=t("help_backtest"))
    st.caption(t("backtest_caption"))
    table = pd.DataFrame(summary["metrics"]["table"]).T
    st.dataframe(table, width="stretch")

shap_bar = ROOT / "assets" / "shap_bar.png"
if shap_bar.exists():
    st.subheader(t("shap_title"), help=t("help_shap"))
    # fixed-width image beside a feature glossary: a full-width stretch of this
    # PNG made the page reflow/flicker on the Space
    img_col, gloss_col = st.columns([3, 2])
    img_col.image(str(shap_bar), width=560)
    img_col.caption(t("shap_caption"))
    gloss_col.markdown(f"**{t('shap_features_title')}**")
    gloss_col.markdown(t("shap_features"))

with st.expander(t("how_title")):
    first, second = (HOW_EL, HOW_EN) if code == "el" else (HOW_EN, HOW_EL)
    st.markdown(first)
    st.divider()
    st.markdown(second)
    st.markdown(SOURCE_LINK)
