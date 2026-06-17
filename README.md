# GR Day-Ahead Electricity Price Forecaster

*English first — [Ελληνικά παρακάτω](#ελληνικά).*

Πρόβλεψη τιμών ηλεκτρικής ενέργειας επόμενης ημέρας για την Eλλάδα (ENTSO-E): leakage-free, LightGBM που νικά τα naive baselines κατά 25%, conformal διαστήματα αβεβαιότητας, και ζωντανή εφαρμογή με καθημερινή αυτόματη ανανέωση.

**Live demo:** https://huggingface.co/spaces/georgekrav/gr-day-ahead-price-forecast

Hourly day-ahead electricity price forecasting for the Greek bidding zone
(EIC `10YGR-HTSO-----Y`), built on ENTSO-E Transparency Platform data.
LightGBM against naive baselines, walk-forward backtesting, split-conformal
prediction intervals, and a Streamlit app fed by a daily GitHub Actions
forecast job.

## Problem

Greek day-ahead prices are set in the SDAC auction at 12:00 CET on day D-1.
The task: predict all 24 hourly prices of day D using only information
available before that gate closure. The market is non-stationary — solar
keeps deepening the midday "duck curve" (negative-price hours grew from
0.0% of 2023 to 6.6% of 2026) — which makes both honest evaluation and
calibrated uncertainty the actual engineering problem.

## Data

Three years (June 2023 – June 2026, 26,568 hours) of:

| series | notes |
|---|---|
| day-ahead prices | hourly until 2025-09-30, 15-min after the SDAC MTU switch |
| actual total load | resolution changed 2025-11-12 |
| day-ahead load forecast | published ~10:00 CET on D-1 |
| generation by type | solar, wind onshore, hydro (reservoir + run-of-river), fossil gas |

Everything is stored on tz-aware UTC indices (DST never duplicates or drops
an hour) and cached as monthly parquet chunks. Sub-hourly periods are
averaged to hourly — these are average-power (MW) and unit-price (EUR/MWh)
quantities, so mean is the correct aggregation, and the resolution of each
chunk is detected rather than assumed. Gaps (about 30 hours in three years)
are reported by a data-quality script and left as NaN, never imputed.

## Features and leakage analysis

Forecast issue time is D-1 just before 12:00 CET. Every feature documents
when it becomes available relative to that moment:

| feature | available because |
|---|---|
| price lags 24/48/168 h | prices through D-1 were set in earlier auctions |
| hour, weekday, month, GR holidays | deterministic calendar (Europe/Athens) |
| day-ahead load forecast for D | published ~10:00 CET on D-1 |
| generation lags **48 h** | actuals publish with ~1 h delay, so D-2 is the last complete day — a 24 h lag would leak the unpublished afternoon of D-1 |
| RES generation forecast (wind, solar), **same-hour** | ENTSO-E publishes the day-ahead wind/solar forecast on D-1 before gate closure — the operator's own forward view of tomorrow's RES output (strongest single feature) |
| weather (solar radiation, wind, temperature, cloud), **same-hour** | a day-ahead forecast available before gate closure (Open-Meteo); kept as redundancy alongside the RES forecast |

Tests enforce this: perturbing data after a cutoff must leave all earlier
features unchanged, perturbing same-hour actuals must leave that hour's
feature row unchanged, and the 48 h generation-lag floor is asserted
directly.

## Results — 12-month walk-forward backtest

Monthly retraining on an expanding window; every scored hour was predicted
by a model that had never seen it or anything after it.

| model | MAE | RMSE | sMAPE |
|---|---|---|---|
| **LightGBM + RES forecast + weather (l1)** | **16.85** | **24.67** | **37.8** |
| LightGBM + weather | 18.39 | 27.06 | 39.2 |
| LightGBM, lags only | 19.27 | 28.30 | 40.6 |
| naive-24h (same hour yesterday) | 22.53 | 34.31 | 46.0 |
| seasonal-naive-168h (same hour last week) | 27.70 | 41.33 | 52.1 |

**25% MAE improvement over the strongest baseline.**

> The numbers in this table are a fixed snapshot. The live app reads
> `forecasts/backtest_summary.json`, which the monthly recalibration job
> refreshes on the latest 12-month window (see [Automation](#automation)), so
> the deployed figures shift slightly month to month while the ~25% margin
> over naive holds.

![MAE by hour of day](assets/backtest_error_by_hour.png)

**Feature ablation (walk-forward).** Each forecast feature was validated
before integration, A/B over the full 12-month walk-forward:

- **RES generation forecast** (ENTSO-E day-ahead wind/solar, MW) is the
  strongest single addition: 18.39 → 16.85 MAE (−8%), stable across seeds
  (−8.11% ± 0.20). It largely subsumes the weather signal — the operator
  has already turned weather into expected MW better than raw meteorology
  would. See `scripts/experiment_res_forecast.py`.
- **Weather** (Open-Meteo, four Greek centres) adds −4.6% on its own and is
  kept as cheap redundancy if the RES forecast is missing for a day. See
  `scripts/experiment_weather.py`.

**Training window.** A sweep from 45 days to expanding shows MAE falling
monotonically as the window grows (45d 19.06 → expanding 16.85). The short-
window finding of [arXiv 2506.10536](https://arxiv.org/html/2506.10536) for
Greece in 2023 does not hold here: with a richer feature set and a more
volatile 2025-26 market, more history is pure signal and monthly retraining
already absorbs the drift. `scripts/experiment_training_window.py`.

**Model family.** LightGBM was compared against a LEAR-style linear model, a
neural net (MLP), and ensembles (`scripts/experiment_models.py`): LightGBM
alone has the best MAE, the neural net is worst, and ensembles trade a
little MAE for a little RMSE — so the single LightGBM is kept.

sMAPE is reported for comparability with the EPF literature but is unstable
when prices cross zero — MAE is the headline.

### Explainability (SHAP)

`scripts/shap_analysis.py` attributes predictions to features with a
TreeExplainer. Yesterday's price dominates (mean |SHAP| ≈ 22 EUR/MWh),
followed by the day-ahead solar forecast (≈ 13) and wind forecast (≈ 6).
The beeswarm confirms the model learned the correct physics: a high
forecast solar value pushes the predicted price *down* (the duck curve),
not a spurious correlation. Weather features rank low — visual confirmation
that the RES forecast subsumes them.

### Prediction intervals

Hand-rolled split-conformal, calibrated per hour of day (errors at 04:00
and 20:00 differ by a factor of three) on the most recent 90 days of
out-of-sample residuals, recalibrated daily:

| method | nominal | empirical coverage |
|---|---|---|
| static | 80% | 78.3% |
| **adaptive (ACI)** | 80% | **80.0%** |
| static | 95% | 94.7% |
| **adaptive (ACI)** | 95% | **95.0%** |

Static conformal slips ~2 points below nominal under drift. **Adaptive
Conformal Inference** ([Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170))
tracks an effective miscoverage rate online — a miss widens the next
interval, a hit narrows it — restoring coverage to the target. The live
calibration uses the ACI-converged level, so deployed intervals track
nominal coverage.

## Automation

A GitHub Actions cron runs every morning before gate closure: refetches
fresh ENTSO-E data, retrains on the full history (seconds for LightGBM),
forecasts day D+1 with intervals, and commits four small artifacts under
`forecasts/` (latest forecast, append-only track record with actuals filled
in as they publish, conformal calibration, backtest summary). The Streamlit
app is pure presentation on top of those files.

Because the daily job retrains from scratch, the model weights are never
stale — but two things it reads, rather than recomputes, can drift: the
conformal calibration widths and the published accuracy metrics, both from a
walk-forward backtest. A second cron (`monthly_recalibration.yml`, 1st of the
month) recomputes them on the latest data behind a **release gate**
(`gr_epf.governance`): the refreshed *challenger* artifacts replace the live
*champion* ones only if a fresh backtest still beats the naive baseline by a
clear margin, stays within an absolute MAE ceiling, and does not regress
sharply against the last accepted run. A failed gate keeps the last-good
artifacts and fails the run — alerting a human instead of silently shipping a
calibration built on a bad month of data. Every accepted run is appended to
`forecasts/model_health.json` as an audit trail.

## Limitations

- Coverage sits ~2 points under nominal at the 80% level — residual drift
  that windowed calibration shrinks but does not eliminate.
- Single bidding zone; no cross-border flow or CO2-price features. Natural
  gas price (TTF) and lignite generation were both tested and did not improve
  the walk-forward forecast — the autoregressive price lags already capture
  the fuel-cost level — so neither was added (`scripts/experiment_fuel.py`).
- One-day-ahead only; the model is not built for longer horizons.

## Run locally

```bash
conda create -n gr-epf python=3.11 -y
conda activate gr-epf
pip install -e ".[dev]"
cp .env.example .env              # add your ENTSOE_API_KEY

python scripts/download_data.py   # ~5 min cold, cached monthly chunks
python scripts/data_quality_report.py
python scripts/train_model.py     # quick 90-day holdout check
python scripts/backtest.py        # definitive 12-month walk-forward
python scripts/conformal_report.py
python scripts/make_forecast.py   # tomorrow's forecast -> forecasts/
python scripts/monthly_revalidation.py --dry-run  # release-gate check, writes nothing
streamlit run app/streamlit_app.py

pytest && ruff check .
```

## Repo layout

```
src/gr_epf/        data, features, models, evaluate, conformal, forecast
scripts/           CLI entry points (download, train, backtest, forecast)
app/               Streamlit app (reads forecasts/ only)
forecasts/         committed artifacts the app and the daily job share
notebooks/         EDA only, nothing imported from here
tests/             pytest, incl. no-leakage and DST/resolution tests
.github/workflows/ daily forecast cron + monthly recalibration gate + HF sync
data/              local parquet cache (gitignored)
```

---

# Ελληνικά

**Ζωντανή εφαρμογή:** https://huggingface.co/spaces/georgekrav/gr-day-ahead-price-forecast

Ωριαία πρόβλεψη των τιμών ηλεκτρικής ενέργειας επόμενης ημέρας για την
ελληνική ζώνη προσφορών (EIC `10YGR-HTSO-----Y`), πάνω σε δεδομένα της
πλατφόρμας διαφάνειας ENTSO-E. LightGBM απέναντι σε naive baselines,
walk-forward backtesting, split-conformal διαστήματα πρόβλεψης, και
εφαρμογή Streamlit που τροφοδοτείται από καθημερινό αυτόματο job στο
GitHub Actions.

## Το πρόβλημα

Οι ελληνικές τιμές επόμενης ημέρας καθορίζονται στη δημοπρασία SDAC στις
12:00 CET της προηγούμενης μέρας (D-1). Ο στόχος: πρόβλεψη και των 24
ωριαίων τιμών της ημέρας D χρησιμοποιώντας μόνο πληροφορία διαθέσιμη πριν
από εκείνο το κλείσιμο. Η αγορά είναι μη στάσιμη — τα ηλιακά βαθαίνουν
συνεχώς τη μεσημεριανή «καμπύλη πάπιας» (οι ώρες με αρνητική τιμή
αυξήθηκαν από 0,0% το 2023 σε 6,6% το 2026) — και αυτό κάνει την τίμια
αξιολόγηση και τη βαθμονομημένη αβεβαιότητα το πραγματικό ζητούμενο.

## Τα δεδομένα

Τρία χρόνια (Ιούνιος 2023 – Ιούνιος 2026, 26.568 ώρες):

| σειρά | σημειώσεις |
|---|---|
| τιμές day-ahead | ωριαίες έως 30/9/2025, 15λεπτες μετά την αλλαγή MTU του SDAC |
| πραγματικό φορτίο | η ανάλυση άλλαξε στις 12/11/2025 |
| πρόβλεψη φορτίου D-1 | δημοσιεύεται ~10:00 CET της προηγούμενης μέρας |
| παραγωγή ανά τύπο | ηλιακά, αιολικά, υδροηλεκτρικά (ταμιευτήρα + ποταμού), φυσικό αέριο |

Όλα αποθηκεύονται σε δείκτες UTC (η αλλαγή ώρας δεν διπλασιάζει ούτε
χάνει ποτέ ώρα) σε μηνιαία αρχεία parquet. Τα υπο-ωριαία διαστήματα
μετατρέπονται σε ωριαία με μέσο όρο — πρόκειται για μέση ισχύ (MW) και
μοναδιαία τιμή (EUR/MWh), άρα ο μέσος είναι η σωστή πράξη — και η ανάλυση
κάθε κομματιού ανιχνεύεται αντί να θεωρείται δεδομένη. Τα κενά (περίπου
30 ώρες σε τρία χρόνια) αναφέρονται από script ποιότητας και μένουν NaN —
ποτέ δεν συμπληρώνονται με εφευρημένες τιμές.

## Χαρακτηριστικά και ανάλυση διαρροής (leakage)

Η πρόβλεψη εκδίδεται την D-1, λίγο πριν τις 12:00 CET. Κάθε feature
τεκμηριώνει πότε γίνεται διαθέσιμο σε σχέση με εκείνη τη στιγμή:

| feature | διαθέσιμο επειδή |
|---|---|
| υστερήσεις τιμής 24/48/168 ω. | οι τιμές έως την D-1 βγήκαν σε προηγούμενες δημοπρασίες |
| ώρα, ημέρα, μήνας, αργίες | ντετερμινιστικό ημερολόγιο (Europe/Athens) |
| πρόβλεψη φορτίου για την D | δημοσιεύεται ~10:00 CET της D-1 |
| υστερήσεις παραγωγής **48 ω.** | τα πραγματικά δημοσιεύονται με ~1 ώρα καθυστέρηση, άρα η D-2 είναι η τελευταία πλήρης μέρα — υστέρηση 24 ω. θα διέρρεε το αδημοσίευτο απόγευμα της D-1 |
| πρόβλεψη παραγωγής ΑΠΕ (αιολικά, ηλιακά), **ίδια ώρα** | το ENTSO-E δημοσιεύει την D-1 πρόβλεψη αιολικής/ηλιακής παραγωγής πριν το κλείσιμο — η ίδια η εκτίμηση του διαχειριστή για την αυριανή παραγωγή ΑΠΕ (το ισχυρότερο feature) |
| καιρός (ηλιακή ακτινοβολία, άνεμος, θερμοκρασία, νέφωση), **ίδια ώρα** | πρόβλεψη επόμενης ημέρας διαθέσιμη πριν το κλείσιμο (Open-Meteo)· κρατιέται ως πλεονασματικότητα δίπλα στην πρόβλεψη ΑΠΕ |

Αυτά επιβάλλονται από tests: διαταραχή των δεδομένων μετά από ένα σημείο
πρέπει να αφήνει όλα τα προγενέστερα features ανέγγιχτα, και το κατώφλι
των 48 ωρών ελέγχεται ρητά.

## Αποτελέσματα — 12μηνο walk-forward backtest

Μηνιαία επανεκπαίδευση σε διευρυνόμενο παράθυρο· κάθε βαθμολογημένη ώρα
προβλέφθηκε από μοντέλο που δεν είχε δει ποτέ ούτε αυτήν ούτε οτιδήποτε
μεταγενέστερο.

| μοντέλο | MAE | RMSE | sMAPE |
|---|---|---|---|
| **LightGBM + πρόβλεψη ΑΠΕ + καιρός (l1)** | **16,85** | **24,67** | **37,8** |
| LightGBM + καιρός | 18,39 | 27,06 | 39,2 |
| LightGBM, μόνο υστερήσεις | 19,27 | 28,30 | 40,6 |
| naive-24h (ίδια ώρα χθες) | 22,53 | 34,31 | 46,0 |
| seasonal-naive-168h (ίδια ώρα πριν 1 εβδ.) | 27,70 | 41,33 | 52,1 |

**Βελτίωση 25% στο MAE από το ισχυρότερο baseline.**

> Οι αριθμοί του πίνακα είναι ένα σταθερό στιγμιότυπο. Η ζωντανή εφαρμογή
> διαβάζει το `forecasts/backtest_summary.json`, που το μηνιαίο job
> επαναβαθμονόμησης ανανεώνει στο πιο πρόσφατο 12μηνο παράθυρο (βλ.
> [Αυτοματοποίηση](#αυτοματοποίηση)), οπότε τα νούμερα στο deployment
> μετατοπίζονται ελαφρώς από μήνα σε μήνα ενώ το ~25% περιθώριο έναντι του
> naive διατηρείται.

![MAE ανά ώρα της ημέρας](assets/backtest_error_by_hour.png)

**Ablation χαρακτηριστικών (walk-forward).** Κάθε forecast feature
επικυρώθηκε πριν την ενσωμάτωση, με A/B στο πλήρες 12μηνο walk-forward:

- **Πρόβλεψη παραγωγής ΑΠΕ** (ENTSO-E day-ahead αιολικά/ηλιακά, MW) είναι
  η ισχυρότερη προσθήκη: 18,39 → 16,85 MAE (−8%), σταθερή σε seeds
  (−8,11% ± 0,20). Καθιστά σχεδόν περιττό τον καιρό — ο διαχειριστής έχει
  ήδη μεταφράσει τον καιρό σε αναμενόμενα MW καλύτερα από ωμά μετεωρολογικά.
  Βλ. `scripts/experiment_res_forecast.py`.
- **Καιρός** (Open-Meteo, τέσσερα ελληνικά κέντρα) προσθέτει −4,6% μόνος του
  και κρατιέται ως φθηνή πλεονασματικότητα αν λείψει η πρόβλεψη ΑΠΕ.
  Βλ. `scripts/experiment_weather.py`.

**Παράθυρο εκπαίδευσης.** Σάρωση από 45 ημέρες έως expanding δείχνει το MAE
να πέφτει μονότονα όσο μεγαλώνει το παράθυρο (45d 19,06 → expanding 16,85).
Το εύρημα «κοντού παραθύρου» του
[arXiv 2506.10536](https://arxiv.org/html/2506.10536) για την Ελλάδα του
2023 δεν ισχύει εδώ: με πλουσιότερο feature set και πιο ευμετάβλητη αγορά
2025-26, η περισσότερη ιστορία είναι καθαρό σήμα και η μηνιαία
επανεκπαίδευση απορροφά ήδη το drift. `scripts/experiment_training_window.py`.

**Οικογένεια μοντέλου.** Το LightGBM συγκρίθηκε με γραμμικό (LEAR-style),
νευρωνικό (MLP) και ensembles (`scripts/experiment_models.py`): το LightGBM
μόνο του έχει το καλύτερο MAE, το νευρωνικό το χειρότερο, και τα ensembles
ανταλλάσσουν λίγο MAE για λίγο RMSE — οπότε κρατάμε το σκέτο LightGBM.

Το sMAPE αναφέρεται για συγκρισιμότητα με τη βιβλιογραφία αλλά είναι
ασταθές όταν οι τιμές περνούν το μηδέν — επίσημη μετρική είναι το MAE.

### Ερμηνευσιμότητα (SHAP)

Το `scripts/shap_analysis.py` αποδίδει τις προβλέψεις στα χαρακτηριστικά με
TreeExplainer. Κυριαρχεί η χθεσινή τιμή (μέσο |SHAP| ≈ 22 EUR/MWh),
ακολουθεί η πρόβλεψη ηλιακής παραγωγής (≈ 13) και αιολικής (≈ 6). Το
beeswarm επιβεβαιώνει ότι το μοντέλο έμαθε τη σωστή φυσική: υψηλή
προβλεπόμενη ηλιακή ωθεί την τιμή *κάτω* (καμπύλη πάπιας), όχι κάποια
ψευδή συσχέτιση. Τα χαρακτηριστικά καιρού κατατάσσονται χαμηλά — οπτική
επιβεβαίωση ότι η πρόβλεψη ΑΠΕ τα υποκαθιστά.

### Διαστήματα πρόβλεψης

Χειροποίητο split-conformal, βαθμονομημένο ανά ώρα της ημέρας (τα
σφάλματα στις 04:00 και στις 20:00 διαφέρουν κατά συντελεστή τρία) στις
πιο πρόσφατες 90 ημέρες out-of-sample σφαλμάτων, με ημερήσια
επαναβαθμονόμηση:

| μέθοδος | ονομαστικό | εμπειρική κάλυψη |
|---|---|---|
| στατική | 80% | 78,3% |
| **adaptive (ACI)** | 80% | **80,0%** |
| στατική | 95% | 94,7% |
| **adaptive (ACI)** | 95% | **95,0%** |

Η στατική conformal υπο-καλύπτει ~2 μονάδες κάτω από το ονομαστικό υπό
drift. Το **Adaptive Conformal Inference**
([Gibbs & Candès 2021](https://arxiv.org/abs/2106.00170)) παρακολουθεί online
ένα ενεργό ποσοστό αστοχίας — μια αστοχία φαρδαίνει την επόμενη ζώνη, μια
επιτυχία τη στενεύει — επαναφέροντας την κάλυψη στον στόχο. Η ζωντανή
βαθμονόμηση χρησιμοποιεί το συγκλίνον επίπεδο του ACI, ώστε οι ζώνες στο
deployment να ακολουθούν την ονομαστική κάλυψη.

## Αυτοματοποίηση

Ένα cron του GitHub Actions τρέχει κάθε πρωί πριν από το κλείσιμο της
αγοράς: ξανακατεβάζει φρέσκα δεδομένα ENTSO-E, επανεκπαιδεύει σε όλη την
ιστορία (δευτερόλεπτα για το LightGBM), προβλέπει την ημέρα D+1 με
διαστήματα, και κάνει commit τέσσερα μικρά αρχεία στο `forecasts/`
(τρέχουσα πρόβλεψη, ιστορικό επιδόσεων που συμπληρώνεται με τις
πραγματικές τιμές μόλις δημοσιευτούν, βαθμονόμηση conformal, σύνοψη
backtest). Η εφαρμογή Streamlit είναι καθαρή παρουσίαση πάνω σε αυτά τα
αρχεία.

Επειδή το ημερήσιο job επανεκπαιδεύει από την αρχή, τα «βάρη» του μοντέλου
δεν μπαγιατεύουν ποτέ — δύο όμως πράγματα που απλώς *διαβάζει* (αντί να τα
ξαναϋπολογίζει) μπορούν να ξεφύγουν: οι ζώνες βαθμονόμησης conformal και οι
δημοσιευμένες μετρικές ακρίβειας, και τα δύο από ένα walk-forward backtest.
Ένα δεύτερο cron (`monthly_recalibration.yml`, την 1η κάθε μήνα) τα
ξαναϋπολογίζει στα πιο πρόσφατα δεδομένα πίσω από μια **πύλη έκδοσης**
(`gr_epf.governance`): τα ανανεωμένα *challenger* αρχεία αντικαθιστούν τα
ζωντανά *champion* μόνο αν ένα φρέσκο backtest εξακολουθεί να νικά το naive
baseline με σαφές περιθώριο, μένει κάτω από ένα απόλυτο όριο MAE, και δεν
οπισθοχωρεί απότομα σε σχέση με την τελευταία αποδεκτή εκτέλεση. Μια
αποτυχία στην πύλη κρατά τα τελευταία καλά αρχεία και ρίχνει την εκτέλεση —
ειδοποιώντας άνθρωπο αντί να προωθήσει σιωπηλά βαθμονόμηση φτιαγμένη πάνω σε
έναν κακό μήνα δεδομένων. Κάθε αποδεκτή εκτέλεση προστίθεται στο
`forecasts/model_health.json` ως ιστορικό ελέγχου.

## Περιορισμοί

- Η κάλυψη μένει ~2 μονάδες κάτω από την ονομαστική στο επίπεδο 80% —
  υπόλειμμα drift που το παράθυρο βαθμονόμησης μικραίνει αλλά δεν
  εξαλείφει.
- Μία ζώνη προσφορών· χωρίς features διασυνδέσεων ή CO₂. Η τιμή φυσικού
  αερίου (TTF) και η παραγωγή λιγνίτη δοκιμάστηκαν και δεν βελτίωσαν την
  πρόβλεψη walk-forward — οι υστερήσεις τιμής ήδη πιάνουν το επίπεδο κόστους
  καυσίμου — οπότε δεν προστέθηκαν (`scripts/experiment_fuel.py`).
- Μόνο μία ημέρα μπροστά· το μοντέλο δεν είναι φτιαγμένο για μεγαλύτερους
  ορίζοντες.

## Τοπική εκτέλεση

```bash
conda create -n gr-epf python=3.11 -y
conda activate gr-epf
pip install -e ".[dev]"
cp .env.example .env              # βάλε το δικό σου ENTSOE_API_KEY

python scripts/download_data.py   # ~5 λεπτά την πρώτη φορά, μηνιαίο cache
python scripts/data_quality_report.py
python scripts/train_model.py     # γρήγορος έλεγχος σε holdout 90 ημερών
python scripts/backtest.py        # το οριστικό 12μηνο walk-forward
python scripts/conformal_report.py
python scripts/make_forecast.py   # αυριανή πρόβλεψη -> forecasts/
python scripts/monthly_revalidation.py --dry-run  # έλεγχος πύλης, δεν γράφει τίποτα
streamlit run app/streamlit_app.py

pytest && ruff check .
```

## Δομή του repository

```
src/gr_epf/        δεδομένα, features, μοντέλα, αξιολόγηση, conformal, forecast
scripts/           εργαλεία γραμμής εντολών (λήψη, εκπαίδευση, backtest, πρόβλεψη)
app/               εφαρμογή Streamlit (διαβάζει μόνο το forecasts/)
forecasts/         τα committed αρχεία που μοιράζονται app και ημερήσιο job
notebooks/         μόνο EDA, τίποτα δεν γίνεται import από εδώ
tests/             pytest, μαζί με tests μη-διαρροής και αλλαγής ώρας/ανάλυσης
.github/workflows/ ημερήσιο cron + μηνιαία πύλη επαναβαθμονόμησης + HF sync
data/              τοπικό parquet cache (εκτός git)
```
