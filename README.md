# ESP PredictAI

### Intelligent Artificial Lift Monitoring & Failure Prediction

An end-to-end Streamlit application for exploring ESP (Electric Submersible Pump) operating data, auditing data quality, training leakage-safe failure classifiers, explaining model behaviour, and scoring manual operating snapshots through a professional dark industrial dashboard.

> **Scope:** This is a **current-state failure classification** prototype. The target describes whether the ESP is recorded as `running` or `failed` on the measurement day. It is **not** a future-failure predictor, a remaining-useful-life model, or a certified diagnostic system. See [Classification vs. future prediction](#classification-vs-future-prediction).

---

## Table of contents

- [What the app does](#what-the-app-does)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [ESP engineering background](#esp-engineering-background)
- [Target definition and leakage controls](#target-definition-and-leakage-controls)
- [Machine-learning methodology](#machine-learning-methodology)
- [Validation strategies and the generalisation problem](#validation-strategies-and-the-generalisation-problem)
- [Explainability](#explainability)
- [Prediction and risk bands](#prediction-and-risk-bands)
- [Application pages](#application-pages)
- [Tests](#tests)
- [Limitations](#limitations)
- [Classification vs. future prediction](#classification-vs-future-prediction)
- [Future improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## What the app does

- Loads the public ESP CSV from GitHub, with a bundled local fallback and CSV upload.
- Audits schema, missingness, duplicates, class balance, panel completeness, label persistence, and outcome leakage.
- Provides reversible data cleaning: mean/median/mode/drop missing values, IQR/Z-score outlier detection, remove/cap treatment, and MinMax/Standard scaling.
- Supports interactive EDA: histograms, box plots, scatter plots, correlation heatmaps, pairwise views, failure distributions, well comparisons, and time series.
- Trains **Logistic Regression**, **Random Forest**, and optional **XGBoost** in leakage-safe scikit-learn pipelines.
- Compares random stratified, time-based, and well-based validation, plus well-grouped cross-validation.
- Evaluates with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrices, ROC/PR curves, threshold analysis, and calibration.
- Saves and loads the fitted preprocessing pipeline plus estimator with joblib.
- Scores manual ESP operating snapshots with configurable model-probability risk bands.
- Explains global and individual model behaviour using native importance, permutation importance, and SHAP when available.
- Provides petroleum-engineering interpretation with explicit limits on what the data can support.

---

## Quick start

### Requirements

- Python 3.11 or 3.12 recommended
- The packages in [`requirements.txt`](requirements.txt)

### Install

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```powershell
streamlit run app.py
```

or, if the `streamlit` executable is not on your PATH:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

### Run the one-day reading check

The static reading check uses the saved model through a small FastAPI service. From the project root, run:

```powershell
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000`. The service loads `models/esp_predictai_model.joblib` at startup and serves the pages in `web/`. If you train and save a replacement model, restart the service to load it. When the model is unavailable, the page disables scoring and explains how to start or restore it.

### Data loading behaviour

At startup the app attempts to download:

```text
https://raw.githubusercontent.com/kings-2025/Electric-Submersible-Pumps/main/esp_downhole_data.csv
```

If the network request fails, it uses the bundled cached copy at `data/esp_downhole_data.csv`. If both fail, use the CSV upload control in **Data Explorer**.

No model is preloaded into the session. Train a model in **ML Model Training**, or load a previously saved bundle, before using **Failure Prediction**.

---

## Project structure

```text
pump ESP failure prediction/
├── api.py                       # FastAPI service for the static reading check
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   └── esp_downhole_data.csv
├── models/
│   └── esp_predictai_model.joblib   # created when a model is saved
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── engineering_insights.py
│   ├── state.py
│   ├── ui.py
│   ├── plotting.py
│   └── pages/
│       ├── overview.py
│       ├── data_explorer.py
│       ├── cleaning.py
│       ├── eda_page.py
│       ├── training.py
│       ├── prediction_view.py
│       ├── explainability_view.py
│       ├── insights.py
│       └── about.py
└── tests/
    ├── test_data_loading.py
    ├── test_preprocessing.py
    └── test_prediction.py
```

---

## Dataset

### Source

- Repository: [kings-2025/Electric-Submersible-Pumps](https://github.com/kings-2025/Electric-Submersible-Pumps)
- Raw CSV: [esp_downhole_data.csv](https://raw.githubusercontent.com/kings-2025/Electric-Submersible-Pumps/main/esp_downhole_data.csv)
- Source commit inspected during development: `4bf2354671ebbccd2cf4f38ca5572b60c0607d0b`
- Bundled CSV SHA-256: `BCDA98519B07F53D7A709FA32E8FFA3867FE93E69CD2F99F7581A73131561A93`

### Shape and contents

The bundled dataset contains:

- **8,766 rows × 14 source columns**
- **6 wells** (`esp_1` through `esp_6`)
- Daily panel from **2015-01-01** to **2018-12-31**
- **5,035 running** observations and **3,731 failed** observations
- **42.56% failure prevalence**

Source columns:

```text
well_id, date, downtime_hours, intake_pressure, discharge_pressure,
motor_temperature, intake_temperature, motor_amps, motor_volt,
vibration, frequency, production, production_loss, esp_status
```

The prepared dataframe adds `esp_failure` (binary target) and an internal `__row_id`. The internal row id is hidden from user-facing column lists.

### Provenance and data-quality warning

**The dataset's provenance is unverified.** The repository does not provide field metadata, equipment specifications, a data-generation script, or independent validation. The app treats the data as an educational example, not field evidence.

Several statistical signatures are consistent with simulated or heavily processed data:

1. Complete, gap-free daily panel with no missing values and no duplicate rows.
2. `downtime_hours` and `production_loss` are exactly zero for every running record and non-zero for every failed record. They perfectly reveal the target.
3. Failure labels show almost no day-to-day persistence; the transition pattern is close to independent daily draws rather than run-to-failure equipment behaviour.
4. A 42.56% daily failure prevalence is implausibly high for a producing ESP fleet.
5. Most sensor channels are integer-quantised and occupy narrow ranges, unusual for raw field sensors.
6. Only six wells are available, so well-level estimates have high variance.

The **Data Explorer** and **Project Information** pages recompute these checks live from the active dataset.

### Target-encoding discrepancy in the source repository

The public repository README states `running = 1` and `failed = 0`, and the source notebook uses that reverse encoding.

**This application deliberately follows the project specification: `running = 0`, `failed = 1`.**

The raw `esp_status` text is preserved for display, and a separate `esp_failure` column is created for modelling. Do not compare metrics to the source notebook without accounting for the reversed encoding.

---

## ESP engineering background

An Electric Submersible Pump is a multistage centrifugal pump installed downhole and driven by an electric motor. A typical system includes:

- Downhole motor
- Motor seal / protector
- Multistage centrifugal pump
- Intake / gas handling section
- Discharge section
- Power cable
- Surface transformer and variable-speed drive (VSD)
- Wellhead and associated instrumentation

### Common failure mechanisms

| Mechanism | Typical indicators | Engineering note |
|---|---|---|
| Electrical / insulation failure | Current imbalance, voltage anomalies, motor temperature rise, sudden trip | Cable or motor insulation damage can cause immediate shutdown. |
| Motor overheating | High winding temperature, high current, low flow / poor cooling | Temperature limits are equipment-specific and must come from the manufacturer. |
| Mechanical wear / bearing failure | Vibration increase, current fluctuation, noise, reduced performance | Vibration interpretation depends on sensor location, mounting, and pump design. |
| Gas interference / gas lock | Unstable intake pressure, current cycling, reduced production, vibration | Gas handling and operating strategy strongly affect behaviour. |
| Sand / solids / erosion | Vibration, current changes, production decline, wear | Sand production can accelerate pump and seal damage. |
| Scale / corrosion / deposition | Gradual performance decline, pressure changes, current changes | Chemistry and completion context are required for diagnosis. |
| Seal / protector failure | Motor temperature, current, insulation changes, vibration | Seal failure can lead to motor damage. |
| Operational / power-quality issues | Frequency, voltage, current anomalies, trips | VSD settings and power quality matter. |

These are general engineering pathways, **not universal alarm rules**. This app does not encode manufacturer limits or make a specific failure diagnosis.

### Operating parameters in this dataset

- **Intake pressure** — suction condition at the pump intake; affected by reservoir pressure, fluid level, gas, and operating point.
- **Discharge pressure** — pressure generated downstream of the pump; related to lift, flow, and system resistance.
- **Motor temperature** — winding/operating temperature; affected by load, cooling, current, and insulation condition.
- **Intake temperature** — fluid temperature at the pump intake; can reflect reservoir/well conditions and thermal effects.
- **Motor current** — related to load and power; abnormal changes can indicate mechanical, electrical, or operating issues.
- **Motor voltage** — supply/operating voltage; deviations can indicate power-quality or electrical problems.
- **Vibration** — mechanical condition indicator; interpretation requires sensor and equipment context.
- **Frequency** — VSD output frequency, linked to pump speed and operating point.
- **Production rate** — delivered production; a decline can reflect pump degradation, reservoir changes, or operating changes.

The app uses these as model features but does **not** impose invented pressure, temperature, current, or vibration thresholds.

---

## Target definition and leakage controls

### Target

- Raw label: `esp_status` (`running` / `failed`)
- Model target: `esp_failure`
  - `running = 0`
  - `failed = 1`
- The target is the status recorded on the **same measurement day**.

### Model features

The nine operational features are used when present:

```text
intake_pressure, discharge_pressure, motor_temperature, intake_temperature,
motor_amps, motor_volt, vibration, frequency, production
```

### Excluded columns

- `esp_status` — raw target, never a model input.
- `downtime_hours` — deterministic leakage in this dataset.
- `production_loss` — deterministic leakage in this dataset.
- `well_id` — identifier used for grouping and well-based validation only.
- `date` — timestamp used for temporal validation and trend analysis only.
- `__row_id` — internal traceability field.

### Leakage-safe pipeline

1. The train/test split is created **before** fitting.
2. Median imputation and standardisation live inside a scikit-learn `Pipeline`.
3. The pipeline is fitted only on the training portion.
4. The target is never imputed, scaled, capped, or otherwise transformed.
5. Cleaning operations are separate from model training and preserve the original dataframe.

---

## Machine-learning methodology

### Algorithms

- **Logistic Regression** — interpretable linear baseline with standardised features and median imputation.
- **Random Forest** — nonlinear tree ensemble; impurity/gain importance.
- **XGBoost** — gradient-boosted trees when the package is installed.

### Evaluation metrics

- Accuracy
- Precision
- Recall for the failed class
- F1
- ROC-AUC
- PR-AUC
- Brier score
- Confusion matrix
- ROC and precision–recall curves
- Threshold sweep
- Calibration curve

**Recall is emphasised** because a missed failure can be more costly than a false alarm. Precision and false-alarm burden are shown alongside recall so the trade-off is explicit.

### Decision threshold

The decision threshold is adjustable. A snapshot is classified as `FAILED` when the model score is at or above the threshold. Lower thresholds detect more failures but produce more false alarms. The app shows the effect on recall, precision, F1, and the confusion matrix.

### Probability caveat

A model score is **not automatically a calibrated field failure probability**. The app displays calibration and Brier score, but calibration on a small, unverified dataset does not guarantee field reliability.

### Model persistence

The fitted pipeline, feature list, training metadata, threshold, training ranges, and metrics are saved together in:

```text
models/esp_predictai_model.joblib
```

Held-out prediction arrays are **not** saved. If a loaded model has no in-memory test set, permutation importance and SHAP use a clearly labelled sample of the current dataset.

---

## Validation strategies and the generalisation problem

| Strategy | What it tests | Key limitation |
|---|---|---|
| **Random stratified** | Row-level generalisation with preserved class balance | Rows from the same well appear in train and test; usually optimistic. |
| **Time-based** | Extrapolation to later dates | Same wells in train and test; does not test a new well. |
| **Well-based** | Cross-well generalisation with entire wells held out | Only six wells, so high variance; one split can be dominated by which wells are held out. |
| **Well-grouped CV** | Cross-well performance over multiple folds | Same small-well limitation, but more robust than a single split. |

### What development runs showed

With the public CSV, random and time-based splits produced very high accuracy and ROC-AUC. Well-based evaluation fell close to chance. In development runs (settings and seed dependent):

- Random-split Random Forest: approximately **0.93 accuracy / 0.99 ROC-AUC**
- Time-based Random Forest: approximately **0.93 accuracy**
- Well-based Random Forest: approximately **0.56 accuracy / 0.59 ROC-AUC**
- Well-grouped CV mean ROC-AUC: approximately **0.51**
- Well-based Logistic Regression recall at 0.5: approximately **0.15**

These are **not** hardcoded app results; the app recomputes metrics from the active data and selected configuration. The pattern matters more than the exact values: the model is learning **well-specific signatures**, not failure physics that transfers to a new well.

**Do not present a high random-split score as evidence of a deployable cross-well model.** The app surfaces this warning prominently and recommends the well-based comparison.

---

## Explainability

The **Model Explainability** page provides:

- **Native feature importance** — impurity/gain for tree models; absolute standardised coefficients for Logistic Regression.
- **Permutation importance** — performance drop when a feature is shuffled, using the holdout split when available.
- **SHAP** — `TreeExplainer` for tree models and `LinearExplainer` for Logistic Regression.
- **Individual attribution** — additive contributions for a manual prediction, a held-out row, or a loaded dataset row.
- **A labelled fallback** if SHAP is unavailable; it is explicitly marked as heuristic, not SHAP.

None of these methods establishes physical causation. Feature importance can reflect well identity, operating strategy, correlated inputs, or data-generation artefacts.

---

## Prediction and risk bands

The **Failure Prediction** page accepts manual values for the active model's features. Inputs are checked for finite values and basic physical validity only. No universal engineering limits are applied.

Risk bands are based **only on the model-estimated failure score** and user-configurable cut-offs:

- **Normal** — below the elevated-risk threshold (default 0.40)
- **Elevated** — at or above 0.40 and below 0.70
- **High** — at or above 0.70

These defaults are **model-score bands, not engineering limits**. They can be changed in the page. The app displays:

- Failure probability / model score
- Classification at the decision threshold
- Risk band and gauge
- Inputs outside the training range
- Top feature contributions for the individual case
- Session-only prediction history
- A calibration and probability caveat

Predictions are disabled until a model is trained or loaded.

---

## Application pages

1. **Overview Dashboard** — fleet summary, failure prevalence, data status, model status, workflow entry points.
2. **Data Explorer** — dataset loading, filtering, preview, schema, missingness, statistics, quality audit, data dictionary.
3. **Data Cleaning** — reversible missing-value, duplicate, outlier, and scaling operations with history and undo/reset.
4. **Exploratory Data Analysis** — histograms, box plots, scatter plots, correlation heatmap, pairwise relationships, failure distributions, well comparisons, and time series.
5. **ML Model Training** — algorithm and hyperparameter selection, split preview, training, model comparison, validation-strategy comparison, grouped CV, threshold analysis, curves, evaluation table, model save/load/reset.
6. **Failure Prediction** — manual operating snapshot, validation, model score, risk band, extrapolation warning, individual explanation, history.
7. **Model Explainability** — native importance, permutation importance, SHAP summary, individual attribution, method notes.
8. **Engineering Insights** — ESP schematic, failure mechanisms, parameter deep dive, evidence layers, deployment path, engineering disclaimer.
9. **Project Information** — scope, dataset provenance and audit, methodology, limitations, architecture, reproducibility, future improvements.

---

## Tests

Run:

```powershell
pytest -q
```

The test suite covers:

- Dataset loading, target encoding, feature exclusions, leakage detection, and audit flags.
- Duplicate handling, random/time/well splits, leakage-safe pipelines, and training statistics.
- Bundle export, save/load, risk bands, invalid inputs, out-of-range detection, and file management.

The bundled test suite currently contains **22 tests**.

---

## Limitations

- **Current-state label only.** No future failure event, lead time, or remaining useful life.
- **Unverified provenance.** The public data has strong simulated-data signatures.
- **Six wells only.** Cross-well estimates are unstable.
- **No equipment context.** No pump curves, motor ratings, cable data, completion details, fluid properties, or manufacturer limits.
- **Daily snapshots.** No high-frequency vibration, transient events, or startup/shutdown dynamics.
- **No maintenance history.** No work orders, failure causes, repairs, or run-to-failure records.
- **Deterministic leakage.** `downtime_hours` and `production_loss` reveal the target and are excluded; other artefacts may remain.
- **Calibration is not guaranteed.** Holdout calibration on a small dataset may not transfer.
- **High failure prevalence.** 42.56% daily failure rate is not representative of a real producing fleet.
- **No causal claims.** Importance and SHAP explain model behaviour, not physical causation.
- **Not a field system.** No safety certification, control action, or maintenance decision should rely on this prototype.

---

## Classification vs. future prediction

This distinction is central to the project:

| Current-state classification (this app) | Future failure prediction (not implemented) |
|---|---|
| Target: `failed` vs `running` on the measurement day | Target: failure within a defined future window |
| Uses contemporaneous sensor values | Uses history and a lead-time horizon |
| Random/time/well holdout evaluation | Temporal, horizon-aware validation |
| Output: model score for the current snapshot | Output: calibrated failure probability over time |
| No maintenance lead time | Explicit lead time and response plan |
| Educational prototype | Requires field data, calibration, monitoring, and trials |

A model can classify a snapshot that already looks like the failed class without being able to predict that a failure will occur next week. The app states this explicitly and does not claim future predictive capability.

---

## Future improvements

- Replace the public dataset with run-to-failure field data and maintenance work orders.
- Define a future-failure target (for example, failure within 30 days) and validate with a horizon-aware temporal split.
- Incorporate high-frequency vibration, motor current signature, and VSD transient data.
- Add equipment specifications, pump curves, motor ratings, cable data, and fluid properties.
- Develop well-specific or hierarchical models and evaluate subgroup performance.
- Calibrate probabilities prospectively and monitor drift and alarm burden.
- Add survival analysis or remaining-useful-life modelling once event times are available.
- Integrate alarm management, maintenance workflows, and cost-sensitive threshold optimisation.
- Add automated page tests and a reproducible experiment log.

---

## Troubleshooting

### `python` or `py` is not recognised

Use the full path to your Python executable, or activate a virtual environment and use `python`. On Windows, the Microsoft Store alias can interfere; disabling the alias or using the full interpreter path resolves it.

### The app cannot download the dataset

It automatically falls back to `data/esp_downhole_data.csv`. If that file is missing, upload a CSV in **Data Explorer**. The required column is `esp_status` with running/failed values (or 0/1).

### XGBoost is unavailable

The app still works with Logistic Regression and Random Forest. Install it with:

```powershell
pip install xgboost
```

### SHAP is unavailable or slow

SHAP is optional. Native and permutation importance remain available, and individual explanations fall back to a clearly labelled heuristic.

### The saved model expects different features

Train or load a model against the current dataset. The app validates the bundle's feature list before loading it for prediction.

### Streamlit widget deprecation warnings

The code uses the current `width="stretch"` API. If you install an older Streamlit version, upgrade with `pip install --upgrade streamlit`.

---

## Disclaimer

ESP PredictAI is an educational and research prototype. It is not a certified industrial safety, diagnostic, or maintenance system. Model scores are not calibrated field failure probabilities and must not be used as the sole basis for operational decisions. The public dataset's provenance is unverified, and its statistical signatures are consistent with simulated data. Always consult equipment specifications, manufacturer limits, field measurements, maintenance records, and qualified engineering judgement.
