# ESP PredictAI redesign brief

## 1. Product summary

The tool classifies one day of electric submersible pump (ESP) readings as **running** or **failed**. It is not a future-failure forecast or a remaining-life estimate. Keep the educational-prototype caveat visible: dataset provenance is unverified, the dataset covers six wells, and cross-well accuracy is near chance. Do not present the score as a calibrated field probability or operational advice. (`src/config.py::TARGET`, `src/data_loader.py::audit_dataset`, `src/pages/about.py::_limitations_tab`, `src/pages/about.py::_methodology_tab`, `README.md`.)

## 2. Page-by-page map

Routes are registered in `app.py::ROUTES`. “Expert” means a link to the existing Streamlit app.

| Current page and code | Inputs → outputs | Who needs it |
|---|---|---|
| Overview, `src/pages/overview.py::render` | Dataset → status, prevalence, integrity | Operators/managers |
| Data Explorer, `src/pages/data_explorer.py::render` | Source and filters → preview, schema, statistics, quality | Data stewards/analysts |
| Data Cleaning, `src/pages/cleaning.py::render` | Dataset and cleaning choices → working copy, history | Analysts |
| Exploratory Data Analysis, `src/pages/eda_page.py::render` | Chart and feature choices → plots, correlations, trends | Analysts/students |
| ML Model Training, `src/pages/training.py::render` | Algorithm, split, settings → metrics, comparisons, model bundle | ML users |
| Failure Prediction, `src/pages/prediction_view.py::render` | Nine readings and thresholds → class, score, band, range note, history | All target users |
| Model Explainability, `src/pages/explainability_view.py::render` | Model, data and method → importance and individual explanation | Analysts/students |
| Engineering Insights, `src/pages/insights.py::render` | Selected parameter and model/data → mechanisms, evidence, cautions | Engineers/students |
| Project Information, `src/pages/about.py::render` | Data audit → scope, provenance, method, limitations | All users |

## 3. Keep, merge, cut or move

| Current function | Decision | Reason |
|---|---|---|
| Overview | Merge into screen 1 | Keep scope and caveat; omit fleet metrics from the main task. |
| Data Explorer | Move to Expert mode | CSV, filters and schema serve data work. |
| Data Cleaning | Move to Expert mode | Reversible edits change the working data and require expertise. |
| Exploratory Data Analysis | Move to Expert mode | Interactive charts serve analyst and student exploration. |
| ML Model Training | Move to Expert mode | Model and validation choices are specialist controls. |
| Failure Prediction | Keep as the core flow | It accepts nine readings and returns a same-day class. |
| Model Explainability | Move to Expert mode | Keep technical methods there; show a short result note. |
| Engineering Insights | Merge into screen 3 | Include one plain-language note with evidence limits. |
| Project Information | Merge into screens 1 and 3 | Keep caveats visible; move long technical details to Expert mode. |

## 4. Three-screen outline

```text
SCREEN 1 · Start
+------------------------------------------------+
| Check today's ESP readings                     |
| Classifies this day's readings only.            |
| Educational prototype · limits and data note    |
| [Enter readings]                                 |
| Expert mode                                      |
+------------------------------------------------+

SCREEN 2 · Readings
+------------------------------------------------+
| Enter readings for one day                      |
| Pressure       [Intake] [Discharge]              |
| Temperature    [Motor]  [Intake]                 |
| Electrical     [Current] [Voltage] [Frequency]   |
| Pump readings  [Vibration] [Production]          |
| [Check readings]                                 |
| No equipment limits are inferred from this data. |
+------------------------------------------------+

SCREEN 3 · Result
+------------------------------------------------+
| Current reading: RUNNING / FAILED               |
| Model score: 64 / 100                            |
| Band: ELEVATED                                   |
| Meaning and limits · range note if applicable    |
| Educational prototype notice                     |
| [Check another reading] [Expert mode]            |
+------------------------------------------------+
```

Use visible labels, units and short hints. Mark up groups with `fieldset` and `legend`; use decimal input mode and inline validation. Keep fields in one column on mobile; use two columns for groups on wider screens. Empty result: “No result yet.” Loading: “Checking these readings…” Normal, Elevated and High each show their text label and score, not colour alone. Out-of-range: identify the reading and say it falls outside the training data range; this is not an equipment limit. Error: explain the field problem inline and focus the first invalid field. Announce completed results and errors through a live region. Keep each screen concise.

## 5. Input glossary

Feature names and units come from `src/config.py::FEATURE_ORDER`, `FEATURE_LABELS`, `FEATURE_UNITS` and `DATA_DICTIONARY`.

| Plain-language label | Hint | Unit |
|---|---|---|
| Intake pressure | Pressure of fluid entering the pump. | psi |
| Discharge pressure | Pressure of fluid leaving the pump. | psi |
| Motor temperature | Temperature at the ESP motor windings. | °F |
| Intake temperature | Temperature of well fluid at the pump intake. | °F |
| Motor current | Electrical current drawn by the motor. | A |
| Motor voltage | Voltage supplied to the motor. | V |
| Vibration | Measured pump vibration. | g |
| Drive frequency | Frequency sent by the variable speed drive. | Hz |
| Production rate | Volume produced per day. | bbl/d |

## 6. Honesty rules

- Say **current reading classification**. Never imply a future failure date or remaining useful life. The target is same-day `esp_status`. (`src/config.py::TARGET_RAW`, `src/data_loader.py::prepare_dataset`.)
- Suggested result copy: “The model classifies this reading as Failed. Model score: 64/100. A higher score means the model leaned more toward Failed for these readings. This is not a calibrated chance of failure and says nothing about future failure or remaining life.” Keep “Educational prototype. Dataset provenance is unverified; it covers six wells, and cross-well performance is near chance. Do not use this result alone for operating decisions.” (`src/config.py::DISCLAIMER`, `src/pages/about.py::_limitations_tab`, `src/pages/about.py::_methodology_tab`.)
- Keep class threshold and risk band distinct. The class threshold defaults to the bundle value or 0.50; risk defaults are Normal below 0.40, Elevated from 0.40 to below 0.70, and High at or above 0.70. (`src/prediction.py::predict_esp`, `risk_level`.)
- An out-of-range note compares readings with per-feature training minimum and maximum only. It does not reject or clip them and does not define safe equipment limits. Physical checks require finite values, nonnegative vibration/production/pressures/current/voltage/frequency, and positive pressure/current/voltage/frequency; no temperature limits are applied. (`src/prediction.py::_physical_checks`, `predict_esp`.)
- Never invent “normal” operating ranges. Ask an engineer or equipment manufacturer for limits.

## 7. Stack decision

**Recommend static HTML/CSS and vanilla JavaScript served by a small FastAPI app with `POST /predict`.** At startup, call `src.prediction::load_model`; the handler passes its bundle and the nine-value object to `src.prediction::predict_esp`. `src/prediction.py`, `src/config.py` and `src/preprocessing.py` do not import Streamlit. `src/state.py::init_state` initializes `trained_bundle` to `None`; `app.py::main` auto-loads data but does not load a model. The tracked `models/esp_predictai_model.joblib` is not excluded by `.gitignore`, yet the current Streamlit user must load or train a model into session state before use. A backend startup load removes that empty-model hurdle.

Streamlit is imported by `app.py`, `src/ui.py`, `src/state.py` and all nine modules in `src/pages/`; the standalone path can omit them. In the current UI, `src/pages/prediction_view.py::render` asks the user to load or train a model when the in-memory bundle is empty, even though the joblib artifact is tracked.

Main risks: pin Python/scikit-learn versions compatible with the joblib artifact; validate requests and return clear errors; keep the endpoint same-origin; avoid logging submitted readings; and preserve score-calibration and training-range caveats. FastAPI is an added dependency. Do not use the dataset's observed ranges as equipment limits.

## 8. Skills used (evidence)

| Skill | What you ran | Top 3 findings |
|---|---|---|
| Hallmark. `.codex/skills/hallmark/SKILL.md`; verbs: `audit <target>`, `redesign <target>` | Read skill and redesign reference; source-audited requested files. Node/npx are absent and Streamlit renders at runtime. | Tokenize colours/fonts and avoid remote font imports; replace `transition: all` and address focus, reduced motion and overflow; avoid eyebrows, tracked capitals, dot metadata/status, emoji, nested cards and dense grids. |
| Web Design Guidelines. `.codex/skills/web-design-guidelines/SKILL.md`; trigger: UI/accessibility review | Fetched current guidelines; reviewed requested UI files. | Associated labels and input modes; inline errors and focus management; visible focus, 4.5:1 contrast, 44px targets, responsive layout, reduced motion and text status. |
| UI/UX Pro Max. `.codex/skills/ui-ux-pro-max/SKILL.md`; trigger: interface design/review | Ran local design-system and focused UX searches; rejected an off-topic pattern; no `--persist`. | Minimal/Swiss direction fits; label and validate numeric fields; pair status text with colour. |
| awesome-design-md. Not a skill; cloned to a temporary folder outside the repository | Read `DESIGN.md` for Linear, Notion and Stripe. | Linear is dark and dense; Notion uses neutral surfaces and readable spacing; Stripe uses branded indigo, gradients and pill controls. See section 9. |
| Writing Guidelines. `.codex/skills/writing-guidelines/SKILL.md`; trigger: prose review | Fetched current guidance; reviewed labels, hints, results and notices. | Active voice and short sentences; explain acronyms and score meaning; avoid reassurance and future-failure claims. |
| Deploy to Vercel. `.codex/skills/deploy-to-vercel/SKILL.md`; trigger: Vercel deployment | Installed and skimmed only; not invoked because deployment is outside this task. | It installs/uses a CLI; may link a Vercel project; can push or deploy externally. No deployment was attempted. |

All `.codex/skills/...` paths are relative to the project root. UI/UX searches were local and non-persistent; guideline fetches were read-only. Hallmark can send optional comps to Together AI and its render audit writes `.hallmark` artifacts. UI/UX `--persist` writes a design-system; Vercel can link or deploy externally. These are expected behaviors, not suspicious; they were not run. Restart Codex for automatic discovery; the skills were followed manually here.

## 9. DESIGN.md recommendation

| Brand | Typography and density | Colour roles and components |
|---|---|---|
| [Linear](https://github.com/VoltAgent/awesome-design-md/blob/main/design-md/linear.app/DESIGN.md) | Custom Linear Sans; compact 4–48px spacing and dense technical screens. | Near-black canvas, pale text, lavender-blue accent; compact rectangular controls. Too dark and dense for first-time operators. |
| [Notion](https://github.com/VoltAgent/awesome-design-md/blob/main/design-md/notion/DESIGN.md) | Notion Sans; moderate 4–40px spacing with generous section gaps. | White/warm-neutral surfaces, clear text hierarchy, simple rectangular controls. **Recommend as token/structure reference only** for neutral roles, spacing and readable forms; omit its many pastel accents. |
| [Stripe](https://github.com/VoltAgent/awesome-design-md/blob/main/design-md/stripe/DESIGN.md) | Sohne; tight 2–32px rhythm and tabular numerals. | White/navy with indigo and gradient accents; polished pill buttons and compact fields. Too brand-specific for this tool. |

Adapt Notion’s neutral token roles and form structure. Hallmark’s modern-minimal genre owns the final visual direction in Milestone 2; do not clone a brand.

## 10. Open questions for the owner

1. Should Expert mode open the current Streamlit app in a separate tab or be hosted beside the new flow?
2. Should the class and band thresholds stay fixed and hidden from non-experts?
3. Who will supply approved operating limits and unit conventions, if the form needs equipment-specific guidance?
4. Should submitted readings remain only in browser memory, with no server-side history?
