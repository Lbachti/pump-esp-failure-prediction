"""Project information, methodology, limitations, and reproducibility page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import (
    APP_VERSION,
    DISCLAIMER,
    FEATURE_LABELS,
    GITHUB_CSV_URL,
    GITHUB_REPO_URL,
    MODEL_PATH,
    TARGET,
)
from ..data_loader import audit_dataset, human_source
from ..state import active_df, active_meta, current_hash, goto, has_data
from ..ui import (
    callout,
    format_float,
    format_int,
    format_pct,
    page_header,
    section_header,
)


def _overview_tab() -> None:
    section_header(
        "Project objective",
        "An educational, end-to-end ESP monitoring and failure-classification prototype.",
        kicker="Overview",
    )
    st.markdown(
        """
**ESP PredictAI** integrates petroleum engineering, artificial lift, data science, machine
learning, explainability, and an interactive industrial dashboard. It is designed for a
university engineering project, technical demonstration, portfolio, or as a foundation for
further research.

The application supports:

- Inspection and auditing of an ESP operating dataset.
- Transparent, reversible data cleaning.
- Interactive exploratory analysis of operating parameters.
- Leakage-safe training of Logistic Regression, Random Forest, and XGBoost classifiers.
- Random, temporal, and well-based validation.
- Recall-first evaluation with an adjustable decision threshold.
- Manual ESP operating-snapshot scoring with model-score risk bands.
- Native feature importance, permutation importance, SHAP, and individual attribution.
- Petroleum-engineering interpretation with explicit epistemic guardrails.
        """
    )
    callout(
        "Core distinction",
        "The dataset target is the operating status recorded on the same day. This is a "
        "current-state failure classification problem. It is not a future-failure prediction "
        "or remaining-useful-life model, because no future label or lead time exists in the data.",
        tone="warn",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Engineering**")
        st.write("ESP lift principles, failure mechanisms, operating parameters, diagnostics.")
    with c2:
        st.markdown("**Data science**")
        st.write("Data audit, cleaning, EDA, leakage control, validation design, calibration.")
    with c3:
        st.markdown("**Software**")
        st.write("Modular Python, Streamlit, Plotly, scikit-learn, XGBoost, SHAP, joblib.")


def _dataset_tab(audit: dict) -> None:
    section_header(
        "Dataset source",
        "Public GitHub repository and raw CSV. The app attempts a live download and falls "
        "back to the bundled cached copy.",
        kicker="Provenance",
    )
    st.markdown(f"- Repository: [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})")
    st.markdown(f"- CSV: [{GITHUB_CSV_URL}]({GITHUB_CSV_URL})")
    st.markdown(f"- Current source: **{human_source(active_meta())}**")
    if audit.get("sha256"):
        st.markdown(f"- Dataset SHA-256: `{audit['sha256']}`")
    if audit.get("date_min") and audit.get("date_max"):
        st.markdown(
            f"- Date range: **{audit['date_min']} → {audit['date_max']}** "
            f"({audit['n_unique_dates']:,} distinct dates)"
        )
    st.markdown(
        f"- Shape: **{audit['n_rows']:,} rows × {audit['n_cols']:,} columns** · "
        f"**{audit['n_wells']} wells**"
    )

    section_header(
        "Target definition and encoding",
        "The application follows the required project specification, not the source notebook's "
        "opposite convention.",
        kicker="Target",
    )
    st.markdown(
        """
- Raw column: `esp_status` with values `running` / `failed`.
- Required model encoding: **running = 0, failed = 1**.
- `esp_status` is preserved for display; a separate `esp_failure` column is created for modelling.
- `esp_status` is never used as a model input.
- `downtime_hours` and `production_loss` are excluded because they deterministically reveal the label.
- `well_id` and `date` are preserved for grouping, temporal analysis, and validation, but are not model features.
        """
    )
    callout(
        "Source encoding discrepancy",
        "The public repository README states running = 1 and failed = 0, and the source notebook "
        "uses that reverse encoding. This application deliberately follows the project "
        "specification: running = 0, failed = 1. Results should not be compared to the notebook "
        "without accounting for this difference.",
        tone="info",
    )
    st.write(
        "The target is a **current-state** label: it describes the status on the measurement "
        "day. It is not a future failure event, a failure-within-N-days label, or a "
        "remaining-useful-life target."
    )

    section_header(
        "Data-integrity audit",
        "All figures are computed live from the active dataset.",
        kicker="Audit",
    )
    rows = [
        ("Rows", format_int(audit["n_rows"])),
        ("Columns", format_int(audit["n_cols"])),
        ("Wells", format_int(audit["n_wells"])),
        ("Missing cells", format_int(audit["missing_cells"])),
        ("Exact duplicate rows", format_int(audit["duplicate_rows"])),
        ("Duplicate well/date pairs", format_int(audit["duplicate_well_dates"])),
        ("Running observations", format_int(audit["target_counts"]["running"])),
        ("Failed observations", format_int(audit["target_counts"]["failed"])),
        ("Failure prevalence", format_pct(audit["failure_rate"])),
        ("Complete gap-free panel", "Yes" if audit["complete_panel"] else "No"),
        ("Near-independent daily labels", "Yes" if audit["near_independent_labels"] else "No"),
        ("Provenance suspicion", audit["suspicion_level"]),
        ("Likely simulated / heavily processed", "Yes" if audit["likely_simulated"] else "No"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Result"]), width="stretch", hide_index=True)

    if audit.get("suspicion_reasons"):
        st.markdown("**Why the dataset is flagged**")
        for reason in audit["suspicion_reasons"]:
            st.markdown(f"- {reason}")
    callout(
        "Leakage finding",
        "In the source data, downtime_hours and production_loss are exactly zero for every "
        "running record and non-zero for every failed record. Any model using them would "
        "trivially reproduce the target. They are excluded everywhere in this application.",
        tone="danger",
    )
    callout(
        "Provenance conclusion",
        "The dataset's origin cannot be independently verified from the repository. It has a "
        "complete balanced panel, no missingness, rounded/narrow sensor ranges, a 42.6% daily "
        "failure prevalence, deterministic outcome columns, and failure labels with almost no "
        "day-to-day persistence. These signatures are consistent with simulated data. Treat "
        "all findings as educational, not as field evidence.",
        tone="warn",
    )


def _methodology_tab() -> None:
    section_header(
        "Feature set and exclusions",
        "Only operational sensor/process variables are model inputs.",
        kicker="Features",
    )
    st.markdown(
        "- Included when available: " + ", ".join(f"`{f}`" for f in FEATURE_LABELS) + "."
    )
    st.markdown(
        "- Excluded: `esp_status` (target), `downtime_hours` and `production_loss` (outcome "
        "leakage), `well_id` and `date` (identifiers used for validation only)."
    )
    st.markdown(
        "- Missing values are imputed with the training-fold median inside the pipeline. "
        "Logistic Regression uses standardisation inside the pipeline. Tree models do not need scaling."
    )

    section_header(
        "Validation strategies",
        "The choice of split is the most important integrity decision in this project.",
        kicker="Validation",
    )
    st.markdown(
        """
**Random stratified split (baseline).** Rows are split at random while preserving class
prevalence. Because the dataset is a daily panel, rows from the same well appear in both
train and test. The model can learn well-specific operating signatures, so this split is
usually optimistic and should not be the only evaluation.

**Time-based split.** Earlier dates train the model and later dates test it. This tests
temporal extrapolation and is stronger than a random row split, but the same wells still
appear in both sets. It does not test a new well.

**Well-based split.** Entire wells are held out. This is the most conservative test of
cross-well generalisation. With only six wells, the estimate has high variance, but it is
still the most honest check available here. A large drop from random/time performance to
well-based performance is evidence that the model is exploiting well identity rather than
transferable failure physics.

**Well-grouped cross-validation.** Multiple folds hold out different groups of wells. It
provides a more robust cross-well estimate than one split, subject to the same small-well
limitation.
        """
    )
    callout(
        "What we observed during development",
        "With the public CSV, random and time-based splits produced very high accuracy and "
        "ROC-AUC, while well-based evaluation fell close to chance. The app therefore presents "
        "the well-based result prominently and includes a validation-strategy comparison. "
        "High random-split metrics are not evidence of a deployable cross-well model.",
        tone="warn",
    )

    section_header(
        "Algorithms and evaluation",
        "Recall is highlighted because a missed failure can be more costly than a false alarm.",
        kicker="Modelling",
    )
    st.markdown(
        """
- **Logistic Regression** — interpretable linear baseline with standardised features.
- **Random Forest** — nonlinear tree ensemble with impurity/gain importance.
- **XGBoost** — gradient-boosted trees when the package is installed.
- **Metrics** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix.
- **Threshold analysis** — the decision threshold is adjustable; lower thresholds detect more
  failures but raise false alarms.
- **Calibration** — reliability curve and Brier score are shown; high discrimination does not
  guarantee calibrated probabilities.
- **Persistence** — the fitted preprocessing pipeline and estimator are saved together with joblib.
- **Leakage control** — the split is created before fitting; imputation/scaling are fitted only
  on the training portion.
        """
    )

    section_header(
        "Explainability",
        "Model explanation is not physical causation.",
        kicker="Explainability",
    )
    st.markdown(
        """
- Native feature importance for tree models; absolute standardised coefficients for Logistic Regression.
- Permutation importance on a holdout split when available.
- SHAP TreeExplainer / LinearExplainer for global summary and individual attribution.
- A clearly labelled fallback for individual cases when SHAP is unavailable.
- No feature-importance result is presented as proof of a physical failure mechanism.
        """
    )


def _limitations_tab(audit: dict) -> None:
    section_header(
        "Project limitations",
        "These are not minor caveats; they determine what can responsibly be claimed.",
        kicker="Limitations",
    )
    limitations = [
        ("Current-state, not future", "The target describes today's status. There is no lead time, no failure-within-N-days label, and no remaining-useful-life target. The app does not claim future failure prediction."),
        ("Unverified provenance", "The public repository provides no field metadata, equipment specifications, generation script, or independent validation. The statistical signatures strongly suggest simulated or heavily processed data."),
        ("Small well count", "Only six wells are available. Well-based validation has high variance, and a single split can be dominated by which wells happen to be held out."),
        ("No equipment context", "Pump curves, motor ratings, cable data, completion details, fluid properties, and manufacturer limits are absent. The app cannot make equipment-specific diagnoses."),
        ("Daily snapshots", "Daily data cannot capture high-frequency vibration, transient slugs, startup/shutdown events, or short-duration electrical faults."),
        ("No maintenance history", "There are no work orders, failure causes, repair outcomes, or run-to-failure histories. The label does not distinguish failure mechanisms."),
        ("Deterministic leakage", "downtime_hours and production_loss perfectly reveal the target and are excluded. Other hidden leakage or artefacts may remain."),
        ("Probability calibration", "Model scores are not guaranteed calibrated field probabilities. Calibration is evaluated on a small holdout and may not transfer."),
        ("Class prevalence", "A 42.6% daily failure rate is implausible for a real producing ESP fleet and makes accuracy misleading."),
        ("No causal claims", "Correlation, feature importance, and SHAP explain model behaviour, not physical causation."),
    ]
    for title, text in limitations:
        st.markdown(f"**{title}.** {text}")

    section_header(
        "What this application does not do",
        "Explicit scope boundaries.",
        kicker="Scope",
    )
    st.markdown(
        """
- It does not predict a future failure date, remaining useful life, or failure probability over a lead-time horizon.
- It does not diagnose a specific failure mechanism from a single observation.
- It does not replace manufacturer limits, equipment specifications, field measurements, or a qualified engineer.
- It does not provide a certified safety, control, or maintenance system.
- It does not claim that the public dataset represents real field operations.
- It does not automatically take operational action.
        """
    )
    callout(
        "Classification versus prediction",
        "A classifier can estimate whether a snapshot looks like the failed class. A predictive-maintenance "
        "system needs a future event label, a defined lead time, temporal validation, calibration, and a "
        "field response plan. This project implements the former and explicitly declines to claim the latter.",
        tone="warn",
    )


def _about_tab(audit: dict) -> None:
    section_header(
        "Architecture",
        "Modular data, modelling, explainability, and presentation layers.",
        kicker="Software",
    )
    st.code(
        """ESP_PredictAI/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── data/
│   └── esp_downhole_data.csv
├── models/
│   └── esp_predictai_model.joblib   (created when you save a model)
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
    └── test_prediction.py""",
        language="text",
    )
    st.markdown(
        f"""
**Version:** {APP_VERSION}  
**Tested locally with:** Python 3.12, Streamlit 1.66, pandas 3.0.2, NumPy 2.2.3,
scikit-learn 1.9.0, XGBoost 2.1.4, SHAP 0.54.1, Plotly 7.1.0.  
**Model file:** `{MODEL_PATH}`
        """
    )

    section_header("Installation and execution", kicker="Run locally")
    st.code(
        """# from the project root
python -m venv .venv
.venv\\Scripts\\activate          # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py""",
        language="powershell",
    )
    st.code("pytest -q", language="powershell")
    st.caption(
        "The app attempts to download the public CSV at startup and falls back to the bundled "
        "copy in data/. If both fail, use the upload control in Data Explorer."
    )

    section_header("Reproducibility", kicker="Data and model")
    st.markdown(
        f"- Active dataset fingerprint: `{current_hash()[:24] or '—'}`\n"
        f"- Source SHA-256: `{audit.get('sha256') or '—'}`\n"
        f"- Target encoding: running = 0, failed = 1\n"
        f"- Model bundle: preprocessing pipeline + estimator + metadata + training fingerprint"
    )
    st.caption(
        "Model results depend on the selected algorithm, validation strategy, test size, random "
        "seed, threshold, and cleaning operations. The app recomputes metrics from the active data; "
        "no metrics are hardcoded."
    )

    section_header("Future improvements", kicker="Roadmap")
    improvements = [
        "Replace the public dataset with run-to-failure field data and maintenance work orders.",
        "Define a future-failure target (for example, failure within 30 days) and validate with a horizon-aware temporal split.",
        "Incorporate high-frequency vibration, motor current signature, and VSD transient data.",
        "Add equipment specifications, pump curves, motor ratings, cable data, and fluid properties.",
        "Develop well-specific or hierarchical models and evaluate subgroup performance.",
        "Calibrate probabilities prospectively and monitor drift and alarm burden.",
        "Add survival analysis or remaining-useful-life modelling once event times are available.",
        "Integrate alarm management, maintenance workflows, and cost-sensitive threshold optimisation.",
        "Add automated tests for all pages and a reproducible experiment log.",
    ]
    for item in improvements:
        st.markdown(f"- {item}")

    callout("Disclaimer", DISCLAIMER, tone="danger", icon="⚠")


def render() -> None:
    page_header(
        "Project information",
        "Scope, data provenance, methodology, limitations, reproducibility, and the boundary "
        "between failure classification and future failure prediction.",
        eyebrow="ESP PREDICTAI / 09 · ABOUT",
        badges=[("Educational prototype", "blue"), ("Not for field decisions", "amber")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="about_no_data"):
            goto("Data Explorer")
            st.rerun()
        return
    audit = audit_dataset(active_df(), active_meta())
    tab_overview, tab_dataset, tab_method, tab_limits, tab_about = st.tabs(
        ["Overview", "Dataset & integrity", "Methodology", "Limitations", "Stack & reproducibility"]
    )
    with tab_overview:
        _overview_tab()
    with tab_dataset:
        _dataset_tab(audit)
    with tab_method:
        _methodology_tab()
    with tab_limits:
        _limitations_tab(audit)
    with tab_about:
        _about_tab(audit)
