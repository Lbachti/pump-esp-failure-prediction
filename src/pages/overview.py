"""Overview dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import DATE_COL, FEATURE_LABELS, FEATURE_UNITS, TARGET, WELL_COL
from ..data_loader import audit_dataset, human_source
from ..eda import (
    effect_size_figure,
    parameter_summary_table,
    status_donut,
    time_failure_rate,
    well_failure_comparison,
)
from ..plotting import empty_figure
from ..state import active_df, active_meta, goto, has_data, is_cleaned
from ..ui import (
    callout,
    format_float,
    format_int,
    format_pct,
    page_header,
    plotly_chart,
    render_kpis,
    section_header,
)


def _quick_actions() -> None:
    section_header("Continue the workflow", kicker="Next steps")
    cols = st.columns(3)
    with cols[0]:
        if st.button("Open Data Explorer", width="stretch", key="ov_go_data"):
            goto("Data Explorer")
            st.rerun()
    with cols[1]:
        if st.button("Train an ML model", width="stretch", key="ov_go_train"):
            goto("ML Model Training")
            st.rerun()
    with cols[2]:
        if st.button("Run a prediction", width="stretch", key="ov_go_pred"):
            goto("Failure Prediction")
            st.rerun()


def render() -> None:
    page_header(
        "Fleet overview",
        "Operational status, failure prevalence, and integrity signals calculated "
        "directly from the loaded ESP dataset.",
        eyebrow="ESP PREDICTAI / 01 · OVERVIEW",
        badges=[("Current-state classification", "amber"), ("Educational prototype", "blue")],
    )

    if not has_data():
        st.error("No dataset is loaded.")
        callout(
            "Dataset required",
            "Use Data Explorer to load the public GitHub dataset or upload a CSV file. "
            "All KPIs and charts are computed from the active dataset; nothing is hardcoded.",
            tone="warn",
        )
        if st.button("Go to Data Explorer", type="primary", key="ov_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    meta = active_meta()
    audit = audit_dataset(df, meta)

    notice = st.session_state.get("data_notice")
    if notice:
        callout("Dataset loaded", notice, tone="success", icon="✓")
    if is_cleaned():
        callout(
            "Cleaned dataset in use",
            "The KPIs below use the cleaned working dataset. The original data is still "
            "preserved in memory and can be restored from Data Cleaning.",
            tone="info",
        )

    if audit.get("likely_simulated"):
        reasons = audit.get("suspicion_reasons", [])
        summary = " ".join(reasons[:2]) if reasons else ""
        callout(
            "Provenance warning — read before using results",
            "The public dataset's provenance cannot be verified. Multiple statistical "
            "signatures are consistent with simulated or heavily processed data. "
            + summary
            + " Treat all model results as educational, not field evidence.",
            tone="warn",
            icon="⚠",
        )
    if audit.get("critical"):
        st.error("Critical data issues: " + " ".join(audit["critical"]))

    counts = audit["target_counts"]
    feature_means = {
        col: pd.to_numeric(df[col], errors="coerce").mean()
        for col in ["motor_temperature", "intake_pressure", "discharge_pressure"]
        if col in df.columns
    }
    cards = [
        {
            "label": "Total observations",
            "value": format_int(audit["n_rows"]),
            "sub": f"{audit['n_cols']} columns · working dataset",
            "tone": "blue",
            "icon": "▦",
        },
        {
            "label": "Wells",
            "value": format_int(audit["n_wells"]),
            "sub": "Unique well identifiers",
            "tone": "cyan",
            "icon": "◉",
        },
        {
            "label": "Running observations",
            "value": format_int(counts["running"]),
            "sub": "Target = 0",
            "tone": "green",
            "icon": "●",
        },
        {
            "label": "Failed observations",
            "value": format_int(counts["failed"]),
            "sub": "Target = 1",
            "tone": "red",
            "icon": "▲",
        },
        {
            "label": "Failure percentage",
            "value": format_pct(audit["failure_rate"]),
            "sub": "Share of labelled observations",
            "tone": "amber",
            "icon": "◆",
        },
        {
            "label": "Avg motor temperature",
            "value": f"{format_float(feature_means.get('motor_temperature'), 1)} °F",
            "sub": "Arithmetic mean · all statuses",
            "tone": "cyan",
            "icon": "T",
        },
        {
            "label": "Avg intake pressure",
            "value": f"{format_float(feature_means.get('intake_pressure'), 0)} psi",
            "sub": "Arithmetic mean · all statuses",
            "tone": "blue",
            "icon": "P",
        },
        {
            "label": "Avg discharge pressure",
            "value": f"{format_float(feature_means.get('discharge_pressure'), 0)} psi",
            "sub": "Arithmetic mean · all statuses",
            "tone": "blue",
            "icon": "P",
        },
    ]
    render_kpis(cards, columns=4)

    section_header(
        "Fleet operating status",
        "Failure prevalence is high in this dataset; accuracy alone would be misleading.",
        kicker="Status & time",
    )
    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        plotly_chart(status_donut(df), key="ov_donut")
    with right:
        if DATE_COL in df.columns and df[DATE_COL].notna().any():
            plotly_chart(time_failure_rate(df), key="ov_time")
        else:
            plotly_chart(
                empty_figure("No reliable date column: time analysis unavailable."),
                key="ov_time_empty",
            )

    section_header(
        "Well-level comparison",
        "Failure rates differ materially across wells; this is a warning that pooled "
        "metrics may hide well-specific behaviour.",
        kicker="Fleet heterogeneity",
    )
    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        plotly_chart(well_failure_comparison(df), key="ov_wells")
    with right:
        plotly_chart(effect_size_figure(df, list(FEATURE_LABELS.keys())), key="ov_effect")

    section_header(
        "Operational parameter overview",
        "Means are descriptive statistics, not operating limits. Standardised differences "
        "show association, not causation.",
        kicker="Operating envelope",
    )
    table = parameter_summary_table(df, [c for c in FEATURE_LABELS if c in df.columns])
    if not table.empty:
        display = table.copy()
        display["unit"] = display["feature"].map(FEATURE_UNITS)
        display = display.rename(
            columns={
                "label": "Parameter",
                "mean_running": "Mean · running",
                "mean_failed": "Mean · failed",
                "median": "Median",
                "min": "Min",
                "max": "Max",
                "std": "Std. dev.",
                "unit": "Unit",
            }
        )
        st.dataframe(
            display[
                [
                    "Parameter",
                    "Unit",
                    "Mean · running",
                    "Mean · failed",
                    "Median",
                    "Min",
                    "Max",
                    "Std. dev.",
                ]
            ].round(3),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Units follow the public dataset description: pressure psi, temperature °F, "
        "current A, voltage V, vibration g, frequency Hz, production bbl/d."
    )

    section_header(
        "Data integrity snapshot",
        "Computed from the active dataset. Excluded outcome columns are never model inputs.",
        kicker="Audit",
    )
    leakage_cols = [c for c, v in audit["leakage"].items() if v["deterministic_separation"]]
    integrity_cards = [
        {
            "label": "Missing cells",
            "value": format_int(audit["missing_cells"]),
            "sub": "Across all rows and columns",
            "tone": "green" if audit["missing_cells"] == 0 else "amber",
            "icon": "◌",
        },
        {
            "label": "Duplicate rows",
            "value": format_int(audit["duplicate_rows"]),
            "sub": "Exact row duplicates",
            "tone": "green" if audit["duplicate_rows"] == 0 else "amber",
            "icon": "⧉",
        },
        {
            "label": "Deterministic leakage",
            "value": str(len(leakage_cols)) if leakage_cols else "None",
            "sub": ", ".join(leakage_cols) if leakage_cols else "No perfect separation found",
            "tone": "red" if leakage_cols else "green",
            "icon": "!",
        },
        {
            "label": "Validation support",
            "value": "3 strategies" if audit["n_wells"] >= 3 and audit["has_date"] else "Limited",
            "sub": "Random · time · well-based",
            "tone": "cyan" if audit["n_wells"] >= 3 and audit["has_date"] else "amber",
            "icon": "✓",
        },
    ]
    render_kpis(integrity_cards, columns=4)

    with st.expander("Why the dataset is flagged as unverified", expanded=False):
        st.markdown(
            "\n".join(f"- {reason}" for reason in audit.get("suspicion_reasons", []))
            or "No specific provenance flags were detected."
        )
        st.caption(
            "These are statistical signatures, not proof of synthetic origin. The public "
            "repository provides no field metadata, generation script, or equipment "
            "specifications. See Project Information for the full audit."
        )

    _quick_actions()
