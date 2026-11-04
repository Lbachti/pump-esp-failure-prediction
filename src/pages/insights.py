"""Petroleum-engineering interpretation page."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from ..config import FEATURE_LABELS, FEATURE_UNITS
from ..data_loader import audit_dataset, get_feature_columns
from ..eda import failure_rate_by_bins, signal_figure, univariate_signal_table
from ..engineering_insights import (
    DEPLOYMENT_REQUIREMENTS,
    EVIDENCE_LAYERS,
    FAILURE_MECHANISMS,
    PARAMETER_KNOWLEDGE,
    classification_vs_prediction,
    dataset_parameter_table,
    engineering_disclaimer,
    esp_schematic_html,
)
from ..explainability import global_importance
from ..state import active_df, active_meta, goto, has_data
from ..ui import (
    callout,
    format_float,
    format_pct,
    page_header,
    plotly_chart,
    section_header,
)


def _parameter_card(feature: str, df: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    knowledge = PARAMETER_KNOWLEDGE[feature]
    stat_table = univariate_signal_table(df, [feature])
    stat = stat_table.iloc[0].to_dict() if not stat_table.empty else None
    st.markdown(
        f"### {knowledge['label']}  \n"
        f"<span class='muted small'>{knowledge['unit']} · {knowledge['summary']}</span>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Established principles**")
        for item in knowledge["established"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Potential mechanisms**")
        for item in knowledge["mechanisms"]:
            st.markdown(f"- {item}")
    with c3:
        st.markdown("**Diagnostic context**")
        for item in knowledge["diagnostics"]:
            st.markdown(f"- {item}")

    if stat:
        st.markdown("**Dataset-derived statistics**")
        cards = st.columns(4)
        cards[0].metric("Mean · running", format_float(stat["mean_running"], 3))
        cards[1].metric("Mean · failed", format_float(stat["mean_failed"], 3))
        cards[2].metric("Univariate ROC-AUC", format_float(stat["auc"], 3))
        cards[3].metric("Mann–Whitney p", f"{stat['p_value']:.3g}")
        st.caption(stat["direction"] + " · This is association, not causation.")
    if bundle is not None:
        imp = global_importance(bundle)
        row = imp[imp["feature"] == feature]
        if not row.empty:
            st.caption(
                f"Model importance share for this feature: "
                f"{float(row['share'].iloc[0]):.3f} ({row['method'].iloc[0]})."
            )
    st.caption(knowledge["caveat"])


def _mechanism_grid() -> None:
    section_header(
        "ESP failure mechanisms",
        "Common physical pathways that motivate condition monitoring. They are not universal "
        "failure rules and must be evaluated against equipment specifications.",
        kicker="Engineering background",
    )
    for i in range(0, len(FAILURE_MECHANISMS), 2):
        cols = st.columns(2, gap="large")
        for col, mechanism in zip(cols, FAILURE_MECHANISMS[i : i + 2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{mechanism['title']}**")
                    st.write(mechanism["description"])
                    st.caption("Indicators: " + " · ".join(mechanism["indicators"]))
                    st.caption(mechanism["note"])


def _evidence_layers() -> None:
    section_header(
        "Evidence layers",
        "Keep these categories separate when discussing a possible failure mechanism.",
        kicker="Epistemics",
    )
    cols = st.columns(2, gap="large")
    tones = ["cyan", "blue", "amber", "red"]
    for i, layer in enumerate(EVIDENCE_LAYERS):
        with cols[i % 2]:
            st.markdown(
                f"""
<div class="param-card">
  <div class="param-title">{html.escape(layer['title'])}</div>
  <div class="param-text">{html.escape(layer['text'])}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def render() -> None:
    page_header(
        "Engineering insights",
        "Petroleum-engineering context for ESP operating parameters, failure mechanisms, and "
        "the limits of what this dataset and model can support.",
        eyebrow="ESP PREDICTAI / 08 · ENGINEERING",
        badges=[("Established principles", "cyan"), ("Dataset statistics", "blue"), ("Hypotheses", "amber")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="insights_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    meta = active_meta()
    audit = audit_dataset(df, meta)
    features = get_feature_columns(df)
    bundle = st.session_state.get("trained_bundle")

    callout(
        "Read this first",
        "The relationships below combine established ESP engineering principles with "
        "statistics from an unverified public dataset. They are not operating limits, "
        "not a diagnosis, and not a substitute for equipment specifications and field data.",
        tone="warn",
        icon="⚠",
    )
    if audit.get("likely_simulated"):
        callout(
            "Dataset provenance",
            "The active dataset has multiple signatures consistent with simulated or heavily "
            "processed data. Any engineering story derived from it is illustrative. "
            "See Project Information for the audit details.",
            tone="danger",
        )

    st.markdown(esp_schematic_html(), unsafe_allow_html=True)
    _mechanism_grid()

    section_header(
        "Parameter deep dive",
        "Select an operating parameter to combine engineering principles, dataset statistics, "
        "and model importance in one view.",
        kicker="Interpretation",
    )
    feature = st.selectbox(
        "Operating parameter",
        features,
        format_func=lambda f: f"{FEATURE_LABELS.get(f, f)} ({FEATURE_UNITS.get(f, '—')})",
        key="insights_feature",
    )
    with st.container(border=True):
        _parameter_card(feature, df, bundle)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        fig, table = failure_rate_by_bins(df, feature, n_bins=8)
        plotly_chart(fig, key="insights_bins")
        st.caption(
            "Bins are quantiles of the observed feature values. A trend across bins is a "
            "dataset association and may be confounded by well identity, operating strategy, "
            "or the data-generation process."
        )

    section_header(
        "All-parameter evidence table",
        "Univariate association with failure, plus model importance when a model is available.",
        kicker="Cross-parameter",
    )
    stats = dataset_parameter_table(df, features)
    if not stats.empty:
        display = stats.copy()
        if bundle is not None:
            imp = global_importance(bundle)
            if not imp.empty:
                display = display.merge(
                    imp[["feature", "share", "method"]],
                    on="feature",
                    how="left",
                )
                display = display.rename(
                    columns={"share": "Model importance share", "method": "Importance method"}
                )
        display = display.rename(
            columns={
                "label": "Parameter",
                "auc": "Univariate ROC-AUC",
                "signal": "Signal strength",
                "direction": "Direction",
                "p_value": "Mann–Whitney p",
                "mean_running": "Mean · running",
                "mean_failed": "Mean · failed",
            }
        )
        preferred = [
            "Parameter",
            "Univariate ROC-AUC",
            "Signal strength",
            "Direction",
            "Mann–Whitney p",
            "Mean · running",
            "Mean · failed",
            "Model importance share",
            "Importance method",
        ]
        cols = [c for c in preferred if c in display.columns]
        st.dataframe(display[cols].round(4), width="stretch", hide_index=True)
        st.caption(
            "Signal strength = max(AUC, 1−AUC). Values near 0.5 mean the single feature "
            "barely separates failure from running. A small p-value is not a causal result."
        )
    plotly_chart(signal_figure(df, features), key="insights_signal")

    _evidence_layers()

    section_header(
        "Classification versus future prediction",
        "This distinction is central to the project's claims.",
        kicker="Scope",
    )
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        callout(
            "What this app does",
            classification_vs_prediction(),
            tone="info",
        )
        st.markdown("**What a true predictive-maintenance target would require**")
        for item in DEPLOYMENT_REQUIREMENTS:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Current evidence status**")
        rows = [
            ("Target type", "Current-day status (failed / running)"),
            ("Future lead time", "Not encoded"),
            ("Wells", f"{audit['n_wells']}"),
            ("Failure prevalence", format_pct(audit["failure_rate"])),
            ("Provenance", "Unverified public upload"),
            ("Well-based generalisation", "Test in ML Model Training"),
        ]
        if bundle is not None:
            metrics = bundle.get("metrics", {})
            rows.append(("Model", bundle.get("model_label", "—")))
            rows.append(("Holdout recall", format_float(metrics.get("recall"), 3)))
            rows.append(("Holdout ROC-AUC", format_float(metrics.get("roc_auc"), 3)))
        st.dataframe(
            pd.DataFrame(rows, columns=["Item", "Status"]),
            width="stretch",
            hide_index=True,
        )

    section_header(
        "From model output to field action",
        "A responsible deployment path is longer than fitting a classifier.",
        kicker="Deployment",
    )
    steps = [
        ("1 · Define the decision", "Specify the maintenance action, lead time, alarm owner, and acceptable false-alarm burden."),
        ("2 · Build a valid target", "Use run-to-failure histories and a future-failure window; avoid contemporaneous outcome leakage."),
        ("3 · Validate by well and time", "Hold out entire wells and future periods; report uncertainty and subgroup performance."),
        ("4 · Calibrate and monitor", "Calibrate probabilities, monitor drift, and retrain with field feedback."),
        ("5 · Field trial", "Compare model alarms with engineering diagnosis and manufacturer limits before operational use."),
    ]
    cols = st.columns(len(steps))
    for col, (title, text) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="param-card"><div class="param-title">{html.escape(title)}</div>'
                f'<div class="param-text">{html.escape(text)}</div></div>',
                unsafe_allow_html=True,
            )

    callout(
        "Engineering disclaimer",
        engineering_disclaimer(),
        tone="warn",
        icon="⚠",
    )

    section_header("Continue", kicker="Workflow")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Train a model", width="stretch", key="insights_go_train"):
            goto("ML Model Training")
            st.rerun()
    with c2:
        if st.button("Run a prediction", width="stretch", key="insights_go_pred"):
            goto("Failure Prediction")
            st.rerun()
    with c3:
        if st.button("Read project information", width="stretch", key="insights_go_about"):
            goto("Project Information")
            st.rerun()
