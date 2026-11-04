"""Interactive data-cleaning page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import COLORS, FEATURE_LABELS, TARGET
from ..data_cleaning import (
    before_after_stats,
    cleanable_numeric_columns,
    duplicate_count,
    impute_missing,
    missing_summary,
    outlier_bounds,
    outlier_mask,
    remove_duplicate_rows,
    scale_numeric,
    treat_outliers,
)
from ..plotting import apply_theme
from ..state import (
    active_df,
    active_meta,
    apply_cleaning,
    current_hash,
    goto,
    has_data,
    reset_cleaning,
    undo_cleaning,
)
from ..ui import (
    callout,
    format_float,
    format_int,
    page_header,
    plotly_chart,
    render_kpis,
    section_header,
)


def _run_and_refresh(new_df, operation: str, details, columns: list[str] | None) -> None:
    apply_cleaning(new_df, operation, details, columns)
    st.rerun()


def _missing_section(df: pd.DataFrame) -> None:
    section_header(
        "Missing values",
        "Mean, median, mode, or row deletion. Only operational features can be selected; "
        "the target and identifiers are protected.",
        kicker="Step 1",
    )
    summary = missing_summary(df)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        if summary.empty:
            st.success("No missing values in the working dataset.")
        else:
            st.dataframe(
                summary.round(3),
                width="stretch",
                hide_index=True,
            )
    with c2:
        columns = cleanable_numeric_columns(df)
        if not columns:
            st.info("No cleanable operational features are available.")
            return
        missing_cols = [
            c for c in columns if df[c].isna().any()
        ] or columns
        selected = st.multiselect(
            "Columns to treat",
            columns,
            default=missing_cols,
            key="clean_missing_cols",
        )
        method = st.selectbox(
            "Missing-value method",
            ["Median", "Mean", "Mode", "Drop rows"],
            key="clean_missing_method",
        )
        if st.button("Apply missing-value treatment", type="primary", key="clean_missing_apply"):
            if not selected:
                st.error("Select at least one column.")
            else:
                try:
                    with st.spinner("Handling missing values..."):
                        new_df, info = impute_missing(df, selected, method)
                    _run_and_refresh(
                        new_df, info["operation"], info.get("details"), selected
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))


def _duplicate_section(df: pd.DataFrame) -> None:
    section_header(
        "Duplicate records",
        "Exact duplicate rows can inflate apparent performance if they cross train/test splits.",
        kicker="Step 2",
    )
    count = duplicate_count(df)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.metric("Exact duplicate rows", f"{count:,}")
    with c2:
        if count == 0:
            st.success("No exact duplicates detected.")
        else:
            st.warning(
                f"{count:,} duplicate rows detected. Removing them changes the working dataset "
                "and invalidates any trained model."
            )
            if st.button("Remove duplicate rows", type="primary", key="clean_dupes_apply"):
                with st.spinner("Removing duplicates..."):
                    new_df, info = remove_duplicate_rows(df)
                _run_and_refresh(new_df, info["operation"], info.get("details"), None)


def _outlier_section(df: pd.DataFrame) -> None:
    section_header(
        "Outliers",
        "IQR or Z-score detection, then remove or cap. Bounds are computed from the selected "
        "features only; the target is never treated.",
        kicker="Step 3",
    )
    columns = cleanable_numeric_columns(df)
    if not columns:
        st.info("No numeric operational features are available.")
        return
    c1, c2, c3, c4 = st.columns([1.3, 0.9, 0.9, 0.9])
    with c1:
        selected = st.multiselect(
            "Columns",
            columns,
            default=columns,
            key="clean_outlier_cols",
        )
    with c2:
        method = st.selectbox("Detection method", ["IQR", "Z-score"], key="clean_outlier_method")
    with c3:
        factor = st.number_input(
            "IQR multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="clean_outlier_factor",
            disabled=method != "IQR",
        )
    with c4:
        z_threshold = st.number_input(
            "Z threshold",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="clean_outlier_z",
            disabled=method != "Z-score",
        )

    if not selected:
        st.info("Select at least one column.")
        return
    try:
        bounds = outlier_bounds(df, selected, method, factor, z_threshold)
        mask = outlier_mask(df, selected, method, factor, z_threshold)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    total_rows = int(mask.sum())
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.metric("Rows with at least one outlier", f"{total_rows:,}")
        st.caption(
            f"{total_rows / max(1, len(df)):.2%} of the working dataset. Detection is "
            "sensitive to the selected method and multiplier."
        )
        if st.button("Detect outliers", key="clean_outlier_detect"):
            st.session_state["show_outlier_report"] = True
    with c2:
        if st.session_state.get("show_outlier_report", False):
            st.dataframe(
                bounds.round(3),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Click Detect outliers to inspect bounds and per-column counts.")

    treatment = st.selectbox(
        "Treatment",
        ["Cap values", "Remove rows"],
        key="clean_outlier_treatment",
    )
    st.caption(
        "Cap values winsorises each selected feature to its computed bounds. Remove rows "
        "drops any row with an outlier in the selected features. Neither operation changes the target."
    )
    if st.button("Apply outlier treatment", type="primary", key="clean_outlier_apply"):
        if total_rows == 0:
            st.info("No outliers to treat with the current settings.")
        else:
            try:
                with st.spinner("Applying outlier treatment..."):
                    new_df, info = treat_outliers(
                        df,
                        selected,
                        method=method,
                        treatment=treatment,
                        factor=factor,
                        z_threshold=z_threshold,
                    )
                _run_and_refresh(new_df, info["operation"], info.get("details"), selected)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


def _scaling_section(df: pd.DataFrame) -> None:
    section_header(
        "Normalisation / standardisation",
        "Optional transformation of the working dataset for inspection. The modelling "
        "pipeline applies its own leakage-safe scaling, so this is not required for training.",
        kicker="Step 4",
    )
    columns = cleanable_numeric_columns(df)
    if not columns:
        st.info("No numeric operational features are available.")
        return
    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    with c1:
        selected = st.multiselect(
            "Columns to scale",
            columns,
            default=columns,
            key="clean_scale_cols",
        )
    with c2:
        method = st.selectbox(
            "Scaling method",
            ["MinMaxScaler", "StandardScaler"],
            key="clean_scale_method",
        )
    with c3:
        st.write("")
        st.write("")
        if st.button("Apply scaling", key="clean_scale_apply"):
            if not selected:
                st.error("Select at least one column.")
            else:
                try:
                    with st.spinner("Scaling selected features..."):
                        new_df, info = scale_numeric(df, selected, method)
                    _run_and_refresh(new_df, info["operation"], info.get("details"), selected)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    callout(
        "Double-scaling caution",
        "If you scale here and then train a Logistic Regression, the pipeline will scale "
        "again inside the training split. Tree models are invariant to monotonic scaling. "
        "For a clean model workflow, prefer cleaning only and let the pipeline handle scaling.",
        tone="warn",
    )


def _comparison_section(raw: pd.DataFrame, current: pd.DataFrame) -> None:
    section_header(
        "Before / after",
        "Compare the original loaded data with the current working dataset. The original "
        "dataframe is preserved and is never mutated by cleaning.",
        kicker="Audit trail",
    )
    columns = cleanable_numeric_columns(current)
    if not columns:
        st.info("No numeric feature is available for comparison.")
        return
    selected = st.selectbox(
        "Feature to compare",
        columns,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
        key="clean_compare_col",
    )
    stats = before_after_stats(raw, current, [selected])
    if not stats.empty:
        st.dataframe(stats.round(3), width="stretch", hide_index=True)
    fig = go.Figure()
    before_values = pd.to_numeric(raw[selected], errors="coerce").dropna()
    after_values = pd.to_numeric(current[selected], errors="coerce").dropna()
    fig.add_trace(
        go.Histogram(
            x=before_values,
            name="Original",
            nbinsx=40,
            marker_color=COLORS["muted"],
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=after_values,
            name="Working",
            nbinsx=40,
            marker_color=COLORS["cyan"],
            opacity=0.6,
        )
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=FEATURE_LABELS.get(selected, selected))
    fig.update_yaxes(title="Count")
    fig = apply_theme(
        fig,
        title=f"Original vs working — {FEATURE_LABELS.get(selected, selected)}",
        height=360,
    )
    plotly_chart(fig, key="clean_compare_chart")


def _history_section() -> None:
    section_header(
        "Cleaning history",
        "Every operation is logged. Undo restores the previous snapshot; reset restores "
        "the original loaded dataset.",
        kicker="Reproducibility",
    )
    history = st.session_state.get("clean_history") or []
    if not history:
        st.info("No cleaning operations have been applied.")
        return
    frame = pd.DataFrame(history)
    st.dataframe(frame, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo last operation", width="stretch", key="clean_undo"):
            removed = undo_cleaning()
            if removed:
                st.rerun()
    with c2:
        if st.button("Reset all cleaning", width="stretch", key="clean_reset"):
            reset_cleaning()
            st.rerun()


def render() -> None:
    page_header(
        "Data cleaning",
        "A transparent, reversible cleaning workspace. The original dataframe is kept "
        "separate from the working copy, and the target is never transformed.",
        eyebrow="ESP PREDICTAI / 03 · CLEANING",
        badges=[("Original data preserved", "green"), ("Target protected", "cyan")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="clean_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    raw = st.session_state["raw_df"]
    current = active_df()
    meta = active_meta()
    history = st.session_state.get("clean_history") or []

    cards = [
        {"label": "Original rows", "value": format_int(len(raw)), "sub": "Immutable source copy", "tone": "blue", "icon": "◫"},
        {"label": "Working rows", "value": format_int(len(current)), "sub": "Current cleaned copy", "tone": "cyan", "icon": "▤"},
        {"label": "Operations", "value": format_int(len(history)), "sub": "Logged cleaning steps", "tone": "amber", "icon": "↺"},
        {"label": "Missing cells", "value": format_int(int(current.isna().sum().sum())), "sub": "Working dataset", "tone": "green" if current.isna().sum().sum() == 0 else "amber", "icon": "◌"},
    ]
    render_kpis(cards, columns=4)

    if st.session_state.get("model_notice"):
        st.info(st.session_state["model_notice"])
    callout(
        "Integrity rule",
        "Cleaning functions exclude the binary target, raw status, well_id, and date. "
        "The model pipeline also excludes downtime_hours and production_loss because they "
        "deterministically reveal failure in this dataset.",
        tone="cyan",
    )

    _missing_section(current)
    _duplicate_section(current)
    _outlier_section(current)
    _scaling_section(current)
    _comparison_section(raw, current)
    _history_section()

    section_header("Continue", kicker="Workflow")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Explore the cleaned data", width="stretch", key="clean_go_eda"):
            goto("Exploratory Data Analysis")
            st.rerun()
    with c2:
        if st.button("Train a model", width="stretch", key="clean_go_train"):
            goto("ML Model Training")
            st.rerun()
