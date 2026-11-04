"""Data loading, filtering, schema inspection, and integrity audit page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import (
    DATA_DICTIONARY,
    DATE_COL,
    FEATURE_LABELS,
    TARGET,
    TARGET_RAW,
    WELL_COL,
)
from ..data_loader import (
    DatasetError,
    audit_dataset,
    human_source,
    load_github_dataset,
    load_uploaded_csv,
)
from ..state import (
    active_df,
    active_meta,
    goto,
    has_data,
    reset_cleaning,
    set_dataset,
)
from ..ui import (
    callout,
    data_badges,
    download_csv_button,
    format_int,
    format_pct,
    page_header,
    render_kpis,
    section_header,
)


def _reload_github() -> None:
    with st.spinner("Downloading the public ESP dataset..."):
        try:
            df, meta = load_github_dataset()
        except DatasetError as exc:
            st.error(str(exc))
            return
    set_dataset(
        df,
        meta,
        notice=(
            f"Loaded {len(df):,} rows and {meta.get('n_wells', 0)} wells from "
            f"{human_source(meta)}."
        ),
    )
    st.session_state["filters_nonce"] += 1
    st.rerun()


def _apply_upload(uploaded) -> None:
    with st.spinner("Reading and validating the uploaded CSV..."):
        try:
            df, meta = load_uploaded_csv(uploaded)
        except DatasetError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the uploaded file: {exc}")
            return
    set_dataset(
        df,
        meta,
        notice=f"Loaded {len(df):,} rows from uploaded file '{uploaded.name}'.",
    )
    st.session_state["filters_nonce"] += 1
    st.rerun()


def _filtered_frame(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    nonce = st.session_state.get("filters_nonce", 0)
    frame = df.copy()
    with st.container(border=True):
        st.markdown('<div class="sidebar-mini-title">Filters</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
        with c1:
            wells = sorted(frame[WELL_COL].dropna().astype(str).unique().tolist()) if WELL_COL in frame else []
            selected_wells = st.multiselect(
                "Well",
                options=wells,
                default=wells,
                key=f"filter_wells_{nonce}",
                help="Filter the preview and statistics by well.",
            )
        with c2:
            status = st.selectbox(
                "Operating status",
                ["All", "Running", "Failed"],
                key=f"filter_status_{nonce}",
            )
        with c3:
            st.write("")
            st.write("")
            if st.button("Reset filters", width="stretch", key=f"reset_filters_{nonce}"):
                st.session_state["filters_nonce"] += 1
                st.rerun()

        if audit.get("has_date") and DATE_COL in frame:
            min_date = frame[DATE_COL].min().date()
            max_date = frame[DATE_COL].max().date()
            date_value = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"filter_dates_{nonce}",
            )
            if isinstance(date_value, tuple) and len(date_value) == 2:
                start, end = date_value
                frame = frame[
                    frame[DATE_COL].dt.date.between(start, end, inclusive="both")
                ]
        else:
            st.caption("Date filtering is unavailable because the dataset has no parsed dates.")

    if WELL_COL in frame:
        frame = frame[frame[WELL_COL].astype(str).isin(selected_wells)]
    if status == "Running":
        frame = frame[frame[TARGET] == 0]
    elif status == "Failed":
        frame = frame[frame[TARGET] == 1]
    return frame


def _quality_tab(df: pd.DataFrame, audit: dict, meta: dict) -> None:
    if audit.get("likely_simulated"):
        callout(
            "Provenance status: unverified",
            "The repository does not provide field metadata, equipment specifications, or "
            "a data-generation record. Several statistical signatures are consistent with "
            "simulated data. This app therefore treats the dataset as an educational "
            "example, not field evidence.",
            tone="warn",
            icon="⚠",
        )
    else:
        callout(
            "Provenance status: unverified",
            "The source is a public repository upload. Verify the provenance before using "
            "these results in any real engineering context.",
            tone="warn",
        )
    if audit.get("critical"):
        st.error(" ".join(audit["critical"]))
    for warning in audit.get("warnings", []):
        st.warning(warning)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Target definition**")
        st.write(audit.get("target_definition") or "—")
        st.markdown("**Class balance**")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Running (0)", audit["target_counts"]["running"]),
                    ("Failed (1)", audit["target_counts"]["failed"]),
                    ("Missing / invalid", audit["target_counts"]["missing_or_invalid"]),
                ],
                columns=["Class", "Count"],
            ),
            width="stretch",
            hide_index=True,
        )
    with c2:
        st.markdown("**Dataset fingerprint**")
        st.code(audit.get("sha256") or "—", language=None)
        st.markdown("**Source**")
        st.write(human_source(meta))
        if audit.get("source_url"):
            st.markdown(f"[Open source CSV]({audit['source_url']})")

    st.markdown("**Deterministic leakage checks**")
    if audit["leakage"]:
        leak_rows = []
        for col, values in audit["leakage"].items():
            leak_rows.append(
                {
                    "column": col,
                    "perfect separation": values["deterministic_separation"],
                    "agreement with failure": values["agreement_with_failure"],
                    "correlation with failure": values["correlation_with_failure"],
                    "non-zero running": values["nonzero_running"],
                    "zero failed": values["zero_failed"],
                }
            )
        st.dataframe(pd.DataFrame(leak_rows), width="stretch", hide_index=True)
        st.caption(
            "A column that is non-zero only for failed records reveals the target. "
            "downtime_hours and production_loss are therefore excluded from all model inputs."
        )
    else:
        st.success("No configured leakage columns were found.")

    st.markdown("**Failure-label persistence by well**")
    if isinstance(audit.get("persistence"), pd.DataFrame) and not audit["persistence"].empty:
        st.dataframe(
            audit["persistence"].round(4),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Status persistence is the probability that today's label equals yesterday's. "
            "The independent baseline is p² + (1−p)². Near-equality indicates labels behave "
            "like independent daily draws rather than persistent equipment states."
        )
    else:
        st.info("Persistence cannot be computed without dates and multiple observations per well.")

    st.markdown("**Provenance flags**")
    if audit.get("suspicion_reasons"):
        st.markdown("\n".join(f"- {r}" for r in audit["suspicion_reasons"]))
    else:
        st.write("No specific provenance flags were detected.")


def _dictionary_tab(df: pd.DataFrame) -> None:
    rows = []
    for col in df.columns:
        if col.startswith("__"):
            continue
        role, unit, description = DATA_DICTIONARY.get(
            col, ("Extra / unrecognised", "—", "Column supplied by the uploaded dataset")
        )
        rows.append(
            {
                "column": col,
                "role": role,
                "unit": unit,
                "description": description,
                "present": True,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Operational features are used by the model only if present. Identifiers and dates "
        "are preserved for grouping and temporal validation but are not model inputs."
    )


def render() -> None:
    meta = active_meta()
    page_header(
        "Data explorer",
        "Load, inspect, filter, and audit the ESP dataset before any cleaning or modelling.",
        eyebrow="ESP PREDICTAI / 02 · DATA",
        badges=data_badges(meta, None),
    )

    section_header("Dataset source", kicker="Load / replace")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with c1:
            if st.button("Load GitHub Dataset", type="primary", width="stretch", key="de_load_github"):
                _reload_github()
        with c2:
            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="de_upload",
                help="The file must include esp_status and operational columns.",
            )
        with c3:
            if st.button(
                "Apply Uploaded CSV",
                width="stretch",
                key="de_apply_upload",
                disabled=uploaded is None,
            ):
                if uploaded is not None:
                    _apply_upload(uploaded)
        with c4:
            if st.button("Reset Dataset", width="stretch", key="de_reset"):
                if has_data():
                    reset_cleaning()
                    st.session_state["filters_nonce"] += 1
                    st.session_state["data_notice"] = (
                        "Working dataset reset to the originally loaded data; cleaning history cleared."
                    )
                    st.rerun()
                else:
                    st.warning("No dataset is loaded.")

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Preview Data", width="stretch", key="de_preview"):
                st.session_state["show_preview"] = not st.session_state.get(
                    "show_preview", True
                )
        with c2:
            st.caption(
                "Load GitHub Dataset replaces the working data. Apply Uploaded CSV replaces "
                "it with your file. Reset Dataset restores the original loaded data and clears cleaning."
            )

    if st.session_state.get("load_error") and not has_data():
        st.error(st.session_state["load_error"])
        callout(
            "Upload fallback",
            "The public dataset could not be loaded. Upload a CSV with the expected columns "
            "using the control above, then click Apply Uploaded CSV.",
            tone="warn",
        )
        return

    if not has_data():
        callout(
            "No dataset loaded",
            "Use Load GitHub Dataset or upload a CSV file. All analysis is disabled until data is available.",
            tone="warn",
        )
        return

    df = active_df()
    meta = active_meta()
    audit = audit_dataset(df, meta)

    if st.session_state.get("data_notice"):
        st.info(st.session_state["data_notice"])

    cards = [
        {"label": "Rows", "value": format_int(audit["n_rows"]), "sub": "Working dataset", "tone": "blue", "icon": "▦"},
        {"label": "Columns", "value": format_int(audit["n_cols"]), "sub": "Including target and identifiers", "tone": "cyan", "icon": "▥"},
        {"label": "Wells", "value": format_int(audit["n_wells"]), "sub": "Distinct well_id values", "tone": "cyan", "icon": "◉"},
        {"label": "Failure rate", "value": format_pct(audit["failure_rate"]), "sub": "Share of labelled rows", "tone": "amber", "icon": "◆"},
        {"label": "Missing cells", "value": format_int(audit["missing_cells"]), "sub": "All columns", "tone": "green" if audit["missing_cells"] == 0 else "amber", "icon": "◌"},
        {"label": "Duplicate rows", "value": format_int(audit["duplicate_rows"]), "sub": "Exact duplicates", "tone": "green" if audit["duplicate_rows"] == 0 else "amber", "icon": "⧉"},
        {"label": "Date coverage", "value": f"{audit['date_min']} → {audit['date_max']}" if audit.get("has_date") else "Unavailable", "sub": "Parsed measurement dates", "tone": "blue" if audit.get("has_date") else "amber", "icon": "◷"},
        {"label": "Source mode", "value": human_source(meta), "sub": "Current data origin", "tone": "blue", "icon": "↓"},
    ]
    render_kpis(cards, columns=4)

    section_header(
        "Explore the data",
        "Filtering below affects the preview, summaries, and download — not the underlying dataset.",
        kicker="Interactive view",
    )
    filtered = _filtered_frame(df, audit)
    st.caption(f"Filtered view: {len(filtered):,} of {len(df):,} rows.")

    tab_preview, tab_schema, tab_stats, tab_quality, tab_dict = st.tabs(
        ["Preview", "Schema & missing", "Statistics", "Data quality", "Data dictionary"]
    )

    with tab_preview:
        if st.session_state.get("show_preview", True):
            all_columns = [c for c in filtered.columns if not c.startswith("__")]
            selected = st.multiselect(
                "Columns to display",
                all_columns,
                default=all_columns,
                key=f"preview_columns_{st.session_state.get('filters_nonce', 0)}",
            )
            if selected:
                st.dataframe(
                    filtered[selected],
                    width="stretch",
                    hide_index=True,
                    height=420,
                )
                download_csv_button(
                    filtered[selected],
                    "Download filtered CSV",
                    "esp_filtered_view.csv",
                    key="de_download_filtered",
                )
            else:
                st.info("Select at least one column.")
        else:
            st.info("Preview hidden. Click Preview Data to show the table.")

    with tab_schema:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Column names and data types**")
            public_columns = [c for c in df.columns if not str(c).startswith("__")]
            st.dataframe(
                pd.DataFrame(
                    {
                        "column": public_columns,
                        "dtype": [str(df[c].dtype) for c in public_columns],
                        "non_null": [int(df[c].notna().sum()) for c in public_columns],
                        "unique": [int(df[c].nunique(dropna=True)) for c in public_columns],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with c2:
            st.markdown("**Missing values**")
            missing = audit["missing_by_col"]
            missing = missing[missing["missing"] > 0]
            if missing.empty:
                st.success("No missing values in the current dataset.")
            else:
                st.dataframe(
                    missing.sort_values("missing", ascending=False),
                    width="stretch",
                    hide_index=True,
                )
        st.markdown("**Dataset dimensions**")
        st.write(f"{len(df):,} rows × {audit['n_cols']:,} public columns")
        if audit["duplicate_well_dates"]:
            st.warning(
                f"{audit['duplicate_well_dates']:,} duplicate well/date pairs detected."
            )

    with tab_stats:
        public_columns = [c for c in filtered.columns if not str(c).startswith("__")]
        numeric = filtered[public_columns].select_dtypes(include="number")
        if numeric.empty:
            st.info("No numeric columns available.")
        else:
            summary = numeric.describe().T
            summary.insert(0, "column", summary.index)
            st.dataframe(
                summary.round(3),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Summary statistics describe the filtered view."
            )
        st.markdown("**Operating-status counts**")
        filtered_target = pd.to_numeric(filtered[TARGET], errors="coerce")
        counts = pd.DataFrame(
            [
                ("Running (0)", int((filtered_target == 0).sum())),
                ("Failed (1)", int((filtered_target == 1).sum())),
                ("Missing / invalid", int(filtered_target.isna().sum())),
            ],
            columns=["Status", "Count"],
        )
        st.dataframe(counts, width="stretch", hide_index=True)

    with tab_quality:
        _quality_tab(df, audit, meta)

    with tab_dict:
        _dictionary_tab(df)

    section_header("Continue", kicker="Workflow")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clean the data", width="stretch", key="de_go_clean"):
            goto("Data Cleaning")
            st.rerun()
    with c2:
        if st.button("Explore with EDA", width="stretch", key="de_go_eda"):
            goto("Exploratory Data Analysis")
            st.rerun()
    with c3:
        if st.button("Train a model", width="stretch", key="de_go_train"):
            goto("ML Model Training")
            st.rerun()
