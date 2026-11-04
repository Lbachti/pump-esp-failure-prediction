"""Exploratory data analysis page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import FEATURE_LABELS, FEATURE_UNITS, TARGET, WELL_COL
from ..data_loader import get_feature_columns
from ..eda import (
    box_by_status,
    correlation_heatmap,
    correlation_with_target_table,
    effect_size_figure,
    failure_rate_by_bins,
    histogram_by_status,
    pair_plot,
    scatter_plot,
    signal_figure,
    time_failure_by_well,
    time_failure_rate,
    time_feature,
    univariate_signal_table,
    violin_by_status,
    well_failure_comparison,
)
from ..engineering_insights import PARAMETER_KNOWLEDGE
from ..state import active_df, goto, has_data
from ..ui import (
    callout,
    format_float,
    format_pct,
    page_header,
    plotly_chart,
    section_header,
)


def _interpretation(feature: str, df: pd.DataFrame) -> None:
    knowledge = PARAMETER_KNOWLEDGE.get(feature)
    if not knowledge:
        return
    stats = univariate_signal_table(df, [feature])
    stat = stats.iloc[0].to_dict() if not stats.empty else None
    section_header(
        "Engineering interpretation",
        "Established principles are separated from dataset statistics and model inference.",
        kicker="Context",
    )
    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.markdown(
            f"**{knowledge['label']} ({knowledge['unit']})** — {knowledge['summary']}"
        )
        st.markdown("**Established ESP principles**")
        for item in knowledge["established"]:
            st.markdown(f"- {item}")
        st.markdown("**Potential mechanisms**")
        for item in knowledge["mechanisms"]:
            st.markdown(f"- {item}")
        st.caption(knowledge["caveat"])
    with c2:
        if stat:
            st.markdown("**Dataset-derived statistics**")
            st.write(f"Mean when running: **{format_float(stat['mean_running'], 3)}**")
            st.write(f"Mean when failed: **{format_float(stat['mean_failed'], 3)}**")
            st.write(f"Univariate ROC-AUC: **{format_float(stat['auc'], 3)}**")
            st.write(f"Mann–Whitney p-value: **{stat['p_value']:.3g}**")
            st.caption(stat["direction"])
        st.markdown("**Diagnostic context**")
        for item in knowledge["diagnostics"]:
            st.markdown(f"- {item}")
    callout(
        "Association is not causation",
        "A statistical difference or model attribution does not prove that this parameter "
        "causes failure. Confounding, operating context, sensor quality, and dataset "
        "provenance must be assessed before drawing engineering conclusions.",
        tone="warn",
    )


def render() -> None:
    page_header(
        "Exploratory analysis",
        "Interactive distributions, relationships, correlation, time trends, and failure "
        "signals across the active ESP dataset.",
        eyebrow="ESP PREDICTAI / 04 · EDA",
        badges=[("Interactive Plotly", "cyan"), ("Association ≠ causation", "amber")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="eda_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    features = get_feature_columns(df)
    if not features:
        st.error("No operational features are available for analysis.")
        return

    with st.container(border=True):
        st.markdown('<div class="sidebar-mini-title">Plot controls</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.0, 0.8])
        with c1:
            plot_type = st.selectbox(
                "Plot type",
                ["Histogram", "Box plot", "Violin", "Scatter", "Time series"],
                key="eda_plot_type",
            )
        with c2:
            x_feature = st.selectbox(
                "X / primary feature",
                features,
                index=features.index("discharge_pressure")
                if "discharge_pressure" in features
                else 0,
                format_func=lambda f: FEATURE_LABELS.get(f, f),
                key="eda_x",
            )
        with c3:
            y_feature = st.selectbox(
                "Y feature (scatter)",
                features,
                index=features.index("intake_pressure")
                if "intake_pressure" in features
                else min(1, len(features) - 1),
                format_func=lambda f: FEATURE_LABELS.get(f, f),
                key="eda_y",
            )
        with c4:
            color_by = st.selectbox(
                "Colour by",
                ["Operating status", "Well"],
                key="eda_color",
            )
        with c5:
            bins = st.slider("Histogram bins", 10, 80, 40, key="eda_bins")

        st.caption(
            f"Working dataset: {len(df):,} rows · {df[WELL_COL].nunique() if WELL_COL in df else 0} wells. "
            "Box, violin, and histogram views group by operating status; colour-by-well "
            "applies to scatter and pair plots."
        )

    tab_dist, tab_corr, tab_wells, tab_drivers = st.tabs(
        ["Distributions & relationships", "Correlation", "Wells & time", "Failure drivers"]
    )

    with tab_dist:
        if plot_type == "Histogram":
            plotly_chart(histogram_by_status(df, x_feature, bins), key="eda_hist")
        elif plot_type == "Box plot":
            plotly_chart(box_by_status(df, x_feature), key="eda_box")
        elif plot_type == "Violin":
            plotly_chart(violin_by_status(df, x_feature), key="eda_violin")
        elif plot_type == "Scatter":
            if x_feature == y_feature:
                st.warning("Choose two different features for a scatter plot.")
            else:
                plotly_chart(
                    scatter_plot(df, x_feature, y_feature, color_by),
                    key="eda_scatter",
                )
                valid = df[[x_feature, y_feature]].dropna()
                if len(valid) > 2:
                    pearson = valid[x_feature].corr(valid[y_feature], method="pearson")
                    spearman = valid[x_feature].corr(valid[y_feature], method="spearman")
                    st.caption(
                        f"Pearson r = {pearson:.3f} · Spearman ρ = {spearman:.3f} · "
                        f"n = {len(valid):,}. Correlation is not proof of causation."
                    )
        else:
            if color_by == "Well":
                plotly_chart(time_feature(df, x_feature, by_well=True), key="eda_time_well")
            else:
                plotly_chart(time_feature(df, x_feature, by_well=False), key="eda_time_all")

        with st.expander("Pairwise parameter relationships (up to five features)", expanded=False):
            pair_features = st.multiselect(
                "Features for pair plot",
                features,
                default=features[: min(4, len(features))],
                format_func=lambda f: FEATURE_LABELS.get(f, f),
                key="eda_pair_features",
            )
            if len(pair_features) >= 2:
                plotly_chart(
                    pair_plot(df, pair_features, color_by=color_by),
                    key="eda_pair_plot",
                )
            else:
                st.info("Select at least two features.")

        _interpretation(x_feature, df)

    with tab_corr:
        method = st.radio(
            "Correlation method",
            ["pearson", "spearman"],
            horizontal=True,
            key="eda_corr_method",
        )
        plotly_chart(
            correlation_heatmap(df, features, include_target=True, method=method),
            key="eda_corr_heatmap",
        )
        st.caption(
            "The target row is shown as 'Failure'. Correlation measures linear (Pearson) "
            "or rank (Spearman) association; it does not establish causation."
        )
        corr_table = correlation_with_target_table(df, features, method=method)
        if not corr_table.empty:
            st.dataframe(
                corr_table.rename(
                    columns={
                        "feature": "feature",
                        "label": "Parameter",
                        "correlation": "Correlation with failure",
                        "abs_correlation": "Absolute correlation",
                    }
                ).round(4),
                width="stretch",
                hide_index=True,
            )
        st.markdown("**Standardised mean difference (failed − running)**")
        plotly_chart(effect_size_figure(df, features), key="eda_corr_effect")

    with tab_wells:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            plotly_chart(well_failure_comparison(df), key="eda_well_compare")
        with c2:
            plotly_chart(time_failure_rate(df), key="eda_well_time")
        if WELL_COL in df.columns:
            st.markdown("**Well-level operating summary**")
            summary = (
                df.groupby(WELL_COL)
                .agg(
                    observations=(TARGET, "size"),
                    failure_rate=(TARGET, "mean"),
                    mean_motor_temperature=("motor_temperature", "mean")
                    if "motor_temperature" in df
                    else (TARGET, "mean"),
                    mean_production=("production", "mean")
                    if "production" in df
                    else (TARGET, "mean"),
                )
                .reset_index()
            )
            st.dataframe(summary.round(3), width="stretch", hide_index=True)
        plotly_chart(time_failure_by_well(df), key="eda_time_by_well")
        st.caption(
            "Well-level differences are a warning: a model trained with random row splits can "
            "learn well-specific signatures rather than transferable failure physics."
        )

    with tab_drivers:
        st.markdown(
            f"**Failure rate by {FEATURE_LABELS.get(x_feature, x_feature)} quantile**"
        )
        fig, table = failure_rate_by_bins(df, x_feature, n_bins=8)
        plotly_chart(fig, key="eda_driver_bins")
        if not table.empty:
            display = table.copy()
            display["_bin"] = display["_bin"].astype(str)
            display = display.rename(
                columns={
                    "_bin": "Quantile bin",
                    "n": "Observations",
                    "failed": "Failures",
                    "mean_feature": "Mean value",
                    "failure_rate": "Failure rate",
                }
            )
            st.dataframe(
                display[["Quantile bin", "Observations", "Failures", "Mean value", "Failure rate"]]
                .round(4),
                width="stretch",
                hide_index=True,
            )
        plotly_chart(signal_figure(df, features), key="eda_signal")
        signal_table = univariate_signal_table(df, features)
        if not signal_table.empty:
            st.dataframe(
                signal_table.rename(
                    columns={
                        "label": "Parameter",
                        "auc": "Univariate ROC-AUC",
                        "signal": "Signal strength",
                        "direction": "Direction",
                        "p_value": "Mann–Whitney p",
                        "mean_running": "Mean running",
                        "mean_failed": "Mean failed",
                    }
                )[
                    [
                        "Parameter",
                        "Univariate ROC-AUC",
                        "Signal strength",
                        "Direction",
                        "Mann–Whitney p",
                        "Mean running",
                        "Mean failed",
                    ]
                ].round(4),
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "Univariate signal strength is max(AUC, 1−AUC). Values near 0.5 indicate no "
            "useful single-feature separation. A significant p-value is not proof of causation."
        )

    section_header("Continue", kicker="Workflow")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train a model on this data", width="stretch", key="eda_go_train"):
            goto("ML Model Training")
            st.rerun()
    with c2:
        if st.button("Open engineering insights", width="stretch", key="eda_go_insights"):
            goto("Engineering Insights")
            st.rerun()
