"""ML model training, validation, evaluation, and model-management page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ..config import FEATURE_LABELS, MODEL_LABELS, MODEL_PATH
from ..data_loader import get_feature_columns, human_source
from ..explainability import global_importance, importance_figure
from ..model_evaluation import (
    actual_vs_predicted_table,
    calibration_figure,
    classification_report_table,
    compute_metrics,
    confusion_matrix_figure,
    pr_curve_figure,
    probability_distribution_figure,
    roc_curve_figure,
    threshold_curve_figure,
)
from ..model_training import (
    compare_validation_strategies,
    group_cv,
    train_and_compare,
    train_model,
)
from ..prediction import (
    bundle_info_table,
    delete_model_file,
    export_bundle,
    load_model,
    model_file_info,
    save_model,
    validate_bundle_features,
)
from ..preprocessing import (
    ModelingError,
    available_split_strategies,
    get_split_indices,
    prepare_training_frame,
    xgboost_available,
)
from ..state import (
    active_df,
    active_meta,
    clear_model,
    current_hash,
    goto,
    has_data,
    model_is_stale,
    set_trained_bundle,
)
from ..ui import (
    callout,
    empty_state,
    format_float,
    format_int,
    format_pct,
    page_header,
    plotly_chart,
    render_kpis,
    section_header,
)

MODEL_OPTIONS = [
    ("logistic", "Logistic Regression"),
    ("random_forest", "Random Forest"),
]
if xgboost_available():
    MODEL_OPTIONS.append(("xgboost", "XGBoost"))


def _model_key_from_label(label: str) -> str:
    for key, text in MODEL_OPTIONS:
        if text == label:
            return key
    return MODEL_OPTIONS[0][0]


def _model_params(model_key: str) -> dict[str, Any]:
    if model_key == "logistic":
        c = st.select_slider(
            "Regularisation strength (C)",
            options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            value=1.0,
            key="train_log_C",
            help="Smaller C means stronger regularisation.",
        )
        return {"C": float(c)}
    if model_key == "random_forest":
        c1, c2, c3 = st.columns(3)
        with c1:
            n_estimators = st.slider(
                "Trees", 100, 600, 350, 50, key="train_rf_trees"
            )
        with c2:
            depth_label = st.selectbox(
                "Max depth",
                ["None (unlimited)", "4", "6", "8", "12", "16"],
                index=0,
                key="train_rf_depth",
            )
            max_depth = None if depth_label.startswith("None") else int(depth_label)
        with c3:
            min_leaf = st.slider(
                "Min samples / leaf", 1, 15, 2, 1, key="train_rf_leaf"
            )
        return {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_leaf),
        }
    if model_key == "xgboost":
        c1, c2 = st.columns(2)
        with c1:
            n_estimators = st.slider(
                "Boosting rounds", 100, 600, 250, 50, key="train_xgb_trees"
            )
            max_depth = st.slider("Max depth", 2, 10, 6, 1, key="train_xgb_depth")
            learning_rate = st.slider(
                "Learning rate", 0.01, 0.30, 0.10, 0.01, key="train_xgb_lr"
            )
        with c2:
            subsample = st.slider(
                "Row subsample", 0.50, 1.00, 0.90, 0.05, key="train_xgb_subsample"
            )
            colsample = st.slider(
                "Column subsample", 0.50, 1.00, 0.90, 0.05, key="train_xgb_colsample"
            )
        return {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
        }
    return {}


def _strategy_selector(df: pd.DataFrame) -> tuple[str, dict[str, str]]:
    strategies = available_split_strategies(df)
    available = [s for s in strategies if s["available"]]
    if not available:
        return "random", {}
    labels = {s["key"]: s["label"] for s in available}
    default_index = 0
    for i, s in enumerate(available):
        if s["key"] == "well":
            default_index = i
            break
    choice = st.selectbox(
        "Validation strategy",
        options=list(labels.keys()),
        index=default_index,
        format_func=lambda key: labels[key],
        key="train_strategy",
    )
    selected = next(s for s in available if s["key"] == choice)
    st.caption(selected["reason"])
    for s in strategies:
        if not s["available"]:
            st.caption(f"Unavailable: {s['label']} — {s['reason']}")
    return choice, labels


def _split_preview(
    df: pd.DataFrame,
    features: list[str],
    strategy: str,
    test_size: float,
    random_state: int,
    drop_duplicates: bool,
) -> dict[str, Any] | None:
    try:
        frame, info = prepare_training_frame(
            df, features=features, drop_duplicates=drop_duplicates
        )
        train_idx, test_idx, split_info = get_split_indices(
            frame, strategy=strategy, test_size=test_size, random_state=random_state
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Split preview unavailable: {exc}")
        return None
    st.markdown('<div class="sidebar-mini-title">Split preview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", f"{split_info['n_train']:,}")
    c2.metric("Test rows", f"{split_info['n_test']:,}")
    c3.metric("Train failure rate", format_pct(split_info.get("train_failure_rate")))
    c4.metric("Test failure rate", format_pct(split_info.get("test_failure_rate")))
    st.caption(
        f"Train wells: {', '.join(split_info.get('train_wells', [])) or '—'} · "
        f"Test wells: {', '.join(split_info.get('test_wells', [])) or '—'}"
    )
    if split_info.get("cutoff_date"):
        st.caption(f"Time cutoff: {split_info['cutoff_date']}")
    if info.get("n_duplicates_removed"):
        st.caption(f"{info['n_duplicates_removed']:,} exact duplicate rows would be removed before splitting.")
    return split_info


def _train_selected(
    df: pd.DataFrame,
    features: list[str],
    model_key: str,
    strategy: str,
    test_size: float,
    random_state: int,
    threshold: float,
    class_weight: bool,
    drop_duplicates: bool,
    params: dict[str, Any],
    source: str,
) -> None:
    try:
        with st.spinner(
            f"Training {MODEL_LABELS.get(model_key, model_key)} with {strategy} split..."
        ):
            result = train_model(
                df,
                model_key=model_key,
                features=features,
                strategy=strategy,
                test_size=test_size,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params=params,
                drop_duplicates=drop_duplicates,
                dataset_source=source,
            )
            bundle = export_bundle(result, threshold=threshold)
        set_trained_bundle(bundle, result)
        st.session_state["eval_threshold"] = float(threshold)
        st.session_state["training_notice"] = (
            f"Trained {bundle['model_label']} with {bundle['split_info'].get('strategy_label', strategy)}. "
            "Review the holdout metrics before saving or deploying it."
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")


def _run_comparison(
    df: pd.DataFrame,
    features: list[str],
    model_key: str,
    strategy: str,
    test_size: float,
    random_state: int,
    threshold: float,
    class_weight: bool,
    drop_duplicates: bool,
    params: dict[str, Any],
    source: str,
) -> None:
    keys = [key for key, _ in MODEL_OPTIONS]
    params_by_model = {key: params if key == model_key else {} for key in keys}
    try:
        with st.spinner("Training and comparing all available algorithms on the same split..."):
            comparison = train_and_compare(
                df,
                model_keys=keys,
                features=features,
                strategy=strategy,
                test_size=test_size,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params_by_model=params_by_model,
                drop_duplicates=drop_duplicates,
                dataset_source=source,
            )
        st.session_state["comparison_result"] = comparison
        st.session_state["training_notice"] = (
            f"Model comparison completed on the same {strategy} split."
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model comparison failed: {exc}")


def _run_strategy_comparison(
    df: pd.DataFrame,
    features: list[str],
    model_key: str,
    test_size: float,
    random_state: int,
    threshold: float,
    class_weight: bool,
    drop_duplicates: bool,
    params: dict[str, Any],
    source: str,
) -> None:
    try:
        with st.spinner("Training the selected model under each validation strategy..."):
            table = compare_validation_strategies(
                df,
                model_key=model_key,
                features=features,
                test_size=test_size,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params=params,
                drop_duplicates=drop_duplicates,
                dataset_source=source,
            )
        st.session_state["strategy_comparison"] = table
        st.session_state["training_notice"] = "Validation-strategy comparison completed."
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Strategy comparison failed: {exc}")


def _run_group_cv(
    df: pd.DataFrame,
    features: list[str],
    model_key: str,
    random_state: int,
    threshold: float,
    class_weight: bool,
    drop_duplicates: bool,
    params: dict[str, Any],
) -> None:
    try:
        with st.spinner("Running well-grouped cross-validation..."):
            result = group_cv(
                df,
                model_key=model_key,
                features=features,
                n_splits=3,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params=params,
                drop_duplicates=drop_duplicates,
            )
        st.session_state["group_cv_result"] = result
        st.session_state["training_notice"] = "Well-grouped cross-validation completed."
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Grouped cross-validation failed: {exc}")


def _show_warnings(bundle: dict[str, Any]) -> None:
    warnings = bundle.get("warnings", [])
    if not warnings:
        return
    with st.expander("Validation warnings", expanded=False):
        for warning in warnings:
            st.warning(warning)


def _show_results(bundle: dict[str, Any], result: dict[str, Any] | None) -> None:
    section_header(
        "Holdout evaluation",
        "Metrics below are computed on the held-out test split. Adjust the decision " \
        "threshold to see the recall / false-alarm trade-off.",
        kicker="Evaluation",
    )
    if model_is_stale():
        callout(
            "Dataset changed since training",
            "The active working dataset no longer matches the model fingerprint. " \
            "Retrain before using this model for decisions.",
            tone="danger",
            icon="⚠",
        )

    strategy = bundle.get("strategy")
    metrics_saved = bundle.get("metrics", {})
    if strategy == "well" and metrics_saved.get("roc_auc") is not None:
        auc = metrics_saved.get("roc_auc")
        if auc == auc and auc < 0.65:
            callout(
                "Cross-well performance is close to chance",
                f"ROC-AUC is {auc:.3f} on unseen wells. The high random/time-split scores " \
                "do not transfer across wells. This model is an educational demonstration, " \
                "not a deployable cross-well failure predictor.",
                tone="danger",
                icon="✕",
            )
    if strategy == "random":
        callout(
            "Random split is optimistic",
            "Rows from the same well appear in both train and test. Use the well-based " \
            "strategy and the validation comparison below before trusting the headline metrics.",
            tone="warn",
        )
    if strategy == "time":
        callout(
            "Temporal split, same wells",
            "This tests extrapolation to later dates but not to a new well. It is stronger " \
            "than a random split but still not evidence of cross-well generalisation.",
            tone="warn",
        )

    if result is not None:
        y_test = result["y_test"]
        proba = result["proba"]
        if "eval_threshold" not in st.session_state:
            st.session_state["eval_threshold"] = float(bundle.get("threshold", 0.5))
        threshold = st.slider(
            "Decision threshold (failure probability)",
            min_value=0.05,
            max_value=0.95,
            step=0.01,
            key="eval_threshold",
            help="Lower thresholds detect more failures but increase false alarms.",
        )
        metrics = compute_metrics(y_test, proba, threshold)
        bundle["threshold"] = float(threshold)
        bundle["metrics"] = metrics
        if result is not None:
            result["threshold"] = float(threshold)
            result["metrics"] = metrics
    else:
        threshold = float(bundle.get("threshold", 0.5))
        metrics = metrics_saved
        st.caption(
            f"This model was loaded from disk, so held-out predictions are not in memory. " \
            f"Showing saved metrics at threshold {threshold:.2f}. Retrain to interactively " \
            "recompute the threshold trade-off."
        )

    cards = [
        {"label": "Recall (failed)", "value": format_float(metrics.get("recall"), 3), "sub": "Priority metric: missed failures are costly", "tone": "cyan", "icon": "◎"},
        {"label": "Precision", "value": format_float(metrics.get("precision"), 3), "sub": "Share of failure alerts that are correct", "tone": "blue", "icon": "◈"},
        {"label": "F1 score", "value": format_float(metrics.get("f1"), 3), "sub": "Harmonic mean of precision and recall", "tone": "amber", "icon": "◆"},
        {"label": "Accuracy", "value": format_float(metrics.get("accuracy"), 3), "sub": "Misleading when prevalence is high", "tone": "muted", "icon": "▦"},
        {"label": "ROC-AUC", "value": format_float(metrics.get("roc_auc"), 3), "sub": "Ranking quality across thresholds", "tone": "cyan", "icon": "◇"},
        {"label": "PR-AUC", "value": format_float(metrics.get("pr_auc"), 3), "sub": "Precision–recall area; useful for imbalance", "tone": "amber", "icon": "◉"},
    ]
    render_kpis(cards, columns=3)
    st.caption(
        f"Threshold {threshold:.2f} · test rows {metrics.get('n', 0):,} · "
        f"actual failures {metrics.get('positives', 0):,} · "
        f"false negatives {metrics.get('fn', 0):,} · false positives {metrics.get('fp', 0):,} · "
        f"Brier score {format_float(metrics.get('brier'), 3)}."
    )

    if result is not None:
        y_test = result["y_test"]
        proba = result["proba"]
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            plotly_chart(
                confusion_matrix_figure(y_test, proba, threshold),
                key="train_confusion",
            )
        with c2:
            st.markdown("**Classification report**")
            report = classification_report_table(y_test, proba, threshold)
            st.dataframe(report.round(3), width="stretch", hide_index=True)
            st.caption(
                "Recall for the failed class is the fraction of actual failures detected. " \
                "Precision is the fraction of predicted failures that are correct."
            )
        c1, c2 = st.columns(2, gap="large")
        with c1:
            plotly_chart(roc_curve_figure(y_test, proba), key="train_roc")
        with c2:
            plotly_chart(pr_curve_figure(y_test, proba), key="train_pr")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            plotly_chart(threshold_curve_figure(y_test, proba, threshold), key="train_threshold")
        with c2:
            plotly_chart(
                probability_distribution_figure(y_test, proba, threshold),
                key="train_probdist",
            )

        with st.expander("Calibration and probability reliability", expanded=False):
            st.markdown(
                "A calibration curve compares the model score with the observed failure " \
                "rate in bins of similar score. A model can rank well (high ROC-AUC) and " \
                "still be poorly calibrated."
            )
            plotly_chart(calibration_figure(y_test, proba), key="train_calibration")
            st.caption(f"Brier score: {format_float(metrics.get('brier'), 4)} (lower is better).")

        with st.expander("Actual vs predicted observations", expanded=False):
            table = actual_vs_predicted_table(result, threshold)
            outcome = st.multiselect(
                "Outcome filter",
                ["TP", "TN", "FP", "FN"],
                default=["TP", "TN", "FP", "FN"],
                key="train_outcome_filter",
            )
            view = table[table["outcome"].isin(outcome)] if outcome else table
            view_display = view.head(500).copy()
            numeric_cols = view_display.select_dtypes(include="number").columns
            if len(numeric_cols):
                view_display[numeric_cols] = view_display[numeric_cols].round(4)
            st.dataframe(
                view_display,
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Showing up to 500 of {len(view):,} rows. TP = true positive, TN = true " \
                "negative, FP = false positive (false alarm), FN = false negative (missed failure)."
            )
            st.download_button(
                "Download evaluation rows (CSV)",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="esp_holdout_predictions.csv",
                mime="text/csv",
                key="train_download_eval",
            )

        with st.expander("Model-native feature importance preview", expanded=False):
            imp = global_importance(bundle)
            if imp.empty:
                st.info("This model does not expose native feature importance.")
            else:
                plotly_chart(importance_figure(imp), key="train_importance_preview")
                st.caption(
                    "Feature importance describes the fitted model, not physical causation. " \
                    "See Model Explainability for SHAP and individual explanations."
                )
    _show_warnings(bundle)


def _show_comparison() -> None:
    comparison = st.session_state.get("comparison_result")
    if not comparison:
        return
    section_header(
        "Algorithm comparison",
        "All algorithms were trained on the same split and evaluated at the same threshold.",
        kicker="Model selection",
    )
    table = comparison.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        display = table.copy()
        numeric_cols = [
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold"
        ]
        for col in numeric_cols:
            if col in display:
                display[col] = pd.to_numeric(display[col], errors="coerce")
        st.dataframe(
            display.rename(
                columns={
                    "model": "Model",
                    "status": "Status",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                    "brier": "Brier",
                    "threshold": "Threshold",
                    "error": "Error",
                }
            ).round(3),
            width="stretch",
            hide_index=True,
        )
        trained = display[display["status"] == "trained"] if "status" in display else display
        if not trained.empty and "recall" in trained:
            best = trained.loc[trained["recall"].idxmax()]
            callout(
                "Highest recall in this comparison",
                f"{best['model']} achieved recall {best['recall']:.3f} at the configured threshold. " \
                "Do not select a model on one metric alone; review precision, calibration, and " \
                "cross-well validation.",
                tone="cyan",
            )
        bundles = comparison.get("bundles", {})
        if bundles:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                choice = st.selectbox(
                    "Adopt a compared model into the prediction workspace",
                    options=list(bundles.keys()),
                    format_func=lambda key: bundles[key]["model_label"],
                    key="adopt_model_choice",
                )
            with c2:
                st.write("")
                st.write("")
                if st.button("Load selected model", width="stretch", key="adopt_model_btn"):
                    chosen = bundles[choice]
                    bundle = export_bundle(chosen)
                    set_trained_bundle(bundle, chosen)
                    st.session_state["eval_threshold"] = float(bundle.get("threshold", 0.5))
                    st.session_state["training_notice"] = (
                        f"Adopted {bundle['model_label']} from the comparison table."
                    )
                    st.rerun()
        if "error" in display:
            errors = display[display.get("status", "") == "failed"] if "status" in display else display
            if not errors.empty:
                st.caption("Models that failed to train are shown with their error message.")


def _show_strategy_comparison() -> None:
    table = st.session_state.get("strategy_comparison")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return
    section_header(
        "Validation-strategy comparison",
        "The same algorithm and hyperparameters under random, temporal, and well-based " \
        "splits. The gap between random and well-based scores is the key integrity signal.",
        kicker="Generalisation check",
    )
    display = table.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if col in display:
            display[col] = pd.to_numeric(display[col], errors="coerce")
    st.dataframe(
        display.rename(
            columns={
                "strategy": "Validation strategy",
                "available": "Available",
                "train_rows": "Train rows",
                "test_rows": "Test rows",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "note": "Note",
            }
        ).round(3),
        width="stretch",
        hide_index=True,
    )
    available = display[display["available"] == True] if "available" in display else display
    if len(available) >= 2 and "roc_auc" in available:
        random_rows = available[available["strategy"].str.contains("Random", na=False)]
        well_rows = available[available["strategy"].str.contains("Well", na=False)]
        if not random_rows.empty and not well_rows.empty:
            random_auc = float(random_rows["roc_auc"].iloc[0])
            well_auc = float(well_rows["roc_auc"].iloc[0])
            if random_auc - well_auc > 0.10:
                callout(
                    "Large generalisation gap",
                    f"ROC-AUC falls from {random_auc:.3f} under a random split to " \
                    f"{well_auc:.3f} under a well-based split. The model is learning " \
                    "well-specific signatures, not failure physics that transfers to new wells.",
                    tone="danger",
                    icon="⚠",
                )
    st.caption(
        "Random split = rows from the same well in train and test. Time split = later " \
        "dates held out, but the same wells. Well split = entire wells held out, the " \
        "most conservative cross-well estimate."
    )


def _show_group_cv() -> None:
    result = st.session_state.get("group_cv_result")
    if not result:
        return
    section_header(
        "Well-grouped cross-validation",
        "Each fold holds out entire wells. This is a more robust check of cross-well " \
        "generalisation than a single well split.",
        kicker="Robustness",
    )
    summary = result["summary"]
    cards = [
        {"label": "Folds completed", "value": f"{summary['folds_completed']}/{summary['n_splits']}", "sub": "Skipped folds lacked both classes", "tone": "blue", "icon": "◫"},
        {"label": "Mean recall", "value": f"{summary['recall_mean']:.3f}", "sub": f"± {summary['recall_std']:.3f} across folds", "tone": "cyan", "icon": "◎"},
        {"label": "Mean F1", "value": f"{summary['f1_mean']:.3f}", "sub": f"± {summary['f1_std']:.3f}", "tone": "amber", "icon": "◆"},
        {"label": "Mean ROC-AUC", "value": f"{summary['roc_auc_mean']:.3f}", "sub": f"± {summary['roc_auc_std']:.3f}", "tone": "cyan", "icon": "◇"},
    ]
    render_kpis(cards, columns=4)
    st.dataframe(result["folds"].round(3), width="stretch", hide_index=True)
    if summary["roc_auc_mean"] < 0.65:
        callout(
            "Grouped CV confirms near-chance cross-well performance",
            "The model does not generalise reliably to unseen wells. This is a property of " \
            "the dataset and evaluation design, not a reason to tune until the number looks better.",
            tone="danger",
            icon="✕",
        )


def _model_management(bundle: dict[str, Any], result: dict[str, Any] | None) -> None:
    section_header(
        "Model management",
        "The preprocessing pipeline and estimator are saved together with joblib. " \
        "Predictions are disabled until a model is trained or loaded.",
        kicker="Persistence",
    )
    info = model_file_info(MODEL_PATH)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if info["exists"]:
            st.markdown(
                f"**Saved model file**  \n`{info['path']}`  \n"
                f"{info['size_bytes'] / (1024 * 1024):.2f} MB · modified {info['modified_at']}"
            )
            with open(info["path"], "rb") as handle:
                st.download_button(
                    "Download model bundle",
                    data=handle.read(),
                    file_name=Path(info["path"]).name,
                    mime="application/octet-stream",
                    key="train_download_model",
                )
        else:
            st.info("No saved model file yet.")
    with c2:
        if st.button("Save current model", type="primary", width="stretch", key="train_save_model"):
            try:
                if result is not None:
                    export = export_bundle(result, threshold=float(bundle.get("threshold", 0.5)))
                else:
                    export = dict(bundle)
                export["saved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
                path = save_model(export)
                st.session_state["training_notice"] = f"Saved model bundle to {path}."
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save model: {exc}")
        if st.button("Load saved model", width="stretch", key="train_load_model", disabled=not info["exists"]):
            try:
                loaded = load_model(MODEL_PATH)
                available = get_feature_columns(active_df())
                missing = validate_bundle_features(loaded, available)
                if missing:
                    st.error(
                        "The saved model expects features that are not available in the current " \
                        "dataset: " + ", ".join(missing)
                    )
                else:
                    set_trained_bundle(loaded, None)
                    st.session_state["eval_threshold"] = float(loaded.get("threshold", 0.5))
                    st.session_state["training_notice"] = "Loaded a saved model bundle from disk."
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load model: {exc}")
        if st.button("Reset in-memory model", width="stretch", key="train_reset_model"):
            clear_model("The in-memory model was reset. Train or load a model to predict.")
            st.rerun()
        if st.button(
            "Delete saved model file",
            width="stretch",
            key="train_delete_model",
            disabled=not info["exists"],
        ):
            deleted = delete_model_file(MODEL_PATH)
            if deleted:
                st.session_state["training_notice"] = "The saved model file was deleted."
                st.rerun()

    st.markdown("**Model information**")
    st.dataframe(bundle_info_table(bundle), width="stretch", hide_index=True)
    st.caption(
        f"Active dataset fingerprint: {current_hash()[:16] or '—'} · "
        f"Model fingerprint: {bundle.get('dataset_hash', '—')[:16] or '—'}."
    )
    if model_is_stale():
        st.error(
            "The active dataset fingerprint differs from the model fingerprint. The model " \
            "was trained on a different data version; retrain before operational use."
        )


def render() -> None:
    page_header(
        "Machine-learning training",
        "Leakage-safe pipelines, three validation strategies, threshold analysis, model " \
        "comparison, and joblib persistence.",
        eyebrow="ESP PREDICTAI / 05 · MODELLING",
        badges=[("0 = running · 1 = failed", "amber"), ("Recall-first evaluation", "cyan")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="train_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    meta = active_meta()
    features = get_feature_columns(df)
    if len(features) < 2:
        st.error("At least two operational features are required to train a model.")
        return

    if st.session_state.get("training_notice"):
        st.info(st.session_state["training_notice"])
    if st.session_state.get("model_notice"):
        st.warning(st.session_state["model_notice"])

    section_header(
        "Training configuration",
        "The split is created before fitting. Imputation and scaling live inside the " \
        "pipeline and are fitted on training data only.",
        kicker="Setup",
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.5, 0.9, 0.7])
        with c1:
            model_label = st.selectbox(
                "Algorithm",
                [label for _, label in MODEL_OPTIONS],
                key="train_model_label",
            )
            model_key = _model_key_from_label(model_label)
        with c2:
            strategy, strategy_labels = _strategy_selector(df)
        with c3:
            test_size = st.slider(
                "Test size", 0.15, 0.40, 0.25, 0.05, key="train_test_size"
            )
        with c4:
            random_state = st.number_input(
                "Random seed", min_value=0, max_value=9999, value=42, step=1, key="train_seed"
            )

        with st.expander("Advanced settings and hyperparameters", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                threshold = st.slider(
                    "Initial decision threshold",
                    0.05,
                    0.95,
                    0.50,
                    0.01,
                    key="train_threshold_cfg",
                    help="Lower thresholds prioritise recall; higher thresholds reduce false alarms.",
                )
                class_weight = st.checkbox(
                    "Use class weighting where supported",
                    value=False,
                    key="train_class_weight",
                    help="Balances classes during fitting. Threshold adjustment is usually more transparent.",
                )
                drop_duplicates = st.checkbox(
                    "Drop exact duplicate rows before splitting",
                    value=True,
                    key="train_drop_dupes",
                )
            with c2:
                st.caption(
                    "XGBoost is available." if xgboost_available() else "XGBoost is not installed."
                )
                st.caption(
                    "Features used: " + ", ".join(FEATURE_LABELS.get(f, f) for f in features)
                )
            params = _model_params(model_key)

        _split_preview(
            df, features, strategy, test_size, int(random_state), drop_duplicates
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            train_clicked = st.button(
                "Train selected model",
                type="primary",
                width="stretch",
                key="train_run_selected",
            )
        with b2:
            compare_clicked = st.button(
                "Train & compare all models",
                width="stretch",
                key="train_run_compare",
            )
        with b3:
            strategy_clicked = st.button(
                "Compare validation strategies",
                width="stretch",
                key="train_run_strategies",
            )
        with b4:
            cv_clicked = st.button(
                "Run well-grouped CV",
                width="stretch",
                key="train_run_groupcv",
                disabled=df["well_id"].nunique() < 3 if "well_id" in df else True,
            )

    if train_clicked:
        _train_selected(
            df, features, model_key, strategy, test_size, int(random_state), threshold,
            class_weight, drop_duplicates, params, human_source(meta),
        )
    if compare_clicked:
        _run_comparison(
            df, features, model_key, strategy, test_size, int(random_state), threshold,
            class_weight, drop_duplicates, params, human_source(meta),
        )
    if strategy_clicked:
        _run_strategy_comparison(
            df, features, model_key, test_size, int(random_state), threshold,
            class_weight, drop_duplicates, params, human_source(meta),
        )
    if cv_clicked:
        _run_group_cv(
            df, features, model_key, int(random_state), threshold,
            class_weight, drop_duplicates, params,
        )

    bundle = st.session_state.get("trained_bundle")
    result = st.session_state.get("training_result")
    if bundle is None:
        empty_state(
            "No model in memory",
            "Choose an algorithm and validation strategy, then train a model. You can also " \
            "load a previously saved model below.",
            icon="◇",
        )
    else:
        _show_results(bundle, result)

    _show_comparison()
    _show_strategy_comparison()
    _show_group_cv()

    if bundle is not None:
        _model_management(bundle, result)

    if bundle is None:
        info = model_file_info(MODEL_PATH)
        if info["exists"]:
            if st.button("Load saved model file", type="primary", key="train_load_empty"):
                try:
                    loaded = load_model(MODEL_PATH)
                    available = get_feature_columns(df)
                    missing = validate_bundle_features(loaded, available)
                    if missing:
                        st.error("Saved model requires missing features: " + ", ".join(missing))
                    else:
                        set_trained_bundle(loaded, None)
                        st.session_state["eval_threshold"] = float(loaded.get("threshold", 0.5))
                        st.session_state["training_notice"] = "Loaded a saved model bundle from disk."
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    section_header("Continue", kicker="Workflow")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Run a prediction", width="stretch", key="train_go_pred"):
            goto("Failure Prediction")
            st.rerun()
    with c2:
        if st.button("Explain the model", width="stretch", key="train_go_explain"):
            goto("Model Explainability")
            st.rerun()
    with c3:
        if st.button("Engineering insights", width="stretch", key="train_go_insights"):
            goto("Engineering Insights")
            st.rerun()
