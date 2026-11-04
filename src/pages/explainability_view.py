"""Model explainability page: native importance, permutation importance, SHAP, and individual attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ..config import FEATURE_LABELS, MODEL_PATH
from ..eda import univariate_signal_table
from ..explainability import (
    compute_shap,
    contribution_figure,
    explain_single,
    explanation_text,
    global_importance,
    importance_figure,
    mean_shap_figure,
    permutation_importance_table,
    shap_summary_figure,
)
from ..prediction import load_model, model_file_info, validate_bundle_features
from ..state import active_df, goto, has_data, set_trained_bundle
from ..ui import (
    callout,
    empty_state,
    format_float,
    page_header,
    plotly_chart,
    section_header,
)

METHOD_NOTES = {
    "native": (
        "Native tree importance measures how much a feature reduces impurity (or gain) " \
        "across the fitted trees. It is model-specific and can favour high-cardinality or " \
        "frequently used features. For Logistic Regression the table shows absolute " \
        "standardised coefficients."
    ),
    "permutation": (
        "Permutation importance measures how much holdout performance drops when one feature " \
        "is randomly shuffled. It is model-agnostic but depends on the evaluation data and " \
        "metric. Correlated features can share or mask importance."
    ),
    "shap": (
        "SHAP decomposes an individual model output into additive feature contributions. " \
        "TreeExplainer is exact for tree models; LinearExplainer is exact for the linear " \
        "model. SHAP values explain the model, not the physical system. They do not establish " \
        "causation and can be unstable when features are correlated."
    ),
}


def _load_saved_model(df: pd.DataFrame) -> None:
    try:
        loaded = load_model(MODEL_PATH)
        missing = validate_bundle_features(loaded, list(df.columns))
        if missing:
            st.error("Saved model requires missing features: " + ", ".join(missing))
            return
        set_trained_bundle(loaded, None)
        st.session_state["training_notice"] = "Loaded a saved model bundle from disk."
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))


def _influence_table(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    imp = global_importance(bundle)
    signal = univariate_signal_table(df, list(bundle["features"]))
    if imp.empty:
        return signal
    merged = imp[["feature", "label", "share"]].merge(
        signal[["feature", "auc", "signal", "direction", "p_value"]],
        on="feature",
        how="left",
        suffixes=("", "_dataset"),
    )
    merged["label"] = merged["label"].fillna(
        merged["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    )
    return merged.sort_values("share", ascending=False).reset_index(drop=True)


def _global_tab(bundle: dict[str, Any], result: dict[str, Any] | None, df: pd.DataFrame) -> None:
    section_header(
        "Native model importance",
        "Which features the fitted model uses most. This is not evidence of physical causation.",
        kicker="Global",
    )
    if bundle.get("model_key") == "logistic":
        callout(
            "Logistic model active",
            "The chart shows absolute standardised coefficients. Tree-specific impurity or " \
            "gain importance is available when a Random Forest or XGBoost model is trained.",
            tone="info",
        )
    imp = global_importance(bundle)
    if imp.empty:
        st.info("This model does not expose native feature importance.")
    else:
        plotly_chart(importance_figure(imp), key="explain_native")
        st.dataframe(
            imp.rename(
                columns={
                    "label": "Parameter",
                    "importance": "Importance",
                    "share": "Share",
                    "method": "Method",
                }
            )[["Parameter", "Importance", "Share", "Method"]].round(4),
            width="stretch",
            hide_index=True,
        )
        st.caption(METHOD_NOTES["native"])

    section_header(
        "Model association vs dataset association",
        "A feature can be important to the model without showing a strong univariate " \
        "difference, because the model may use interactions and well-specific signatures.",
        kicker="Cross-check",
    )
    influence = _influence_table(bundle, df)
    if not influence.empty:
        st.dataframe(
            influence.rename(
                columns={
                    "label": "Parameter",
                    "share": "Model importance share",
                    "auc": "Univariate ROC-AUC",
                    "signal": "Univariate signal",
                    "direction": "Dataset direction",
                    "p_value": "Mann–Whitney p",
                }
            ).round(4),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Model importance share and univariate signal answer different questions. " \
            "Neither is a causal effect."
        )

    section_header(
        "Permutation importance",
        "Model-agnostic check: shuffle one feature at a time and measure the performance drop.",
        kicker="Holdout",
    )
    if result is not None:
        X_eval = result["X_test"]
        y_eval = result["y_test"]
        source = "held-out test split"
    else:
        X_eval = df[list(bundle["features"])].dropna(how="all").head(1500)
        y_eval = df.loc[X_eval.index, "esp_failure"]
        source = "current dataset (may overlap training data)"
        callout(
            "No held-out predictions in memory",
            "This model was loaded from disk, so permutation importance uses a sample of the " \
            "current dataset. If that data overlaps the training data, importance may be " \
            "optimistic. Retrain to compute it on the holdout split.",
            tone="warn",
        )
    if st.button("Compute permutation importance", key="explain_perm_btn"):
        try:
            with st.spinner("Shuffling features and re-scoring..."):
                perm = permutation_importance_table(
                    bundle, X_eval, y_eval, n_repeats=8, random_state=42
                )
            st.session_state["perm_result"] = perm
            st.session_state["perm_source"] = source
        except Exception as exc:  # noqa: BLE001
            st.error(f"Permutation importance failed: {exc}")
    perm = st.session_state.get("perm_result")
    if isinstance(perm, pd.DataFrame) and not perm.empty:
        plotly_chart(
            importance_figure(
                perm.rename(columns={"importance_mean": "importance"}),
                value_col="importance",
                title="Permutation importance (performance drop)",
            ),
            key="explain_perm_fig",
        )
        st.dataframe(
            perm.rename(
                columns={
                    "label": "Parameter",
                    "importance_mean": "Mean importance",
                    "importance_std": "Std. dev.",
                    "scoring": "Scoring",
                }
            )[["Parameter", "Mean importance", "Std. dev.", "Scoring"]].round(4),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Evaluation data: {st.session_state.get('perm_source', source)}. {METHOD_NOTES['permutation']}")


def _shap_tab(bundle: dict[str, Any], result: dict[str, Any] | None, df: pd.DataFrame) -> None:
    section_header(
        "SHAP summary",
        "Global additive attributions on a sample of observations. Red/high feature values " \
        "are not inherently failure-causing; the colour shows the feature value.",
        kicker="SHAP",
    )
    if result is not None:
        X_source = result["X_test"]
        source = "held-out test split"
    else:
        X_source = df[list(bundle["features"])].dropna(how="all")
        source = "current dataset (may overlap training data)"
    c1, c2 = st.columns([1, 1])
    with c1:
        max_samples = st.slider(
            "SHAP sample size", 50, 600, 250, 50, key="explain_shap_n"
        )
    with c2:
        st.write("")
        st.write("")
        compute = st.button("Compute SHAP summary", type="primary", key="explain_shap_btn")
    if compute:
        with st.spinner("Computing SHAP values..."):
            shap_result = compute_shap(
                bundle, X_source, max_samples=int(max_samples), random_state=42
            )
        st.session_state["shap_result"] = shap_result
        st.session_state["shap_source"] = source
        st.session_state["shap_model"] = bundle.get("trained_at")
    shap_result = st.session_state.get("shap_result")
    if shap_result and st.session_state.get("shap_model") != bundle.get("trained_at"):
        shap_result = None
    if not shap_result:
        st.info(
            "Click Compute SHAP summary to generate the beeswarm and mean-absolute-SHAP charts."
        )
        return
    if "error" in shap_result:
        st.error(shap_result["error"])
        st.caption(
            "If SHAP is unavailable, use native or permutation importance. The prediction page " \
            "also provides a clearly labelled fallback for individual attribution."
        )
        return
    plotly_chart(shap_summary_figure(shap_result), key="explain_shap_summary")
    plotly_chart(mean_shap_figure(shap_result), key="explain_shap_bar")
    st.caption(
        f"Method: {shap_result['method']} · n = {shap_result['n']} · "
        f"evaluation data: {st.session_state.get('shap_source', source)}. "
        f"Base value: {format_float(shap_result.get('base_value'), 3)} (log-odds)."
    )
    st.caption(METHOD_NOTES["shap"])


def _individual_tab(bundle: dict[str, Any], result: dict[str, Any] | None, df: pd.DataFrame) -> None:
    section_header(
        "Individual explanation",
        "Inspect the additive contribution of each feature to one model score.",
        kicker="Local",
    )
    options = []
    if st.session_state.get("last_prediction"):
        options.append("Last manual prediction")
    if result is not None:
        options.append("Held-out test row")
    options.append("Loaded dataset row")
    mode = st.radio("Explanation source", options, horizontal=True, key="explain_local_mode")

    explanation = None
    context = ""
    if mode == "Last manual prediction":
        last = st.session_state["last_prediction"]
        explanation = explain_single(bundle, last["inputs"])
        context = (
            f"Manual prediction · model score {last['probability']:.3f} · "
            f"risk band {last['risk_level']}"
        )
    elif mode == "Held-out test row" and result is not None:
        n = len(result["X_test"])
        row = st.number_input(
            "Test row index",
            min_value=0,
            max_value=max(0, n - 1),
            value=0,
            step=1,
            key="explain_test_row",
        )
        X_row = result["X_test"].iloc[[int(row)]]
        explanation = explain_single(bundle, X_row)
        actual = int(result["y_test"].iloc[int(row)])
        prob = float(result["proba"][int(row)])
        meta = result.get("test_meta")
        meta_text = ""
        if isinstance(meta, pd.DataFrame) and len(meta) > int(row):
            parts = []
            for col in ["well_id", "date"]:
                if col in meta.columns:
                    parts.append(f"{col}={meta.iloc[int(row)][col]}")
            meta_text = " · " + " · ".join(parts) if parts else ""
        context = (
            f"Held-out row {int(row)} · actual={'Failed' if actual else 'Running'} · "
            f"model score {prob:.3f}{meta_text}"
        )
    else:
        features = list(bundle["features"])
        available = df.dropna(subset=[f for f in features if f in df.columns], how="all")
        if available.empty:
            st.warning("No rows available for individual explanation.")
            return
        n = min(len(available), 2000)
        row = st.number_input(
            "Dataset row position",
            min_value=0,
            max_value=max(0, n - 1),
            value=0,
            step=1,
            key="explain_dataset_row",
        )
        row_frame = available.iloc[[int(row)]]
        explanation = explain_single(bundle, row_frame[features])
        actual = int(row_frame["esp_failure"].iloc[0]) if "esp_failure" in row_frame else -1
        context = (
            f"Loaded dataset row {int(row)} · actual="
            f"{'Failed' if actual == 1 else 'Running' if actual == 0 else 'unknown'}"
        )

    if explanation is None:
        st.info("No explanation is available for this selection.")
        return
    st.caption(context)
    plotly_chart(contribution_figure(explanation), key="explain_local_fig")
    st.caption(
        f"Method: {explanation['method']} · base value {format_float(explanation.get('base_value'), 3)} "
        "(log-odds). Positive contributions push the model output toward failure."
    )
    st.write(explanation_text(explanation))
    st.dataframe(
        explanation["table"].rename(
            columns={
                "label": "Parameter",
                "value": "Input value",
                "contribution": "Contribution",
                "abs_contribution": "Absolute contribution",
                "direction": "Direction",
            }
        )[["Parameter", "Input value", "Contribution", "Direction"]].round(4),
        width="stretch",
        hide_index=True,
    )
    if not explanation.get("is_shap") and "Logistic" not in explanation.get("method", ""):
        st.warning(
            "SHAP could not be computed for this model/case. The chart uses a clearly labelled " \
            "heuristic (global importance × standardised deviation), not SHAP."
        )
    callout(
        "Interpretation guardrail",
        "An individual attribution is conditional on the fitted model, the input values, and " \
        "the training distribution. It is not a root-cause diagnosis and does not replace " \
        "equipment specifications, field measurements, or maintenance records.",
        tone="warn",
    )


def render() -> None:
    page_header(
        "Model explainability",
        "Understand global model behaviour and individual predictions with native importance, " \
        "permutation importance, SHAP, and additive feature attribution.",
        eyebrow="ESP PREDICTAI / 07 · EXPLAINABILITY",
        badges=[("SHAP when available", "cyan"), ("Model association, not causation", "amber")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="explain_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    bundle = st.session_state.get("trained_bundle")
    if bundle is None:
        empty_state(
            "No model in memory",
            "Train a model or load a saved bundle to inspect feature importance and SHAP values.",
            icon="◇",
        )
        info = model_file_info(MODEL_PATH)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Go to model training", type="primary", width="stretch", key="explain_go_train"):
                goto("ML Model Training")
                st.rerun()
        with c2:
            if st.button(
                "Load saved model",
                width="stretch",
                key="explain_load_saved",
                disabled=not info["exists"],
            ):
                _load_saved_model(df)
        return

    st.markdown(
        f"**{bundle.get('model_label', 'Model')}** · "
        f"{bundle.get('split_info', {}).get('strategy_label', 'Unknown validation')} · "
        f"features: {', '.join(FEATURE_LABELS.get(f, f) for f in bundle.get('features', []))}"
    )
    result = st.session_state.get("training_result")
    if result is None:
        st.caption(
            "This model was loaded from disk; held-out predictions are not in memory. " \
            "SHAP and permutation importance will use a sample of the current dataset."
        )

    tab_global, tab_shap, tab_individual, tab_notes = st.tabs(
        ["Global importance", "SHAP", "Individual explanation", "Method notes"]
    )
    with tab_global:
        _global_tab(bundle, result, df)
    with tab_shap:
        _shap_tab(bundle, result, df)
    with tab_individual:
        _individual_tab(bundle, result, df)
    with tab_notes:
        section_header(
            "How to read these explanations",
            "Different methods answer different questions. None of them proves physical causation.",
            kicker="Methods",
        )
        st.markdown("**Native feature importance**")
        st.write(METHOD_NOTES["native"])
        st.markdown("**Permutation importance**")
        st.write(METHOD_NOTES["permutation"])
        st.markdown("**SHAP**")
        st.write(METHOD_NOTES["shap"])
        st.markdown("**Logistic regression coefficients**")
        st.write(
            "For Logistic Regression, the displayed coefficients are absolute standardised " \
            "coefficients. A positive coefficient increases the log-odds of failure; a negative " \
            "coefficient decreases it. The pipeline scales features inside the training split."
        )
        st.markdown("**Random Forest and XGBoost**")
        st.write(
            "Random Forest uses impurity-based importance by default; XGBoost uses gain-based " \
            "importance. These are useful diagnostics but can be biased by feature scale, " \
            "cardinality, and correlated inputs. Use permutation importance and SHAP as " \
            "complementary checks."
        )
        st.markdown("**Why importance is not causation**")
        st.write(
            "A feature can be important because it is correlated with well identity, operating " \
            "strategy, or another unmeasured variable. The dataset's provenance is unverified, " \
            "so feature attributions may reflect data-generation artefacts rather than ESP physics."
        )
        st.markdown("**Validation matters more than explanation**")
        st.write(
            "An explainable model that fails on unseen wells is still not a reliable predictor. " \
            "Review the well-based validation results before using any attribution in an " \
            "engineering discussion."
        )

    section_header("Continue", kicker="Workflow")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run a prediction", width="stretch", key="explain_go_pred"):
            goto("Failure Prediction")
            st.rerun()
    with c2:
        if st.button("Engineering insights", width="stretch", key="explain_go_insights"):
            goto("Engineering Insights")
            st.rerun()
