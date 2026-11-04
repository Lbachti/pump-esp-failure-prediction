"""Manual ESP prediction interface."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import FEATURE_LABELS, FEATURE_UNITS, MODEL_PATH
from ..data_loader import get_feature_columns
from ..explainability import contribution_figure, explain_single, explanation_text
from ..plotting import apply_theme
from ..prediction import (
    bundle_info_table,
    input_template,
    load_model,
    model_file_info,
    predict_esp,
    prediction_interpretation,
    risk_color,
    risk_level,
    validate_bundle_features,
)
from ..state import (
    active_df,
    current_hash,
    goto,
    has_data,
    log_prediction,
    model_is_stale,
    set_trained_bundle,
)
from ..ui import (
    callout,
    empty_state,
    format_float,
    page_header,
    plotly_chart,
    section_header,
)

_NONNEGATIVE = {
    "intake_pressure",
    "discharge_pressure",
    "motor_amps",
    "motor_volt",
    "vibration",
    "frequency",
    "production",
}

_STEPS = {
    "intake_pressure": 10.0,
    "discharge_pressure": 10.0,
    "motor_temperature": 1.0,
    "intake_temperature": 1.0,
    "motor_amps": 0.5,
    "motor_volt": 5.0,
    "vibration": 0.01,
    "frequency": 0.5,
    "production": 10.0,
}


def _load_saved_model(df: pd.DataFrame) -> None:
    try:
        loaded = load_model(MODEL_PATH)
        missing = validate_bundle_features(loaded, get_feature_columns(df))
        if missing:
            st.error("Saved model requires missing features: " + ", ".join(missing))
            return
        set_trained_bundle(loaded, None)
        st.session_state["eval_threshold"] = float(loaded.get("threshold", 0.5))
        st.session_state["training_notice"] = "Loaded a saved model bundle from disk."
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))


def _threshold_controls(bundle: dict[str, Any]) -> tuple[float, float, float]:
    bundle_key = (
        f"{bundle.get('model_key', '')}|{bundle.get('trained_at', '')}|"
        f"{bundle.get('dataset_hash', '')}"
    )
    if st.session_state.get("pred_threshold_bundle") != bundle_key:
        # Initialise widget state before the sliders are instantiated so a newly
        # loaded model starts from its own decision threshold and default bands.
        st.session_state["pred_threshold_bundle"] = bundle_key
        st.session_state["pred_decision"] = float(bundle.get("threshold", 0.5))
        st.session_state["pred_warning"] = 0.40
        st.session_state["pred_high"] = 0.70

    with st.expander("Risk thresholds", expanded=False):
        st.markdown(
            "Risk bands are based **only on the model-estimated failure score** and these " \
            "configurable cut-offs. They are not motor-temperature, pressure, or vibration " \
            "engineering limits."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            decision = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                step=0.01,
                key="pred_decision",
                help="At or above this score, the classification is FAILED.",
            )
        with c2:
            warning = st.slider(
                "Elevated-risk threshold",
                min_value=0.05,
                max_value=0.90,
                step=0.01,
                key="pred_warning",
                help="Amber band starts here.",
            )
        with c3:
            high = st.slider(
                "High-risk threshold",
                min_value=0.10,
                max_value=0.99,
                step=0.01,
                key="pred_high",
                help="Red band starts here. Must exceed the elevated threshold.",
            )
        if warning >= high:
            st.error("The elevated-risk threshold must be lower than the high-risk threshold.")
        st.caption(
            "Changing these values re-bands the stored model score without retraining. " \
            "Save the model if you want the decision threshold persisted in the bundle."
        )
    return decision, warning, high


def _gauge(probability: float, level: str, decision: float, warning: float, high: float) -> go.Figure:
    color = risk_color(level)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(
                suffix="%",
                font=dict(size=38, color="#e6eef8", family="JetBrains Mono, monospace"),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=0,
                    tickcolor="#8aa4bf",
                    tickfont=dict(color="#8aa4bf", size=10),
                ),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    {"range": [0, warning * 100], "color": "rgba(34,197,94,0.15)"},
                    {"range": [warning * 100, high * 100], "color": "rgba(245,158,11,0.16)"},
                    {"range": [high * 100, 100], "color": "rgba(239,68,68,0.17)"},
                ],
                threshold=dict(
                    line=dict(color="#22d3ee", width=3),
                    thickness=0.8,
                    value=decision * 100,
                ),
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(title="Model-estimated failure score")
    return apply_theme(fig, title="Model-estimated failure score", height=280, showlegend=False)


def _risk_card(probability: float, level: str, decision: float, predicted: str) -> None:
    color = risk_color(level)
    st.markdown(
        f"""
<div class="risk-card" style="--risk: {color}">
  <div class="risk-top">
    <span class="risk-dot"></span>
    <span class="risk-level">{html.escape(level)} model-estimated risk</span>
  </div>
  <div class="risk-prob">{probability:.1%}</div>
  <div class="risk-caption">
    Classification at threshold {decision:.2f}: <b>{html.escape(predicted)}</b>.
    This score is model output, not automatically a calibrated real-world failure probability.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction(
    bundle: dict[str, Any],
    result: dict[str, Any],
    decision: float,
    warning: float,
    high: float,
) -> None:
    probability = float(result["probability"])
    try:
        level = risk_level(probability, warning, high)
    except Exception:  # noqa: BLE001
        level = result.get("risk_level", "Normal")
    predicted = "FAILED" if probability >= decision else "RUNNING"
    _risk_card(probability, level, decision, predicted)

    c1, c2 = st.columns([1, 1.05], gap="large")
    with c1:
        plotly_chart(_gauge(probability, level, decision, warning, high), key="pred_gauge")
    with c2:
        st.markdown("**What this means**")
        st.write(prediction_interpretation({**result, "risk_level": level, "predicted_label": predicted, "decision_threshold": decision, "warning_threshold": warning, "high_threshold": high}))
        st.caption(result.get("confidence_note", ""))
        if result.get("brier") is not None and result.get("brier") == result.get("brier"):
            st.caption(
                f"Holdout Brier score at training: {result['brier']:.3f}. This measures " \
                "probability accuracy on that evaluation set, not field reliability."
            )
        callout(
            "Probability caveat",
            "A model score is not a guaranteed failure probability. It may be miscalibrated, " \
            "especially for wells or operating conditions not represented in training. " \
            "Review the calibration curve and validation strategy before acting.",
            tone="warn",
        )

    if result.get("out_of_range"):
        st.markdown("**Inputs outside the training range**")
        rows = []
        for feature, values in result["out_of_range"].items():
            rows.append(
                {
                    "Parameter": FEATURE_LABELS.get(feature, feature),
                    "Input": values["value"],
                    "Training min": values["min"],
                    "Training median": values.get("median"),
                    "Training max": values["max"],
                }
            )
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
        st.warning(
            "These inputs are outside the range observed during training. The model is " \
            "extrapolating; treat the score with extra caution."
        )
    else:
        st.success("All inputs lie within the observed training range (range check only).")

    section_header(
        "Why this prediction?",
        "Feature attribution explains the fitted model's output for this input. It does not " \
        "prove a physical failure mechanism.",
        kicker="Explainability",
    )
    try:
        explanation = explain_single(bundle, result["inputs"])
        plotly_chart(contribution_figure(explanation), key="pred_contrib")
        st.caption(
            f"Method: {explanation['method']}. Positive values push the model toward failure; " \
            "negative values push toward running."
        )
        st.write(explanation_text(explanation))
        top = explanation["table"].head(3)
        if not top.empty:
            st.markdown("**Top model associations for this case**")
            for _, row in top.iterrows():
                st.markdown(
                    f"- **{row['label']}** = {format_float(row['value'], 3)} · "
                    f"contribution {row['contribution']:+.3f} ({row['direction'].lower()})"
                )
        st.caption(
            "These are model associations, not engineering limits. Do not adjust equipment " \
            "set-points based on this chart alone."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Individual explanation could not be computed: {exc}")


def render() -> None:
    page_header(
        "Failure prediction",
        "Enter an ESP operating snapshot and pass it through the saved preprocessing " \
        "pipeline and trained classifier.",
        eyebrow="ESP PREDICTAI / 06 · PREDICTION",
        badges=[("Model-score risk bands", "cyan"), ("No invented engineering limits", "amber")],
    )
    if not has_data():
        st.error("No dataset is loaded. Open Data Explorer first.")
        if st.button("Go to Data Explorer", key="pred_no_data"):
            goto("Data Explorer")
            st.rerun()
        return

    df = active_df()
    bundle = st.session_state.get("trained_bundle")
    if bundle is None:
        empty_state(
            "Predictions are disabled",
            "No model is in memory. Train a model in ML Model Training or load a saved model " \
            "bundle. Predictions cannot run before a model exists.",
            icon="◇",
        )
        info = model_file_info(MODEL_PATH)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Go to model training", type="primary", width="stretch", key="pred_go_train"):
                goto("ML Model Training")
                st.rerun()
        with c2:
            if st.button(
                "Load saved model",
                width="stretch",
                key="pred_load_saved_empty",
                disabled=not info["exists"],
            ):
                _load_saved_model(df)
        if not info["exists"]:
            st.caption("No saved model file exists yet.")
        return

    if model_is_stale():
        callout(
            "Model / dataset mismatch",
            "The active dataset fingerprint differs from the model's training fingerprint. " \
            "Predictions still run, but the model may not match the current data version. " \
            "Retrain for a valid comparison.",
            tone="danger",
            icon="⚠",
        )

    st.markdown(
        f"**{bundle.get('model_label', 'Model')}** · "
        f"{bundle.get('split_info', {}).get('strategy_label', 'Unknown validation')} · "
        f"threshold {bundle.get('threshold', 0.5):.2f} · "
        f"trained {bundle.get('trained_at', '—')}"
    )
    with st.expander("Model information", expanded=False):
        st.dataframe(bundle_info_table(bundle), width="stretch", hide_index=True)
        st.caption(
            f"Active dataset fingerprint: {current_hash()[:16] or '—'} · "
            f"Model fingerprint: {bundle.get('dataset_hash', '')[:16] or '—'}"
        )

    decision, warning, high = _threshold_controls(bundle)
    thresholds_valid = warning < high
    template = input_template(bundle)
    features = list(bundle["features"])

    section_header(
        "Operating snapshot",
        "Defaults are training-set medians. Edit any value, then run the model. " \
        "Inputs are checked for finite values and basic physical validity only.",
        kicker="Manual input",
    )
    with st.form("esp_prediction_form", clear_on_submit=False):
        cols = st.columns(3)
        inputs: dict[str, float] = {}
        for i, feature in enumerate(features):
            label = FEATURE_LABELS.get(feature, feature)
            unit = FEATURE_UNITS.get(feature, "")
            with cols[i % 3]:
                inputs[feature] = st.number_input(
                    f"{label} ({unit})" if unit else label,
                    value=float(template.get(feature, 0.0)),
                    step=float(_STEPS.get(feature, 1.0)),
                    min_value=0.0 if feature in _NONNEGATIVE else None,
                    format="%.2f" if feature == "vibration" else "%.1f",
                    key=f"pred_input_{feature}",
                    help=f"Typical training median: {template.get(feature, 0.0):.3f}",
                )
        st.caption(
            "No universal pressure, temperature, current, or vibration limits are applied. " \
            "Risk categories are based on model scores and the thresholds above."
        )
        submitted = st.form_submit_button(
            "PREDICT ESP STATUS",
            type="primary",
            width="stretch",
        )

    if submitted:
        if not thresholds_valid:
            st.error("Fix the risk thresholds before predicting.")
        else:
            try:
                with st.spinner("Scoring the ESP operating snapshot..."):
                    result = predict_esp(
                        bundle,
                        inputs,
                        decision_threshold=decision,
                        warning_threshold=warning,
                        high_threshold=high,
                    )
                log_prediction(result)
                st.session_state["prediction_notice"] = "Prediction completed."
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    last = st.session_state.get("last_prediction")
    if last is not None:
        st.caption(
            f"Scored at {last.get('scored_at', '—')} with {last.get('model_label', 'model')}. "
            "Changing thresholds re-bands the same score; submit again to rescore after " \
            "changing inputs."
        )
        _render_prediction(bundle, last, decision, warning, high)
    else:
        empty_state(
            "No prediction yet",
            "Enter the operating values and click PREDICT ESP STATUS. The result will show " \
            "the model score, classification, risk band, and feature attribution.",
            icon="◎",
        )

    history = st.session_state.get("prediction_history") or []
    if history:
        section_header(
            "Prediction history",
            "Session-only log of manual scoring requests. It is not a maintenance record.",
            kicker="Audit trail",
        )
        frame = pd.DataFrame(history)
        st.dataframe(frame.round(4), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download prediction history (CSV)",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name="esp_prediction_history.csv",
                mime="text/csv",
                key="pred_download_history",
            )
        with c2:
            if st.button("Clear prediction history", width="stretch", key="pred_clear_history"):
                st.session_state["prediction_history"] = []
                st.session_state["last_prediction"] = None
                st.rerun()

    section_header("Continue", kicker="Workflow")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Explain the model", width="stretch", key="pred_go_explain"):
            goto("Model Explainability")
            st.rerun()
    with c2:
        if st.button("Review engineering insights", width="stretch", key="pred_go_insights"):
            goto("Engineering Insights")
            st.rerun()
