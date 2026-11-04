"""Model-agnostic and SHAP-based explainability utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

from .config import COLORS, FEATURE_LABELS
from .plotting import apply_theme
from .preprocessing import transform_features


def _model(bundle: dict[str, Any]):
    return bundle["pipeline"].named_steps["model"]


def global_importance(bundle: dict[str, Any]) -> pd.DataFrame:
    """Native model importance: impurity/gain for trees, |coef| for logistic."""
    model = _model(bundle)
    features = list(bundle["features"])
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "Tree importance (impurity / gain)"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        method = "Absolute standardised coefficient"
    else:
        return pd.DataFrame(columns=["feature", "label", "importance", "share", "method"])
    total = float(values.sum())
    frame = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "importance": values,
            "share": values / total if total else 0.0,
            "method": method,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on a held-out or explicitly labelled dataset."""
    features = list(bundle["features"])
    X = X[features].copy()
    y = pd.Series(y).astype(int).reset_index(drop=True)
    if X.empty:
        return pd.DataFrame()
    scoring = "roc_auc" if y.nunique() > 1 else "accuracy"
    result = permutation_importance(
        bundle["pipeline"],
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "scoring": scoring,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _normalise_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        # Newer SHAP returns (n_samples, n_features, n_classes).
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {arr.shape}")
    return arr


def _normalise_expected(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return float("nan")
        return float(arr[1] if arr.size > 1 else arr[0])
    return float(value)


def compute_shap(
    bundle: dict[str, Any],
    X: pd.DataFrame,
    max_samples: int = 300,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute SHAP values for a sample, returning a serialisable result dict."""
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        return {"error": f"SHAP is not installed: {exc}"}

    features = list(bundle["features"])
    frame = X[features].dropna(how="all").copy()
    if frame.empty:
        return {"error": "No rows available for SHAP."}
    if len(frame) > max_samples:
        frame = frame.sample(max_samples, random_state=random_state)
    frame = frame.reset_index(drop=True)
    try:
        transformed = transform_features(bundle["pipeline"], frame)
        model = _model(bundle)
        if bundle.get("model_key") == "logistic":
            explainer = shap.LinearExplainer(model, transformed)
            method = "SHAP LinearExplainer"
        else:
            explainer = shap.TreeExplainer(model)
            method = "SHAP TreeExplainer"
        raw_values = explainer.shap_values(transformed)
        values = _normalise_shap_values(raw_values)
        base = _normalise_expected(explainer.expected_value)
        if values.shape[1] != len(features):
            raise ValueError(
                f"SHAP returned {values.shape[1]} features, expected {len(features)}."
            )
        return {
            "values": pd.DataFrame(values, columns=features),
            "data": frame,
            "base_value": base,
            "method": method,
            "n": int(len(frame)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"SHAP computation failed: {exc}"}


def explain_single(bundle: dict[str, Any], inputs: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
    """Explain one observation with SHAP or a clearly labelled fallback."""
    features = list(bundle["features"])
    if isinstance(inputs, dict):
        frame = pd.DataFrame([{f: inputs[f] for f in features}], columns=features)
    else:
        frame = inputs[features].head(1).copy()
    transformed = transform_features(bundle["pipeline"], frame)
    model = _model(bundle)
    method = "Unknown"
    base = float("nan")
    contributions = None

    if bundle.get("model_key") == "logistic" and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        contributions = coef * transformed[0]
        base = float(np.asarray(model.intercept_).ravel()[0])
        method = "Logistic regression coefficient × standardised input (log-odds)"
    else:
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(transformed)
            arr = _normalise_shap_values(raw)
            contributions = arr[0]
            base = _normalise_expected(explainer.expected_value)
            method = "SHAP TreeExplainer (log-odds)"
        except Exception:  # noqa: BLE001
            contributions = None

    is_shap = method.startswith("SHAP")
    if contributions is None:
        # Fallback: global importance × signed standardised deviation.
        imp = global_importance(bundle)
        stats = bundle.get("training_stats", {}) or {}
        signed = []
        for feature in features:
            value = float(frame.iloc[0][feature])
            stat = stats.get(feature, {})
            mean = float(stat.get("mean", 0.0))
            std = float(stat.get("std", 1.0)) or 1.0
            z = (value - mean) / std
            weight = float(
                imp.loc[imp["feature"] == feature, "share"].sum() if not imp.empty else 0.0
            )
            signed.append(weight * z)
        contributions = np.asarray(signed, dtype=float)
        method = "Heuristic: global importance × standardised deviation (not SHAP)"
        is_shap = False

    frame_out = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "value": [float(frame.iloc[0][f]) for f in features],
            "contribution": contributions,
        }
    )
    frame_out["abs_contribution"] = frame_out["contribution"].abs()
    frame_out["direction"] = np.where(
        frame_out["contribution"] > 0,
        "Pushes toward failure",
        "Pushes toward running",
    )
    frame_out = frame_out.sort_values("abs_contribution", ascending=False).reset_index(
        drop=True
    )
    return {
        "table": frame_out,
        "base_value": base,
        "method": method,
        "is_shap": is_shap,
        "inputs": frame,
    }


def importance_figure(
    frame: pd.DataFrame,
    value_col: str = "importance",
    title: str = "Global feature importance",
) -> go.Figure:
    if frame is None or frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No importance values available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"]),
        )
        return apply_theme(fig, title=title, height=300)
    data = frame.sort_values(value_col, ascending=True)
    colors = [
        COLORS["cyan"] if i == len(data) - 1 else COLORS["blue"]
        for i in range(len(data))
    ]
    fig = go.Figure(
        go.Bar(
            x=data[value_col],
            y=data["label"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.3f}" for v in data[value_col]],
            textposition="outside",
            textfont=dict(color=COLORS["muted"], size=10),
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Relative importance")
    fig.update_yaxes(title="")
    return apply_theme(fig, title=title, height=max(320, 60 + 30 * len(data)))


def shap_summary_figure(shap_result: dict[str, Any]) -> go.Figure:
    if "error" in shap_result:
        fig = go.Figure()
        fig.add_annotation(
            text=shap_result["error"],
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"]),
        )
        return apply_theme(fig, title="SHAP summary", height=300)
    values = shap_result["values"]
    data = shap_result["data"]
    order = values.abs().mean().sort_values(ascending=True).index.tolist()
    fig = go.Figure()
    rng = np.random.default_rng(42)
    for i, feature in enumerate(order):
        x = values[feature].to_numpy()
        raw = data[feature].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(raw), np.nanmax(raw)
        norm = (raw - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(raw, 0.5)
        jitter = rng.uniform(-0.22, 0.22, size=len(x))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full(len(x), i, dtype=float) + jitter,
                mode="markers",
                marker=dict(
                    size=7,
                    color=norm,
                    colorscale=[
                        [0.0, "#2563eb"],
                        [0.45, "#22d3ee"],
                        [0.75, "#f59e0b"],
                        [1.0, "#ef4444"],
                    ],
                    cmin=0,
                    cmax=1,
                    showscale=i == len(order) - 1,
                    colorbar=dict(
                        title=dict(text="Feature value", font=dict(size=10)),
                        tickvals=[0, 1],
                        ticktext=["Low", "High"],
                        thickness=10,
                        len=0.6,
                    ),
                    line=dict(width=0),
                    opacity=0.85,
                ),
                customdata=np.column_stack([raw, x]),
                hovertemplate=(
                    f"<b>{FEATURE_LABELS.get(feature, feature)}</b><br>"
                    "Feature value: %{customdata[0]:.3f}<br>"
                    "SHAP value: %{customdata[1]:.3f}<extra></extra>"
                ),
                name=FEATURE_LABELS.get(feature, feature),
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line=dict(color=COLORS["muted"], width=1, dash="dot"))
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(order))),
        ticktext=[FEATURE_LABELS.get(f, f) for f in order],
        title="",
    )
    fig.update_xaxes(title="SHAP value (impact on model output, log-odds)")
    return apply_theme(
        fig,
        title=f"SHAP summary — {shap_result['method']} (n = {shap_result['n']})",
        height=max(380, 100 + 38 * len(order)),
        showlegend=False,
        margin=dict(l=10, r=70, t=52, b=10),
    )


def mean_shap_figure(shap_result: dict[str, Any]) -> go.Figure:
    if "error" in shap_result:
        return shap_summary_figure(shap_result)
    values = shap_result["values"]
    frame = (
        values.abs()
        .mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    frame["label"] = frame["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    return importance_figure(
        frame, value_col="mean_abs_shap", title="Mean absolute SHAP value"
    )


def contribution_figure(explanation: dict[str, Any], top_n: int = 9) -> go.Figure:
    table = explanation["table"].head(top_n).sort_values("contribution")
    colors = [
        COLORS["red"] if value > 0 else COLORS["green"]
        for value in table["contribution"]
    ]
    fig = go.Figure(
        go.Bar(
            x=table["contribution"],
            y=table["label"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in table["contribution"]],
            textposition="outside",
            textfont=dict(color=COLORS["muted"], size=10),
            hovertemplate="%{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=COLORS["muted"], width=1))
    fig.update_xaxes(title="Contribution to model output (log-odds)")
    fig.update_yaxes(title="")
    return apply_theme(
        fig,
        title="Individual feature contributions",
        height=max(340, 70 + 30 * len(table)),
        showlegend=False,
    )


def explanation_text(explanation: dict[str, Any], top_n: int = 3) -> str:
    table = explanation["table"].head(top_n)
    if table.empty:
        return "No feature contributions are available for this prediction."
    parts = []
    for _, row in table.iterrows():
        direction = "increased" if row["contribution"] > 0 else "decreased"
        parts.append(
            f"{row['label']} {direction} the model score "
            f"({row['contribution']:+.3f} log-odds)"
        )
    method = str(explanation.get("method", ""))
    if explanation.get("is_shap"):
        prefix = "SHAP attribution"
    elif method.startswith("Logistic regression coefficient"):
        prefix = "Logistic coefficient attribution"
    else:
        prefix = "Heuristic attribution (not SHAP)"
    return (
        f"{prefix}: " + "; ".join(parts) + ". These are model associations, not "
        "physical causation or engineering limits."
    )
