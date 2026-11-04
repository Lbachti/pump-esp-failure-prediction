"""Model-evaluation metrics and evaluation figures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import COLORS
from .plotting import apply_theme


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, kind: str = "roc") -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if kind == "pr":
            return float(average_precision_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score))
    except Exception:  # noqa: BLE001
        return float("nan")


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    if len(y) != len(score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y) == 0:
        raise ValueError("No evaluation rows.")
    threshold = float(min(max(threshold, 0.0), 1.0))
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = accuracy_score(y, pred)
    brier = brier_score_loss(y, score) if len(np.unique(y)) == 2 else float("nan")
    return {
        "threshold": threshold,
        "n": int(len(y)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "predicted_positives": int(pred.sum()),
        "predicted_positive_rate": float(pred.mean()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": float((recall + specificity) / 2)
        if not np.isnan(specificity)
        else float("nan"),
        "roc_auc": _safe_auc(y, score, "roc"),
        "pr_auc": _safe_auc(y, score, "pr"),
        "brier": float(brier),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def confusion_matrix_array(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return confusion_matrix(y, pred, labels=[0, 1])


def classification_report_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    rows = [
        {
            "class": "Running (0)",
            "precision": precision[0],
            "recall": recall[0],
            "f1": f1[0],
            "support": int(support[0]),
        },
        {
            "class": "Failed (1)",
            "precision": precision[1],
            "recall": recall[1],
            "f1": f1[1],
            "support": int(support[1]),
        },
    ]
    return pd.DataFrame(rows)


def roc_curve_points(
    y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, thresholds = roc_curve(y, score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def pr_curve_points(
    y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray
) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["recall", "precision", "threshold"])
    precision, recall, thresholds = precision_recall_curve(y, score)
    # precision_recall_curve returns one more precision/recall point than thresholds.
    return pd.DataFrame(
        {
            "recall": recall[:-1],
            "precision": precision[:-1],
            "threshold": thresholds,
        }
    )


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for threshold in thresholds:
        m = compute_metrics(y_true, y_score, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "false_negatives": m["fn"],
                "false_positives": m["fp"],
                "predicted_failure_rate": m["predicted_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def calibration_points(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(y) == 0:
        return pd.DataFrame(columns=["mean_score", "observed_rate", "count"])
    n_bins = max(2, min(n_bins, len(np.unique(score))))
    try:
        bins = pd.qcut(score, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["mean_score", "observed_rate", "count"])
    frame = pd.DataFrame({"score": score, "target": y, "bin": bins})
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_score=("score", "mean"),
            observed_rate=("target", "mean"),
            count=("target", "size"),
        )
        .reset_index(drop=True)
        .sort_values("mean_score")
    )
    return grouped


def actual_vs_predicted_table(
    result: dict[str, Any],
    threshold: float | None = None,
) -> pd.DataFrame:
    threshold = result.get("threshold", 0.5) if threshold is None else threshold
    y = np.asarray(result["y_test"], dtype=int)
    score = np.asarray(result["proba"], dtype=float)
    pred = (score >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "actual": y,
            "predicted": pred,
            "failure_probability": score,
        }
    )
    frame["outcome"] = np.select(
        [
            (frame["actual"] == 1) & (frame["predicted"] == 1),
            (frame["actual"] == 0) & (frame["predicted"] == 0),
            (frame["actual"] == 1) & (frame["predicted"] == 0),
            (frame["actual"] == 0) & (frame["predicted"] == 1),
        ],
        ["TP", "TN", "FN", "FP"],
        default="—",
    )
    frame["actual_label"] = np.where(frame["actual"] == 1, "Failed", "Running")
    frame["predicted_label"] = np.where(
        frame["predicted"] == 1, "Failed", "Running"
    )
    meta = result.get("test_meta")
    if isinstance(meta, pd.DataFrame) and len(meta) == len(frame):
        for col in meta.columns:
            if col not in frame.columns:
                frame.insert(0, col, meta[col].to_numpy())
    return frame


def confusion_matrix_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
    title: str = "Confusion matrix",
) -> go.Figure:
    cm = confusion_matrix_array(y_true, y_score, threshold)
    labels = ["Running", "Failed"]
    text = [
        [f"TN<br>{cm[0, 0]:,}", f"FP<br>{cm[0, 1]:,}"],
        [f"FN<br>{cm[1, 0]:,}", f"TP<br>{cm[1, 1]:,}"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorscale=[
                [0.0, "rgba(11,26,45,0.9)"],
                [1.0, COLORS["blue"]],
            ],
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_yaxes(autorange="reversed", title="Actual")
    fig.update_xaxes(title="Predicted")
    fig.update_layout(title=title)
    return apply_theme(fig, title=title, height=330, showlegend=False)


def roc_curve_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    title: str = "ROC curve",
) -> go.Figure:
    points = roc_curve_points(y_true, y_score)
    auc = _safe_auc(np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float))
    fig = go.Figure()
    if points.empty:
        fig.add_annotation(
            text="ROC curve requires both classes in the evaluation set.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"]),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=points["fpr"],
                y=points["tpr"],
                mode="lines",
                name=f"Model (AUC = {auc:.3f})",
                line=dict(color=COLORS["cyan"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(34,211,238,0.08)",
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(color=COLORS["muted"], width=1, dash="dot"),
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    return apply_theme(fig, title=title, height=350)


def pr_curve_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    title: str = "Precision–Recall curve",
) -> go.Figure:
    points = pr_curve_points(y_true, y_score)
    ap = _safe_auc(
        np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), "pr"
    )
    fig = go.Figure()
    if points.empty:
        fig.add_annotation(
            text="PR curve requires both classes in the evaluation set.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"]),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=points["recall"],
                y=points["precision"],
                mode="lines",
                name=f"Model (AP = {ap:.3f})",
                line=dict(color=COLORS["amber"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.07)",
                hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return apply_theme(fig, title=title, height=350)


def threshold_curve_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    current_threshold: float = 0.5,
) -> go.Figure:
    table = threshold_sweep(y_true, y_score)
    fig = go.Figure()
    for col, color, label in [
        ("recall", COLORS["cyan"], "Recall"),
        ("precision", COLORS["green"], "Precision"),
        ("f1", COLORS["amber"], "F1"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=table["threshold"],
                y=table[col],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate="Threshold %{x:.2f}<br>"
                + label
                + " %{y:.3f}<extra></extra>",
            )
        )
    fig.add_vline(
        x=current_threshold,
        line=dict(color=COLORS["red"], width=1.5, dash="dash"),
        annotation_text=f"Selected {current_threshold:.2f}",
        annotation_font_color=COLORS["red"],
    )
    fig.update_xaxes(title="Decision threshold", range=[0.03, 0.97])
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return apply_theme(fig, title="Threshold trade-off", height=360)


def calibration_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> go.Figure:
    points = calibration_points(y_true, y_score)
    fig = go.Figure()
    if points.empty:
        fig.add_annotation(
            text="Not enough probability variation to build a calibration curve.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"]),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=points["mean_score"],
                y=points["observed_rate"],
                mode="lines+markers",
                name="Observed failure rate",
                line=dict(color=COLORS["cyan"], width=2),
                marker=dict(size=8, color=COLORS["cyan"]),
                customdata=points["count"],
                hovertemplate="Mean score %{x:.3f}<br>Observed %{y:.3f}"
                "<br>n = %{customdata}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color=COLORS["muted"], dash="dot", width=1),
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed failure rate", range=[0, 1])
    return apply_theme(fig, title="Calibration (reliability) curve", height=350)


def probability_distribution_figure(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> go.Figure:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    fig = go.Figure()
    for label, mask, color in [
        ("Running (actual)", y == 0, COLORS["green"]),
        ("Failed (actual)", y == 1, COLORS["red"]),
    ]:
        fig.add_trace(
            go.Histogram(
                x=score[mask],
                name=label,
                nbinsx=30,
                opacity=0.65,
                marker_color=color,
                hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
            )
        )
    fig.add_vline(
        x=threshold,
        line=dict(color=COLORS["cyan"], width=1.5, dash="dash"),
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_font_color=COLORS["cyan"],
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Model-estimated failure probability", range=[0, 1])
    fig.update_yaxes(title="Count")
    return apply_theme(fig, title="Prediction score distribution", height=340)
