"""Exploratory data analysis helpers and interactive Plotly figures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from .config import COLORS, DATE_COL, FEATURE_LABELS, TARGET, WELL_COL
from .plotting import COLOR_SEQUENCE, apply_theme, empty_figure

STATUS_LABELS = {0: "Running", 1: "Failed"}
STATUS_COLORS = {"Running": COLORS["green"], "Failed": COLORS["red"]}


def with_status_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_status"] = out[TARGET].map(STATUS_LABELS)
    return out


def status_counts(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=[TARGET])
    counts = valid[TARGET].astype(int).value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame(
        {
            "status": ["Running", "Failed"],
            "count": [int(counts.loc[0]), int(counts.loc[1])],
        }
    )


def status_donut(df: pd.DataFrame) -> go.Figure:
    counts = status_counts(df)
    fig = go.Figure(
        go.Pie(
            labels=counts["status"],
            values=counts["count"],
            hole=0.68,
            marker=dict(
                colors=[COLORS["green"], COLORS["red"]],
                line=dict(color=COLORS["panel"], width=3),
            ),
            textinfo="percent",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="%{label}<br>%{value:,} observations<br>%{percent}<extra></extra>",
            sort=False,
        )
    )
    total = int(counts["count"].sum())
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:11px;color:{COLORS['muted']}'>OBSERVATIONS</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color=COLORS["text"]),
    )
    return apply_theme(fig, title="Operating-status distribution", height=330)


def histogram_by_status(
    df: pd.DataFrame,
    feature: str,
    bins: int = 40,
) -> go.Figure:
    data = with_status_label(df.dropna(subset=[feature, TARGET]))
    fig = go.Figure()
    for label, color in [("Running", COLORS["green"]), ("Failed", COLORS["red"])]:
        subset = data[data["_status"] == label]
        fig.add_trace(
            go.Histogram(
                x=subset[feature],
                name=label,
                nbinsx=bins,
                marker_color=color,
                opacity=0.62,
                hovertemplate="%{x}<br>Count %{y}<extra></extra>",
            )
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=f"{FEATURE_LABELS.get(feature, feature)}")
    fig.update_yaxes(title="Count")
    return apply_theme(
        fig,
        title=f"{FEATURE_LABELS.get(feature, feature)} distribution by status",
        height=380,
    )


def box_by_status(df: pd.DataFrame, feature: str) -> go.Figure:
    data = with_status_label(df.dropna(subset=[feature, TARGET]))
    fig = go.Figure()
    for label, color in [("Running", COLORS["green"]), ("Failed", COLORS["red"])]:
        subset = data[data["_status"] == label]
        fig.add_trace(
            go.Box(
                y=subset[feature],
                name=label,
                marker_color=color,
                boxpoints="outliers",
                line=dict(width=1.5),
                fillcolor=color,
                opacity=0.55,
                hovertemplate="%{y}<extra></extra>",
            )
        )
    fig.update_yaxes(title=FEATURE_LABELS.get(feature, feature))
    return apply_theme(
        fig,
        title=f"{FEATURE_LABELS.get(feature, feature)} by operating status",
        height=380,
    )


def violin_by_status(df: pd.DataFrame, feature: str) -> go.Figure:
    data = with_status_label(df.dropna(subset=[feature, TARGET]))
    fig = go.Figure()
    for label, color in [("Running", COLORS["green"]), ("Failed", COLORS["red"])]:
        subset = data[data["_status"] == label]
        fig.add_trace(
            go.Violin(
                y=subset[feature],
                name=label,
                line_color=color,
                fillcolor=color,
                opacity=0.45,
                box_visible=True,
                meanline_visible=True,
                points=False,
            )
        )
    fig.update_yaxes(title=FEATURE_LABELS.get(feature, feature))
    return apply_theme(
        fig,
        title=f"{FEATURE_LABELS.get(feature, feature)} distribution by status",
        height=380,
    )


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str = "Operating status",
    max_points: int = 3000,
    random_state: int = 42,
) -> go.Figure:
    data = df.dropna(subset=[x, y]).copy()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=random_state)
    if color_by == "Well":
        color_col = WELL_COL
        color_map = None
        color_seq = COLOR_SEQUENCE
    else:
        data["_status"] = data[TARGET].map(STATUS_LABELS)
        color_col = "_status"
        color_map = STATUS_COLORS
        color_seq = None
    hover = [c for c in [WELL_COL, DATE_COL, TARGET] if c in data.columns]
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color_col,
        color_discrete_map=color_map,
        color_discrete_sequence=color_seq,
        hover_data=hover,
        opacity=0.62,
        render_mode="webgl" if len(data) > 1500 else "svg",
        labels={
            x: FEATURE_LABELS.get(x, x),
            y: FEATURE_LABELS.get(y, y),
            "_status": "Status",
        },
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return apply_theme(
        fig,
        title=f"{FEATURE_LABELS.get(x, x)} vs {FEATURE_LABELS.get(y, y)}",
        height=440,
    )


def correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
    include_target: bool = True,
    method: str = "pearson",
) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    frame = df[cols + ([TARGET] if include_target else [])].copy()
    if include_target:
        frame = frame.rename(columns={TARGET: "Failure (1=failed)"})
    corr = frame.corr(numeric_only=True, method=method.lower())
    return corr


def correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    include_target: bool = True,
    method: str = "pearson",
) -> go.Figure:
    corr = correlation_matrix(df, features, include_target, method)
    if corr.empty:
        return empty_figure("No numeric columns available for correlation.")
    labels = [
        FEATURE_LABELS.get(c, "Failure" if c.startswith("Failure") else c)
        for c in corr.columns
    ]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#2563eb"],
                [0.5, "#0b1a2d"],
                [1.0, "#ef4444"],
            ],
            colorbar=dict(title="r", thickness=10),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_xaxes(tickangle=-35)
    return apply_theme(
        fig,
        title=f"{method.title()} correlation matrix",
        height=620,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )


def pair_plot(
    df: pd.DataFrame,
    features: list[str],
    color_by: str = "Operating status",
    max_points: int = 700,
    random_state: int = 42,
) -> go.Figure:
    features = [f for f in features if f in df.columns][:5]
    if len(features) < 2:
        return empty_figure("Select at least two features for a pair plot.")
    data = df.dropna(subset=[TARGET]).copy()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=random_state)
    if color_by == "Well":
        color_col = WELL_COL
        color_map = None
        color_seq = COLOR_SEQUENCE
    else:
        data["_status"] = data[TARGET].map(STATUS_LABELS)
        color_col = "_status"
        color_map = STATUS_COLORS
        color_seq = None
    fig = px.scatter_matrix(
        data,
        dimensions=features,
        color=color_col,
        color_discrete_map=color_map,
        color_discrete_sequence=color_seq,
        labels={f: FEATURE_LABELS.get(f, f) for f in features},
        opacity=0.5,
        height=720,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return apply_theme(fig, title="Pairwise parameter relationships", height=760)


def failure_rate_by_bins(
    df: pd.DataFrame,
    feature: str,
    n_bins: int = 8,
) -> tuple[go.Figure, pd.DataFrame]:
    data = df.dropna(subset=[feature, TARGET]).copy()
    if len(data) < 20:
        return empty_figure("Not enough data to bin."), pd.DataFrame()
    try:
        data["_bin"] = pd.qcut(
            data[feature], q=min(n_bins, max(2, data[feature].nunique())), duplicates="drop"
        )
    except ValueError:
        return empty_figure("Could not create quantile bins."), pd.DataFrame()
    grouped = (
        data.groupby("_bin", observed=True)
        .agg(
            n=(TARGET, "size"),
            failed=(TARGET, "sum"),
            mean_feature=(feature, "mean"),
        )
        .reset_index()
    )
    grouped["failure_rate"] = grouped["failed"] / grouped["n"]
    labels = [
        f"{interval.left:.3g}–{interval.right:.3g}" for interval in grouped["_bin"]
    ]
    overall = float(data[TARGET].mean())
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=grouped["failure_rate"],
            marker_color=COLORS["blue"],
            text=[f"{r:.0%}<br>n={n:,}" for r, n in zip(grouped["failure_rate"], grouped["n"])],
            textposition="outside",
            textfont=dict(size=9, color=COLORS["muted"]),
            hovertemplate="Range %{x}<br>Failure rate %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=overall,
        line=dict(color=COLORS["amber"], dash="dash", width=1.3),
        annotation_text=f"Overall {overall:.1%}",
        annotation_font_color=COLORS["amber"],
    )
    fig.update_yaxes(title="Failure rate", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title=f"{FEATURE_LABELS.get(feature, feature)} quantile bin")
    fig = apply_theme(
        fig,
        title=f"Failure rate by {FEATURE_LABELS.get(feature, feature)}",
        height=390,
        showlegend=False,
    )
    return fig, grouped


def well_failure_comparison(df: pd.DataFrame) -> go.Figure:
    if WELL_COL not in df.columns:
        return empty_figure("No well identifier available.")
    data = df.dropna(subset=[TARGET])
    grouped = (
        data.groupby(WELL_COL)[TARGET]
        .agg(observations="size", failed="sum")
        .reset_index()
    )
    grouped["running"] = grouped["observations"] - grouped["failed"]
    grouped["failure_rate"] = grouped["failed"] / grouped["observations"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grouped[WELL_COL],
            y=grouped["running"],
            name="Running",
            marker_color=COLORS["green"],
            opacity=0.75,
            hovertemplate="%{x}<br>Running %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=grouped[WELL_COL],
            y=grouped["failed"],
            name="Failed",
            marker_color=COLORS["red"],
            opacity=0.8,
            hovertemplate="%{x}<br>Failed %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped[WELL_COL],
            y=grouped["failure_rate"],
            name="Failure rate",
            mode="lines+markers",
            line=dict(color=COLORS["amber"], width=2),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Failure rate %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Observations", secondary_y=False)
    fig.update_yaxes(
        title="Failure rate", tickformat=".0%", range=[0, 1], secondary_y=True
    )
    return apply_theme(fig, title="Well-level failure comparison", height=390)


def time_failure_rate(
    df: pd.DataFrame,
    freq: str = "ME",
) -> go.Figure:
    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        return empty_figure("Date information is unavailable.")
    data = df.dropna(subset=[DATE_COL, TARGET]).set_index(DATE_COL).sort_index()
    if data.empty:
        return empty_figure("No dated observations.")
    grouped = data.resample(freq)[TARGET].agg(["mean", "size", "sum"]).dropna()
    grouped = grouped[grouped["size"] > 0]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grouped.index,
            y=grouped["sum"],
            name="Failed observations",
            marker_color=COLORS["red"],
            opacity=0.45,
            hovertemplate="%{x|%b %Y}<br>Failures %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped.index,
            y=grouped["mean"],
            name="Failure rate",
            mode="lines+markers",
            line=dict(color=COLORS["cyan"], width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>Failure rate %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title="Failed observations", secondary_y=False)
    fig.update_yaxes(title="Failure rate", tickformat=".0%", range=[0, 1], secondary_y=True)
    return apply_theme(fig, title="Failure occurrence over time", height=390)


def time_failure_by_well(df: pd.DataFrame, freq: str = "ME") -> go.Figure:
    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        return empty_figure("Date information is unavailable.")
    data = df.dropna(subset=[DATE_COL, TARGET]).copy()
    data["_period"] = data[DATE_COL].dt.to_period(freq.replace("ME", "M")).dt.to_timestamp()
    grouped = (
        data.groupby(["_period", WELL_COL])[TARGET].mean().reset_index()
    )
    fig = px.line(
        grouped,
        x="_period",
        y=TARGET,
        color=WELL_COL,
        markers=True,
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={"_period": "Period", TARGET: "Failure rate", WELL_COL: "Well"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return apply_theme(fig, title="Failure rate over time by well", height=390)


def time_feature(
    df: pd.DataFrame,
    feature: str,
    freq: str = "ME",
    by_well: bool = True,
) -> go.Figure:
    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        return empty_figure("Date information is unavailable.")
    data = df.dropna(subset=[DATE_COL, feature]).copy()
    data["_period"] = data[DATE_COL].dt.to_period(freq.replace("ME", "M")).dt.to_timestamp()
    if by_well and WELL_COL in data.columns:
        grouped = data.groupby(["_period", WELL_COL])[feature].mean().reset_index()
        fig = px.line(
            grouped,
            x="_period",
            y=feature,
            color=WELL_COL,
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"_period": "Period", feature: FEATURE_LABELS.get(feature, feature)},
        )
    else:
        grouped = data.groupby("_period")[feature].mean().reset_index()
        fig = px.line(
            grouped,
            x="_period",
            y=feature,
            labels={"_period": "Period", feature: FEATURE_LABELS.get(feature, feature)},
        )
    return apply_theme(
        fig,
        title=f"{FEATURE_LABELS.get(feature, feature)} trend",
        height=390,
    )


def standardized_mean_difference(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        failed = pd.to_numeric(
            df.loc[df[TARGET] == 1, feature], errors="coerce"
        ).dropna()
        running = pd.to_numeric(
            df.loc[df[TARGET] == 0, feature], errors="coerce"
        ).dropna()
        if len(failed) < 2 or len(running) < 2:
            continue
        pooled = np.sqrt((failed.var(ddof=1) + running.var(ddof=1)) / 2)
        effect = (failed.mean() - running.mean()) / pooled if pooled else 0.0
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "effect_size": float(effect),
                "abs_effect": abs(float(effect)),
                "mean_failed": float(failed.mean()),
                "mean_running": float(running.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_effect", ascending=False).reset_index(drop=True)


def effect_size_figure(df: pd.DataFrame, features: list[str]) -> go.Figure:
    table = standardized_mean_difference(df, features)
    if table.empty:
        return empty_figure("Not enough data to compute effect sizes.")
    data = table.sort_values("effect_size")
    colors = [COLORS["red"] if v > 0 else COLORS["green"] for v in data["effect_size"]]
    fig = go.Figure(
        go.Bar(
            x=data["effect_size"],
            y=data["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in data["effect_size"]],
            textposition="outside",
            textfont=dict(size=10, color=COLORS["muted"]),
            hovertemplate="%{y}<br>Standardised mean difference %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=COLORS["muted"], width=1))
    fig.update_xaxes(title="Cohen's d (failed − running)")
    fig.update_yaxes(title="")
    return apply_theme(
        fig,
        title="Standardised difference between failed and running observations",
        height=max(320, 70 + 30 * len(data)),
        showlegend=False,
    )


def parameter_summary_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        failed = pd.to_numeric(df.loc[df[TARGET] == 1, feature], errors="coerce")
        running = pd.to_numeric(df.loc[df[TARGET] == 0, feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "mean_running": running.mean(),
                "mean_failed": failed.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "std": values.std(),
            }
        )
    return pd.DataFrame(rows)


def univariate_signal_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Univariate ROC-AUC and Mann–Whitney p-value for each feature."""
    data = df.dropna(subset=[TARGET])
    rows = []
    for feature in features:
        if feature not in data.columns:
            continue
        valid = data.dropna(subset=[feature])
        failed = pd.to_numeric(valid.loc[valid[TARGET] == 1, feature], errors="coerce")
        running = pd.to_numeric(valid.loc[valid[TARGET] == 0, feature], errors="coerce")
        if len(failed) < 2 or len(running) < 2:
            continue
        try:
            auc = float(roc_auc_score(valid[TARGET].astype(int), valid[feature]))
            p_value = float(
                mannwhitneyu(failed, running, alternative="two-sided").pvalue
            )
        except Exception:  # noqa: BLE001
            continue
        direction = (
            "Higher values associated with failure"
            if auc > 0.5
            else "Lower values associated with failure"
        )
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "auc": auc,
                "signal": max(auc, 1 - auc),
                "direction": direction,
                "p_value": p_value,
                "mean_running": float(running.mean()),
                "mean_failed": float(failed.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("signal", ascending=False).reset_index(drop=True)


def signal_figure(df: pd.DataFrame, features: list[str]) -> go.Figure:
    table = univariate_signal_table(df, features)
    if table.empty:
        return empty_figure("Not enough data for univariate signal analysis.")
    data = table.sort_values("signal")
    fig = go.Figure(
        go.Bar(
            x=data["signal"],
            y=data["label"],
            orientation="h",
            marker_color=COLORS["cyan"],
            text=[f"{v:.3f}" for v in data["signal"]],
            textposition="outside",
            textfont=dict(size=10, color=COLORS["muted"]),
            customdata=np.column_stack([data["auc"], data["p_value"]]),
            hovertemplate="%{y}<br>Signal (max AUC, 1−AUC): %{x:.3f}"
            "<br>Raw AUC: %{customdata[0]:.3f}"
            "<br>p-value: %{customdata[1]:.3g}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Univariate signal strength", range=[0.4, 1.0])
    fig.update_yaxes(title="")
    return apply_theme(
        fig,
        title="Single-feature association with failure",
        height=max(320, 70 + 30 * len(data)),
        showlegend=False,
    )


def correlation_with_target_table(
    df: pd.DataFrame, features: list[str], method: str = "pearson"
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        valid = df.dropna(subset=[feature, TARGET])
        if valid.empty:
            continue
        corr = valid[feature].corr(valid[TARGET], method=method.lower())
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "correlation": float(corr),
                "abs_correlation": abs(float(corr)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
