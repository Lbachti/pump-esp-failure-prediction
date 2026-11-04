"""Shared Plotly styling for the industrial dark theme."""

from __future__ import annotations

import plotly.graph_objects as go

from .config import COLORS

FONT_FAMILY = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
COLOR_SEQUENCE = [
    COLORS["cyan"],
    COLORS["blue"],
    COLORS["green"],
    COLORS["amber"],
    COLORS["purple"],
    COLORS["red"],
    "#60a5fa",
    "#34d399",
    "#fbbf24",
]

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def apply_theme(
    fig: go.Figure,
    title: str | None = None,
    height: int = 360,
    showlegend: bool = True,
    margin: dict | None = None,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
        title=dict(
            text=title or "",
            font=dict(size=15, color=COLORS["text"]),
            x=0.01,
            xanchor="left",
        ),
        height=height,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["muted"]),
        ),
        margin=margin or dict(l=10, r=10, t=48, b=10),
        hoverlabel=dict(
            bgcolor=COLORS["panel_2"],
            bordercolor=COLORS["border"],
            font=dict(family=FONT_FAMILY, color=COLORS["text"]),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(45,125,210,0.12)",
        zerolinecolor="rgba(45,125,210,0.25)",
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"], size=11),
        title_font=dict(color=COLORS["muted"], size=12),
    )
    fig.update_yaxes(
        gridcolor="rgba(45,125,210,0.12)",
        zerolinecolor="rgba(45,125,210,0.25)",
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["muted"], size=11),
        title_font=dict(color=COLORS["muted"], size=12),
    )
    return fig


def empty_figure(message: str = "No data to display", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=COLORS["muted"], size=13),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, height=height, showlegend=False)
