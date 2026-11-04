"""Shared visual language for the ESP PredictAI dashboard."""

from __future__ import annotations

import html
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from .config import COLORS, DISCLAIMER
from .plotting import PLOTLY_CONFIG

_TONES = {"cyan", "blue", "green", "amber", "red", "muted"}


def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --bg: #050d18;
  --panel: #0b1a2d;
  --panel-2: #0e2138;
  --border: #17324f;
  --text: #e6eef8;
  --muted: #8aa4bf;
  --blue: #2d7dd2;
  --cyan: #22d3ee;
  --green: #22c55e;
  --amber: #f59e0b;
  --red: #ef4444;
}

html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 12% -10%, rgba(45,125,210,0.13), transparent 60%),
    radial-gradient(900px 420px at 95% 0%, rgba(34,211,238,0.07), transparent 55%),
    #050d18;
  color: var(--text);
}

header[data-testid="stHeader"] {
  background: rgba(5, 13, 24, 0.35);
  backdrop-filter: blur(8px);
}

#MainMenu, footer { visibility: hidden; }

.block-container {
  padding-top: 1.7rem;
  padding-bottom: 4rem;
  max-width: 1500px;
}

h1, h2, h3, h4 {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  letter-spacing: -0.02em;
  color: var(--text);
}

p, li, label, span { color: var(--text); }

a { color: var(--cyan); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #071321 0%, #06101d 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }

.brand {
  display: flex; align-items: center; gap: 12px;
  padding: 0.4rem 0.2rem 1.1rem 0.2rem;
}
.brand-mark {
  width: 42px; height: 42px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(145deg, #0e2138, #12395c);
  border: 1px solid rgba(34,211,238,0.45);
  color: var(--cyan); font-weight: 700; font-size: 12px;
  letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace;
  box-shadow: 0 0 24px rgba(34,211,238,0.12);
}
.brand-title { font-size: 1.02rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.brand-sub { font-size: 0.68rem; color: var(--muted); margin-top: 3px; letter-spacing: 0.02em; }

.sidebar-label {
  font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.13em;
  color: var(--muted); margin: 0.9rem 0 0.35rem 0.2rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.1rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding: 0.48rem 0.65rem;
  border-radius: 7px;
  border: 1px solid transparent;
  transition: background 120ms ease, border-color 120ms ease;
  cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(34,211,238,0.055);
  border-color: rgba(34,211,238,0.12);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: rgba(34,211,238,0.10);
  border-color: rgba(34,211,238,0.28);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
  color: #c8f7ff; font-weight: 600;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
  display: none;
}
[data-testid="stSidebar"] div[role="radiogroup"] p {
  font-size: 0.86rem; color: #b9cbe0;
}

.sidebar-card {
  background: rgba(14,33,56,0.62);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.75rem 0.8rem;
  margin: 0.55rem 0.15rem;
}
.sidebar-row {
  display: flex; justify-content: space-between; gap: 8px;
  font-size: 0.72rem; color: var(--muted); margin: 0.22rem 0;
}
.sidebar-row b { color: var(--text); font-weight: 600; }
.sidebar-mini-title {
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.11em;
  color: var(--muted); margin-bottom: 0.45rem;
}
.sidebar-note {
  font-size: 0.66rem; line-height: 1.45; color: #6f8aa6;
  padding: 0.4rem 0.3rem 0 0.3rem;
}

/* ---------- Hero / section ---------- */
.page-hero {
  border-bottom: 1px solid var(--border);
  padding: 0.15rem 0 1.15rem 0;
  margin-bottom: 1.35rem;
}
.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.64rem; letter-spacing: 0.17em; text-transform: uppercase;
  color: var(--cyan); margin-bottom: 0.5rem;
}
.page-title {
  font-size: clamp(1.65rem, 2.4vw, 2.3rem);
  font-weight: 700; line-height: 1.12; margin: 0; color: var(--text);
}
.page-sub {
  color: var(--muted); font-size: 0.91rem; line-height: 1.55;
  max-width: 940px; margin-top: 0.55rem;
}
.badge-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }

.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 0.22rem 0.55rem; border-radius: 999px;
  font-size: 0.66rem; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid var(--border); color: var(--muted);
  background: rgba(14,33,56,0.65); white-space: nowrap;
}
.badge.cyan { color: #a5f3fc; border-color: rgba(34,211,238,0.35); background: rgba(34,211,238,0.08); }
.badge.blue { color: #bfdbfe; border-color: rgba(45,125,210,0.45); background: rgba(45,125,210,0.12); }
.badge.green { color: #bbf7d0; border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.08); }
.badge.amber { color: #fde68a; border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.08); }
.badge.red { color: #fecaca; border-color: rgba(239,68,68,0.38); background: rgba(239,68,68,0.09); }

.section-head { margin: 1.55rem 0 0.75rem 0; }
.section-kicker {
  font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 0.25rem;
}
.section-title { font-size: 1.08rem; font-weight: 650; margin: 0; color: var(--text); }
.section-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.45; }

/* ---------- KPI cards ---------- */
.kpi-card {
  position: relative; overflow: hidden;
  background: linear-gradient(155deg, rgba(14,33,56,0.95), rgba(8,21,37,0.92));
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 0.9rem 1rem 0.85rem 1rem;
  min-height: 112px;
  box-shadow: 0 8px 26px rgba(0,0,0,0.14);
}
.kpi-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent, var(--blue)); opacity: 0.9;
}
.kpi-card.cyan { --accent: var(--cyan); }
.kpi-card.blue { --accent: var(--blue); }
.kpi-card.green { --accent: var(--green); }
.kpi-card.amber { --accent: var(--amber); }
.kpi-card.red { --accent: var(--red); }
.kpi-card.muted { --accent: var(--muted); }
.kpi-top { display: flex; justify-content: space-between; align-items: center; }
.kpi-label {
  font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.11em;
  color: var(--muted); font-weight: 600;
}
.kpi-icon { font-size: 0.8rem; color: var(--accent, var(--muted)); opacity: 0.9; }
.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.55rem; font-weight: 600; color: var(--text);
  margin-top: 0.5rem; line-height: 1.05; letter-spacing: -0.03em;
}
.kpi-sub { font-size: 0.69rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.3; }

/* ---------- Callouts ---------- */
.callout {
  border: 1px solid var(--border);
  border-left: 3px solid var(--blue);
  border-radius: 8px;
  background: rgba(14,33,56,0.62);
  padding: 0.8rem 0.95rem;
  margin: 0.55rem 0 0.85rem 0;
}
.callout.info { border-left-color: var(--blue); }
.callout.success { border-left-color: var(--green); }
.callout.warn { border-left-color: var(--amber); }
.callout.danger { border-left-color: var(--red); }
.callout.cyan { border-left-color: var(--cyan); }
.callout-title {
  font-size: 0.77rem; font-weight: 650; color: var(--text);
  margin-bottom: 0.3rem; letter-spacing: 0.01em;
}
.callout-body { font-size: 0.79rem; line-height: 1.55; color: #b9cbe0; }
.callout-body b { color: var(--text); }
.callout-body code {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
  background: rgba(34,211,238,0.08); color: #a5f3fc;
  padding: 0.05rem 0.3rem; border-radius: 4px;
}

/* ---------- Empty state ---------- */
.empty-state {
  border: 1px dashed rgba(45,125,210,0.35);
  border-radius: 10px;
  background: rgba(11,26,45,0.45);
  padding: 1.4rem 1.2rem;
  text-align: center;
  margin: 0.6rem 0 0.9rem 0;
}
.empty-icon { font-size: 1.3rem; color: var(--cyan); margin-bottom: 0.35rem; }
.empty-title { font-size: 0.95rem; font-weight: 650; color: var(--text); }
.empty-body { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; line-height: 1.5; }

/* ---------- Risk card ---------- */
.risk-card {
  border: 1px solid var(--border);
  border-left: 5px solid var(--risk, var(--green));
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(14,33,56,0.97), rgba(8,21,37,0.9));
  padding: 1.05rem 1.15rem;
  margin: 0.5rem 0 1rem 0;
}
.risk-top { display: flex; align-items: center; gap: 9px; }
.risk-dot {
  width: 11px; height: 11px; border-radius: 50%;
  background: var(--risk, var(--green));
  box-shadow: 0 0 16px var(--risk, var(--green));
}
.risk-level {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; letter-spacing: 0.13em; text-transform: uppercase;
  color: var(--risk, var(--green)); font-weight: 600;
}
.risk-prob {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.25rem; font-weight: 600; line-height: 1.1;
  color: var(--text); margin-top: 0.5rem; letter-spacing: -0.04em;
}
.risk-caption { color: var(--muted); font-size: 0.8rem; margin-top: 0.35rem; line-height: 1.5; }

/* ---------- Schematic ---------- */
.schematic-card {
  background: linear-gradient(150deg, rgba(14,33,56,0.9), rgba(7,19,33,0.92));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1rem 0.7rem 1rem;
}
.schematic-title {
  font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--muted); margin-bottom: 0.2rem;
}
.schematic-card svg { width: 100%; height: auto; display: block; }
.schematic-legend {
  font-size: 0.68rem; color: #6f8aa6; margin-top: 0.35rem;
}

/* ---------- Small utilities ---------- */
.mono { font-family: 'JetBrains Mono', monospace; }
.muted { color: var(--muted); }
.small { font-size: 0.74rem; }
.divider { height: 1px; background: var(--border); margin: 1.1rem 0; }
.param-card {
  border: 1px solid var(--border); border-radius: 10px;
  background: rgba(14,33,56,0.55); padding: 0.9rem 1rem; height: 100%;
}
.param-title { font-size: 0.92rem; font-weight: 650; color: var(--text); }
.param-text { font-size: 0.78rem; color: #b9cbe0; line-height: 1.55; margin-top: 0.35rem; }
.param-label {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--muted); margin-top: 0.55rem;
}
.status-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.7rem; font-weight: 600; padding: 0.2rem 0.55rem;
  border-radius: 999px; border: 1px solid var(--border);
}
.status-pill.green { color: #bbf7d0; border-color: rgba(34,197,94,0.4); background: rgba(34,197,94,0.08); }
.status-pill.amber { color: #fde68a; border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.08); }
.status-pill.red { color: #fecaca; border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.08); }
.status-pill.cyan { color: #a5f3fc; border-color: rgba(34,211,238,0.4); background: rgba(34,211,238,0.08); }

/* ---------- Widgets ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 7px;
  border: 1px solid #24517c;
  background: #0e2138;
  color: #dbeafe;
  font-weight: 600;
  font-size: 0.8rem;
  padding: 0.42rem 0.9rem;
  transition: all 120ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--cyan);
  color: #ecfeff;
  background: #12395c;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #1d6fc4, #155a9e);
  border-color: #2d7dd2;
  color: #f0f9ff;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #2d7dd2, #1d6fc4);
  border-color: var(--cyan);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"],
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
  background: #0a1829 !important;
  border-color: #1d3d5e !important;
  color: var(--text) !important;
  border-radius: 7px !important;
}
div[data-baseweb="select"] span { color: var(--text) !important; }
div[data-baseweb="popover"] { background: #0b1a2d; }

.stSlider [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--cyan) !important;
  border-color: var(--cyan) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.2rem; border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 6px 6px 0 0;
  color: var(--muted); font-size: 0.79rem; font-weight: 600;
  padding: 0.5rem 0.8rem;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  background: rgba(34,211,238,0.06) !important;
  border-bottom: 2px solid var(--cyan);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
[data-testid="stDataFrame"] * { font-size: 0.75rem; }

div[data-testid="stExpander"] {
  background: rgba(11,26,45,0.45);
  border: 1px solid var(--border);
  border-radius: 8px;
}
div[data-testid="stExpander"] summary { color: var(--muted); font-size: 0.8rem; }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(11,26,45,0.55);
  border: 1px dashed #24517c;
  border-radius: 8px;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted); }

.stAlert { border-radius: 8px; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stMarkdownContainer"] hr { border-color: var(--border); }

.stProgress > div > div > div { background-color: var(--cyan); }

@media (max-width: 900px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .kpi-value { font-size: 1.3rem; }
  .page-title { font-size: 1.55rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _badge(text: str, tone: str = "muted") -> str:
    tone = tone if tone in _TONES else "muted"
    return f'<span class="badge {tone}">{html.escape(str(text))}</span>'


def page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "ESP PREDICTAI",
    badges: Iterable[tuple[str, str]] | None = None,
) -> None:
    badge_html = ""
    if badges:
        badge_html = '<div class="badge-row">' + "".join(
            _badge(text, tone) for text, tone in badges
        ) + "</div>"
    st.markdown(
        f"""
<div class="page-hero">
  <div class="eyebrow">{html.escape(eyebrow)}</div>
  <h1 class="page-title">{html.escape(title)}</h1>
  <div class="page-sub">{html.escape(subtitle)}</div>
  {badge_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    kick = f'<div class="section-kicker">{html.escape(kicker)}</div>' if kicker else ""
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-head">{kick}<div class="section-title">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    tone: str = "blue",
    icon: str = "",
) -> str:
    tone = tone if tone in _TONES else "blue"
    return f"""
<div class="kpi-card {tone}">
  <div class="kpi-top">
    <span class="kpi-label">{html.escape(str(label))}</span>
    <span class="kpi-icon">{html.escape(str(icon))}</span>
  </div>
  <div class="kpi-value">{html.escape(str(value))}</div>
  <div class="kpi-sub">{html.escape(str(sub))}</div>
</div>
"""


def render_kpis(cards: list[dict[str, Any]], columns: int = 4) -> None:
    if not cards:
        return
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        cols = st.columns(len(row), gap="small")
        for col, card in zip(cols, row):
            with col:
                st.markdown(
                    kpi_card(
                        card.get("label", ""),
                        card.get("value", "—"),
                        card.get("sub", ""),
                        card.get("tone", "blue"),
                        card.get("icon", ""),
                    ),
                    unsafe_allow_html=True,
                )


def callout(title: str, body: str, tone: str = "info", icon: str = "") -> None:
    tone = tone if tone in {"info", "success", "warn", "danger", "cyan"} else "info"
    icon_html = f"{html.escape(icon)} " if icon else ""
    body_html = html.escape(str(body)).replace("\n", "<br>")
    st.markdown(
        f"""
<div class="callout {tone}">
  <div class="callout-title">{icon_html}{html.escape(title)}</div>
  <div class="callout-body">{body_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, icon: str = "◈") -> None:
    st.markdown(
        f"""
<div class="empty-state">
  <div class="empty-icon">{html.escape(icon)}</div>
  <div class="empty-title">{html.escape(title)}</div>
  <div class="empty-body">{html.escape(body)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, tone: str = "cyan") -> str:
    return f'<span class="status-pill {tone}">{html.escape(label)}</span>'


def sidebar_brand() -> None:
    st.markdown(
        """
<div class="brand">
  <div class="brand-mark">ESP</div>
  <div>
    <div class="brand-title">PredictAI</div>
    <div class="brand-sub">Artificial Lift Intelligence</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_dataset_card(brief: dict[str, Any]) -> None:
    if not brief.get("loaded"):
        st.markdown(
            '<div class="sidebar-card"><div class="sidebar-mini-title">Dataset</div>'
            '<div class="sidebar-row"><span>Status</span><b>Not loaded</b></div></div>',
            unsafe_allow_html=True,
        )
        return
    failure = brief.get("failure_rate")
    failure_text = f"{failure:.1%}" if failure == failure else "—"
    cleaned = "Cleaned" if brief.get("cleaned") else "Original"
    st.markdown(
        f"""
<div class="sidebar-card">
  <div class="sidebar-mini-title">Dataset</div>
  <div class="sidebar-row"><span>Rows</span><b>{brief.get('rows', 0):,}</b></div>
  <div class="sidebar-row"><span>Wells</span><b>{brief.get('wells', 0):,}</b></div>
  <div class="sidebar-row"><span>Failure rate</span><b>{failure_text}</b></div>
  <div class="sidebar-row"><span>Version</span><b>{cleaned}</b></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_model_card(bundle: dict[str, Any] | None) -> None:
    if not bundle:
        st.markdown(
            '<div class="sidebar-card"><div class="sidebar-mini-title">Model</div>'
            '<div class="sidebar-row"><span>Status</span><b>Not trained</b></div>'
            '<div class="sidebar-note">Train or load a model to enable predictions.</div></div>',
            unsafe_allow_html=True,
        )
        return
    metrics = bundle.get("metrics", {})
    recall = metrics.get("recall", float("nan"))
    auc = metrics.get("roc_auc", float("nan"))
    recall_text = f"{recall:.2f}" if recall == recall else "—"
    auc_text = f"{auc:.2f}" if auc == auc else "—"
    st.markdown(
        f"""
<div class="sidebar-card">
  <div class="sidebar-mini-title">Model</div>
  <div class="sidebar-row"><span>Type</span><b>{html.escape(bundle.get('model_label', '—'))}</b></div>
  <div class="sidebar-row"><span>Recall</span><b>{recall_text}</b></div>
  <div class="sidebar-row"><span>ROC-AUC</span><b>{auc_text}</b></div>
  <div class="sidebar-row"><span>Threshold</span><b>{bundle.get('threshold', 0.5):.2f}</b></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    st.markdown(
        f'<div class="sidebar-note">{html.escape(DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


def format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def format_pct(value: Any, digits: int = 1) -> str:
    try:
        value = float(value)
        if value != value:
            return "—"
        return f"{value * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def format_float(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
        if value != value:
            return "—"
        return f"{value:,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def data_badges(meta: dict[str, Any], audit: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    badges = [(meta.get("source_label", "Dataset"), "blue")]
    if audit:
        badges.append((f"{audit['n_rows']:,} rows", "cyan"))
        badges.append((f"{audit['n_wells']} wells", "cyan"))
        if audit.get("likely_simulated"):
            badges.append(("Provenance: unverified", "amber"))
    return badges


def plotly_chart(fig, key: str | None = None, height: int | None = None) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key=key,
    )


def download_csv_button(
    df: pd.DataFrame,
    label: str,
    file_name: str,
    key: str,
) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
