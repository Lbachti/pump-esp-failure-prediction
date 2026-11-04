"""ESP PredictAI — Intelligent Artificial Lift Monitoring & Failure Prediction.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ESP PredictAI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import APP_NAME, APP_SUBTITLE  # noqa: E402
from src.data_loader import DatasetError, human_source, load_github_dataset  # noqa: E402
from src.pages import (  # noqa: E402
    about,
    cleaning,
    data_explorer,
    eda_page,
    explainability_view,
    insights,
    overview,
    prediction_view,
    training,
)
from src.state import (  # noqa: E402
    PAGES,
    active_df,
    dataset_brief,
    goto,
    has_data,
    init_state,
    set_dataset,
)
from src.ui import (  # noqa: E402
    inject_theme,
    sidebar_brand,
    sidebar_dataset_card,
    sidebar_footer,
    sidebar_model_card,
)

ROUTES = {
    "Overview Dashboard": overview.render,
    "Data Explorer": data_explorer.render,
    "Data Cleaning": cleaning.render,
    "Exploratory Data Analysis": eda_page.render,
    "ML Model Training": training.render,
    "Failure Prediction": prediction_view.render,
    "Model Explainability": explainability_view.render,
    "Engineering Insights": insights.render,
    "Project Information": about.render,
}


def _auto_load() -> None:
    """Load the public dataset once per session, with a local cached fallback."""
    if has_data() or st.session_state.get("auto_load_attempted"):
        return
    st.session_state["auto_load_attempted"] = True
    try:
        with st.spinner("Loading the public ESP dataset..."):
            df, meta = load_github_dataset()
        notice = (
            f"Loaded {len(df):,} rows from {human_source(meta)}. "
            "The app performed a data-integrity audit before enabling modelling."
        )
        if meta.get("fallback_reason"):
            notice += " The live GitHub download failed, so the bundled cached copy was used."
        set_dataset(df, meta, notice=notice)
    except DatasetError as exc:
        st.session_state["load_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state["load_error"] = f"Unexpected dataset loading error: {exc}"


def _sidebar() -> str:
    with st.sidebar:
        sidebar_brand()
        st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

        # Programmatic navigation is deferred through pending_nav so it does not
        # conflict with the radio widget's own state.
        if st.session_state.get("pending_nav"):
            st.session_state["nav_radio"] = st.session_state["pending_nav"]
            st.session_state["pending_nav"] = None

        def _nav_changed() -> None:
            st.session_state["page"] = st.session_state["nav_radio"]

        st.radio(
            "Navigation",
            PAGES,
            key="nav_radio",
            label_visibility="collapsed",
            on_change=_nav_changed,
        )

        st.markdown('<div class="sidebar-label">Status</div>', unsafe_allow_html=True)
        sidebar_dataset_card(dataset_brief())
        sidebar_model_card(st.session_state.get("trained_bundle"))

        if has_data():
            st.markdown('<div class="sidebar-label">Quick action</div>', unsafe_allow_html=True)
            if st.button("Open model training", width="stretch", key="side_train"):
                goto("ML Model Training")
                st.rerun()

        st.markdown('<div class="sidebar-label">Scope</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-note">Current-state failure classification. '
            "No future-failure or remaining-useful-life claim. Public dataset provenance "
            "is unverified.</div>",
            unsafe_allow_html=True,
        )
        sidebar_footer()

    page = st.session_state.get("page", PAGES[0])
    if page not in ROUTES:
        page = PAGES[0]
        st.session_state["page"] = page
    return page


def main() -> None:
    init_state()
    inject_theme()
    _auto_load()
    page = _sidebar()

    if not has_data() and st.session_state.get("load_error"):
        st.warning(
            "The public dataset could not be loaded automatically. Open Data Explorer to "
            "retry or upload a CSV."
        )

    renderer = ROUTES.get(page, overview.render)
    try:
        renderer()
    except Exception as exc:  # noqa: BLE001 - show a useful page-level error
        st.error(
            "This page encountered an error. The rest of the app remains available; "
            "check the technical details below."
        )
        with st.expander("Technical details", expanded=False):
            st.exception(exc)


if __name__ == "__main__":
    main()
