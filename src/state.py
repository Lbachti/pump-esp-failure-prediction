"""Streamlit session-state helpers.

All cross-page state lives here so pages do not invent their own keys.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from .config import DATE_COL, TARGET, WELL_COL
from .data_cleaning import make_history_entry
from .data_loader import dataset_hash

PAGES = [
    "Overview Dashboard",
    "Data Explorer",
    "Data Cleaning",
    "Exploratory Data Analysis",
    "ML Model Training",
    "Failure Prediction",
    "Model Explainability",
    "Engineering Insights",
    "Project Information",
]


def init_state() -> None:
    defaults: dict[str, Any] = {
        "raw_df": None,
        "raw_meta": {},
        "clean_df": None,
        "clean_history": [],
        "clean_snapshots": [],
        "data_notice": None,
        "load_error": None,
        "trained_bundle": None,
        "training_result": None,
        "comparison_result": None,
        "strategy_comparison": None,
        "group_cv_result": None,
        "model_notice": None,
        "training_notice": None,
        "last_prediction": None,
        "prediction_history": [],
        "outlier_report": None,
        "outlier_report_hash": None,
        "filters_nonce": 0,
        "page": PAGES[0],
        "nav_radio": PAGES[0],
        "pending_nav": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_data() -> bool:
    return st.session_state.get("raw_df") is not None


def active_df() -> pd.DataFrame | None:
    df = st.session_state.get("clean_df")
    if df is None:
        df = st.session_state.get("raw_df")
    return df


def active_meta() -> dict[str, Any]:
    return dict(st.session_state.get("raw_meta") or {})


def is_cleaned() -> bool:
    return st.session_state.get("clean_df") is not None


def set_dataset(df: pd.DataFrame, meta: dict[str, Any], notice: str | None = None) -> None:
    had_model = st.session_state.get("trained_bundle") is not None
    st.session_state["raw_df"] = df.copy()
    st.session_state["raw_meta"] = dict(meta or {})
    st.session_state["clean_df"] = None
    st.session_state["clean_history"] = []
    st.session_state["clean_snapshots"] = []
    st.session_state["data_notice"] = notice
    st.session_state["load_error"] = None
    st.session_state["outlier_report"] = None
    st.session_state["outlier_report_hash"] = None
    st.session_state["prediction_history"] = []
    st.session_state["last_prediction"] = None
    clear_model(
        "The dataset changed; retrain or load a compatible model."
        if had_model
        else None
    )


def clear_model(reason: str | None = None) -> None:
    st.session_state["trained_bundle"] = None
    st.session_state["training_result"] = None
    st.session_state["comparison_result"] = None
    st.session_state["strategy_comparison"] = None
    st.session_state["group_cv_result"] = None
    st.session_state["last_prediction"] = None
    st.session_state["training_notice"] = None
    st.session_state["model_notice"] = reason


def apply_cleaning(
    new_df: pd.DataFrame,
    operation: str,
    details: Any,
    columns: list[str] | None = None,
) -> None:
    current = active_df()
    if current is None:
        raise RuntimeError("No dataset is loaded.")
    st.session_state["clean_snapshots"].append(current.copy())
    if len(st.session_state["clean_snapshots"]) > 20:
        st.session_state["clean_snapshots"] = st.session_state["clean_snapshots"][-20:]
    st.session_state["clean_df"] = new_df.copy()
    entry = make_history_entry(
        operation=operation,
        details=details,
        rows_before=len(current),
        rows_after=len(new_df),
        columns=columns,
    )
    st.session_state["clean_history"].append(entry)
    st.session_state["outlier_report"] = None
    st.session_state["outlier_report_hash"] = None
    clear_model(
        "Cleaning changed the working dataset; the previous model was invalidated."
    )


def undo_cleaning() -> dict[str, Any] | None:
    snapshots = st.session_state.get("clean_snapshots") or []
    history = st.session_state.get("clean_history") or []
    if not snapshots:
        return None
    previous = snapshots.pop()
    st.session_state["clean_snapshots"] = snapshots
    st.session_state["clean_df"] = previous if history[:-1] else previous
    # If this was the only operation, the snapshot is the raw data; keeping it as
    # clean_df is equivalent but the UI can still show it as cleaned.  Clear it.
    if not history[:-1]:
        st.session_state["clean_df"] = None
    removed = history.pop() if history else None
    st.session_state["clean_history"] = history
    st.session_state["outlier_report"] = None
    st.session_state["outlier_report_hash"] = None
    clear_model("Cleaning was undone; retrain the model on the current data.")
    return removed


def reset_cleaning() -> None:
    st.session_state["clean_df"] = None
    st.session_state["clean_history"] = []
    st.session_state["clean_snapshots"] = []
    st.session_state["outlier_report"] = None
    st.session_state["outlier_report_hash"] = None
    clear_model("Cleaning was reset; retrain the model on the original data.")


def set_trained_bundle(bundle: dict[str, Any], result: dict[str, Any] | None = None) -> None:
    st.session_state["trained_bundle"] = bundle
    st.session_state["training_result"] = result
    st.session_state["model_notice"] = None


def current_hash() -> str:
    df = active_df()
    return dataset_hash(df) if df is not None else ""


def model_is_stale() -> bool:
    bundle = st.session_state.get("trained_bundle")
    if not bundle:
        return False
    stored = bundle.get("dataset_hash", "")
    return bool(stored) and stored != current_hash()


def log_prediction(result: dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model": result.get("model_label", "—"),
        "probability": result["probability"],
        "risk": result["risk_level"],
        "classification": result["predicted_label"],
        "threshold": result["decision_threshold"],
    }
    for feature, value in result.get("inputs", {}).items():
        entry[feature] = value
    history = st.session_state.get("prediction_history") or []
    history.append(entry)
    st.session_state["prediction_history"] = history[-50:]
    st.session_state["last_prediction"] = result


def dataset_brief() -> dict[str, Any]:
    df = active_df()
    if df is None:
        return {"loaded": False}
    valid = df.dropna(subset=[TARGET]) if TARGET in df.columns else df
    return {
        "loaded": True,
        "rows": int(len(df)),
        "wells": int(df[WELL_COL].nunique()) if WELL_COL in df.columns else 0,
        "failure_rate": float(valid[TARGET].mean()) if len(valid) and TARGET in df else float("nan"),
        "cleaned": is_cleaned(),
        "source": (st.session_state.get("raw_meta") or {}).get("source_label", "—"),
        "has_date": bool(
            DATE_COL in df.columns and pd.to_datetime(df[DATE_COL], errors="coerce").notna().any()
        ),
    }


def goto(page: str) -> None:
    if page in PAGES:
        st.session_state["page"] = page
        st.session_state["pending_nav"] = page


def clear_outlier_report() -> None:
    st.session_state["outlier_report"] = None
    st.session_state["outlier_report_hash"] = None
