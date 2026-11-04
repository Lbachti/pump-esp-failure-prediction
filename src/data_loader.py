"""Dataset loading, schema normalisation, target encoding, and integrity audit."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import requests

from .config import (
    DATE_COL,
    FEATURE_ORDER,
    GITHUB_CSV_URL,
    LEAKAGE_COLUMNS,
    LOCAL_CSV_PATH,
    SOURCE_DATASET_NAME,
    TARGET,
    TARGET_RAW,
    WELL_COL,
)


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded or prepared."""


_COLUMN_ALIASES = {
    "well": WELL_COL,
    "wellid": WELL_COL,
    "well_name": WELL_COL,
    "timestamp": DATE_COL,
    "datetime": DATE_COL,
    "motor_current": "motor_amps",
    "motor_amp": "motor_amps",
    "amps": "motor_amps",
    "motor_volts": "motor_volt",
    "voltage": "motor_volt",
    "motor_voltage": "motor_volt",
    "motor_temp": "motor_temperature",
    "intake_temp": "intake_temperature",
    "vibration_g": "vibration",
    "production_rate": "production",
    "oil_production": "production",
    "status": TARGET_RAW,
    "esp_failure_status": TARGET_RAW,
    "downtime": "downtime_hours",
    "prod_loss": "production_loss",
}

_NUMERIC_COLUMNS = set(FEATURE_ORDER + LEAKAGE_COLUMNS)

_RUNNING_TOKENS = {
    "running",
    "run",
    "normal",
    "ok",
    "healthy",
    "on",
    "operating",
    "active",
    "up",
}
_FAILED_TOKENS = {
    "failed",
    "failure",
    "fail",
    "down",
    "off",
    "stopped",
    "stop",
    "fault",
    "faulted",
    "tripped",
    "shutdown",
}


def _clean_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return _COLUMN_ALIASES.get(text, text)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower_snake_case, de-aliased column names."""
    out = df.copy()
    out.columns = [_clean_column_name(c) for c in out.columns]
    # If aliases collapse two source columns, keep the first occurrence.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse a date column without assuming a single locale format."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    missing = parsed.isna() & series.notna()
    if missing.any():
        fallback = pd.to_datetime(series[missing], format="mixed", errors="coerce")
        parsed.loc[missing] = fallback
    return parsed


def _map_target_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) == 0.0:
            return 0.0
        if float(value) == 1.0:
            return 1.0
        return np.nan
    token = str(value).strip().lower()
    if token in _RUNNING_TOKENS or token in {"0", "0.0", "false"}:
        return 0.0
    if token in _FAILED_TOKENS or token in {"1", "1.0", "true"}:
        return 1.0
    return np.nan


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding)
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The CSV file is empty.") from exc
    raise DatasetError(f"Could not decode the CSV file: {last_error}")


def prepare_dataset(
    df: pd.DataFrame,
    source_label: str = "Uploaded CSV",
    source_url: str | None = None,
    load_mode: str = "upload",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise a raw dataframe and add the binary target.

    The raw ``esp_status`` column is preserved.  A separate ``esp_failure``
    column is created with the required encoding: running = 0, failed = 1.
    """
    if df is None or len(df) == 0:
        raise DatasetError("The dataset contains no rows.")

    out = normalise_columns(df)
    if TARGET_RAW not in out.columns:
        raise DatasetError(
            "The dataset must contain an 'esp_status' column with running/failed values."
        )

    # Preserve a readable status label even if the source uses numeric labels.
    out[TARGET_RAW] = out[TARGET_RAW].astype("object")

    invalid_numeric: dict[str, int] = {}
    for col in [c for c in _NUMERIC_COLUMNS if c in out.columns]:
        original_notna = out[col].notna().sum()
        converted = pd.to_numeric(out[col], errors="coerce")
        new_nan = int(original_notna - converted.notna().sum())
        if new_nan > 0:
            invalid_numeric[col] = new_nan
        out[col] = converted

    mapped = out[TARGET_RAW].map(_map_target_value)
    unmapped_mask = mapped.isna() & out[TARGET_RAW].notna()
    unmapped_values = (
        out.loc[unmapped_mask, TARGET_RAW].astype(str).value_counts().head(10).to_dict()
    )
    if mapped.notna().sum() == 0:
        raise DatasetError(
            "No valid target labels were found. Expected running/failed (or 0/1). "
            f"Values seen: {list(unmapped_values)}"
        )
    out[TARGET] = mapped.astype("Int64")

    if DATE_COL not in out.columns:
        out[DATE_COL] = pd.NaT
        date_available = False
    else:
        out[DATE_COL] = _parse_date_series(out[DATE_COL])
        date_available = bool(out[DATE_COL].notna().any())
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    if WELL_COL not in out.columns:
        out[WELL_COL] = "well_1"
        well_available = False
    else:
        out[WELL_COL] = (
            out[WELL_COL]
            .astype("object")
            .where(out[WELL_COL].notna(), "unknown_well")
            .astype(str)
            .str.strip()
        )
        well_available = True

    # Keep a stable row index for traceability.
    out = out.reset_index(drop=True)
    out["__row_id"] = np.arange(1, len(out) + 1)

    meta: dict[str, Any] = {
        "source_label": source_label,
        "source_url": source_url,
        "load_mode": load_mode,
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "has_date": date_available,
        "has_well": well_available,
        "n_wells": int(out[WELL_COL].nunique()),
        "invalid_numeric": invalid_numeric,
        "unmapped_target_values": unmapped_values,
        "unmapped_target_rows": int(unmapped_mask.sum()),
        "target_definition": (
            "esp_status is the operating status on the measurement day "
            "(current-state classification). No future-failure lead time is encoded."
        ),
    }
    return out, meta


def _load_local(path: Path = LOCAL_CSV_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise DatasetError(
            f"No cached dataset found at {path}. Check your internet connection "
            "and use the 'Load GitHub Dataset' button, or upload a CSV file."
        )
    payload = path.read_bytes()
    raw = _read_csv_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    df, meta = prepare_dataset(
        raw,
        source_label=f"{SOURCE_DATASET_NAME} (local cache)",
        source_url=GITHUB_CSV_URL,
        load_mode="local_cache",
    )
    meta["sha256"] = sha
    meta["local_path"] = str(path)
    return df, meta


def load_github_dataset(timeout: int = 25) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download the public dataset, falling back to the bundled cached copy.

    A clear error is raised only when both the network and the local cache fail.
    """
    try:
        response = requests.get(
            GITHUB_CSV_URL,
            timeout=timeout,
            headers={"User-Agent": "ESP-PredictAI/1.0 educational prototype"},
        )
        response.raise_for_status()
        payload = response.content
        if len(payload) < 1000:
            raise DatasetError("The downloaded CSV is unexpectedly small.")
        raw = _read_csv_bytes(payload)
        df, meta = prepare_dataset(
            raw,
            source_label=SOURCE_DATASET_NAME,
            source_url=GITHUB_CSV_URL,
            load_mode="github",
        )
        meta["sha256"] = hashlib.sha256(payload).hexdigest()
        return df, meta
    except Exception as network_error:  # noqa: BLE001 - fallback is intentional
        try:
            df, meta = _load_local()
            meta["fallback_reason"] = f"GitHub download failed: {network_error}"
            return df, meta
        except Exception as local_error:  # noqa: BLE001
            raise DatasetError(
                "Could not load the GitHub dataset and no valid local cache is "
                f"available.\n\nNetwork error: {network_error}\n"
                f"Local cache error: {local_error}\n\n"
                "Use the CSV upload feature in Data Explorer to continue."
            ) from local_error


def load_uploaded_csv(uploaded_file: BinaryIO) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a user-uploaded CSV file from a Streamlit uploader."""
    try:
        payload = uploaded_file.getvalue()
    except AttributeError:
        payload = uploaded_file.read()
    if not payload:
        raise DatasetError("The uploaded file is empty.")
    raw = _read_csv_bytes(payload)
    df, meta = prepare_dataset(
        raw,
        source_label=getattr(uploaded_file, "name", "Uploaded CSV"),
        load_mode="upload",
    )
    meta["sha256"] = hashlib.sha256(payload).hexdigest()
    return df, meta


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Operational features present in the dataset with at least one value."""
    return [
        c
        for c in FEATURE_ORDER
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint used to detect dataset/model mismatch."""
    cols = [
        c
        for c in [TARGET, *FEATURE_ORDER, WELL_COL, DATE_COL]
        if c in df.columns
    ]
    if not cols:
        return ""
    payload = df[cols].copy()
    for col in payload.columns:
        if pd.api.types.is_datetime64_any_dtype(payload[col]):
            payload[col] = payload[col].astype(str)
    text = payload.to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return value[:length] if value else "—"


def _is_complete_panel(df: pd.DataFrame) -> bool:
    if not {WELL_COL, DATE_COL}.issubset(df.columns):
        return False
    valid = df.dropna(subset=[DATE_COL, TARGET])
    if valid[DATE_COL].nunique() < 2 or valid[WELL_COL].nunique() < 2:
        return False
    sets = valid.groupby(WELL_COL)[DATE_COL].agg(lambda s: frozenset(s.dropna()))
    if len(sets) < 2:
        return False
    first = sets.iloc[0]
    if any(s != first for s in sets.iloc[1:]):
        return False
    # Ensure the union is a gap-free daily sequence.
    dates = pd.Series(sorted(first))
    diffs = dates.diff().dropna().dt.days
    return bool(len(diffs) == 0 or (diffs == 1).all())


def _persistence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not {WELL_COL, DATE_COL, TARGET}.issubset(df.columns):
        return pd.DataFrame()
    valid = df.dropna(subset=[TARGET, DATE_COL]).sort_values([WELL_COL, DATE_COL])
    for well, group in valid.groupby(WELL_COL):
        series = group[TARGET].astype(int).to_numpy()
        if len(series) < 3:
            continue
        prevalence = float(series.mean())
        persistence = float(np.mean(series[1:] == series[:-1]))
        independent = prevalence**2 + (1 - prevalence) ** 2
        rows.append(
            {
                "well_id": well,
                "n": int(len(series)),
                "failure_rate": prevalence,
                "status_persistence": persistence,
                "independent_baseline": independent,
                "difference": persistence - independent,
            }
        )
    return pd.DataFrame(rows)


def audit_dataset(
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a data-integrity and provenance audit from the loaded data.

    All values are derived from the dataframe; none are hardcoded.
    """
    meta = dict(meta or {})
    if TARGET not in df.columns:
        raise DatasetError("The prepared dataset is missing the binary target column.")

    valid_target = df[TARGET].dropna().astype(int)
    n_rows = len(df)
    n_wells = int(df[WELL_COL].nunique()) if WELL_COL in df.columns else 0
    n_unique_dates = int(df[DATE_COL].nunique()) if DATE_COL in df.columns else 0
    date_min = date_max = None
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        date_min = str(df[DATE_COL].min().date())
        date_max = str(df[DATE_COL].max().date())

    public_columns = [c for c in df.columns if not str(c).startswith("__")]
    missing_by_col = pd.DataFrame(
        {
            "column": public_columns,
            "dtype": [str(df[c].dtype) for c in public_columns],
            "missing": [int(df[c].isna().sum()) for c in public_columns],
            "missing_pct": [float(df[c].isna().mean() * 100) for c in public_columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in public_columns],
        }
    )
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(
        df.drop(columns=["__row_id"], errors="ignore").duplicated().sum()
    )
    duplicate_well_dates = 0
    if {WELL_COL, DATE_COL}.issubset(df.columns):
        duplicate_well_dates = int(
            df.dropna(subset=[DATE_COL]).duplicated([WELL_COL, DATE_COL]).sum()
        )

    target_counts = {
        "running": int((valid_target == 0).sum()),
        "failed": int((valid_target == 1).sum()),
        "missing_or_invalid": int(df[TARGET].isna().sum()),
    }
    failure_rate = (
        float((valid_target == 1).mean()) if len(valid_target) else float("nan")
    )

    leakage: dict[str, dict[str, Any]] = {}
    for col in LEAKAGE_COLUMNS:
        if col not in df.columns:
            continue
        valid = df.dropna(subset=[TARGET, col])
        if valid.empty:
            continue
        indicator = (pd.to_numeric(valid[col], errors="coerce").fillna(0) > 0).astype(int)
        target = valid[TARGET].astype(int)
        agreement = float((indicator == target).mean())
        corr = float(pd.to_numeric(valid[col], errors="coerce").corr(target))
        leakage[col] = {
            "deterministic_separation": bool(agreement > 0.999),
            "agreement_with_failure": agreement,
            "correlation_with_failure": corr,
            "nonzero_running": int(((indicator == 1) & (target == 0)).sum()),
            "zero_failed": int(((indicator == 0) & (target == 1)).sum()),
        }

    persistence = _persistence_table(df)
    near_independent = False
    if not persistence.empty:
        near_independent = bool(
            (persistence["difference"].abs() < 0.03).mean() >= 0.8
        )

    complete_panel = _is_complete_panel(df)
    numeric_features = get_feature_columns(df)
    integer_like = 0
    narrow_ranges = 0
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        if bool(np.allclose(values, np.round(values))):
            integer_like += 1
        if values.nunique() < 60:
            narrow_ranges += 1

    reasons: list[str] = []
    if missing_cells == 0 and duplicate_rows == 0 and complete_panel:
        reasons.append(
            "Complete, gap-free panel with no missing values and no duplicate rows."
        )
    if any(v["deterministic_separation"] for v in leakage.values()):
        cols = [c for c, v in leakage.items() if v["deterministic_separation"]]
        reasons.append(
            "Outcome columns perfectly separate the classes: "
            + ", ".join(cols)
            + " are zero for every running record and non-zero for every failed record."
        )
    if near_independent:
        reasons.append(
            "Failure labels show almost no day-to-day persistence; the transition "
            "pattern is close to independent daily coin flips rather than run-to-failure "
            "equipment behaviour."
        )
    if not np.isnan(failure_rate) and failure_rate > 0.25:
        reasons.append(
            f"A {failure_rate:.1%} daily failure prevalence is unusually high for a "
            "producing ESP fleet and is more consistent with a simulated label."
        )
    if integer_like >= 5 and narrow_ranges >= 5:
        reasons.append(
            "Most sensor channels are integer-quantized and occupy narrow ranges, "
            "which is unusual for raw field sensor data."
        )
    if n_wells <= 6:
        reasons.append(
            f"Only {n_wells} wells are available; well-level validation has high variance."
        )

    suspicion_level = "High" if len(reasons) >= 4 else "Moderate" if len(reasons) >= 2 else "Low"
    likely_simulated = len(reasons) >= 3

    feature_ranges = []
    for col in numeric_features:
        values = pd.to_numeric(df[col], errors="coerce")
        feature_ranges.append(
            {
                "feature": col,
                "n": int(values.notna().sum()),
                "min": float(values.min()) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().any() else np.nan,
            }
        )

    well_failure = pd.DataFrame()
    if WELL_COL in df.columns:
        grouped = (
            df.dropna(subset=[TARGET])
            .groupby(WELL_COL)[TARGET]
            .agg(observations="size", failed="sum")
            .reset_index()
        )
        if not grouped.empty:
            grouped["running"] = grouped["observations"] - grouped["failed"]
            grouped["failure_rate"] = grouped["failed"] / grouped["observations"]
            well_failure = grouped

    warnings: list[str] = []
    critical: list[str] = []
    if meta.get("unmapped_target_rows", 0):
        warnings.append(
            f"{meta['unmapped_target_rows']} rows have unrecognised target values and "
            "will be excluded from model training."
        )
    if missing_cells:
        warnings.append(f"{missing_cells} missing cells detected.")
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} exact duplicate rows detected.")
    if duplicate_well_dates:
        warnings.append(
            f"{duplicate_well_dates} duplicate well/date pairs detected; check panel integrity."
        )
    if meta.get("invalid_numeric"):
        warnings.append(
            "Non-numeric values were coerced to missing in: "
            + ", ".join(meta["invalid_numeric"].keys())
        )
    if len(valid_target) < 50:
        critical.append("Fewer than 50 labelled observations; model results are unstable.")
    if n_wells < 3:
        warnings.append(
            "Fewer than three wells: well-based validation is not meaningful."
        )
    if not meta.get("has_date", False):
        warnings.append("No usable date column: temporal validation is unavailable.")
    if df[TARGET].nunique(dropna=True) < 2:
        critical.append("The target contains only one class; classification is impossible.")
    if failure_rate > 0.25:
        warnings.append(
            "Class prevalence is high; accuracy alone will be misleading. Review recall, "
            "precision, and the confusion matrix."
        )

    return {
        "n_rows": int(n_rows),
        "n_cols": int(len(public_columns)),
        "n_wells": n_wells,
        "wells": sorted(df[WELL_COL].dropna().astype(str).unique().tolist())
        if WELL_COL in df.columns
        else [],
        "n_unique_dates": n_unique_dates,
        "date_min": date_min,
        "date_max": date_max,
        "has_date": bool(meta.get("has_date", False)),
        "complete_panel": complete_panel,
        "missing_cells": missing_cells,
        "missing_pct": float(missing_cells / max(1, n_rows * df.shape[1]) * 100),
        "missing_by_col": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "duplicate_well_dates": duplicate_well_dates,
        "target_counts": target_counts,
        "failure_rate": failure_rate,
        "leakage": leakage,
        "persistence": persistence,
        "near_independent_labels": near_independent,
        "feature_ranges": pd.DataFrame(feature_ranges),
        "well_failure": well_failure,
        "suspicion_reasons": reasons,
        "suspicion_level": suspicion_level,
        "likely_simulated": likely_simulated,
        "provenance_status": "Unverified public repository upload",
        "warnings": warnings,
        "critical": critical,
        "target_definition": meta.get("target_definition"),
        "source_label": meta.get("source_label"),
        "source_url": meta.get("source_url"),
        "sha256": meta.get("sha256"),
        "load_mode": meta.get("load_mode"),
        "loaded_at": meta.get("loaded_at"),
        "integer_like_features": int(integer_like),
        "narrow_range_features": int(narrow_ranges),
    }


def target_note() -> str:
    return (
        "The target is the operating status recorded on the same measurement day. "
        "It is a current-state failure classification label, not a future-failure "
        "event or remaining-useful-life target."
    )


def human_source(meta: dict[str, Any] | None) -> str:
    meta = meta or {}
    mode = meta.get("load_mode")
    if mode == "github":
        return "GitHub (live download)"
    if mode == "local_cache":
        return "Local cached copy of GitHub CSV"
    if mode == "upload":
        return "User-uploaded CSV"
    return meta.get("source_label", "Unknown source")
