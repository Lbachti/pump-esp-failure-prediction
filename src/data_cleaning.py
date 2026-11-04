"""Interactive data-cleaning operations.

Every function is pure: it returns a new dataframe and never mutates the
caller's data.  Target, identifier, and date columns are protected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import FEATURE_ORDER, PROTECTED_COLUMNS, TARGET, TARGET_RAW
from .data_loader import get_feature_columns


class CleaningError(Exception):
    """Raised for invalid cleaning requests."""


def cleanable_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Operational numeric features that may be cleaned."""
    available = get_feature_columns(df)
    return [c for c in FEATURE_ORDER if c in available and c not in PROTECTED_COLUMNS]


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if not columns:
        raise CleaningError("Select at least one column.")
    protected = [c for c in columns if c in PROTECTED_COLUMNS]
    if protected:
        raise CleaningError(
            "These columns are protected and cannot be transformed: "
            + ", ".join(protected)
        )
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CleaningError("Columns not found: " + ", ".join(missing))
    return columns


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count:
            rows.append(
                {
                    "column": col,
                    "missing": count,
                    "missing_pct": count / max(1, len(df)) * 100,
                    "dtype": str(df[col].dtype),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["column", "missing", "missing_pct", "dtype"]
        )
    return pd.DataFrame(rows).sort_values("missing", ascending=False)


def duplicate_count(df: pd.DataFrame) -> int:
    return int(
        df.drop(columns=["__row_id"], errors="ignore").duplicated().sum()
    )


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    subset = [c for c in df.columns if c != "__row_id"]
    out = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return out, {
        "operation": "Remove duplicate rows",
        "rows_before": before,
        "rows_after": len(out),
        "rows_removed": before - len(out),
        "details": "Exact duplicate rows removed.",
    }


def impute_missing(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Handle missing values without touching the target or identifiers."""
    columns = _validate_columns(df, columns)
    out = df.copy()
    before_missing = int(out[columns].isna().sum().sum())
    filled: dict[str, Any] = {}

    if method in {"Mean", "Median"}:
        for col in columns:
            values = pd.to_numeric(out[col], errors="coerce")
            if values.notna().sum() == 0:
                filled[col] = "all missing — unchanged"
                continue
            fill_value = values.mean() if method == "Mean" else values.median()
            missing_count = int(values.isna().sum())
            out[col] = values.fillna(fill_value)
            filled[col] = {"value": float(fill_value), "filled": missing_count}
    elif method == "Mode":
        for col in columns:
            mode = out[col].mode(dropna=True)
            if mode.empty:
                filled[col] = "all missing — unchanged"
                continue
            fill_value = mode.iloc[0]
            missing_count = int(out[col].isna().sum())
            out[col] = out[col].fillna(fill_value)
            filled[col] = {"value": fill_value, "filled": missing_count}
    elif method == "Drop rows":
        before = len(out)
        out = out.dropna(subset=columns, how="any").reset_index(drop=True)
        filled = {"rows_removed": before - len(out)}
    else:
        raise CleaningError(f"Unsupported missing-value method: {method}")

    after_missing = int(out[columns].isna().sum().sum())
    return out, {
        "operation": f"Handle missing values ({method})",
        "columns": columns,
        "rows_before": len(df),
        "rows_after": len(out),
        "missing_before": before_missing,
        "missing_after": after_missing,
        "details": filled,
    }


def outlier_bounds(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "IQR",
    factor: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    columns = _validate_columns(df, columns)
    if factor <= 0:
        raise CleaningError("The IQR multiplier must be positive.")
    if z_threshold <= 0:
        raise CleaningError("The Z-score threshold must be positive.")

    rows = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "column": col,
                    "method": method,
                    "lower": np.nan,
                    "upper": np.nan,
                    "outliers": 0,
                    "outlier_pct": 0.0,
                }
            )
            continue
        if method == "IQR":
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
        elif method == "Z-score":
            mean, std = values.mean(), values.std()
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std
        else:
            raise CleaningError(f"Unsupported outlier method: {method}")
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": col,
                "method": method,
                "lower": float(lower),
                "upper": float(upper),
                "outliers": count,
                "outlier_pct": count / max(1, len(values)) * 100,
            }
        )
    return pd.DataFrame(rows)


def outlier_mask(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "IQR",
    factor: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.Series:
    bounds = outlier_bounds(df, columns, method, factor, z_threshold)
    mask = pd.Series(False, index=df.index)
    for _, row in bounds.iterrows():
        if pd.isna(row["lower"]) or pd.isna(row["upper"]):
            continue
        values = pd.to_numeric(df[row["column"]], errors="coerce")
        mask |= (values < row["lower"]) | (values > row["upper"])
    return mask


def treat_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "IQR",
    treatment: str = "Cap values",
    factor: float = 1.5,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = _validate_columns(df, columns)
    bounds = outlier_bounds(df, columns, method, factor, z_threshold)
    out = df.copy()
    if treatment == "Remove rows":
        mask = outlier_mask(df, columns, method, factor, z_threshold)
        out = out.loc[~mask].reset_index(drop=True)
        details = {"rows_removed": int(mask.sum())}
    elif treatment == "Cap values":
        clipped = 0
        details = {}
        for _, row in bounds.iterrows():
            col = row["column"]
            if pd.isna(row["lower"]) or pd.isna(row["upper"]):
                continue
            values = pd.to_numeric(out[col], errors="coerce")
            below = int((values < row["lower"]).sum())
            above = int((values > row["upper"]).sum())
            out[col] = values.clip(lower=row["lower"], upper=row["upper"])
            clipped += below + above
            details[col] = {"clipped": below + above, "lower": row["lower"], "upper": row["upper"]}
        details["total_values_clipped"] = clipped
    else:
        raise CleaningError(f"Unsupported outlier treatment: {treatment}")

    return out, {
        "operation": f"Outlier treatment ({method} → {treatment})",
        "columns": columns,
        "rows_before": len(df),
        "rows_after": len(out),
        "bounds": bounds.to_dict(orient="records"),
        "details": details,
    }


def scale_numeric(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale selected numeric features; the target is never transformed.

    Missing values remain missing; the scaler is fitted on observed values only.
    Model pipelines perform their own leakage-safe scaling.
    """
    columns = _validate_columns(df, columns)
    if method not in {"MinMaxScaler", "StandardScaler"}:
        raise CleaningError(f"Unsupported scaling method: {method}")
    out = df.copy()
    scalers: dict[str, Any] = {}
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce")
        mask = values.notna()
        if not mask.any():
            continue
        scaler = MinMaxScaler() if method == "MinMaxScaler" else StandardScaler()
        arr = values[mask].to_numpy(dtype=float).reshape(-1, 1)
        scaler.fit(arr)
        out.loc[mask, col] = scaler.transform(arr).ravel()
        scalers[col] = scaler
    return out, {
        "operation": f"Scale features ({method})",
        "columns": columns,
        "rows_before": len(df),
        "rows_after": len(out),
        "details": f"{len(scalers)} columns transformed; target untouched.",
    }


def before_after_stats(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in before.columns or col not in after.columns:
            continue
        b = pd.to_numeric(before[col], errors="coerce")
        a = pd.to_numeric(after[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "before_mean": b.mean(),
                "after_mean": a.mean(),
                "before_std": b.std(),
                "after_std": a.std(),
                "before_min": b.min(),
                "after_min": a.min(),
                "before_max": b.max(),
                "after_max": a.max(),
                "before_missing": int(b.isna().sum()),
                "after_missing": int(a.isna().sum()),
                "rows_before": len(before),
                "rows_after": len(after),
            }
        )
    return pd.DataFrame(rows)


def make_history_entry(
    operation: str,
    details: Any,
    rows_before: int,
    rows_after: int,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "operation": operation,
        "columns": ", ".join(columns) if columns else "—",
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "details": details if isinstance(details, str) else str(details),
    }


def target_integrity_check(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Confirm a cleaning operation did not change the target column."""
    if TARGET not in before.columns or TARGET not in after.columns:
        return True
    if len(before) != len(after):
        # Row removal is allowed, but the retained target counts cannot increase.
        before_counts = before[TARGET].value_counts(dropna=False)
        after_counts = after[TARGET].value_counts(dropna=False)
        return bool(
            all(
                int(after_counts.get(value, 0)) <= int(before_counts.get(value, 0))
                for value in after_counts.index
            )
        )
    return before[TARGET].reset_index(drop=True).equals(
        after[TARGET].reset_index(drop=True)
    )


def protected_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in [TARGET, TARGET_RAW, "well_id", "date"] if c in df.columns]
