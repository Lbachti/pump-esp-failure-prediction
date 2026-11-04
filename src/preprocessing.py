"""Leakage-safe preprocessing pipelines and validation splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DATE_COL,
    FEATURE_LABELS,
    FEATURE_ORDER,
    SPLIT_LABELS,
    TARGET,
    WELL_COL,
)
from .data_loader import get_feature_columns


class ModelingError(Exception):
    """Raised when a model or split cannot be constructed."""


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return ordered operational features, excluding leakage and identifiers."""
    return get_feature_columns(df)


def prepare_training_frame(
    df: pd.DataFrame,
    features: list[str] | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop unlabelled rows (and optionally exact duplicates) before splitting."""
    if df is None or df.empty:
        raise ModelingError("No data is loaded.")
    if TARGET not in df.columns:
        raise ModelingError("The binary target column is missing.")
    features = features or get_model_features(df)
    if len(features) < 2:
        raise ModelingError(
            "At least two operational features are required to train a model."
        )
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ModelingError("Features not found: " + ", ".join(missing))

    out = df.copy()
    n_available = len(out)
    n_target_missing = int(out[TARGET].isna().sum())
    out = out.dropna(subset=[TARGET]).copy()
    out[TARGET] = out[TARGET].astype(int)
    if out.empty:
        raise ModelingError("No rows with a valid target remain after filtering.")

    n_duplicates_removed = 0
    if drop_duplicates:
        before = len(out)
        subset = [c for c in out.columns if c != "__row_id"]
        out = out.drop_duplicates(subset=subset).reset_index(drop=True)
        n_duplicates_removed = before - len(out)
    else:
        out = out.reset_index(drop=True)

    if out[TARGET].nunique() < 2:
        raise ModelingError(
            "The target contains only one class after filtering; cannot train a classifier."
        )
    if out[TARGET].value_counts().min() < 2:
        raise ModelingError(
            "Each class needs at least two observations to create a stratified split."
        )

    info = {
        "n_available": int(n_available),
        "n_used": int(len(out)),
        "n_target_missing": n_target_missing,
        "n_duplicates_removed": int(n_duplicates_removed),
        "features": features,
        "failure_rate": float(out[TARGET].mean()),
    }
    return out, info


def build_estimator(
    model_key: str,
    params: dict[str, Any] | None = None,
    class_weight: bool = False,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
):
    params = dict(params or {})
    if model_key == "logistic":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 2000)),
            class_weight="balanced" if class_weight else None,
            random_state=random_state,
            solver="lbfgs",
        )
    if model_key == "random_forest":
        depth = params.get("max_depth", None)
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=None if depth in (None, 0, "None") else int(depth),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features="sqrt",
            class_weight="balanced" if class_weight else None,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_key == "xgboost":
        if not xgboost_available():
            raise ModelingError(
                "XGBoost is not installed. Install it with 'pip install xgboost' "
                "or select another model."
            )
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 250)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            gamma=float(params.get("gamma", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            scale_pos_weight=float(scale_pos_weight) if class_weight else 1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )
    raise ModelingError(f"Unknown model: {model_key}")


def build_pipeline(
    model_key: str,
    params: dict[str, Any] | None = None,
    class_weight: bool = False,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build a pipeline that imputes, optionally scales, then fits the model.

    Scaling is only used for Logistic Regression; tree models do not need it.
    The pipeline is fitted inside each training split to prevent leakage.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if model_key == "logistic":
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            build_estimator(
                model_key,
                params=params,
                class_weight=class_weight,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            ),
        )
    )
    return Pipeline(steps)


def _class_counts(y: pd.Series | np.ndarray) -> dict[int, int]:
    values = pd.Series(y).astype(int)
    return {int(k): int(v) for k, v in values.value_counts().to_dict().items()}


def _split_info(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategy: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    info: dict[str, Any] = {
        "strategy": strategy,
        "strategy_label": SPLIT_LABELS.get(strategy, strategy),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_failure_rate": float(train[TARGET].mean()) if len(train) else np.nan,
        "test_failure_rate": float(test[TARGET].mean()) if len(test) else np.nan,
        "train_class_counts": _class_counts(train[TARGET]) if len(train) else {},
        "test_class_counts": _class_counts(test[TARGET]) if len(test) else {},
        "train_wells": sorted(train[WELL_COL].astype(str).unique().tolist())
        if WELL_COL in train.columns
        else [],
        "test_wells": sorted(test[WELL_COL].astype(str).unique().tolist())
        if WELL_COL in test.columns
        else [],
    }
    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        info["train_date_min"] = (
            str(train[DATE_COL].min().date()) if train[DATE_COL].notna().any() else "—"
        )
        info["train_date_max"] = (
            str(train[DATE_COL].max().date()) if train[DATE_COL].notna().any() else "—"
        )
        info["test_date_min"] = (
            str(test[DATE_COL].min().date()) if test[DATE_COL].notna().any() else "—"
        )
        info["test_date_max"] = (
            str(test[DATE_COL].max().date()) if test[DATE_COL].notna().any() else "—"
        )
    if extra:
        info.update(extra)
    return info


def _ensure_two_classes(y: pd.Series, context: str) -> None:
    if y.nunique() < 2:
        raise ModelingError(
            f"The {context} split contains only one class. Choose another seed, "
            "test size, or validation strategy."
        )


def get_split_indices(
    df: pd.DataFrame,
    strategy: str = "random",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return positional train/test indices and a human-readable split summary."""
    if not 0.05 <= test_size <= 0.6:
        raise ModelingError("Test size must be between 5% and 60%.")
    if TARGET not in df.columns:
        raise ModelingError("The target column is missing.")
    frame = df.reset_index(drop=True)
    y = frame[TARGET].astype(int)

    if strategy == "random":
        _ensure_two_classes(y, "random")
        counts = y.value_counts()
        if counts.min() < 2:
            raise ModelingError("Each class needs at least two rows.")
        train_idx, test_idx = train_test_split(
            np.arange(len(frame)),
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return (
            np.asarray(train_idx),
            np.asarray(test_idx),
            _split_info(
                frame,
                np.asarray(train_idx),
                np.asarray(test_idx),
                "random",
                {"note": "Rows from the same well can appear in both sets."},
            ),
        )

    if strategy == "time":
        if DATE_COL not in frame.columns or frame[DATE_COL].isna().all():
            raise ModelingError("Time-based splitting requires a usable date column.")
        dates = np.sort(frame[DATE_COL].dropna().unique())
        if len(dates) < 5:
            raise ModelingError("At least five distinct dates are needed for a time split.")
        n_test_dates = max(1, int(round(len(dates) * test_size)))
        cutoff_position = len(dates) - n_test_dates
        cutoff_position = min(max(1, cutoff_position), len(dates) - 1)
        cutoff = dates[cutoff_position]
        train_mask = frame[DATE_COL] < cutoff
        test_mask = frame[DATE_COL] >= cutoff
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ModelingError("The time split produced an empty train or test set.")
        _ensure_two_classes(frame.iloc[train_idx][TARGET], "training")
        _ensure_two_classes(frame.iloc[test_idx][TARGET], "test")
        return (
            train_idx,
            test_idx,
            _split_info(
                frame,
                train_idx,
                test_idx,
                "time",
                {
                    "cutoff_date": str(pd.Timestamp(cutoff).date()),
                    "note": "Earlier dates train the model; later dates test temporal extrapolation. The same wells appear in both sets.",
                },
            ),
        )

    if strategy == "well":
        if WELL_COL not in frame.columns:
            raise ModelingError("Well-based splitting requires a well_id column.")
        groups = frame[WELL_COL].astype(str)
        n_wells = groups.nunique()
        if n_wells < 3:
            raise ModelingError(
                "Well-based splitting requires at least three distinct wells."
            )
        effective_test_size = min(max(test_size, 1 / n_wells), 0.6)
        last_error: Exception | None = None
        for seed in [random_state, *range(0, 60)]:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=effective_test_size, random_state=seed
            )
            try:
                train_idx, test_idx = next(splitter.split(frame, y, groups=groups))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            train_y = frame.iloc[train_idx][TARGET]
            test_y = frame.iloc[test_idx][TARGET]
            if train_y.nunique() < 2 or test_y.nunique() < 2:
                last_error = ModelingError(
                    "A candidate well split had only one class in train or test."
                )
                continue
            return (
                np.asarray(train_idx),
                np.asarray(test_idx),
                _split_info(
                    frame,
                    np.asarray(train_idx),
                    np.asarray(test_idx),
                    "well",
                    {
                        "seed_used": seed,
                        "note": "Entire wells are held out, testing cross-well generalisation. With few wells the estimate has high variance.",
                    },
                ),
            )
        raise ModelingError(
            "Could not construct a well-based split with both classes in train and test. "
            f"Try another seed. ({last_error})"
        )

    raise ModelingError(f"Unknown validation strategy: {strategy}")


def available_split_strategies(df: pd.DataFrame) -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    strategies.append(
        {
            "key": "random",
            "label": SPLIT_LABELS["random"],
            "available": True,
            "reason": "Row-level baseline; may overstate cross-well performance.",
        }
    )
    has_date = DATE_COL in df.columns and df[DATE_COL].notna().any()
    n_dates = int(df[DATE_COL].nunique()) if has_date else 0
    strategies.append(
        {
            "key": "time",
            "label": SPLIT_LABELS["time"],
            "available": bool(has_date and n_dates >= 5),
            "reason": (
                "Tests temporal extrapolation."
                if has_date and n_dates >= 5
                else "Requires a usable date column with at least five dates."
            ),
        }
    )
    n_wells = int(df[WELL_COL].nunique()) if WELL_COL in df.columns else 0
    strategies.append(
        {
            "key": "well",
            "label": SPLIT_LABELS["well"],
            "available": n_wells >= 3,
            "reason": (
                "Most conservative test of cross-well generalisation."
                if n_wells >= 3
                else "Requires at least three distinct wells."
            ),
        }
    )
    return strategies


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply all preprocessing steps, excluding the final estimator."""
    steps = pipeline.steps[:-1]
    if not steps:
        return np.asarray(X, dtype=float)
    transformer = Pipeline(steps)
    transformed = transformer.transform(X)
    return np.asarray(transformed, dtype=float)


def training_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        values = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "count": int(values.notna().sum()),
                "min": float(values.min()) if values.notna().any() else np.nan,
                "q1": float(values.quantile(0.25)) if values.notna().any() else np.nan,
                "median": float(values.median()) if values.notna().any() else np.nan,
                "mean": float(values.mean()) if values.notna().any() else np.nan,
                "q3": float(values.quantile(0.75)) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
                "std": float(values.std()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def training_ranges(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    stats = training_stats(df, features)
    return {
        row["feature"]: {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "median": float(row["median"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
        for _, row in stats.iterrows()
    }


def positive_class_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in classes:
        raise ModelingError("The fitted model has no failed class (1).")
    return proba[:, classes.index(1)]
