"""Training orchestration for ESP failure classifiers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .config import (
    APP_VERSION,
    DATE_COL,
    FEATURE_LABELS,
    MODEL_LABELS,
    TARGET,
    WELL_COL,
)
from .data_loader import dataset_hash, get_feature_columns
from .model_evaluation import compute_metrics
from .preprocessing import (
    ModelingError,
    available_split_strategies,
    build_pipeline,
    get_split_indices,
    positive_class_probability,
    prepare_training_frame,
    training_ranges,
    training_stats,
    xgboost_available,
)


def _warnings_for_result(
    split_info: dict[str, Any],
    metrics: dict[str, Any],
    n_wells: int,
    n_features: int,
    info: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    strategy = split_info["strategy"]
    if strategy == "random" and n_wells > 1:
        warnings.append(
            "Random row-level split: observations from the same well appear in both "
            "train and test. This can overstate cross-well generalisation."
        )
    if strategy == "time":
        warnings.append(
            "Time-based split tests temporal extrapolation, but the same wells are "
            "present in train and test; it does not test a new well."
        )
    if strategy == "well":
        warnings.append(
            "Well-based split is the most conservative test, but only "
            f"{n_wells} wells are available, so the estimate has high variance."
        )
    if metrics.get("roc_auc") is not None and not np.isnan(metrics["roc_auc"]):
        if metrics["roc_auc"] < 0.65:
            warnings.append(
                "ROC-AUC is close to chance on this holdout. The model should not be "
                "presented as a reliable failure predictor."
            )
    if metrics["recall"] < 0.7:
        warnings.append(
            "Recall is below 0.70 at the selected threshold: a substantial number of "
            "actual failures would be missed. Consider a lower threshold and review "
            "the false-alarm trade-off."
        )
    if abs(
        split_info.get("train_failure_rate", 0.5)
        - split_info.get("test_failure_rate", 0.5)
    ) > 0.10:
        warnings.append(
            "Train and test failure prevalence differ by more than 10 percentage "
            "points; metrics may not transfer across periods or wells."
        )
    if split_info["n_test"] < 100:
        warnings.append(
            f"Only {split_info['n_test']} test observations; metric uncertainty is large."
        )
    if n_features < 9:
        warnings.append(
            f"Only {n_features} of the nine recommended operational features are "
            "available in this dataset."
        )
    if info.get("n_target_missing"):
        warnings.append(
            f"{info['n_target_missing']} rows without a valid target were excluded."
        )
    if info.get("n_duplicates_removed"):
        warnings.append(
            f"{info['n_duplicates_removed']} exact duplicate rows were removed before "
            "splitting to reduce leakage."
        )
    if metrics["predicted_positive_rate"] > 0.8:
        warnings.append(
            "The model flags most observations as failures at this threshold; review "
            "precision and the business cost of false alarms."
        )
    return warnings


def train_model(
    df: pd.DataFrame,
    model_key: str = "random_forest",
    features: list[str] | None = None,
    strategy: str = "well",
    test_size: float = 0.25,
    random_state: int = 42,
    threshold: float = 0.5,
    class_weight: bool = False,
    params: dict[str, Any] | None = None,
    drop_duplicates: bool = True,
    dataset_source: str = "Unknown",
) -> dict[str, Any]:
    """Train one model on a leakage-safe split and return a rich result bundle."""
    if model_key not in MODEL_LABELS:
        raise ModelingError(f"Unknown model: {model_key}")
    if model_key == "xgboost" and not xgboost_available():
        raise ModelingError("XGBoost is not installed.")

    features = features or get_feature_columns(df)
    frame, info = prepare_training_frame(
        df, features=features, drop_duplicates=drop_duplicates
    )
    train_idx, test_idx, split_info = get_split_indices(
        frame, strategy=strategy, test_size=test_size, random_state=random_state
    )
    X_train = frame.iloc[train_idx][features]
    y_train = frame.iloc[train_idx][TARGET].astype(int)
    X_test = frame.iloc[test_idx][features]
    y_test = frame.iloc[test_idx][TARGET].astype(int)

    if class_weight and model_key == "xgboost":
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        scale_pos_weight = negatives / max(1, positives)
    else:
        scale_pos_weight = 1.0

    pipeline = build_pipeline(
        model_key,
        params=params,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    proba = positive_class_probability(pipeline, X_test)
    metrics = compute_metrics(y_test, proba, threshold=threshold)

    meta_cols = [c for c in [WELL_COL, DATE_COL, "__row_id"] if c in frame.columns]
    test_meta = (
        frame.iloc[test_idx][meta_cols].reset_index(drop=True)
        if meta_cols
        else pd.DataFrame(index=range(len(test_idx)))
    )

    ranges = training_ranges(frame.iloc[train_idx], features)
    stats = training_stats(frame.iloc[train_idx], features)
    warnings = _warnings_for_result(
        split_info,
        metrics,
        n_wells=int(frame[WELL_COL].nunique()) if WELL_COL in frame.columns else 0,
        n_features=len(features),
        info=info,
    )

    return {
        "model_key": model_key,
        "model_label": MODEL_LABELS[model_key],
        "pipeline": pipeline,
        "features": list(features),
        "feature_labels": {f: FEATURE_LABELS.get(f, f) for f in features},
        "params": dict(params or {}),
        "class_weight": bool(class_weight),
        "strategy": strategy,
        "split_info": split_info,
        "test_size": float(test_size),
        "random_state": int(random_state),
        "threshold": float(threshold),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_hash": dataset_hash(df),
        "dataset_source": dataset_source,
        "n_available": info["n_available"],
        "n_used": info["n_used"],
        "n_target_missing": info["n_target_missing"],
        "n_duplicates_removed": info["n_duplicates_removed"],
        "train_index": np.asarray(train_idx),
        "test_index": np.asarray(test_idx),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "test_meta": test_meta,
        "proba": proba,
        "metrics": metrics,
        "training_stats": ranges,
        "training_stats_table": stats,
        "warnings": warnings,
        "sklearn_version": sklearn.__version__,
        "app_version": APP_VERSION,
        "info": info,
    }


def comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, result in results.items():
        if "error" in result:
            rows.append(
                {
                    "model": MODEL_LABELS.get(key, key),
                    "status": "failed",
                    "error": result["error"],
                }
            )
            continue
        m = result["metrics"]
        rows.append(
            {
                "model": result["model_label"],
                "status": "trained",
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "threshold": result["threshold"],
            }
        )
    return pd.DataFrame(rows)


def train_and_compare(
    df: pd.DataFrame,
    model_keys: list[str] | None = None,
    features: list[str] | None = None,
    strategy: str = "well",
    test_size: float = 0.25,
    random_state: int = 42,
    threshold: float = 0.5,
    class_weight: bool = False,
    params_by_model: dict[str, dict[str, Any]] | None = None,
    drop_duplicates: bool = True,
    dataset_source: str = "Unknown",
) -> dict[str, Any]:
    if model_keys is None:
        model_keys = ["logistic", "random_forest"]
        if xgboost_available():
            model_keys.append("xgboost")
    results: dict[str, dict[str, Any]] = {}
    for key in model_keys:
        try:
            results[key] = train_model(
                df,
                model_key=key,
                features=features,
                strategy=strategy,
                test_size=test_size,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params=(params_by_model or {}).get(key),
                drop_duplicates=drop_duplicates,
                dataset_source=dataset_source,
            )
        except Exception as exc:  # noqa: BLE001 - show per-model failure
            results[key] = {"error": str(exc)}
    table = comparison_table(results)
    trained = {k: v for k, v in results.items() if "error" not in v}
    return {"table": table, "bundles": trained, "raw": results}


def compare_validation_strategies(
    df: pd.DataFrame,
    model_key: str = "random_forest",
    features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    threshold: float = 0.5,
    class_weight: bool = False,
    params: dict[str, Any] | None = None,
    drop_duplicates: bool = True,
    dataset_source: str = "Unknown",
) -> pd.DataFrame:
    rows = []
    for item in available_split_strategies(df):
        if not item["available"]:
            rows.append(
                {
                    "strategy": item["label"],
                    "available": False,
                    "note": item["reason"],
                }
            )
            continue
        try:
            result = train_model(
                df,
                model_key=model_key,
                features=features,
                strategy=item["key"],
                test_size=test_size,
                random_state=random_state,
                threshold=threshold,
                class_weight=class_weight,
                params=params,
                drop_duplicates=drop_duplicates,
                dataset_source=dataset_source,
            )
            m = result["metrics"]
            rows.append(
                {
                    "strategy": item["label"],
                    "available": True,
                    "train_rows": result["split_info"]["n_train"],
                    "test_rows": result["split_info"]["n_test"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "note": result["split_info"].get("note", ""),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "strategy": item["label"],
                    "available": False,
                    "note": f"Could not evaluate: {exc}",
                }
            )
    return pd.DataFrame(rows)


def group_cv(
    df: pd.DataFrame,
    model_key: str = "random_forest",
    features: list[str] | None = None,
    n_splits: int = 3,
    random_state: int = 42,
    threshold: float = 0.5,
    class_weight: bool = False,
    params: dict[str, Any] | None = None,
    drop_duplicates: bool = True,
) -> dict[str, Any]:
    """Well-grouped cross-validation for a more robust cross-well estimate."""
    features = features or get_feature_columns(df)
    frame, info = prepare_training_frame(
        df, features=features, drop_duplicates=drop_duplicates
    )
    if WELL_COL not in frame.columns:
        raise ModelingError("Grouped CV requires a well_id column.")
    groups = frame[WELL_COL].astype(str)
    n_wells = groups.nunique()
    if n_wells < 3:
        raise ModelingError("Grouped CV requires at least three wells.")
    n_splits = max(2, min(int(n_splits), n_wells))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows = []
    skipped = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame[TARGET].astype(int), groups=groups), start=1
    ):
        y_train = frame.iloc[train_idx][TARGET].astype(int)
        y_test = frame.iloc[test_idx][TARGET].astype(int)
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            skipped.append(fold)
            continue
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        pipeline = build_pipeline(
            model_key,
            params=params,
            class_weight=class_weight,
            scale_pos_weight=negatives / max(1, positives),
            random_state=random_state,
        )
        pipeline.fit(frame.iloc[train_idx][features], y_train)
        proba = positive_class_probability(pipeline, frame.iloc[test_idx][features])
        m = compute_metrics(y_test, proba, threshold=threshold)
        rows.append(
            {
                "fold": fold,
                "test_wells": ", ".join(sorted(groups.iloc[test_idx].unique())),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
            }
        )
    if not rows:
        raise ModelingError("No grouped fold had both classes in train and test.")
    folds = pd.DataFrame(rows)
    summary = {
        "n_splits": n_splits,
        "folds_completed": len(folds),
        "skipped_folds": skipped,
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_std": float(folds["accuracy"].std(ddof=0)),
        "recall_mean": float(folds["recall"].mean()),
        "recall_std": float(folds["recall"].std(ddof=0)),
        "f1_mean": float(folds["f1"].mean()),
        "f1_std": float(folds["f1"].std(ddof=0)),
        "roc_auc_mean": float(folds["roc_auc"].mean()),
        "roc_auc_std": float(folds["roc_auc"].std(ddof=0)),
        "pr_auc_mean": float(folds["pr_auc"].mean()),
        "pr_auc_std": float(folds["pr_auc"].std(ddof=0)),
    }
    return {"folds": folds, "summary": summary, "info": info}


def feature_importance_table(result: dict[str, Any]) -> pd.DataFrame:
    """Model-native feature importance (not causal inference)."""
    pipeline = result["pipeline"]
    model = pipeline.named_steps["model"]
    features = result["features"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "Impurity / gain importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        method = "Absolute standardised coefficient"
    else:
        return pd.DataFrame(columns=["feature", "label", "importance", "method"])
    frame = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "importance": values,
            "method": method,
        }
    )
    total = frame["importance"].sum()
    frame["share"] = frame["importance"] / total if total else 0.0
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
