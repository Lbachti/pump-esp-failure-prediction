"""Model persistence and single-observation ESP prediction."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    APP_VERSION,
    FEATURE_LABELS,
    FEATURE_UNITS,
    MODEL_PATH,
    RISK_COLORS,
    TARGET,
)
from .preprocessing import ModelingError, positive_class_probability


class PredictionError(Exception):
    """Raised when a prediction request is invalid."""


class ModelNotFoundError(PredictionError):
    """Raised when no model is available."""


def export_bundle(result: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    """Create a compact, serialisable model bundle from a training result."""
    if "pipeline" not in result:
        raise PredictionError("The training result does not contain a fitted pipeline.")
    return {
        "pipeline": result["pipeline"],
        "features": list(result["features"]),
        "feature_labels": dict(result.get("feature_labels", {})),
        "model_key": result["model_key"],
        "model_label": result["model_label"],
        "params": dict(result.get("params", {})),
        "class_weight": bool(result.get("class_weight", False)),
        "strategy": result.get("strategy"),
        "split_info": dict(result.get("split_info", {})),
        "threshold": float(threshold if threshold is not None else result.get("threshold", 0.5)),
        "trained_at": result.get("trained_at", datetime.now(timezone.utc).isoformat()),
        "dataset_hash": result.get("dataset_hash", ""),
        "dataset_source": result.get("dataset_source", "Unknown"),
        "training_stats": dict(result.get("training_stats", {})),
        "metrics": dict(result.get("metrics", {})),
        "warnings": list(result.get("warnings", [])),
        "n_used": int(result.get("n_used", 0)),
        "n_available": int(result.get("n_available", 0)),
        "sklearn_version": result.get("sklearn_version"),
        "app_version": result.get("app_version", APP_VERSION),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_model(bundle: dict[str, Any], path: Path | str = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "pipeline" not in bundle or "features" not in bundle:
        raise PredictionError("The model bundle is incomplete.")
    joblib.dump(bundle, path)
    return path


def load_model(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"No saved model was found at {path}. Train a model first, then save it."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Could not load the saved model: {exc}") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise PredictionError("The saved file is not a valid ESP PredictAI model bundle.")
    bundle["loaded_from"] = str(path)
    return bundle


def model_file_info(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    }


def delete_model_file(path: Path | str = MODEL_PATH) -> bool:
    path = Path(path)
    if path.exists():
        os.remove(path)
        return True
    return False


def model_matches_dataset(bundle: dict[str, Any], current_hash: str) -> bool:
    stored = bundle.get("dataset_hash", "")
    return bool(stored) and stored == current_hash


def _physical_checks(values: dict[str, float]) -> None:
    """Basic physical-validity checks only; no invented engineering limits."""
    nonnegative = {
        "intake_pressure",
        "discharge_pressure",
        "motor_amps",
        "motor_volt",
        "vibration",
        "frequency",
        "production",
    }
    positive = {"intake_pressure", "discharge_pressure", "motor_amps", "motor_volt", "frequency"}
    for feature, value in values.items():
        if not np.isfinite(value):
            raise PredictionError(f"{FEATURE_LABELS.get(feature, feature)} must be finite.")
        if feature in nonnegative and value < 0:
            raise PredictionError(
                f"{FEATURE_LABELS.get(feature, feature)} cannot be negative."
            )
        if feature in positive and value <= 0:
            raise PredictionError(
                f"{FEATURE_LABELS.get(feature, feature)} must be greater than zero."
            )


def risk_level(
    probability: float, warning_threshold: float = 0.4, high_threshold: float = 0.7
) -> str:
    if not 0 <= warning_threshold < high_threshold <= 1:
        raise PredictionError(
            "Risk thresholds must satisfy 0 ≤ warning < high ≤ 1."
        )
    if probability >= high_threshold:
        return "High"
    if probability >= warning_threshold:
        return "Elevated"
    return "Normal"


def risk_color(level: str) -> str:
    return RISK_COLORS.get(level, "#8aa4bf")


def predict_esp(
    bundle: dict[str, Any],
    inputs: dict[str, Any],
    decision_threshold: float | None = None,
    warning_threshold: float = 0.4,
    high_threshold: float = 0.7,
) -> dict[str, Any]:
    """Run a single ESP observation through the saved preprocessing pipeline."""
    if bundle is None or "pipeline" not in bundle:
        raise ModelNotFoundError("Train or load a model before requesting a prediction.")
    features = list(bundle["features"])
    missing = [f for f in features if f not in inputs]
    if missing:
        raise PredictionError(
            "Missing input features: "
            + ", ".join(FEATURE_LABELS.get(f, f) for f in missing)
        )
    values: dict[str, float] = {}
    for feature in features:
        raw = inputs[feature]
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            raise PredictionError(
                f"Enter a value for {FEATURE_LABELS.get(feature, feature)}."
            )
        try:
            values[feature] = float(raw)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"{FEATURE_LABELS.get(feature, feature)} must be numeric."
            ) from exc
    _physical_checks(values)

    threshold = float(
        decision_threshold if decision_threshold is not None else bundle.get("threshold", 0.5)
    )
    if not 0 <= threshold <= 1:
        raise PredictionError("The decision threshold must be between 0 and 1.")
    if not 0 <= warning_threshold < high_threshold <= 1:
        raise PredictionError(
            "Risk thresholds must satisfy 0 ≤ warning < high ≤ 1."
        )

    X = pd.DataFrame([values], columns=features)
    try:
        probability = float(positive_class_probability(bundle["pipeline"], X)[0])
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"The model could not score this input: {exc}") from exc
    probability = float(np.clip(probability, 0.0, 1.0))
    predicted = int(probability >= threshold)
    level = risk_level(probability, warning_threshold, high_threshold)

    ranges = bundle.get("training_stats", {}) or {}
    out_of_range = {}
    for feature, value in values.items():
        stats = ranges.get(feature)
        if not stats:
            continue
        low, high = stats.get("min"), stats.get("max")
        if low is None or high is None:
            continue
        if value < low or value > high:
            out_of_range[feature] = {
                "value": value,
                "min": float(low),
                "max": float(high),
                "median": float(stats.get("median", np.nan)),
            }

    margin = probability - threshold
    if abs(margin) < 0.10:
        confidence_note = (
            "The score is close to the decision threshold; this case is borderline. "
            "Small input changes could flip the classification."
        )
    elif margin >= 0.10:
        confidence_note = (
            "The score is above the decision threshold by a clear margin. This is a "
            "model score margin, not a statistical confidence interval."
        )
    else:
        confidence_note = (
            "The score is below the decision threshold by a clear margin. This is a "
            "model score margin, not a statistical confidence interval."
        )

    return {
        "probability": probability,
        "predicted": predicted,
        "predicted_label": "FAILED" if predicted == 1 else "RUNNING",
        "risk_level": level,
        "risk_color": risk_color(level),
        "decision_threshold": threshold,
        "warning_threshold": float(warning_threshold),
        "high_threshold": float(high_threshold),
        "margin": float(margin),
        "confidence_note": confidence_note,
        "out_of_range": out_of_range,
        "inputs": values,
        "features": features,
        "model_label": bundle.get("model_label", "Model"),
        "model_key": bundle.get("model_key", ""),
        "brier": bundle.get("metrics", {}).get("brier"),
        "trained_at": bundle.get("trained_at"),
        "dataset_hash": bundle.get("dataset_hash", ""),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def input_template(bundle: dict[str, Any]) -> dict[str, float]:
    template: dict[str, float] = {}
    stats = bundle.get("training_stats", {}) or {}
    for feature in bundle.get("features", []):
        value = stats.get(feature, {}).get("median", 0.0)
        if value is None or not np.isfinite(float(value)):
            value = 0.0
        template[feature] = float(value)
    return template


def prediction_interpretation(result: dict[str, Any]) -> str:
    prob = result["probability"]
    threshold = result["decision_threshold"]
    level = result["risk_level"]
    return (
        f"Model score {prob:.3f} at decision threshold {threshold:.2f} → "
        f"{result['predicted_label']} classification. Risk band: {level} "
        f"(warning ≥ {result['warning_threshold']:.2f}, high ≥ {result['high_threshold']:.2f}). "
        "This is a model-estimated score, not automatically a calibrated real-world "
        "failure probability."
    )


def bundle_info_table(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Model", bundle.get("model_label", "—")),
        ("Trained at (UTC)", bundle.get("trained_at", "—")),
        ("Validation strategy", bundle.get("split_info", {}).get("strategy_label", "—")),
        ("Features used", ", ".join(bundle.get("features", []))),
        ("Decision threshold", f"{bundle.get('threshold', 0.5):.2f}"),
        ("Training rows", f"{bundle.get('n_used', 0):,}"),
        ("Dataset source", bundle.get("dataset_source", "—")),
        ("Dataset fingerprint", (bundle.get("dataset_hash", "") or "—")[:16]),
        ("Saved at (UTC)", bundle.get("saved_at", "—")),
        ("App version", bundle.get("app_version", APP_VERSION)),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def validate_bundle_features(bundle: dict[str, Any], available: list[str]) -> list[str]:
    missing = [f for f in bundle.get("features", []) if f not in available]
    return missing
