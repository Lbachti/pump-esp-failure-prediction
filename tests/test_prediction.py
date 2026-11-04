"""Tests for model persistence, risk bands, and single-observation prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_ORDER, TARGET
from src.data_loader import prepare_dataset
from src.model_training import train_model
from src.prediction import (
    PredictionError,
    delete_model_file,
    export_bundle,
    input_template,
    load_model,
    model_file_info,
    predict_esp,
    prediction_interpretation,
    risk_level,
    save_model,
    validate_bundle_features,
)


@pytest.fixture(scope="module")
def trained_bundle():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rows = []
    for well in ["w1", "w2", "w3"]:
        for date in dates:
            failure = int(rng.random() < 0.35)
            rows.append(
                {
                    "well_id": well,
                    "date": date.strftime("%m/%d/%Y"),
                    "esp_status": "failed" if failure else "running",
                    "intake_pressure": 1800 + rng.normal(0, 50),
                    "discharge_pressure": 2500 + rng.normal(0, 70),
                    "motor_temperature": 140 + rng.normal(0, 4),
                    "intake_temperature": 145 + rng.normal(0, 3),
                    "motor_amps": 30 + rng.normal(0, 2),
                    "motor_volt": 1600 + rng.normal(0, 30),
                    "vibration": 0.5 + rng.normal(0, 0.04),
                    "frequency": 46 + rng.normal(0, 0.8),
                    "production": 1400 + rng.normal(0, 50),
                    "downtime_hours": 8.0 if failure else 0.0,
                    "production_loss": 100.0 if failure else 0.0,
                }
            )
    raw = pd.DataFrame(rows)
    df, _ = prepare_dataset(raw, source_label="synthetic prediction fixture")
    result = train_model(
        df,
        model_key="logistic",
        features=FEATURE_ORDER,
        strategy="random",
        test_size=0.25,
        random_state=42,
        threshold=0.5,
        params={"C": 1.0},
        drop_duplicates=True,
    )
    return export_bundle(result, threshold=0.5)


def test_bundle_is_compact_and_contains_pipeline(trained_bundle):
    assert "pipeline" in trained_bundle
    assert "features" in trained_bundle
    assert trained_bundle["features"] == FEATURE_ORDER
    assert "X_test" not in trained_bundle
    assert "y_test" not in trained_bundle
    assert "proba" not in trained_bundle
    assert "training_stats" in trained_bundle


def test_save_load_and_file_management(trained_bundle, tmp_path):
    path = tmp_path / "model.joblib"
    saved = save_model(trained_bundle, path)
    assert saved == path
    assert path.exists()
    info = model_file_info(path)
    assert info["exists"] is True
    assert info["size_bytes"] > 0

    loaded = load_model(path)
    assert loaded["model_key"] == "logistic"
    assert loaded["features"] == FEATURE_ORDER
    assert "pipeline" in loaded

    assert validate_bundle_features(loaded, FEATURE_ORDER) == []
    assert validate_bundle_features(loaded, FEATURE_ORDER[:-1]) == [FEATURE_ORDER[-1]]

    assert delete_model_file(path) is True
    assert delete_model_file(path) is False


def test_prediction_returns_probability_and_risk_band(trained_bundle):
    inputs = input_template(trained_bundle)
    result = predict_esp(
        trained_bundle,
        inputs,
        decision_threshold=0.5,
        warning_threshold=0.4,
        high_threshold=0.7,
    )
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_label"] in {"RUNNING", "FAILED"}
    assert result["risk_level"] in {"Normal", "Elevated", "High"}
    assert result["decision_threshold"] == 0.5
    assert set(result["inputs"]) == set(FEATURE_ORDER)
    assert "model score" in prediction_interpretation(result).lower()


def test_risk_bands_are_model_probability_bands():
    assert risk_level(0.10, 0.4, 0.7) == "Normal"
    assert risk_level(0.40, 0.4, 0.7) == "Elevated"
    assert risk_level(0.70, 0.4, 0.7) == "High"
    with pytest.raises(PredictionError):
        risk_level(0.5, 0.8, 0.7)


def test_invalid_inputs_are_rejected(trained_bundle):
    inputs = input_template(trained_bundle)
    bad = dict(inputs)
    bad["vibration"] = -0.1
    with pytest.raises(PredictionError):
        predict_esp(trained_bundle, bad)
    with pytest.raises(PredictionError):
        predict_esp(trained_bundle, inputs, warning_threshold=0.8, high_threshold=0.7)
    with pytest.raises(PredictionError):
        predict_esp(trained_bundle, {k: v for k, v in inputs.items() if k != "vibration"})


def test_out_of_range_inputs_are_flagged(trained_bundle):
    inputs = input_template(trained_bundle)
    inputs["motor_temperature"] = trained_bundle["training_stats"]["motor_temperature"]["max"] + 50
    result = predict_esp(trained_bundle, inputs)
    assert "motor_temperature" in result["out_of_range"]


def test_empty_bundle_cannot_predict():
    with pytest.raises(PredictionError):
        predict_esp(None, {})
