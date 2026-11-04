"""Tests for dataset loading, target encoding, and integrity auditing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import FEATURE_ORDER, LOCAL_CSV_PATH, TARGET, TARGET_RAW
from src.data_loader import (
    audit_dataset,
    dataset_hash,
    get_feature_columns,
    prepare_dataset,
)


@pytest.fixture(scope="module")
def loaded_dataset():
    raw = pd.read_csv(LOCAL_CSV_PATH)
    return prepare_dataset(raw, source_label="test fixture", load_mode="test")


def test_target_encoding_is_running_zero_failed_one(loaded_dataset):
    df, _ = loaded_dataset
    assert TARGET in df.columns
    assert TARGET_RAW in df.columns
    assert set(df[TARGET].dropna().unique()) == {0, 1}
    running = df.loc[df[TARGET_RAW] == "running", TARGET]
    failed = df.loc[df[TARGET_RAW] == "failed", TARGET]
    assert (running == 0).all()
    assert (failed == 1).all()


def test_source_shape_and_class_counts(loaded_dataset):
    df, meta = loaded_dataset
    assert len(df) == 8766
    assert meta["n_wells"] == 6
    assert int((df[TARGET] == 0).sum()) == 5035
    assert int((df[TARGET] == 1).sum()) == 3731


def test_model_features_exclude_target_and_leakage(loaded_dataset):
    df, _ = loaded_dataset
    features = get_feature_columns(df)
    assert features == [f for f in FEATURE_ORDER if f in df.columns]
    assert TARGET not in features
    assert TARGET_RAW not in features
    assert "downtime_hours" not in features
    assert "production_loss" not in features
    assert "well_id" not in features
    assert "date" not in features


def test_audit_detects_deterministic_leakage(loaded_dataset):
    df, meta = loaded_dataset
    audit = audit_dataset(df, meta)
    assert audit["leakage"]["downtime_hours"]["deterministic_separation"] is True
    assert audit["leakage"]["production_loss"]["deterministic_separation"] is True
    assert audit["leakage"]["downtime_hours"]["nonzero_running"] == 0
    assert audit["leakage"]["downtime_hours"]["zero_failed"] == 0


def test_audit_flags_unverified_likely_simulated_data(loaded_dataset):
    df, meta = loaded_dataset
    audit = audit_dataset(df, meta)
    assert audit["n_rows"] == 8766
    assert audit["n_cols"] == 15  # 14 source columns + encoded target, internal row id hidden
    assert audit["duplicate_rows"] == 0
    assert audit["missing_cells"] == 0
    assert audit["complete_panel"] is True
    assert audit["near_independent_labels"] is True
    assert audit["likely_simulated"] is True
    assert audit["suspicion_level"] == "High"


def test_dataset_hash_is_stable_and_sensitive_to_target(loaded_dataset):
    df, _ = loaded_dataset
    first = dataset_hash(df)
    second = dataset_hash(df.copy())
    assert first == second
    changed = df.copy()
    changed.loc[changed.index[0], TARGET] = 1 - int(changed.loc[changed.index[0], TARGET])
    assert dataset_hash(changed) != first


def test_prepare_dataset_maps_synthetic_labels():
    raw = pd.DataFrame(
        {
            "well_id": ["w1", "w1", "w2", "w2"],
            "date": ["1/1/2024", "1/2/2024", "1/1/2024", "1/2/2024"],
            "esp_status": ["running", "failed", "RUNNING", "FAILED"],
            "intake_pressure": [100, 110, 120, 130],
        }
    )
    df, meta = prepare_dataset(raw)
    assert df[TARGET].tolist() == [0, 1, 0, 1]
    assert df["date"].notna().all()
    assert meta["n_wells"] == 2
