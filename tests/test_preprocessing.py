"""Tests for leakage-safe preprocessing and validation splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_ORDER, TARGET
from src.data_loader import prepare_dataset
from src.preprocessing import (
    available_split_strategies,
    build_pipeline,
    get_split_indices,
    prepare_training_frame,
    training_ranges,
    training_stats,
)


def _synthetic_raw(n_per_well: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    wells = ["well_a", "well_b", "well_c", "well_d"]
    dates = pd.date_range("2020-01-01", periods=n_per_well, freq="D")
    rows = []
    for well_index, well in enumerate(wells):
        for day_index, date in enumerate(dates):
            failure = int(rng.random() < 0.32)
            rows.append(
                {
                    "well_id": well,
                    "date": date.strftime("%m/%d/%Y"),
                    "esp_status": "failed" if failure else "running",
                    "intake_pressure": 1800 + 200 * well_index + rng.normal(0, 40),
                    "discharge_pressure": 2500 + 250 * well_index + rng.normal(0, 55),
                    "motor_temperature": 135 + 8 * well_index + rng.normal(0, 3),
                    "intake_temperature": 140 + 5 * well_index + rng.normal(0, 2),
                    "motor_amps": 28 + 3 * well_index + rng.normal(0, 1.2),
                    "motor_volt": 1550 + 60 * well_index + rng.normal(0, 25),
                    "vibration": 0.45 + 0.05 * well_index + rng.normal(0, 0.03),
                    "frequency": 45 + 2 * well_index + rng.normal(0, 0.6),
                    "production": 1300 + 120 * well_index + rng.normal(0, 40),
                    "downtime_hours": 12.0 if failure else 0.0,
                    "production_loss": 300.0 if failure else 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df():
    df, _ = prepare_dataset(_synthetic_raw(), source_label="synthetic")
    return df


def test_training_frame_uses_only_operational_features(synthetic_df):
    frame, info = prepare_training_frame(synthetic_df)
    assert len(frame) == len(synthetic_df)
    assert info["n_used"] == len(synthetic_df)
    assert all(feature in frame.columns for feature in FEATURE_ORDER)
    assert TARGET in frame.columns


def test_duplicate_rows_are_removed_ignoring_internal_row_id():
    raw = _synthetic_raw(20)
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    df, _ = prepare_dataset(raw, source_label="synthetic duplicates")
    frame, info = prepare_training_frame(df, drop_duplicates=True)
    assert info["n_duplicates_removed"] == 1
    assert len(frame) == len(df) - 1


def test_random_split_is_stratified_and_disjoint(synthetic_df):
    frame, _ = prepare_training_frame(synthetic_df)
    train_idx, test_idx, info = get_split_indices(
        frame, strategy="random", test_size=0.25, random_state=42
    )
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert info["n_train"] + info["n_test"] == len(frame)
    assert set(frame.iloc[test_idx][TARGET].unique()) == {0, 1}


def test_well_split_holds_out_entire_wells(synthetic_df):
    frame, _ = prepare_training_frame(synthetic_df)
    train_idx, test_idx, info = get_split_indices(
        frame, strategy="well", test_size=0.25, random_state=42
    )
    assert set(info["train_wells"]).isdisjoint(set(info["test_wells"]))
    assert set(frame.iloc[train_idx]["well_id"]).isdisjoint(
        set(frame.iloc[test_idx]["well_id"])
    )
    assert set(frame.iloc[test_idx][TARGET].unique()) == {0, 1}


def test_time_split_uses_later_dates_for_test(synthetic_df):
    frame, _ = prepare_training_frame(synthetic_df)
    train_idx, test_idx, info = get_split_indices(
        frame, strategy="time", test_size=0.25, random_state=42
    )
    assert info["cutoff_date"]
    assert frame.iloc[train_idx]["date"].max() < frame.iloc[test_idx]["date"].min()
    assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_pipelines_are_leakage_safe_and_model_specific(synthetic_df):
    frame, _ = prepare_training_frame(synthetic_df)
    X = frame[FEATURE_ORDER]
    y = frame[TARGET].astype(int)

    logistic = build_pipeline("logistic", params={"C": 0.5}, random_state=42)
    assert "scaler" in logistic.named_steps
    logistic.fit(X.iloc[:200], y.iloc[:200])
    probabilities = logistic.predict_proba(X.iloc[200:220])[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    forest = build_pipeline("random_forest", params={"n_estimators": 20}, random_state=42)
    assert "scaler" not in forest.named_steps
    forest.fit(X.iloc[:200], y.iloc[:200])
    assert forest.predict_proba(X.iloc[200:220]).shape == (20, 2)


def test_training_statistics_are_derived_from_features(synthetic_df):
    frame, _ = prepare_training_frame(synthetic_df)
    stats = training_stats(frame, FEATURE_ORDER)
    ranges = training_ranges(frame, FEATURE_ORDER)
    assert len(stats) == len(FEATURE_ORDER)
    assert set(ranges) == set(FEATURE_ORDER)
    assert ranges["motor_temperature"]["min"] <= ranges["motor_temperature"]["median"]
    assert ranges["motor_temperature"]["median"] <= ranges["motor_temperature"]["max"]


def test_available_split_strategies_report_reasons(synthetic_df):
    strategies = available_split_strategies(synthetic_df)
    keys = {item["key"] for item in strategies}
    assert keys == {"random", "time", "well"}
    assert all(item["available"] for item in strategies)
