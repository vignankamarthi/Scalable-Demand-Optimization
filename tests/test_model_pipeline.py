"""
Tests for src/model_pipeline.py
Train/test splitting, scaling, model training, and evaluation.
"""

import numpy as np
import pandas as pd
import pytest
from src.model_pipeline import (
    mission_stratified_split,
    prepare_features_and_target,
    evaluate_model,
    get_model_configs,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def feature_matrix():
    """
    Synthetic feature matrix from 4 missions, 2 routes.
    50 rows per mission = 200 rows total.
    Includes mission_name, itcs_numberOfPassengers (target), and numeric features.
    """
    rows = []
    missions = [
        ("m1", "83", 15.0),
        ("m2", "83", 25.0),
        ("m3", "33", 10.0),
        ("m4", "33", 30.0),
    ]
    for mission_name, route, base_pax in missions:
        for i in range(50):
            rows.append({
                "mission_name": mission_name,
                "itcs_numberOfPassengers": base_pax + np.sin(i) * 5,
                "hour": (7 + i % 12),
                "dayofweek": 1,
                "month": 4,
                "is_weekend": False,
                "is_rush_hour": True if (7 + i % 12) in [7, 8, 16, 17, 18] else False,
                "speed_kmh": 20.0 + i * 0.5,
                "temp_C": 10.0,
                "acceleration": np.random.default_rng(42).normal(0, 1),
                "route_83": 1 if route == "83" else 0,
                "gnss_altitude": 408.0,
                "electric_powerDemand": 10000.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_feature_matrix():
    """Minimal feature matrix for quick tests (20 rows, 2 missions)."""
    rows = []
    for mission, pax in [("m1", 5.0), ("m2", 25.0)]:
        for i in range(10):
            rows.append({
                "mission_name": mission,
                "itcs_numberOfPassengers": pax,
                "hour": 8,
                "speed_kmh": 10.0 + i,
                "route_83": 1,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# mission_stratified_split tests
# -----------------------------------------------------------------------

class TestMissionStratifiedSplit:
    def test_no_mission_overlap(self, feature_matrix):
        train, test = mission_stratified_split(feature_matrix, test_size=0.5, seed=42)
        train_missions = set(train["mission_name"].unique())
        test_missions = set(test["mission_name"].unique())
        assert train_missions.isdisjoint(test_missions)

    def test_all_rows_accounted_for(self, feature_matrix):
        train, test = mission_stratified_split(feature_matrix, test_size=0.5, seed=42)
        assert len(train) + len(test) == len(feature_matrix)

    def test_approximate_split_ratio(self, feature_matrix):
        train, test = mission_stratified_split(feature_matrix, test_size=0.25, seed=42)
        # With 4 missions of 50 rows each, 25% test = 1 mission = 50 rows
        # But stratified by route, so 1 from each route = 2 missions = 100 rows
        # Actual ratio depends on stratification; just check it's roughly right
        test_ratio = len(test) / len(feature_matrix)
        assert 0.1 <= test_ratio <= 0.6  # loose bounds for 4 missions

    def test_deterministic(self, feature_matrix):
        t1_train, t1_test = mission_stratified_split(feature_matrix, test_size=0.5, seed=42)
        t2_train, t2_test = mission_stratified_split(feature_matrix, test_size=0.5, seed=42)
        assert set(t1_train["mission_name"].unique()) == set(t2_train["mission_name"].unique())

    def test_preserves_columns(self, feature_matrix):
        train, test = mission_stratified_split(feature_matrix, test_size=0.5, seed=42)
        assert set(train.columns) == set(feature_matrix.columns)
        assert set(test.columns) == set(feature_matrix.columns)


# -----------------------------------------------------------------------
# prepare_features_and_target tests
# -----------------------------------------------------------------------

class TestPrepareFeaturesAndTarget:
    def test_separates_x_and_y(self, feature_matrix):
        X, y = prepare_features_and_target(
            feature_matrix,
            target_col="itcs_numberOfPassengers",
            drop_cols=["mission_name"],
        )
        assert "itcs_numberOfPassengers" not in X.columns
        assert "mission_name" not in X.columns
        assert len(y) == len(feature_matrix)

    def test_x_all_numeric(self, feature_matrix):
        X, y = prepare_features_and_target(
            feature_matrix,
            target_col="itcs_numberOfPassengers",
            drop_cols=["mission_name"],
        )
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is not numeric"

    def test_y_is_series(self, feature_matrix):
        X, y = prepare_features_and_target(
            feature_matrix,
            target_col="itcs_numberOfPassengers",
            drop_cols=["mission_name"],
        )
        assert isinstance(y, pd.Series)

    def test_preserves_row_count(self, feature_matrix):
        X, y = prepare_features_and_target(
            feature_matrix,
            target_col="itcs_numberOfPassengers",
            drop_cols=["mission_name"],
        )
        assert len(X) == len(feature_matrix)


# -----------------------------------------------------------------------
# evaluate_model tests
# -----------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_dict(self):
        y_true = pd.Series(["low", "medium", "high", "low", "medium", "high"])
        y_pred = pd.Series(["low", "medium", "high", "low", "low", "high"])
        result = evaluate_model(y_true, y_pred)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        y_true = pd.Series(["low", "medium", "high"] * 10)
        y_pred = pd.Series(["low", "medium", "high"] * 10)
        result = evaluate_model(y_true, y_pred)
        assert "macro_f1" in result
        assert "balanced_accuracy" in result
        assert "confusion_matrix" in result
        assert "per_class" in result

    def test_perfect_prediction(self):
        y_true = pd.Series(["low", "medium", "high"] * 10)
        y_pred = pd.Series(["low", "medium", "high"] * 10)
        result = evaluate_model(y_true, y_pred)
        assert np.isclose(result["macro_f1"], 1.0)
        assert np.isclose(result["balanced_accuracy"], 1.0)

    def test_per_class_has_all_labels(self):
        y_true = pd.Series(["low", "medium", "high"] * 5)
        y_pred = pd.Series(["low", "low", "high"] * 5)
        result = evaluate_model(y_true, y_pred)
        assert "low" in result["per_class"]
        assert "medium" in result["per_class"]
        assert "high" in result["per_class"]

    def test_confusion_matrix_shape(self):
        y_true = pd.Series(["low", "medium", "high"] * 5)
        y_pred = pd.Series(["low", "medium", "high"] * 5)
        result = evaluate_model(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert cm.shape == (3, 3)


# -----------------------------------------------------------------------
# get_model_configs tests
# -----------------------------------------------------------------------

class TestGetModelConfigs:
    def test_returns_dict(self):
        configs = get_model_configs()
        assert isinstance(configs, dict)

    def test_has_all_models(self):
        configs = get_model_configs()
        expected = {"decision_tree", "random_forest"}
        assert expected == set(configs.keys())

    def test_each_config_has_model_key(self):
        configs = get_model_configs()
        for name, cfg in configs.items():
            assert "model" in cfg, f"{name} missing 'model' key"

    def test_each_config_has_params_key(self):
        configs = get_model_configs()
        for name, cfg in configs.items():
            assert "params" in cfg, f"{name} missing 'params' key"
            assert isinstance(cfg["params"], dict)
