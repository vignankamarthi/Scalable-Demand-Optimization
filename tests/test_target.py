"""
Tests for src/target.py
Target variable construction: tercile binning and class assignment.
"""

import numpy as np
import pandas as pd
import pytest
from src.target import (
    compute_tercile_boundaries,
    assign_demand_class,
    compute_class_distribution,
)


class TestComputeTercileBoundaries:
    def test_uniform_distribution(self):
        """Uniform 0-99: terciles at ~33 and ~66"""
        s = pd.Series(range(100), dtype=float)
        q1, q2 = compute_tercile_boundaries(s)
        assert np.isclose(q1, 33.0, atol=1.0)
        assert np.isclose(q2, 66.0, atol=1.0)

    def test_known_values(self, mock_passenger_series):
        """
        30 values: [0-5, 10-15, 25-35].
        33rd percentile should fall between 5 and 10.
        67th percentile should fall between 15 and 25.
        """
        q1, q2 = compute_tercile_boundaries(mock_passenger_series)
        assert 4.0 <= q1 <= 11.0
        assert 14.0 <= q2 <= 26.0

    def test_returns_floats(self, mock_passenger_series):
        q1, q2 = compute_tercile_boundaries(mock_passenger_series)
        assert isinstance(q1, float)
        assert isinstance(q2, float)

    def test_q1_less_than_q2(self, mock_passenger_series):
        q1, q2 = compute_tercile_boundaries(mock_passenger_series)
        assert q1 < q2

    def test_ignores_nans(self):
        s = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        q1, q2 = compute_tercile_boundaries(s)
        # Should compute on non-NaN values [1-9]
        assert 2.0 <= q1 <= 4.0
        assert 5.0 <= q2 <= 7.0

    def test_single_value(self):
        """Edge case: all same values -> boundaries equal"""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        q1, q2 = compute_tercile_boundaries(s)
        assert q1 == q2 == 5.0


class TestAssignDemandClass:
    def test_below_low_boundary(self):
        s = pd.Series([5.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert result.iloc[0] == "low"

    def test_at_low_boundary(self):
        """Value exactly at low boundary -> low"""
        s = pd.Series([10.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert result.iloc[0] == "low"

    def test_between_boundaries(self):
        s = pd.Series([15.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert result.iloc[0] == "medium"

    def test_at_high_boundary(self):
        """Value exactly at high boundary -> medium"""
        s = pd.Series([20.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert result.iloc[0] == "medium"

    def test_above_high_boundary(self):
        s = pd.Series([25.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert result.iloc[0] == "high"

    def test_nan_input_produces_nan(self):
        s = pd.Series([np.nan])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        # np.select with NaN input should produce the default (nan)
        assert result.iloc[0] == "nan" or pd.isna(result.iloc[0])

    def test_multiple_values(self):
        s = pd.Series([5.0, 10.0, 15.0, 20.0, 25.0])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        expected = ["low", "low", "medium", "medium", "high"]
        assert list(result) == expected

    def test_preserves_index(self):
        s = pd.Series([5.0, 15.0, 25.0], index=[100, 200, 300])
        result = assign_demand_class(s, low_boundary=10.0, high_boundary=20.0)
        assert list(result.index) == [100, 200, 300]


class TestComputeClassDistribution:
    def test_balanced(self):
        labels = pd.Series(["low"] * 10 + ["medium"] * 10 + ["high"] * 10)
        result = compute_class_distribution(labels)
        assert result == {"low": 10, "medium": 10, "high": 10}

    def test_imbalanced(self):
        labels = pd.Series(["low"] * 50 + ["medium"] * 30 + ["high"] * 20)
        result = compute_class_distribution(labels)
        assert result == {"low": 50, "medium": 30, "high": 20}

    def test_missing_class(self):
        """If a class has 0 observations, it should still appear with count 0"""
        labels = pd.Series(["low"] * 10 + ["medium"] * 10)
        result = compute_class_distribution(labels)
        assert result["high"] == 0

    def test_all_keys_present(self):
        labels = pd.Series(["low"])
        result = compute_class_distribution(labels)
        assert set(result.keys()) == {"low", "medium", "high"}
