"""
Tests for src/timeseries.py -- time series analysis utilities.
All tests use synthetic data (no real dataset access).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.timeseries import (
    extract_stop_events,
    run_stationarity_tests,
    build_hourly_aggregate,
)


# -----------------------------------------------------------------------
# extract_stop_events
# -----------------------------------------------------------------------

class TestExtractStopEvents:
    """Tests for extracting stop events (non-NaN passenger rows)."""

    def test_mixed_nan_and_values(self):
        """Standard case: some rows have passengers, some NaN."""
        df = pd.DataFrame({
            "time_iso": pd.date_range("2020-01-01", periods=10, freq="s"),
            "itcs_numberOfPassengers": [np.nan, np.nan, 5, np.nan, np.nan, 8, np.nan, 3, np.nan, np.nan],
            "itcs_stopName": ["-", "-", "StopA", "-", "-", "StopB", "-", "StopC", "-", "-"],
        })
        result = extract_stop_events(df)
        assert len(result) == 3
        assert list(result["itcs_numberOfPassengers"]) == [5, 8, 3]
        assert list(result["itcs_stopName"]) == ["StopA", "StopB", "StopC"]

    def test_empty_dataframe(self):
        """Empty input returns empty output."""
        df = pd.DataFrame({
            "time_iso": pd.Series(dtype="datetime64[ns]"),
            "itcs_numberOfPassengers": pd.Series(dtype="float64"),
        })
        result = extract_stop_events(df)
        assert len(result) == 0

    def test_no_stop_events(self):
        """All passengers are NaN -- no stop events."""
        df = pd.DataFrame({
            "time_iso": pd.date_range("2020-01-01", periods=5, freq="s"),
            "itcs_numberOfPassengers": [np.nan] * 5,
        })
        result = extract_stop_events(df)
        assert len(result) == 0

    def test_all_stop_events(self):
        """Every row has a passenger count (no NaN)."""
        df = pd.DataFrame({
            "time_iso": pd.date_range("2020-01-01", periods=4, freq="s"),
            "itcs_numberOfPassengers": [1, 2, 3, 4],
        })
        result = extract_stop_events(df)
        assert len(result) == 4

    def test_preserves_all_columns(self):
        """Output retains every column from input."""
        df = pd.DataFrame({
            "time_iso": pd.date_range("2020-01-01", periods=3, freq="s"),
            "itcs_numberOfPassengers": [np.nan, 10, np.nan],
            "extra_col": ["a", "b", "c"],
        })
        result = extract_stop_events(df)
        assert "extra_col" in result.columns
        assert result["extra_col"].iloc[0] == "b"

    def test_does_not_modify_input(self):
        """Input DataFrame is not mutated."""
        df = pd.DataFrame({
            "time_iso": pd.date_range("2020-01-01", periods=3, freq="s"),
            "itcs_numberOfPassengers": [np.nan, 5, np.nan],
        })
        original_len = len(df)
        _ = extract_stop_events(df)
        assert len(df) == original_len


# -----------------------------------------------------------------------
# run_stationarity_tests
# -----------------------------------------------------------------------

class TestRunStationarityTests:
    """Tests for ADF + KPSS stationarity testing wrapper."""

    def test_stationary_series(self):
        """White noise should be detected as stationary."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=500)
        result = run_stationarity_tests(series, "white_noise")

        assert result["name"] == "white_noise"
        assert "adf_stat" in result
        assert "adf_p" in result
        assert "kpss_stat" in result
        assert "kpss_p" in result
        assert "verdict" in result
        # ADF should reject (stationary), KPSS should not reject (stationary)
        assert result["adf_reject"] is True, f"ADF should reject for white noise, p={result['adf_p']}"
        assert result["kpss_reject"] is False, f"KPSS should not reject for white noise, p={result['kpss_p']}"
        assert result["verdict"] == "STATIONARY"

    def test_nonstationary_series(self):
        """Random walk should be detected as non-stationary."""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, size=500))
        result = run_stationarity_tests(series, "random_walk")

        assert result["name"] == "random_walk"
        # ADF should fail to reject (non-stationary), KPSS should reject (non-stationary)
        assert result["adf_reject"] is False, f"ADF should not reject for random walk, p={result['adf_p']}"
        assert result["kpss_reject"] is True, f"KPSS should reject for random walk, p={result['kpss_p']}"
        assert result["verdict"] == "NON-STATIONARY"

    def test_result_keys(self):
        """Result dict has all expected keys."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        result = run_stationarity_tests(series, "test")

        expected_keys = {"name", "adf_stat", "adf_p", "adf_reject", "kpss_stat", "kpss_p", "kpss_reject", "verdict"}
        assert set(result.keys()) == expected_keys

    def test_short_series(self):
        """Series with minimum viable length still works (>= 20 points)."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=25)
        result = run_stationarity_tests(series, "short")
        assert "verdict" in result


# -----------------------------------------------------------------------
# build_hourly_aggregate
# -----------------------------------------------------------------------

class TestBuildHourlyAggregate:
    """Tests for building hourly aggregate ridership from mission CSVs."""

    def test_single_mission_single_hour(self, tmp_path):
        """One mission with all stop events in the same hour."""
        # Create a fake mission CSV
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()
        df = pd.DataFrame({
            "time_iso": [
                "2020-06-15 10:05:00",
                "2020-06-15 10:15:00",
                "2020-06-15 10:30:00",
            ],
            "itcs_numberOfPassengers": [4, 8, 12],
            "itcs_stopName": ["A", "B", "C"],
        })
        df.to_csv(mission_dir / "mission1.csv", index=False)

        meta = pd.DataFrame({"name": ["mission1"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(8.0)  # mean(4, 8, 12)

    def test_two_missions_same_hour(self, tmp_path):
        """Two missions contributing to the same calendar hour get averaged."""
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()

        # Mission 1: mean pax in 10:00 hour = 10
        df1 = pd.DataFrame({
            "time_iso": ["2020-06-15 10:10:00", "2020-06-15 10:20:00"],
            "itcs_numberOfPassengers": [8, 12],
            "itcs_stopName": ["A", "B"],
        })
        df1.to_csv(mission_dir / "m1.csv", index=False)

        # Mission 2: mean pax in 10:00 hour = 6
        df2 = pd.DataFrame({
            "time_iso": ["2020-06-15 10:05:00", "2020-06-15 10:25:00"],
            "itcs_numberOfPassengers": [4, 8],
            "itcs_stopName": ["C", "D"],
        })
        df2.to_csv(mission_dir / "m2.csv", index=False)

        meta = pd.DataFrame({"name": ["m1", "m2"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert len(result) == 1
        # m1 contributes mean=10 for that hour, m2 contributes mean=6
        # All 4 stop events pooled: mean(8,12,4,8) = 8.0
        assert result.iloc[0] == pytest.approx(8.0)

    def test_multiple_hours(self, tmp_path):
        """Stop events across different hours produce multiple data points."""
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()

        df = pd.DataFrame({
            "time_iso": [
                "2020-06-15 10:10:00",
                "2020-06-15 11:10:00",
                "2020-06-15 12:10:00",
            ],
            "itcs_numberOfPassengers": [5, 15, 25],
            "itcs_stopName": ["A", "B", "C"],
        })
        df.to_csv(mission_dir / "m1.csv", index=False)

        meta = pd.DataFrame({"name": ["m1"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(15.0)
        assert result.iloc[2] == pytest.approx(25.0)

    def test_skips_nan_rows(self, tmp_path):
        """NaN passenger rows (non-stop-events) are excluded."""
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()

        df = pd.DataFrame({
            "time_iso": [
                "2020-06-15 10:00:00",
                "2020-06-15 10:01:00",
                "2020-06-15 10:02:00",
            ],
            "itcs_numberOfPassengers": [np.nan, 20, np.nan],
            "itcs_stopName": ["-", "A", "-"],
        })
        df.to_csv(mission_dir / "m1.csv", index=False)

        meta = pd.DataFrame({"name": ["m1"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(20.0)

    def test_missing_mission_file_skipped(self, tmp_path):
        """Missions with no CSV file are silently skipped."""
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()

        meta = pd.DataFrame({"name": ["nonexistent_mission"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert len(result) == 0

    def test_returns_sorted_series(self, tmp_path):
        """Result is sorted by datetime index."""
        mission_dir = tmp_path / "data"
        mission_dir.mkdir()

        df = pd.DataFrame({
            "time_iso": [
                "2020-06-15 14:10:00",
                "2020-06-15 08:10:00",
            ],
            "itcs_numberOfPassengers": [30, 10],
            "itcs_stopName": ["B", "A"],
        })
        df.to_csv(mission_dir / "m1.csv", index=False)

        meta = pd.DataFrame({"name": ["m1"]})
        usecols = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

        result = build_hourly_aggregate(meta, str(mission_dir), usecols)
        assert result.index[0] < result.index[1]
        assert result.iloc[0] == pytest.approx(10.0)  # 08:00 hour
        assert result.iloc[1] == pytest.approx(30.0)  # 14:00 hour
