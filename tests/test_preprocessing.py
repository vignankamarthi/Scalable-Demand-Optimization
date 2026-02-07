"""
Tests for src/preprocessing.py
All tests use mock data -- no real dataset access.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    kelvin_to_celsius,
    radians_to_degrees,
    mps_to_kmh,
    extract_temporal_features,
    detect_missing_values,
    forward_fill_within_mission,
    apply_unit_conversions,
    forward_fill_stop_columns,
)


# -----------------------------------------------------------------------
# Unit conversion tests
# -----------------------------------------------------------------------

class TestKelvinToCelsius:
    def test_freezing_point(self):
        s = pd.Series([273.15])
        result = kelvin_to_celsius(s)
        assert np.isclose(result.iloc[0], 0.0)

    def test_boiling_point(self):
        s = pd.Series([373.15])
        result = kelvin_to_celsius(s)
        assert np.isclose(result.iloc[0], 100.0)

    def test_zurich_range(self):
        """Typical Zurich temps: 273K-311K -> 0C-38C"""
        s = pd.Series([273.15, 283.15, 293.15, 303.15, 311.15])
        result = kelvin_to_celsius(s)
        expected = [0.0, 10.0, 20.0, 30.0, 38.0]
        np.testing.assert_allclose(result.values, expected)

    def test_nan_preserved(self):
        s = pd.Series([283.15, np.nan, 293.15])
        result = kelvin_to_celsius(s)
        assert np.isnan(result.iloc[1])
        assert np.isclose(result.iloc[0], 10.0)

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        result = kelvin_to_celsius(s)
        assert len(result) == 0


class TestRadiansToDegrees:
    def test_zero(self):
        s = pd.Series([0.0])
        result = radians_to_degrees(s)
        assert np.isclose(result.iloc[0], 0.0)

    def test_pi(self):
        s = pd.Series([np.pi])
        result = radians_to_degrees(s)
        assert np.isclose(result.iloc[0], 180.0)

    def test_zurich_latitude(self):
        """Zurich ~47.37 degrees -> ~0.8267 rad"""
        s = pd.Series([0.8267])
        result = radians_to_degrees(s)
        assert np.isclose(result.iloc[0], 47.37, atol=0.1)

    def test_zurich_longitude(self):
        """Zurich ~8.54 degrees -> ~0.1491 rad"""
        s = pd.Series([0.1491])
        result = radians_to_degrees(s)
        assert np.isclose(result.iloc[0], 8.54, atol=0.1)

    def test_nan_preserved(self):
        s = pd.Series([0.5, np.nan])
        result = radians_to_degrees(s)
        assert np.isnan(result.iloc[1])


class TestMpsToKmh:
    def test_zero(self):
        s = pd.Series([0.0])
        result = mps_to_kmh(s)
        assert result.iloc[0] == 0.0

    def test_known_conversion(self):
        """10 m/s = 36 km/h"""
        s = pd.Series([10.0])
        result = mps_to_kmh(s)
        assert np.isclose(result.iloc[0], 36.0)

    def test_bus_speed_range(self):
        """Typical bus: 0-15 m/s -> 0-54 km/h"""
        s = pd.Series([0.0, 5.0, 10.0, 15.0])
        result = mps_to_kmh(s)
        expected = [0.0, 18.0, 36.0, 54.0]
        np.testing.assert_allclose(result.values, expected)

    def test_nan_preserved(self):
        s = pd.Series([5.0, np.nan])
        result = mps_to_kmh(s)
        assert np.isnan(result.iloc[1])


# -----------------------------------------------------------------------
# Temporal feature extraction tests
# -----------------------------------------------------------------------

class TestExtractTemporalFeatures:
    def test_hour_extraction(self, mock_mission_df):
        result = extract_temporal_features(mock_mission_df)
        # 2019-04-30 03:18:56 UTC -> hour = 3
        assert result["hour"].iloc[0] == 3
        assert all(result["hour"] == 3)

    def test_day_of_week(self, mock_mission_df):
        # 2019-04-30 is Tuesday -> dayofweek = 1
        result = extract_temporal_features(mock_mission_df)
        assert all(result["dayofweek"] == 1)

    def test_month(self, mock_mission_df):
        result = extract_temporal_features(mock_mission_df)
        assert all(result["month"] == 4)

    def test_year(self, mock_mission_df):
        result = extract_temporal_features(mock_mission_df)
        assert all(result["year"] == 2019)

    def test_weekend_detection_weekday(self, mock_mission_df):
        """Tuesday should not be weekend"""
        result = extract_temporal_features(mock_mission_df)
        assert not any(result["is_weekend"])

    def test_weekend_detection_saturday(self, weekend_mission_df):
        """Saturday should be weekend"""
        result = extract_temporal_features(weekend_mission_df)
        assert all(result["is_weekend"])

    def test_rush_hour_weekday_am(self, rush_hour_mission_df):
        result = extract_temporal_features(rush_hour_mission_df)
        # 6:59 on Wednesday -> NOT rush
        assert result["is_rush_hour"].iloc[0] == False
        # 7:00 on Wednesday -> AM rush
        assert result["is_rush_hour"].iloc[1] == True

    def test_rush_hour_weekday_pm(self, rush_hour_mission_df):
        result = extract_temporal_features(rush_hour_mission_df)
        # 15:59 on Wednesday -> NOT rush
        assert result["is_rush_hour"].iloc[2] == False
        # 16:00 on Wednesday -> PM rush
        assert result["is_rush_hour"].iloc[3] == True

    def test_rush_hour_weekend(self, weekend_mission_df):
        """Weekend hours should never be rush hour"""
        result = extract_temporal_features(weekend_mission_df)
        assert not any(result["is_rush_hour"])

    def test_does_not_modify_input(self, mock_mission_df):
        original_cols = set(mock_mission_df.columns)
        _ = extract_temporal_features(mock_mission_df)
        assert set(mock_mission_df.columns) == original_cols

    def test_output_has_all_expected_columns(self, mock_mission_df):
        result = extract_temporal_features(mock_mission_df)
        expected_new = {"hour", "dayofweek", "month", "year", "is_weekend", "is_rush_hour"}
        assert expected_new.issubset(set(result.columns))


# -----------------------------------------------------------------------
# Missing data detection tests
# -----------------------------------------------------------------------

class TestDetectMissingValues:
    def test_no_missing(self, mock_mission_df):
        result = detect_missing_values(mock_mission_df)
        assert all(result["nan_count"] == 0)

    def test_nan_detection(self, mock_mission_with_nans):
        result = detect_missing_values(mock_mission_with_nans)
        lat_row = result[result["column"] == "gnss_latitude"]
        assert lat_row["nan_count"].iloc[0] == 5

    def test_dash_detection(self, mock_mission_with_dashes):
        result = detect_missing_values(mock_mission_with_dashes)
        route_row = result[result["column"] == "itcs_busRoute"]
        assert route_row["dash_count"].iloc[0] == 5
        stop_row = result[result["column"] == "itcs_stopName"]
        assert stop_row["dash_count"].iloc[0] == 3

    def test_numeric_no_dash(self, mock_mission_df):
        """Numeric columns should always have 0 dash count"""
        result = detect_missing_values(mock_mission_df)
        numeric_rows = result[result["column"].isin(["gnss_latitude", "temperature_ambient"])]
        assert all(numeric_rows["dash_count"] == 0)

    def test_percentage_calculation(self, mock_mission_with_nans):
        result = detect_missing_values(mock_mission_with_nans)
        lat_row = result[result["column"] == "gnss_latitude"]
        # 5 NaN out of 20 rows = 25%
        assert lat_row["nan_pct"].iloc[0] == 25.0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = detect_missing_values(df)
        assert all(result["nan_pct"] == 0.0)


# -----------------------------------------------------------------------
# Forward fill tests
# -----------------------------------------------------------------------

class TestForwardFillWithinMission:
    def test_basic_fill(self):
        s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan])
        result = forward_fill_within_mission(s)
        expected = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0])
        pd.testing.assert_series_equal(result, expected)

    def test_leading_nan_stays(self):
        """NaN at start with no prior value stays NaN"""
        s = pd.Series([np.nan, 1.0, 2.0])
        result = forward_fill_within_mission(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 1.0

    def test_no_nans(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = forward_fill_within_mission(s)
        pd.testing.assert_series_equal(result, s)

    def test_all_nans(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = forward_fill_within_mission(s)
        assert all(result.isna())


# -----------------------------------------------------------------------
# Forward fill stop columns tests
# -----------------------------------------------------------------------

class TestForwardFillStopColumns:
    def test_fills_passengers(self):
        """Passenger count forward-filled between stops."""
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, np.nan, 15.0, np.nan, np.nan, 25.0, np.nan],
            "itcs_busRoute": ["-", "-", "83", "-", "-", "83", "-"],
            "itcs_stopName": ["-", "-", "StopA", "-", "-", "StopB", "-"],
        })
        result = forward_fill_stop_columns(df)
        # Before first stop: still NaN
        assert np.isnan(result["itcs_numberOfPassengers"].iloc[0])
        assert np.isnan(result["itcs_numberOfPassengers"].iloc[1])
        # At and after first stop, before second stop: 15
        assert result["itcs_numberOfPassengers"].iloc[2] == 15.0
        assert result["itcs_numberOfPassengers"].iloc[3] == 15.0
        assert result["itcs_numberOfPassengers"].iloc[4] == 15.0
        # At and after second stop: 25
        assert result["itcs_numberOfPassengers"].iloc[5] == 25.0
        assert result["itcs_numberOfPassengers"].iloc[6] == 25.0

    def test_fills_route(self):
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, 10.0, np.nan],
            "itcs_busRoute": ["-", "83", "-"],
            "itcs_stopName": ["-", "StopA", "-"],
        })
        result = forward_fill_stop_columns(df)
        assert result["itcs_busRoute"].iloc[0] == "-"   # before first stop stays "-"
        assert result["itcs_busRoute"].iloc[1] == "83"
        assert result["itcs_busRoute"].iloc[2] == "83"   # forward-filled

    def test_fills_stop_name(self):
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, 10.0, np.nan, 20.0, np.nan],
            "itcs_busRoute": ["-", "83", "-", "83", "-"],
            "itcs_stopName": ["-", "Alpha", "-", "Beta", "-"],
        })
        result = forward_fill_stop_columns(df)
        assert result["itcs_stopName"].iloc[2] == "Alpha"
        assert result["itcs_stopName"].iloc[4] == "Beta"

    def test_does_not_modify_input(self):
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, 10.0, np.nan],
            "itcs_busRoute": ["-", "83", "-"],
            "itcs_stopName": ["-", "StopA", "-"],
        })
        original_pax = df["itcs_numberOfPassengers"].copy()
        _ = forward_fill_stop_columns(df)
        pd.testing.assert_series_equal(df["itcs_numberOfPassengers"], original_pax)

    def test_preserves_other_columns(self):
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, 10.0],
            "itcs_busRoute": ["-", "83"],
            "itcs_stopName": ["-", "StopA"],
            "temperature": [283.15, 284.0],
        })
        result = forward_fill_stop_columns(df)
        assert list(result["temperature"]) == [283.15, 284.0]

    def test_no_stop_events(self):
        """All NaN passengers should remain NaN."""
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, np.nan, np.nan],
            "itcs_busRoute": ["-", "-", "-"],
            "itcs_stopName": ["-", "-", "-"],
        })
        result = forward_fill_stop_columns(df)
        assert result["itcs_numberOfPassengers"].isna().all()

    def test_drops_pre_first_stop_rows(self):
        """Rows before first stop should have NaN passengers (not filled)."""
        df = pd.DataFrame({
            "itcs_numberOfPassengers": [np.nan, np.nan, np.nan, 5.0, np.nan],
            "itcs_busRoute": ["-", "-", "-", "83", "-"],
            "itcs_stopName": ["-", "-", "-", "StopA", "-"],
        })
        result = forward_fill_stop_columns(df)
        assert result["itcs_numberOfPassengers"].iloc[0:3].isna().all()
        assert result["itcs_numberOfPassengers"].iloc[3] == 5.0
        assert result["itcs_numberOfPassengers"].iloc[4] == 5.0


# -----------------------------------------------------------------------
# Apply unit conversions tests
# -----------------------------------------------------------------------

class TestApplyUnitConversions:
    def test_creates_temp_c(self, mock_mission_df):
        result = apply_unit_conversions(mock_mission_df)
        assert "temp_C" in result.columns
        assert np.isclose(result["temp_C"].iloc[0], 10.0)

    def test_creates_lat_deg(self, mock_mission_df):
        result = apply_unit_conversions(mock_mission_df)
        assert "lat_deg" in result.columns
        assert np.isclose(result["lat_deg"].iloc[0], 47.37, atol=0.1)

    def test_creates_lon_deg(self, mock_mission_df):
        result = apply_unit_conversions(mock_mission_df)
        assert "lon_deg" in result.columns
        assert np.isclose(result["lon_deg"].iloc[0], 8.54, atol=0.1)

    def test_creates_speed_kmh(self, mock_mission_df):
        result = apply_unit_conversions(mock_mission_df)
        assert "speed_kmh" in result.columns
        # First 5 rows have speed 0 m/s -> 0 km/h
        assert result["speed_kmh"].iloc[0] == 0.0
        # Row 9 has speed 12.0 m/s -> 43.2 km/h
        assert np.isclose(result["speed_kmh"].iloc[9], 43.2)

    def test_does_not_modify_input(self, mock_mission_df):
        original_cols = set(mock_mission_df.columns)
        _ = apply_unit_conversions(mock_mission_df)
        assert set(mock_mission_df.columns) == original_cols

    def test_handles_missing_columns(self):
        """If a column is missing, skip that conversion without error"""
        df = pd.DataFrame({"temperature_ambient": [283.15]})
        result = apply_unit_conversions(df)
        assert "temp_C" in result.columns
        assert "lat_deg" not in result.columns
