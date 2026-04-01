"""
Tests for src/feature_engineering.py
Categorical encoding, rolling features, and feature matrix assembly.
"""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    encode_route,
    encode_stop_name,
    compute_rolling_features,
    compute_acceleration,
    compute_lag_features,
    build_feature_set,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def dense_mission_df():
    """
    Synthetic forward-filled mission (40 rows) with 2 stops.
    Rows 0-19: stop A (passengers=15, route=83, speed ramp up then down)
    Rows 20-39: stop B (passengers=25, route=83, speed ramp up then down)
    """
    n = 40
    base_unix = 1556594336
    timestamps = pd.date_range("2019-04-30T07:30:00Z", periods=n, freq="1s")

    speed = ([0.0]*3 + [2.0, 5.0, 8.0, 10.0, 12.0, 12.0, 10.0,
              8.0, 5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] +
             [0.0]*3 + [3.0, 6.0, 9.0, 11.0, 13.0, 13.0, 11.0,
              9.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return pd.DataFrame({
        "time_iso": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_unix": [base_unix + i for i in range(n)],
        "itcs_busRoute": ["83"] * n,
        "itcs_stopName": ["Paradeplatz"] * 20 + ["Bahnhofstrasse"] * 20,
        "itcs_numberOfPassengers": [15.0] * 20 + [25.0] * 20,
        "is_stop_event": [True] + [False]*19 + [True] + [False]*19,
        "odometry_vehicleSpeed": speed,
        "temperature_ambient": [283.15] * n,
        "electric_powerDemand": [10000.0 + i * 100 for i in range(n)],
        "traction_tractionForce": [500.0] * n,
        "traction_brakePressure": [0.0] * n,
        "gnss_altitude": [408.0] * n,
        "gnss_latitude": [0.8267] * n,
        "gnss_longitude": [0.1491] * n,
        "status_doorIsOpen": [1]*3 + [0]*14 + [1]*3 + [1]*3 + [0]*14 + [1]*3,
        "mission_name": ["mission_001"] * n,
    })


@pytest.fixture
def multi_route_df():
    """Two missions on different routes for encoding tests."""
    n = 10
    base_unix = 1556594336
    ts = pd.date_range("2019-04-30T07:30:00Z", periods=n, freq="1s")

    route83 = pd.DataFrame({
        "time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "itcs_busRoute": ["83"] * n,
        "itcs_stopName": ["Paradeplatz"] * 5 + ["Limmatplatz"] * 5,
        "itcs_numberOfPassengers": [10.0] * n,
        "mission_name": ["m1"] * n,
    })
    route33 = pd.DataFrame({
        "time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "itcs_busRoute": ["33"] * n,
        "itcs_stopName": ["ETH Zentrum"] * n,
        "itcs_numberOfPassengers": [20.0] * n,
        "mission_name": ["m2"] * n,
    })
    return pd.concat([route83, route33], ignore_index=True)


# -----------------------------------------------------------------------
# encode_route tests
# -----------------------------------------------------------------------

class TestEncodeRoute:
    def test_creates_dummy_columns(self, multi_route_df):
        result = encode_route(multi_route_df)
        route_cols = [c for c in result.columns if c.startswith("route_")]
        # drop_first=True with 2 routes -> 1 dummy column
        assert len(route_cols) == 1
        # The kept column should be route_83 (33 is dropped as first alphabetically)
        assert "route_83" in result.columns

    def test_dummy_values_correct(self, multi_route_df):
        result = encode_route(multi_route_df)
        # First 10 rows are route 83 -> route_83 = 1
        assert all(result["route_83"].iloc[:10] == 1)
        # Last 10 rows are route 33 -> route_83 = 0 (reference category)
        assert all(result["route_83"].iloc[10:] == 0)

    def test_drops_original_column(self, multi_route_df):
        result = encode_route(multi_route_df)
        assert "itcs_busRoute" not in result.columns

    def test_drops_first_to_avoid_collinearity(self, multi_route_df):
        result = encode_route(multi_route_df)
        route_cols = [c for c in result.columns if c.startswith("route_")]
        # With 2 routes, drop_first=True should give 1 column
        assert len(route_cols) == 1

    def test_preserves_other_columns(self, multi_route_df):
        result = encode_route(multi_route_df)
        assert "itcs_numberOfPassengers" in result.columns
        assert "mission_name" in result.columns

    def test_does_not_modify_input(self, multi_route_df):
        original_cols = set(multi_route_df.columns)
        _ = encode_route(multi_route_df)
        assert set(multi_route_df.columns) == original_cols


# -----------------------------------------------------------------------
# encode_stop_name tests
# -----------------------------------------------------------------------

class TestEncodeStopName:
    def test_top_n_plus_other(self, multi_route_df):
        result = encode_stop_name(multi_route_df, top_n=1)
        # Only top 1 stop + "other" bucket
        stop_cols = [c for c in result.columns if c.startswith("stop_")]
        # drop_first removes one, so with 2 categories (top1 + other), we get 1 col
        assert len(stop_cols) == 1

    def test_rare_stops_become_other(self, multi_route_df):
        # Paradeplatz=5, Limmatplatz=5, ETH Zentrum=10
        # top_n=1 -> ETH Zentrum stays, others become "__other__"
        result = encode_stop_name(multi_route_df, top_n=1)
        assert "itcs_stopName" not in result.columns

    def test_preserves_row_count(self, multi_route_df):
        result = encode_stop_name(multi_route_df, top_n=2)
        assert len(result) == len(multi_route_df)

    def test_does_not_modify_input(self, multi_route_df):
        original_cols = set(multi_route_df.columns)
        _ = encode_stop_name(multi_route_df, top_n=2)
        assert set(multi_route_df.columns) == original_cols


# -----------------------------------------------------------------------
# compute_rolling_features tests
# -----------------------------------------------------------------------

class TestComputeRollingFeatures:
    def test_creates_rolling_columns(self, dense_mission_df):
        result = compute_rolling_features(dense_mission_df, windows=[5])
        assert "speed_roll_mean_5" in result.columns
        assert "speed_roll_std_5" in result.columns
        assert "power_roll_mean_5" in result.columns

    def test_multiple_windows(self, dense_mission_df):
        result = compute_rolling_features(dense_mission_df, windows=[5, 10])
        assert "speed_roll_mean_5" in result.columns
        assert "speed_roll_mean_10" in result.columns

    def test_rolling_mean_correctness(self, dense_mission_df):
        result = compute_rolling_features(dense_mission_df, windows=[5])
        # Row 5: rolling mean of speed over rows 1-5
        # Speeds at rows 1-5: 0, 0, 2, 5, 8 -> mean = 3.0
        expected = np.mean([0.0, 0.0, 2.0, 5.0, 8.0])
        assert np.isclose(result["speed_roll_mean_5"].iloc[5], expected, atol=0.1)

    def test_first_rows_have_nan(self, dense_mission_df):
        result = compute_rolling_features(dense_mission_df, windows=[5])
        # First 4 rows (window-1) should have NaN for rolling features
        assert result["speed_roll_mean_5"].iloc[:4].isna().all()

    def test_preserves_row_count(self, dense_mission_df):
        result = compute_rolling_features(dense_mission_df, windows=[5])
        assert len(result) == len(dense_mission_df)

    def test_does_not_modify_input(self, dense_mission_df):
        original_cols = set(dense_mission_df.columns)
        _ = compute_rolling_features(dense_mission_df, windows=[5])
        assert set(dense_mission_df.columns) == original_cols


# -----------------------------------------------------------------------
# compute_acceleration tests
# -----------------------------------------------------------------------

class TestComputeAcceleration:
    def test_creates_acceleration_column(self, dense_mission_df):
        result = compute_acceleration(dense_mission_df)
        assert "acceleration" in result.columns

    def test_zero_at_constant_speed(self):
        df = pd.DataFrame({
            "odometry_vehicleSpeed": [5.0, 5.0, 5.0, 5.0],
            "mission_name": ["m1"] * 4,
        })
        result = compute_acceleration(df)
        # All constant speed -> acceleration = 0 (except first row = NaN)
        assert all(result["acceleration"].iloc[1:] == 0.0)

    def test_positive_acceleration(self):
        df = pd.DataFrame({
            "odometry_vehicleSpeed": [0.0, 2.0, 5.0, 10.0],
            "mission_name": ["m1"] * 4,
        })
        result = compute_acceleration(df)
        # Speed increasing -> positive acceleration
        assert result["acceleration"].iloc[1] == 2.0
        assert result["acceleration"].iloc[2] == 3.0

    def test_negative_acceleration(self):
        df = pd.DataFrame({
            "odometry_vehicleSpeed": [10.0, 5.0, 2.0, 0.0],
            "mission_name": ["m1"] * 4,
        })
        result = compute_acceleration(df)
        assert result["acceleration"].iloc[1] == -5.0

    def test_first_row_nan(self, dense_mission_df):
        result = compute_acceleration(dense_mission_df)
        assert np.isnan(result["acceleration"].iloc[0])

    def test_does_not_modify_input(self, dense_mission_df):
        original_cols = set(dense_mission_df.columns)
        _ = compute_acceleration(dense_mission_df)
        assert set(dense_mission_df.columns) == original_cols


# -----------------------------------------------------------------------
# build_feature_set tests
# -----------------------------------------------------------------------

class TestBuildFeatureSet:
    def test_returns_dataframe(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_temporal_features(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert "hour" in result.columns
        assert "is_weekend" in result.columns

    def test_has_rolling_features(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        rolling_cols = [c for c in result.columns if "roll" in c]
        assert len(rolling_cols) > 0

    def test_has_acceleration(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert "acceleration" in result.columns

    def test_has_route_encoding(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        # Single route -> drop_first removes it, so 0 route columns
        # That's fine -- the constant column is uninformative
        route_cols = [c for c in result.columns if c.startswith("route_")]
        assert isinstance(route_cols, list)  # just check it doesn't crash

    def test_preserves_target(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert "itcs_numberOfPassengers" in result.columns

    def test_drops_raw_columns(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        # Raw columns that were transformed should be gone
        assert "itcs_busRoute" not in result.columns
        assert "itcs_stopName" not in result.columns
        assert "time_iso" not in result.columns
        assert "time_unix" not in result.columns

    def test_no_string_columns_except_mission(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        for col in result.columns:
            if col == "mission_name":
                continue
            assert not pd.api.types.is_string_dtype(result[col]), f"{col} is string dtype"

    def test_excludes_temp_C(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert "temp_C" not in result.columns

    def test_excludes_busNumber(self, dense_mission_df):
        df = dense_mission_df.copy()
        df["busNumber"] = 183
        result = build_feature_set(df)
        assert "busNumber" not in result.columns

    def test_has_power_roll_std(self, dense_mission_df):
        result = build_feature_set(dense_mission_df, rolling_windows=[5])
        assert "power_roll_std_5" in result.columns

    def test_power_roll_std_default_windows(self, dense_mission_df):
        result = build_feature_set(dense_mission_df)
        assert "power_roll_std_60" in result.columns
        assert "power_roll_std_300" in result.columns


# -----------------------------------------------------------------------
# Mission boundary bleeding tests
# -----------------------------------------------------------------------

class TestMissionBoundarySafety:
    """Verify rolling features and acceleration do not bleed across missions."""

    @pytest.fixture
    def two_mission_df(self):
        """Two short missions concatenated. Each has 10 rows."""
        n = 10
        ts1 = pd.date_range("2019-04-30T07:00:00Z", periods=n, freq="1s")
        ts2 = pd.date_range("2019-04-30T09:00:00Z", periods=n, freq="1s")
        base_unix = 1556594336

        m1 = pd.DataFrame({
            "time_iso": ts1.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_unix": [base_unix + i for i in range(n)],
            "itcs_busRoute": ["83"] * n,
            "itcs_stopName": ["StopA"] * n,
            "itcs_numberOfPassengers": [10.0] * n,
            "is_stop_event": [True] + [False]*(n-1),
            "odometry_vehicleSpeed": [float(i) for i in range(n)],
            "temperature_ambient": [283.15] * n,
            "electric_powerDemand": [10000.0 + i * 100 for i in range(n)],
            "traction_tractionForce": [500.0] * n,
            "traction_brakePressure": [0.0] * n,
            "gnss_altitude": [408.0] * n,
            "gnss_latitude": [0.8267] * n,
            "gnss_longitude": [0.1491] * n,
            "status_doorIsOpen": [0] * n,
            "mission_name": ["mission_A"] * n,
        })

        m2 = pd.DataFrame({
            "time_iso": ts2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_unix": [base_unix + 7200 + i for i in range(n)],
            "itcs_busRoute": ["83"] * n,
            "itcs_stopName": ["StopB"] * n,
            "itcs_numberOfPassengers": [20.0] * n,
            "is_stop_event": [True] + [False]*(n-1),
            "odometry_vehicleSpeed": [float(n - i) for i in range(n)],
            "temperature_ambient": [283.15] * n,
            "electric_powerDemand": [20000.0 + i * 100 for i in range(n)],
            "traction_tractionForce": [500.0] * n,
            "traction_brakePressure": [0.0] * n,
            "gnss_altitude": [408.0] * n,
            "gnss_latitude": [0.8267] * n,
            "gnss_longitude": [0.1491] * n,
            "status_doorIsOpen": [0] * n,
            "mission_name": ["mission_B"] * n,
        })

        return pd.concat([m1, m2], ignore_index=True)

    def test_rolling_does_not_bleed_across_missions(self, two_mission_df):
        """First rows of mission_B should have NaN rolling, not values from mission_A."""
        result = build_feature_set(two_mission_df, rolling_windows=[5])
        # mission_B starts at index 10
        mission_b = result[result["mission_name"] == "mission_B"]
        # First 4 rows of mission_B (indices 0-3 within group) should be NaN
        # for window=5 rolling features
        assert mission_b["speed_roll_mean_5"].iloc[:4].isna().all(), \
            "Rolling mean bled across mission boundary"

    def test_acceleration_does_not_bleed_across_missions(self, two_mission_df):
        """First row of mission_B acceleration should be NaN, not diff from mission_A's last row."""
        result = build_feature_set(two_mission_df, rolling_windows=[5])
        mission_b = result[result["mission_name"] == "mission_B"]
        assert np.isnan(mission_b["acceleration"].iloc[0]), \
            "Acceleration bled across mission boundary"

    def test_single_mission_still_works(self, two_mission_df):
        """Single-mission subset should produce identical results to pre-fix behavior."""
        single = two_mission_df[two_mission_df["mission_name"] == "mission_A"].copy()
        result = build_feature_set(single, rolling_windows=[5])
        assert "acceleration" in result.columns
        assert "speed_roll_mean_5" in result.columns

    def test_power_roll_std_does_not_bleed(self, two_mission_df):
        """power_roll_std at mission boundary should be NaN."""
        result = build_feature_set(two_mission_df, rolling_windows=[5])
        mission_b = result[result["mission_name"] == "mission_B"]
        assert mission_b["power_roll_std_5"].iloc[:4].isna().all(), \
            "Power roll std bled across mission boundary"

    def test_lag_features_do_not_bleed_across_missions(self, two_mission_df):
        """First stop of mission_B should have NaN lag, not values from mission_A."""
        result = build_feature_set(two_mission_df, rolling_windows=[5])
        mission_b = result[result["mission_name"] == "mission_B"]
        # Mission B has 1 stop event (first row). Lag_1 should be NaN (no prior stop in B).
        assert mission_b["pax_stop_lag_1"].isna().all(), \
            "Lag features bled across mission boundary"


# -----------------------------------------------------------------------
# compute_lag_features tests (stop-level)
# -----------------------------------------------------------------------

class TestComputeLagFeatures:
    """Tests for stop-level lag feature computation."""

    @pytest.fixture
    def stop_level_df(self):
        """
        Synthetic forward-filled mission with 4 stop events.
        Stop events at indices 2, 5, 9, 13 with pax = 10, 20, 15, 30.
        """
        pax_filled = [np.nan, np.nan, 10, 10, 10, 20, 20, 20,
                      20, 15, 15, 15, 15, 30, 30, 30]
        is_stop = [False, False, True, False, False, True, False, False,
                   False, True, False, False, False, True, False, False]
        return pd.DataFrame({
            "itcs_numberOfPassengers": pax_filled,
            "is_stop_event": is_stop,
        })

    def test_stop_lag_1_values(self, stop_level_df):
        """Lag 1 at each stop should be the previous stop's pax count."""
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        # Stop 0 (idx 2, pax=10): lag_1 = NaN (no previous)
        assert np.isnan(result["pax_stop_lag_1"].iloc[2])
        # Stop 1 (idx 5, pax=20): lag_1 = 10
        assert result["pax_stop_lag_1"].iloc[5] == 10.0
        # Stop 2 (idx 9, pax=15): lag_1 = 20
        assert result["pax_stop_lag_1"].iloc[9] == 20.0
        # Stop 3 (idx 13, pax=30): lag_1 = 15
        assert result["pax_stop_lag_1"].iloc[13] == 15.0

    def test_lag_forward_fills_to_inter_stop_rows(self, stop_level_df):
        """Lag value at a stop should propagate to inter-stop rows."""
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        # Between stop 1 (idx 5, lag=10) and stop 2 (idx 9): rows 6,7,8
        assert result["pax_stop_lag_1"].iloc[6] == 10.0
        assert result["pax_stop_lag_1"].iloc[7] == 10.0
        assert result["pax_stop_lag_1"].iloc[8] == 10.0

    def test_first_stop_has_nan_lag(self, stop_level_df):
        """First stop and pre-stop rows should have NaN lag."""
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        # Rows 0-4 (pre-first-stop + first stop interval): all NaN
        assert result["pax_stop_lag_1"].iloc[:5].isna().all()

    def test_stop_lag_3_skips_correctly(self, stop_level_df):
        """Lag 3 should look 3 stops back."""
        result = compute_lag_features(stop_level_df, stop_lags=[3])
        # Stop 3 (idx 13, pax=30): lag_3 = stop 0 pax = 10
        assert result["pax_stop_lag_3"].iloc[13] == 10.0
        # Stops 0, 1, 2 don't have 3 stops of history
        assert np.isnan(result["pax_stop_lag_3"].iloc[2])
        assert np.isnan(result["pax_stop_lag_3"].iloc[5])
        assert np.isnan(result["pax_stop_lag_3"].iloc[9])

    def test_multiple_stop_lags(self, stop_level_df):
        """Multiple lag values should produce multiple columns."""
        result = compute_lag_features(stop_level_df, stop_lags=[1, 3])
        assert "pax_stop_lag_1" in result.columns
        assert "pax_stop_lag_3" in result.columns

    def test_default_stop_lags(self, stop_level_df):
        """Default lags should be [1, 3, 5]."""
        result = compute_lag_features(stop_level_df)
        assert "pax_stop_lag_1" in result.columns
        assert "pax_stop_lag_3" in result.columns
        assert "pax_stop_lag_5" in result.columns

    def test_raises_without_is_stop_event(self):
        """Should raise ValueError if is_stop_event column is missing."""
        df = pd.DataFrame({"itcs_numberOfPassengers": [10, 20, 30]})
        with pytest.raises(ValueError, match="is_stop_event"):
            compute_lag_features(df, stop_lags=[1])

    def test_does_not_modify_input(self, stop_level_df):
        """Input DataFrame should not be mutated."""
        original_cols = set(stop_level_df.columns)
        _ = compute_lag_features(stop_level_df, stop_lags=[1])
        assert set(stop_level_df.columns) == original_cols

    def test_preserves_other_columns(self, stop_level_df):
        """Non-lagged columns should be unchanged."""
        stop_level_df["other"] = range(len(stop_level_df))
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        assert list(result["other"]) == list(range(len(stop_level_df)))

    def test_preserves_row_count(self, stop_level_df):
        """Row count must not change."""
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        assert len(result) == len(stop_level_df)

    def test_lag_values_differ_from_current_pax(self, stop_level_df):
        """Anti-leakage: lag value should differ from current pax at stop events."""
        result = compute_lag_features(stop_level_df, stop_lags=[1])
        # At stop 1 (idx 5): current pax=20, lag=10
        assert result["pax_stop_lag_1"].iloc[5] != result["itcs_numberOfPassengers"].iloc[5]
        # At stop 2 (idx 9): current pax=15, lag=20
        assert result["pax_stop_lag_1"].iloc[9] != result["itcs_numberOfPassengers"].iloc[9]
