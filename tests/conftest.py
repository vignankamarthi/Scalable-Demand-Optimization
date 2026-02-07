"""
Shared test fixtures: mock DataFrames that mimic ZTBus schema.
No real data is loaded -- all synthetic, deterministic, and fast.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_mission_df():
    """
    A small synthetic mission DataFrame (20 rows) with known values.
    Mimics a 20-second window of a bus at a stop.
    """
    n = 20
    base_unix = 1556594336  # 2019-04-30 03:18:56 UTC (a Tuesday)
    timestamps = pd.date_range("2019-04-30T03:18:56Z", periods=n, freq="1s")

    return pd.DataFrame({
        "time_iso": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_unix": [base_unix + i for i in range(n)],
        "gnss_latitude": [0.8267] * n,           # ~47.37 degrees
        "gnss_longitude": [0.1491] * n,           # ~8.54 degrees
        "gnss_altitude": [408.0] * n,
        "itcs_busRoute": ["83"] * n,
        "itcs_numberOfPassengers": [10, 10, 12, 12, 15, 15, 15, 20, 20, 20,
                                     25, 25, 30, 30, 30, 28, 25, 20, 15, 10],
        "itcs_stopName": ["Paradeplatz"] * 10 + ["Bahnhofstrasse"] * 10,
        "odometry_vehicleSpeed": [0.0] * 5 + [2.0, 5.0, 8.0, 10.0, 12.0,
                                               12.0, 11.0, 10.0, 8.0, 5.0,
                                               3.0, 1.0, 0.0, 0.0, 0.0],
        "status_doorIsOpen": [True]*5 + [False]*10 + [True]*5,
        "status_gridIsAvailable": [True] * n,
        "status_haltBrakeIsActive": [True]*5 + [False]*10 + [True]*5,
        "status_parkBrakeIsActive": [False] * n,
        "temperature_ambient": [283.15] * n,       # 10.0 C exactly
        "electric_powerDemand": [5000.0] * n,
        "traction_tractionForce": [0.0]*5 + [1000.0]*10 + [0.0]*5,
        "traction_brakePressure": [0.0] * n,
    })


@pytest.fixture
def mock_mission_with_nans(mock_mission_df):
    """
    Same as mock_mission_df but with realistic NaN patterns:
    - GNSS has gaps (rows 3-7)
    - Passenger count has one NaN (row 0)
    """
    df = mock_mission_df.copy()
    df.loc[3:7, "gnss_latitude"] = np.nan
    df.loc[3:7, "gnss_longitude"] = np.nan
    df.loc[0, "itcs_numberOfPassengers"] = np.nan
    return df


@pytest.fixture
def mock_mission_with_dashes(mock_mission_df):
    """
    Same as mock_mission_df but with dash placeholders in string columns.
    """
    df = mock_mission_df.copy()
    df.loc[0:4, "itcs_busRoute"] = "-"
    df.loc[0:2, "itcs_stopName"] = "-"
    return df


@pytest.fixture
def mock_metadata_df():
    """
    A small synthetic metadata DataFrame (5 missions).
    """
    return pd.DataFrame({
        "name": [
            "B183_2019-04-30_03-18-56_2019-04-30_08-44-20",
            "B183_2019-04-30_13-22-07_2019-04-30_17-54-02",
            "B183_2019-05-01_05-58-51_2019-05-01_22-32-30",
            "B208_2019-05-03_02-50-21_2019-05-03_05-53-20",
            "B208_2019-05-03_15-41-57_2019-05-03_23-06-24",
        ],
        "busNumber": [183, 183, 183, 208, 208],
        "startTime_iso": [
            "2019-04-30T03:18:56Z",
            "2019-04-30T13:22:07Z",
            "2019-05-01T05:58:51Z",
            "2019-05-03T02:50:21Z",
            "2019-05-03T15:41:57Z",
        ],
        "startTime_unix": [1556594336, 1556630527, 1556690331, 1556851821, 1556898117],
        "endTime_iso": [
            "2019-04-30T08:44:20Z",
            "2019-04-30T17:54:02Z",
            "2019-05-01T22:32:30Z",
            "2019-05-03T05:53:20Z",
            "2019-05-03T23:06:24Z",
        ],
        "endTime_unix": [1556613860, 1556646842, 1556749950, 1556862800, 1556924784],
        "drivenDistance": [77213.87, 59029.6, 240900.4, 42565.48, 120000.0],
        "busRoute": ["83", "31", "33", "83", "83"],
        "energyConsumption": [4.78e8, 4.02e8, 1.44e9, 2.82e8, 6.00e8],
        "itcs_numberOfPassengers_mean": [5.54, 33.11, 19.69, 1.69, 15.0],
        "itcs_numberOfPassengers_min": [0.0, 4.0, 0.0, 0.0, 2.0],
        "itcs_numberOfPassengers_max": [20.0, 74.0, 55.0, 8.0, 45.0],
        "status_gridIsAvailable_mean": [0.74, 0.86, 0.78, 0.77, 0.80],
        "temperature_ambient_mean": [282.38, 287.54, 288.75, 282.41, 285.0],
        "temperature_ambient_min": [281.15, 285.15, 280.15, 281.15, 283.15],
        "temperature_ambient_max": [293.15, 293.15, 294.15, 292.15, 290.15],
    })


@pytest.fixture
def mock_passenger_series():
    """
    A Series of passenger counts with known distribution for target variable testing.
    30 values: 10 low (0-5), 10 medium (10-15), 10 high (25-35).
    """
    low = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    medium = [10, 11, 11, 12, 12, 13, 13, 14, 14, 15]
    high = [25, 26, 27, 28, 29, 30, 31, 32, 33, 35]
    return pd.Series(low + medium + high, dtype=float)


@pytest.fixture
def weekend_mission_df():
    """
    A synthetic mission on a Saturday (2019-05-04) for weekend detection testing.
    """
    n = 10
    timestamps = pd.date_range("2019-05-04T10:00:00Z", periods=n, freq="1s")
    return pd.DataFrame({
        "time_iso": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "itcs_numberOfPassengers": [20.0] * n,
    })


@pytest.fixture
def rush_hour_mission_df():
    """
    A synthetic mission spanning rush hour boundaries for precise rush hour testing.
    4 observations: 6:59 (not rush), 7:00 (AM rush), 15:59 (not rush), 16:00 (PM rush)
    All on a Wednesday (2019-05-01).
    """
    return pd.DataFrame({
        "time_iso": [
            "2019-05-01T06:59:00Z",  # Wednesday 6:59 - NOT rush hour
            "2019-05-01T07:00:00Z",  # Wednesday 7:00 - AM rush
            "2019-05-01T15:59:00Z",  # Wednesday 15:59 - NOT rush hour
            "2019-05-01T16:00:00Z",  # Wednesday 16:00 - PM rush
        ],
    })
