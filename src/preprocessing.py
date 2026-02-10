"""
Data preprocessing: unit conversions, feature extraction, missing data handling.
All functions are pure (input -> output), no side effects.
"""

import numpy as np
import pandas as pd
from src.config import KELVIN_OFFSET, RUSH_HOUR_AM, RUSH_HOUR_PM


def kelvin_to_celsius(series: pd.Series) -> pd.Series:
    """Convert a Series of temperatures from Kelvin to Celsius."""
    return series - KELVIN_OFFSET


def radians_to_degrees(series: pd.Series) -> pd.Series:
    """Convert a Series of angles from radians to degrees."""
    return np.degrees(series)


def mps_to_kmh(series: pd.Series) -> pd.Series:
    """Convert a Series of speeds from m/s to km/h."""
    return series * 3.6


def extract_temporal_features(df: pd.DataFrame, time_col: str = "time_iso") -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.

    Adds columns: hour, dayofweek, month, year, is_weekend, is_rush_hour
    Returns a new DataFrame (does not modify input).
    """
    result = df.copy()
    dt = pd.to_datetime(result[time_col])

    result["hour"] = dt.dt.hour
    result["dayofweek"] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
    result["month"] = dt.dt.month
    result["year"] = dt.dt.year
    result["is_weekend"] = result["dayofweek"].isin([5, 6])

    am_start, am_end = RUSH_HOUR_AM
    pm_start, pm_end = RUSH_HOUR_PM
    is_weekday = ~result["is_weekend"]
    is_am_rush = result["hour"].between(am_start, am_end - 1)
    is_pm_rush = result["hour"].between(pm_start, pm_end - 1)
    result["is_rush_hour"] = is_weekday & (is_am_rush | is_pm_rush)

    return result


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing value statistics for each column.
    Counts both NaN and dash ("-") placeholders.

    Returns DataFrame with columns: column, nan_count, nan_pct, dash_count, dash_pct
    """
    records = []
    for col in df.columns:
        nan_count = int(df[col].isnull().sum())
        nan_pct = nan_count / len(df) * 100 if len(df) > 0 else 0.0

        if pd.api.types.is_string_dtype(df[col]):
            dash_count = int((df[col] == "-").sum())
        else:
            dash_count = 0
        dash_pct = dash_count / len(df) * 100 if len(df) > 0 else 0.0

        records.append({
            "column": col,
            "nan_count": nan_count,
            "nan_pct": round(nan_pct, 2),
            "dash_count": dash_count,
            "dash_pct": round(dash_pct, 2),
        })

    return pd.DataFrame(records)


def forward_fill_within_mission(series: pd.Series) -> pd.Series:
    """
    Forward-fill NaN values within a single mission's time series.
    Returns a new Series.
    """
    return series.ffill()


def forward_fill_stop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill ITCS stop columns within a mission.

    Physically: passenger count is constant between stops (passengers only
    board/alight at stops). Forward-filling propagates the last known count
    until the next stop event updates it.

    Fills: itcs_numberOfPassengers, itcs_busRoute, itcs_stopName.
    Rows before the first stop remain NaN/"-" (no prior data to fill from).
    Returns a new DataFrame (does not modify input).
    """
    result = df.copy()

    if "itcs_numberOfPassengers" in result.columns:
        result["itcs_numberOfPassengers"] = result["itcs_numberOfPassengers"].ffill()

    for col in ["itcs_busRoute", "itcs_stopName"]:
        if col in result.columns:
            with pd.option_context("future.no_silent_downcasting", True):
                result[col] = (
                    result[col]
                    .where(result[col] != "-", np.nan)
                    .ffill()
                    .fillna("-")
                )

    return result


def apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all standard unit conversions to a mission DataFrame.
    Adds new columns; does not modify originals.

    Adds: temp_C, lat_deg, lon_deg, speed_kmh
    """
    result = df.copy()

    if "temperature_ambient" in result.columns:
        result["temp_C"] = kelvin_to_celsius(result["temperature_ambient"])

    if "gnss_latitude" in result.columns:
        result["lat_deg"] = radians_to_degrees(result["gnss_latitude"])

    if "gnss_longitude" in result.columns:
        result["lon_deg"] = radians_to_degrees(result["gnss_longitude"])

    if "odometry_vehicleSpeed" in result.columns:
        result["speed_kmh"] = mps_to_kmh(result["odometry_vehicleSpeed"])

    return result
