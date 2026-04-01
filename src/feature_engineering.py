"""
Feature engineering: categorical encoding, rolling windows, acceleration,
and full feature matrix assembly from forward-filled mission data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import ROLLING_WINDOWS
from src.covid import build_covid_features
from src.preprocessing import extract_temporal_features, apply_unit_conversions


def encode_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode itcs_busRoute with drop_first=True.
    Drops original column. Prefixes dummy columns with 'route_'.
    """
    result = df.copy()
    dummies = pd.get_dummies(result["itcs_busRoute"], prefix="route", drop_first=True)
    # Ensure int type for dummy columns
    dummies = dummies.astype(int)
    result = pd.concat([result.drop(columns=["itcs_busRoute"]), dummies], axis=1)
    return result


def encode_stop_name(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    One-hot encode stop names, keeping only the top_n most frequent.
    Less frequent stops are bucketed into '__other__'.
    Drops original column. Prefixes dummy columns with 'stop_'.
    """
    result = df.copy()
    top_stops = result["itcs_stopName"].value_counts().head(top_n).index.tolist()
    result["_stop_grouped"] = result["itcs_stopName"].where(
        result["itcs_stopName"].isin(top_stops), other="__other__"
    )
    dummies = pd.get_dummies(result["_stop_grouped"], prefix="stop", drop_first=True)
    dummies = dummies.astype(int)
    result = pd.concat(
        [result.drop(columns=["itcs_stopName", "_stop_grouped"]), dummies], axis=1
    )
    return result


def compute_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling mean and std for speed and power demand.
    Window sizes are in number of rows (= seconds at 1Hz).

    Creates columns: speed_roll_mean_{w}, speed_roll_std_{w},
                     power_roll_mean_{w}, power_roll_std_{w} for each window w.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    result = df.copy()

    for w in windows:
        if "odometry_vehicleSpeed" in result.columns:
            result[f"speed_roll_mean_{w}"] = (
                result["odometry_vehicleSpeed"].rolling(w, min_periods=w).mean()
            )
            result[f"speed_roll_std_{w}"] = (
                result["odometry_vehicleSpeed"].rolling(w, min_periods=w).std()
            )
        if "electric_powerDemand" in result.columns:
            result[f"power_roll_mean_{w}"] = (
                result["electric_powerDemand"].rolling(w, min_periods=w).mean()
            )
            result[f"power_roll_std_{w}"] = (
                result["electric_powerDemand"].rolling(w, min_periods=w).std()
            )

    return result


def compute_lag_features(
    df: pd.DataFrame,
    stop_lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Create lagged passenger counts at the STOP-EVENT level.

    Instead of shifting by N rows (which on forward-filled data returns
    the current value ~95% of the time), this computes the passenger count
    at the Nth previous stop event, then forward-fills the lag value to
    all 1Hz rows in the current stop interval.

    Requires an 'is_stop_event' boolean column (added by
    forward_fill_stop_columns) to identify stop boundaries.

    Default stop_lags: [1, 3, 5] (previous stop, 3 stops ago, 5 stops ago).
    Output columns: pax_stop_lag_1, pax_stop_lag_3, pax_stop_lag_5.
    """
    if stop_lags is None:
        stop_lags = [1, 3, 5]

    if "is_stop_event" not in df.columns:
        raise ValueError(
            "compute_lag_features requires 'is_stop_event' column. "
            "Ensure forward_fill_stop_columns() was called first."
        )

    result = df.copy()
    stop_mask = result["is_stop_event"]
    stop_rows = result.loc[stop_mask, "itcs_numberOfPassengers"]

    for lag in stop_lags:
        col = f"pax_stop_lag_{lag}"
        lagged = stop_rows.shift(lag)
        result[col] = np.nan
        result.loc[stop_mask, col] = lagged.values
        result[col] = result[col].ffill()

    return result


def compute_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute acceleration as the first difference of vehicle speed.
    At 1Hz sampling, diff = m/s^2.
    """
    result = df.copy()
    if "odometry_vehicleSpeed" in result.columns:
        result["acceleration"] = result["odometry_vehicleSpeed"].diff()
    return result


def build_feature_set(
    df: pd.DataFrame,
    top_n_stops: int = 20,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for forward-filled mission data.

    Steps:
    1. Unit conversions (K->C, rad->deg, m/s->km/h)
    2. Temporal features (hour, dayofweek, month, year, is_weekend, is_rush_hour)
    3. Rolling window features (speed, power)
    4. Acceleration
    5. Route one-hot encoding
    6. Stop name top-N encoding
    7. Drop raw/intermediate columns

    Preserves: itcs_numberOfPassengers (target), mission_name (for splitting).
    """
    result = df.copy()

    # Unit conversions
    result = apply_unit_conversions(result)

    # Temporal features
    result = extract_temporal_features(result)

    # COVID restriction features (Oxford Stringency Index, Switzerland)
    result = build_covid_features(result, time_col="time_iso")

    # Rolling features (per-mission to prevent boundary bleeding)
    if "mission_name" in result.columns:
        parts = []
        for _, group in result.groupby("mission_name", sort=False):
            parts.append(compute_rolling_features(group, windows=rolling_windows))
        result = pd.concat(parts)
    else:
        result = compute_rolling_features(result, windows=rolling_windows)

    # Acceleration (per-mission to prevent boundary bleeding)
    if "mission_name" in result.columns:
        parts = []
        for _, group in result.groupby("mission_name", sort=False):
            parts.append(compute_acceleration(group))
        result = pd.concat(parts)
    else:
        result = compute_acceleration(result)

    # Lag features on passenger count (per-mission to prevent boundary bleeding)
    if "mission_name" in result.columns:
        parts = []
        for _, group in result.groupby("mission_name", sort=False):
            parts.append(compute_lag_features(group))
        result = pd.concat(parts)
    else:
        result = compute_lag_features(result)

    # Categorical encoding
    if "itcs_busRoute" in result.columns:
        result = encode_route(result)
    if "itcs_stopName" in result.columns:
        result = encode_stop_name(result, top_n=top_n_stops)

    # Drop raw columns that have been transformed or are not features
    # Feature inclusion/exclusion decisions: see ML-EXPERIMENT_DESIGN.md
    drop_cols = [
        "time_iso", "time_unix",
        "gnss_latitude", "gnss_longitude",   # replaced by lat_deg, lon_deg
        "temperature_ambient",                # raw Kelvin, excluded entirely
        "odometry_vehicleSpeed",              # replaced by speed_kmh + rolling
        "temp_C",                             # EDA Fig 08: r=-0.011, no signal
        "busNumber",                          # bus identifier, not a predictor
        "is_stop_event",                      # pipeline marker, not a feature
    ]
    for col in drop_cols:
        if col in result.columns:
            result = result.drop(columns=[col])

    return result
