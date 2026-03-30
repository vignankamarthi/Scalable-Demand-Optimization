"""
Time series analysis utilities: stop event extraction, stationarity testing,
and hourly aggregate construction.
"""
from __future__ import annotations

import warnings
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def extract_stop_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rows where itcs_numberOfPassengers is non-NaN.
    These are the actual stop events (before forward-fill).
    Returns a copy; does not modify input.
    """
    return df[df["itcs_numberOfPassengers"].notna()].copy()


def run_stationarity_tests(series: np.ndarray, name: str) -> dict:
    """
    Run ADF and KPSS stationarity tests on a numeric array.

    ADF H0: unit root exists (non-stationary). Reject => stationary.
    KPSS H0: series is stationary. Reject => non-stationary.

    Returns dict with test statistics, p-values, rejection booleans, and
    a 2x2 verdict string.
    """
    result = {"name": name}

    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag="AIC")
    result["adf_stat"] = adf_stat
    result["adf_p"] = adf_p
    result["adf_reject"] = bool(adf_p < 0.05)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*p-value is smaller than.*")
        warnings.filterwarnings("ignore", message=".*p-value is greater than.*")
        kpss_stat, kpss_p, _, _ = kpss(series, regression="c", nlags="auto")
    result["kpss_stat"] = kpss_stat
    result["kpss_p"] = kpss_p
    result["kpss_reject"] = bool(kpss_p < 0.05)

    if result["adf_reject"] and not result["kpss_reject"]:
        result["verdict"] = "STATIONARY"
    elif not result["adf_reject"] and result["kpss_reject"]:
        result["verdict"] = "NON-STATIONARY"
    elif result["adf_reject"] and result["kpss_reject"]:
        result["verdict"] = "TREND-STATIONARY"
    else:
        result["verdict"] = "INCONCLUSIVE"

    return result


def build_hourly_aggregate(
    meta: pd.DataFrame,
    data_dir: str,
    usecols: list[str],
) -> pd.Series:
    """
    Load missions, extract stop events, and compute mean passenger count
    per calendar hour across all missions.

    Returns a pd.Series indexed by hourly datetime, sorted chronologically.
    Empty Series if no data found.
    """
    records: list[dict] = []

    for _, row in meta.iterrows():
        fpath = os.path.join(data_dir, row["name"] + ".csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception:
            continue

        stops = extract_stop_events(df)
        if len(stops) == 0:
            continue

        stops["time_iso"] = pd.to_datetime(stops["time_iso"])
        stops["hour_bucket"] = stops["time_iso"].dt.floor("h")
        for ts, val in stops.groupby("hour_bucket")["itcs_numberOfPassengers"].mean().items():
            records.append({"timestamp": ts, "mean_pax": val})

    if not records:
        return pd.Series(dtype="float64")

    hourly_df = pd.DataFrame(records)
    hourly_agg = hourly_df.groupby("timestamp")["mean_pax"].mean().sort_index()
    return hourly_agg
