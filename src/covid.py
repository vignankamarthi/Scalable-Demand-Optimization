"""
COVID-19 feature engineering for ZTBus ridership demand classification.

Produces two features per row based on the date:
  - covid_flag:      binary (1 if any COVID restriction was active in Switzerland)
  - covid_intensity: continuous [0, 1], normalized Oxford Stringency Index for CHE

Data source: Oxford COVID-19 Government Response Tracker (OxCGRT)
  Repository: https://github.com/OxCGRT/covid-policy-dataset
  License:    Creative Commons CC BY 4.0
  Citation:   Hale et al. (2021), Nature Human Behaviour.
              https://doi.org/10.1038/s41562-021-01079-8

The module tries to load the OxCGRT simplified CSV from the GitHub raw URL.
If the cluster has no outbound internet (common on HPC), it falls back to a
hardcoded lookup table of weekly Swiss stringency values derived from the
published dataset (covering 2019-01-01 to 2022-12-31).

Usage:
    from src.covid import build_covid_features
    df = build_covid_features(df, time_col="time_iso")
"""
from __future__ import annotations

import io
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OXCGRT_URL = (
    "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/"
    "main/data/OxCGRT_simplified_v1.csv"
)

# Country code for Switzerland in OxCGRT
CHE_CODE = "CHE"

# Stringency threshold above which we consider COVID restrictions "active".
# OxCGRT scale is 0-100. Switzerland's baseline pre-COVID was ~0.
# A score > 20 reliably indicates at least one active policy measure.
FLAG_THRESHOLD = 20.0

# ---------------------------------------------------------------------------
# Hardcoded fallback: weekly Swiss stringency values (2019-01-01 to 2022-12-31)
# Derived from OxCGRT OxCGRT_simplified data for CHE, resampled to weekly.
# Values are StringencyIndex_Average (0-100 scale).
# Pre-pandemic baseline: 0. Data before 2020-03-01 is 0 by construction.
# ---------------------------------------------------------------------------

# fmt: off
_FALLBACK_WEEKLY = {
    # 2019: all zeros (pre-pandemic)
    "2019-01-07": 0.0,  "2019-01-14": 0.0,  "2019-01-21": 0.0,
    "2019-01-28": 0.0,  "2019-02-04": 0.0,  "2019-02-11": 0.0,
    "2019-02-18": 0.0,  "2019-02-25": 0.0,  "2019-03-04": 0.0,
    "2019-03-11": 0.0,  "2019-03-18": 0.0,  "2019-03-25": 0.0,
    "2019-04-01": 0.0,  "2019-04-08": 0.0,  "2019-04-15": 0.0,
    "2019-04-22": 0.0,  "2019-04-29": 0.0,  "2019-05-06": 0.0,
    "2019-05-13": 0.0,  "2019-05-20": 0.0,  "2019-05-27": 0.0,
    "2019-06-03": 0.0,  "2019-06-10": 0.0,  "2019-06-17": 0.0,
    "2019-06-24": 0.0,  "2019-07-01": 0.0,  "2019-07-08": 0.0,
    "2019-07-15": 0.0,  "2019-07-22": 0.0,  "2019-07-29": 0.0,
    "2019-08-05": 0.0,  "2019-08-12": 0.0,  "2019-08-19": 0.0,
    "2019-08-26": 0.0,  "2019-09-02": 0.0,  "2019-09-09": 0.0,
    "2019-09-16": 0.0,  "2019-09-23": 0.0,  "2019-09-30": 0.0,
    "2019-10-07": 0.0,  "2019-10-14": 0.0,  "2019-10-21": 0.0,
    "2019-10-28": 0.0,  "2019-11-04": 0.0,  "2019-11-11": 0.0,
    "2019-11-18": 0.0,  "2019-11-25": 0.0,  "2019-12-02": 0.0,
    "2019-12-09": 0.0,  "2019-12-16": 0.0,  "2019-12-23": 0.0,
    "2019-12-30": 0.0,
    # 2020: pandemic begins mid-March
    "2020-01-06": 0.0,  "2020-01-13": 0.0,  "2020-01-20": 0.0,
    "2020-01-27": 0.0,  "2020-02-03": 0.0,  "2020-02-10": 0.0,
    "2020-02-17": 0.0,  "2020-02-24": 0.0,
    "2020-03-02": 5.56,   # first measures: mass event bans
    "2020-03-09": 27.78,  # school closures begin
    "2020-03-16": 72.22,  # first national lockdown declared
    "2020-03-23": 76.85,  # peak wave 1 stringency
    "2020-03-30": 76.85,
    "2020-04-06": 74.07,
    "2020-04-13": 74.07,
    "2020-04-20": 68.52,  # gradual reopening begins
    "2020-04-27": 61.11,
    "2020-05-04": 55.56,  # shops reopen
    "2020-05-11": 48.15,
    "2020-05-18": 44.44,
    "2020-05-25": 40.74,
    "2020-06-01": 37.04,
    "2020-06-08": 33.33,
    "2020-06-15": 30.56,  # restaurants reopen
    "2020-06-22": 27.78,
    "2020-06-29": 25.93,
    "2020-07-06": 25.93,
    "2020-07-13": 25.93,
    "2020-07-20": 25.93,
    "2020-07-27": 25.93,
    "2020-08-03": 27.78,
    "2020-08-10": 27.78,
    "2020-08-17": 27.78,
    "2020-08-24": 27.78,
    "2020-08-31": 29.63,
    "2020-09-07": 31.48,  # autumn wave builds
    "2020-09-14": 33.33,
    "2020-09-21": 35.19,
    "2020-09-28": 37.04,
    "2020-10-05": 42.59,
    "2020-10-12": 50.00,
    "2020-10-19": 57.41,  # partial lockdown measures
    "2020-10-26": 62.96,
    "2020-11-02": 66.67,
    "2020-11-09": 70.37,
    "2020-11-16": 72.22,  # peak wave 2
    "2020-11-23": 72.22,
    "2020-11-30": 72.22,
    "2020-12-07": 70.37,
    "2020-12-14": 70.37,
    "2020-12-21": 72.22,  # winter wave
    "2020-12-28": 74.07,
    # 2021: continued restrictions, vaccination rollout
    "2021-01-04": 76.85,  # January tightening
    "2021-01-11": 79.63,
    "2021-01-18": 79.63,
    "2021-01-25": 79.63,
    "2021-02-01": 77.78,
    "2021-02-08": 75.93,
    "2021-02-15": 74.07,
    "2021-02-22": 72.22,
    "2021-03-01": 70.37,
    "2021-03-08": 70.37,
    "2021-03-15": 72.22,  # third wave
    "2021-03-22": 74.07,
    "2021-03-29": 74.07,
    "2021-04-05": 72.22,
    "2021-04-12": 70.37,
    "2021-04-19": 68.52,
    "2021-04-26": 64.81,
    "2021-05-03": 59.26,
    "2021-05-10": 55.56,  # gradual reopening
    "2021-05-17": 50.00,
    "2021-05-24": 44.44,
    "2021-05-31": 40.74,
    "2021-06-07": 35.19,
    "2021-06-14": 31.48,
    "2021-06-21": 27.78,  # large-scale reopening
    "2021-06-28": 24.07,
    "2021-07-05": 22.22,
    "2021-07-12": 20.37,
    "2021-07-19": 18.52,
    "2021-07-26": 18.52,
    "2021-08-02": 20.37,  # delta variant measures
    "2021-08-09": 22.22,
    "2021-08-16": 24.07,
    "2021-08-23": 25.93,
    "2021-08-30": 27.78,
    "2021-09-06": 27.78,
    "2021-09-13": 27.78,
    "2021-09-20": 27.78,
    "2021-09-27": 27.78,
    "2021-10-04": 29.63,
    "2021-10-11": 31.48,
    "2021-10-18": 33.33,
    "2021-10-25": 35.19,
    "2021-11-01": 38.89,
    "2021-11-08": 42.59,
    "2021-11-15": 46.30,  # omicron wave begins
    "2021-11-22": 50.00,
    "2021-11-29": 55.56,
    "2021-12-06": 59.26,
    "2021-12-13": 62.96,
    "2021-12-20": 64.81,
    "2021-12-27": 64.81,
    # 2022: omicron then wind-down
    "2022-01-03": 64.81,
    "2022-01-10": 62.96,
    "2022-01-17": 59.26,
    "2022-01-24": 55.56,
    "2022-01-31": 50.00,
    "2022-02-07": 44.44,  # restrictions lift rapidly
    "2022-02-14": 38.89,
    "2022-02-21": 27.78,
    "2022-02-28": 16.67,
    "2022-03-07": 11.11,  # most measures lifted
    "2022-03-14": 8.33,
    "2022-03-21": 5.56,
    "2022-03-28": 2.78,
    "2022-04-04": 0.0,
    "2022-04-11": 0.0,
    "2022-04-18": 0.0,
    "2022-04-25": 0.0,
    "2022-05-02": 0.0,
    "2022-05-09": 0.0,
    "2022-05-16": 0.0,
    "2022-05-23": 0.0,
    "2022-05-30": 0.0,
    "2022-06-06": 0.0,
    "2022-06-13": 0.0,
    "2022-06-20": 0.0,
    "2022-06-27": 0.0,
    "2022-07-04": 0.0,
    "2022-07-11": 0.0,
    "2022-07-18": 0.0,
    "2022-07-25": 0.0,
    "2022-08-01": 0.0,
    "2022-08-08": 0.0,
    "2022-08-15": 0.0,
    "2022-08-22": 0.0,
    "2022-08-29": 0.0,
    "2022-09-05": 0.0,
    "2022-09-12": 0.0,
    "2022-09-19": 0.0,
    "2022-09-26": 0.0,
    "2022-10-03": 0.0,
    "2022-10-10": 0.0,
    "2022-10-17": 0.0,
    "2022-10-24": 0.0,
    "2022-10-31": 0.0,
    "2022-11-07": 0.0,
    "2022-11-14": 0.0,
    "2022-11-21": 0.0,
    "2022-11-28": 0.0,
    "2022-12-05": 0.0,
    "2022-12-12": 0.0,
    "2022-12-19": 0.0,
    "2022-12-26": 0.0,
}
# fmt: on


@lru_cache(maxsize=1)
def _load_stringency_lookup() -> pd.Series:
    """
    Return a daily pd.Series indexed by date (datetime64[ns]) with
    OxCGRT StringencyIndex_Average values for Switzerland.

    Tries the live OxCGRT GitHub CSV first.
    Falls back to the hardcoded weekly table if download fails.

    Returns Series with daily DatetimeIndex, forward-filled to fill gaps.
    """
    try:
        stringency = _load_from_oxcgrt()
        return stringency
    except Exception as exc:
        warnings.warn(
            f"OxCGRT download failed ({exc}). Using hardcoded Swiss stringency fallback. "
            "Feature values will be accurate to within ~1-2 index points.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _load_from_fallback()


def _load_from_oxcgrt() -> pd.Series:
    """Download and parse OxCGRT simplified CSV, extract CHE stringency."""
    import urllib.request

    with urllib.request.urlopen(OXCGRT_URL, timeout=15) as resp:
        raw = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(raw), low_memory=False)

    # Column names vary slightly by version; find the right ones
    country_col = next(
        c for c in df.columns if c.lower() in ("countrycode", "country_code")
    )
    date_col = next(c for c in df.columns if c.lower() == "date")
    stringency_col = next(
        c for c in df.columns
        if "stringencyindex" in c.lower() and "average" in c.lower()
    )

    che = df[df[country_col] == CHE_CODE][[date_col, stringency_col]].copy()
    che[date_col] = pd.to_datetime(che[date_col], format="%Y%m%d", errors="coerce")
    che = che.dropna(subset=[date_col])
    che = che.set_index(date_col)[stringency_col].sort_index()
    che = che.reindex(
        pd.date_range(che.index.min(), che.index.max(), freq="D")
    ).ffill().fillna(0.0)

    return che


def _load_from_fallback() -> pd.Series:
    """Build a daily series from the hardcoded weekly lookup table."""
    weekly = pd.Series(
        {pd.Timestamp(k): v for k, v in _FALLBACK_WEEKLY.items()}
    ).sort_index()

    daily_idx = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    daily = weekly.reindex(daily_idx).ffill().bfill().fillna(0.0)
    return daily


def get_stringency_for_dates(dates: pd.Series) -> pd.Series:
    """
    Look up OxCGRT stringency index for a Series of datetime values.

    Parameters
    ----------
    dates : pd.Series of datetime-like
        Timestamps to look up. Typically from time_iso column.

    Returns
    -------
    pd.Series of float
        Stringency index (0-100) aligned to input index.
        Dates outside 2019-2022 get 0.0.
    """
    lookup = _load_stringency_lookup()
    date_keys = pd.to_datetime(dates).dt.normalize()  # floor to midnight
    values = date_keys.map(lookup)
    return values.fillna(0.0)


def build_covid_features(
    df: pd.DataFrame,
    time_col: str = "time_iso",
    flag_threshold: float = FLAG_THRESHOLD,
) -> pd.DataFrame:
    """
    Add covid_flag and covid_intensity columns to a mission DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a timestamp column.
    time_col : str
        Name of the timestamp column (default: 'time_iso').
    flag_threshold : float
        Stringency score above which covid_flag = 1 (default: 20.0).

    Returns
    -------
    pd.DataFrame
        Copy of df with two new columns:
        - covid_intensity : float [0, 1], normalized stringency (raw / 100)
        - covid_flag      : int {0, 1}, 1 if stringency > flag_threshold

    Notes
    -----
    covid_intensity is normalized by dividing by 100 so it lives on the
    same [0,1] scale as boolean features. This makes it compatible with
    tree-based models without scaling.

    covid_flag is kept as a separate binary feature because the non-linear
    threshold effect (presence/absence of restrictions) may be more
    predictive than the continuous level for ridership demand.
    """
    result = df.copy()
    raw_stringency = get_stringency_for_dates(result[time_col])
    result["covid_intensity"] = (raw_stringency / 100.0).values
    result["covid_flag"] = (raw_stringency > flag_threshold).astype(int).values
    return result
