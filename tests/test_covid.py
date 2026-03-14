"""
Tests for src/covid.py

Follows the project's TDD conventions:
  - Synthetic mock data only, no real dataset access
  - 100% deterministic
  - Matches style of test_preprocessing.py / test_target.py

Run:
    python -m pytest tests/test_covid.py -v
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.covid import (
    build_covid_features,
    get_stringency_for_dates,
    FLAG_THRESHOLD,
    _load_from_fallback,
    _load_stringency_lookup,
)


# ---------------------------------------------------------------------------
# Force deterministic fallback data for all tests (no network dependency)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_fallback_stringency():
    """Clear lru_cache and mock live download to force fallback table.

    The live OxCGRT data has minor residual stringency values (e.g., 11.11)
    for dates that the fallback table approximates as 0.0. Forcing the
    fallback path makes all tests deterministic regardless of network access.
    """
    _load_stringency_lookup.cache_clear()
    with patch("src.covid._load_from_oxcgrt", side_effect=Exception("mocked")):
        yield
    _load_stringency_lookup.cache_clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pre_covid_df():
    """Mission rows from 2019 — all stringency should be 0."""
    return pd.DataFrame({
        "time_iso": pd.date_range("2019-06-01", periods=10, freq="D").astype(str),
        "some_col": range(10),
    })


@pytest.fixture
def first_lockdown_df():
    """Mission rows from March 2020 peak lockdown."""
    return pd.DataFrame({
        "time_iso": pd.date_range("2020-03-16", periods=10, freq="D").astype(str),
        "some_col": range(10),
    })


@pytest.fixture
def second_wave_df():
    """Mission rows from November 2020 — elevated but below peak."""
    return pd.DataFrame({
        "time_iso": pd.date_range("2020-11-09", periods=7, freq="D").astype(str),
        "some_col": range(7),
    })


@pytest.fixture
def post_restriction_df():
    """Mission rows from June 2022 — restrictions fully lifted."""
    return pd.DataFrame({
        "time_iso": pd.date_range("2022-06-06", periods=7, freq="D").astype(str),
        "some_col": range(7),
    })


@pytest.fixture
def mixed_df():
    """DataFrame spanning pre-COVID, peak lockdown, and post-restriction."""
    return pd.DataFrame({
        "time_iso": [
            "2019-06-15",
            "2020-03-20",
            "2021-01-15",
            "2022-06-10",
        ],
    })


# ---------------------------------------------------------------------------
# TestBuildCovidFeatures
# ---------------------------------------------------------------------------

class TestBuildCovidFeatures:

    def test_returns_copy_not_inplace(self, pre_covid_df):
        """build_covid_features must not modify the input DataFrame."""
        original_cols = list(pre_covid_df.columns)
        _ = build_covid_features(pre_covid_df)
        assert list(pre_covid_df.columns) == original_cols

    def test_adds_covid_intensity_column(self, pre_covid_df):
        result = build_covid_features(pre_covid_df)
        assert "covid_intensity" in result.columns

    def test_adds_covid_flag_column(self, pre_covid_df):
        result = build_covid_features(pre_covid_df)
        assert "covid_flag" in result.columns

    def test_preserves_existing_columns(self, pre_covid_df):
        result = build_covid_features(pre_covid_df)
        assert "time_iso" in result.columns
        assert "some_col" in result.columns

    def test_output_row_count_unchanged(self, first_lockdown_df):
        result = build_covid_features(first_lockdown_df)
        assert len(result) == len(first_lockdown_df)

    def test_output_index_preserved(self, first_lockdown_df):
        first_lockdown_df.index = range(100, 110)
        result = build_covid_features(first_lockdown_df)
        assert list(result.index) == list(first_lockdown_df.index)


# ---------------------------------------------------------------------------
# TestCovidIntensity
# ---------------------------------------------------------------------------

class TestCovidIntensity:

    def test_pre_covid_intensity_is_zero(self, pre_covid_df):
        result = build_covid_features(pre_covid_df)
        assert (result["covid_intensity"] == 0.0).all(), (
            "Pre-COVID rows should have intensity=0"
        )

    def test_intensity_in_unit_interval(self, first_lockdown_df):
        result = build_covid_features(first_lockdown_df)
        assert result["covid_intensity"].between(0.0, 1.0).all()

    def test_first_lockdown_intensity_elevated(self, first_lockdown_df):
        """March 2020 first lockdown should have intensity > 0.5."""
        result = build_covid_features(first_lockdown_df)
        assert result["covid_intensity"].mean() > 0.5, (
            "March 2020 first lockdown intensity should exceed 0.5"
        )

    def test_post_restriction_intensity_is_zero(self, post_restriction_df):
        result = build_covid_features(post_restriction_df)
        assert (result["covid_intensity"] == 0.0).all()

    def test_intensity_is_float_dtype(self, first_lockdown_df):
        result = build_covid_features(first_lockdown_df)
        assert pd.api.types.is_float_dtype(result["covid_intensity"])

    def test_intensity_normalized_not_raw(self, first_lockdown_df):
        """Intensity must be in [0,1], not raw 0-100 scale."""
        result = build_covid_features(first_lockdown_df)
        assert result["covid_intensity"].max() <= 1.0, (
            "covid_intensity must be normalized to [0,1], not raw 0-100"
        )

    def test_no_nan_in_intensity(self, mixed_df):
        result = build_covid_features(mixed_df)
        assert result["covid_intensity"].notna().all()


# ---------------------------------------------------------------------------
# TestCovidFlag
# ---------------------------------------------------------------------------

class TestCovidFlag:

    def test_pre_covid_flag_is_zero(self, pre_covid_df):
        result = build_covid_features(pre_covid_df)
        assert (result["covid_flag"] == 0).all()

    def test_first_lockdown_flag_is_one(self, first_lockdown_df):
        result = build_covid_features(first_lockdown_df)
        assert (result["covid_flag"] == 1).all(), (
            "March 2020 first lockdown rows should all have covid_flag=1"
        )

    def test_post_restriction_flag_is_zero(self, post_restriction_df):
        result = build_covid_features(post_restriction_df)
        assert (result["covid_flag"] == 0).all()

    def test_flag_is_binary(self, mixed_df):
        result = build_covid_features(mixed_df)
        assert set(result["covid_flag"].unique()).issubset({0, 1})

    def test_flag_is_int_dtype(self, first_lockdown_df):
        result = build_covid_features(first_lockdown_df)
        assert pd.api.types.is_integer_dtype(result["covid_flag"])

    def test_flag_threshold_respected(self):
        """
        Flag should be 0 when intensity is just below threshold,
        1 when at or above threshold.
        Custom threshold of 50 tested explicitly.
        """
        df = pd.DataFrame({"time_iso": ["2020-11-16"]})  # ~72 stringency
        result_default = build_covid_features(df, flag_threshold=FLAG_THRESHOLD)
        result_strict = build_covid_features(df, flag_threshold=80.0)
        assert result_default["covid_flag"].iloc[0] == 1
        assert result_strict["covid_flag"].iloc[0] == 0

    def test_no_nan_in_flag(self, mixed_df):
        result = build_covid_features(mixed_df)
        assert result["covid_flag"].notna().all()


# ---------------------------------------------------------------------------
# TestGetStringencyForDates
# ---------------------------------------------------------------------------

class TestGetStringencyForDates:

    def test_returns_series(self):
        dates = pd.Series(["2020-03-16", "2021-01-10"])
        result = get_stringency_for_dates(dates)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self):
        dates = pd.Series(["2019-01-01", "2020-06-01", "2021-06-01"])
        result = get_stringency_for_dates(dates)
        assert len(result) == len(dates)

    def test_pre_covid_returns_zero(self):
        dates = pd.Series(["2019-01-01", "2019-12-31"])
        result = get_stringency_for_dates(dates)
        assert (result == 0.0).all()

    def test_out_of_range_returns_zero(self):
        dates = pd.Series(["2018-01-01", "2023-06-01"])
        result = get_stringency_for_dates(dates)
        assert (result == 0.0).all()

    def test_lockdown_period_elevated(self):
        dates = pd.Series(["2020-03-23"])  # peak wave 1
        result = get_stringency_for_dates(dates)
        assert result.iloc[0] > FLAG_THRESHOLD


# ---------------------------------------------------------------------------
# TestFallbackLoader
# ---------------------------------------------------------------------------

class TestFallbackLoader:

    def test_returns_series(self):
        result = _load_from_fallback()
        assert isinstance(result, pd.Series)

    def test_daily_frequency(self):
        result = _load_from_fallback()
        diffs = result.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta("1D")).all()

    def test_covers_dataset_range(self):
        result = _load_from_fallback()
        assert pd.Timestamp("2019-01-01") in result.index
        assert pd.Timestamp("2022-12-31") in result.index

    def test_no_nan_values(self):
        result = _load_from_fallback()
        assert result.notna().all()

    def test_values_in_valid_range(self):
        result = _load_from_fallback()
        assert (result >= 0.0).all() and (result <= 100.0).all()

    def test_2019_all_zeros(self):
        result = _load_from_fallback()
        pre_covid = result["2019-01-01":"2019-12-31"]
        assert (pre_covid == 0.0).all()

    def test_late_2022_all_zeros(self):
        result = _load_from_fallback()
        post_restriction = result["2022-04-04":"2022-12-31"]
        assert (post_restriction == 0.0).all()
