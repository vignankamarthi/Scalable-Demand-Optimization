"""
Tests for src/ts_evaluation.py

Follows project TDD conventions:
  - Synthetic mock data only, no real dataset access
  - 100% deterministic
  - statsmodels is mocked so tests run without it installed

Run:
    python -m pytest tests/test_ts_evaluation.py -v
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub out statsmodels before importing ts_evaluation so the module loads
# even when statsmodels is not installed (e.g. local dev machine).
# ---------------------------------------------------------------------------

def _make_statsmodels_stub():
    sm = types.ModuleType("statsmodels")
    tsa = types.ModuleType("statsmodels.tsa")
    stattools = types.ModuleType("statsmodels.tsa.stattools")

    def fake_acf(x, nlags=40, fft=True):
        """Return deterministic ACF: lag-0 = 1, all others decay geometrically."""
        result = np.array([0.9 ** i for i in range(nlags + 1)])
        result[0] = 1.0
        return result

    stattools.acf = fake_acf
    tsa.stattools = stattools
    sm.tsa = tsa
    sys.modules.setdefault("statsmodels", sm)
    sys.modules.setdefault("statsmodels.tsa", tsa)
    sys.modules.setdefault("statsmodels.tsa.stattools", stattools)

_make_statsmodels_stub()

# Also stub matplotlib/seaborn so plots don't open windows or write files
import matplotlib
matplotlib.use("Agg")

from src.ts_evaluation import (
    build_prediction_frame,
    macro_f1_from_frame,
    label_covid_period,
    compute_monthly_f1,
    compute_within_mission_accuracy,
    compute_error_acf,
    compute_covid_breakdown,
    compute_transition_accuracy,
    compute_class_distribution_over_time,
    COVID_PERIODS,
    ACF_NLAGS,
    TRIP_BINS,
    DEMAND_LABELS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_pred_frame(
    n_rows: int = 200,
    n_missions: int = 4,
    start_date: str = "2020-01-01",
    correct_frac: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic prediction frame matching what build_prediction_frame produces.
    Rows are spread evenly across missions and consecutive seconds.
    """
    rng = np.random.default_rng(seed)
    rows_per_mission = n_rows // n_missions

    frames = []
    base = pd.Timestamp(start_date)
    for i in range(n_missions):
        ts = pd.date_range(
            base + pd.Timedelta(days=i * 10),
            periods=rows_per_mission,
            freq="s",
        )
        true_labels = rng.choice(DEMAND_LABELS, size=rows_per_mission)
        # Simulate predictions: correct_frac of the time = true label
        correct_mask = rng.random(rows_per_mission) < correct_frac
        pred_labels = np.where(correct_mask, true_labels,
                               rng.choice(DEMAND_LABELS, size=rows_per_mission))
        frames.append(pd.DataFrame({
            "time_iso":     ts,
            "mission_name": f"mission_{i:03d}",
            "y_true":       true_labels,
            "y_pred":       pred_labels,
            "correct":      (true_labels == pred_labels).astype(int),
        }))

    df = pd.concat(frames, ignore_index=True)
    df["year"]  = df["time_iso"].dt.year
    df["month"] = df["time_iso"].dt.month
    df["date"]  = df["time_iso"].dt.normalize()
    df["ym"]    = df["time_iso"].dt.to_period("M")
    return df


def _make_mock_model(predictions: np.ndarray):
    """Return a mock sklearn estimator that always returns `predictions`."""
    model = MagicMock()
    model.predict.return_value = predictions
    return model


def _make_feat_df(n_rows: int = 50, start_date: str = "2021-06-01") -> pd.DataFrame:
    """
    Build a minimal feature DataFrame as would be passed to build_prediction_frame.
    Contains time_iso, mission_name, and two numeric feature columns.
    """
    ts = pd.date_range(start_date, periods=n_rows, freq="s")
    return pd.DataFrame({
        "time_iso":     ts,
        "mission_name": ["mission_000"] * n_rows,
        "feature_a":    np.random.default_rng(0).random(n_rows),
        "feature_b":    np.random.default_rng(1).random(n_rows),
    })


# ---------------------------------------------------------------------------
# TestBuildPredictionFrame
# ---------------------------------------------------------------------------

class TestBuildPredictionFrame:

    def test_returns_dataframe(self):
        feat_df = _make_feat_df()
        y_true = pd.Series(
            np.random.default_rng(0).choice(DEMAND_LABELS, size=len(feat_df)),
            index=feat_df.index,
        )
        model = _make_mock_model(y_true.values)
        result = build_prediction_frame(feat_df, model, y_true)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        feat_df = _make_feat_df()
        y_true = pd.Series(
            ["low"] * len(feat_df), index=feat_df.index
        )
        model = _make_mock_model(np.array(["low"] * len(feat_df)))
        result = build_prediction_frame(feat_df, model, y_true)
        for col in ["time_iso", "mission_name", "y_true", "y_pred", "correct", "ym"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_correct_column_is_binary(self):
        feat_df = _make_feat_df()
        y_true = pd.Series(["low"] * len(feat_df), index=feat_df.index)
        model = _make_mock_model(np.array(["low"] * len(feat_df)))
        result = build_prediction_frame(feat_df, model, y_true)
        assert set(result["correct"].unique()).issubset({0, 1})

    def test_perfect_predictions_all_correct(self):
        feat_df = _make_feat_df(n_rows=20)
        labels = np.array(["low", "medium", "high"] * 7)[:20]
        y_true = pd.Series(labels, index=feat_df.index)
        model = _make_mock_model(labels.copy())
        result = build_prediction_frame(feat_df, model, y_true)
        assert result["correct"].all()

    def test_wrong_predictions_all_incorrect(self):
        feat_df = _make_feat_df(n_rows=20)
        y_true = pd.Series(["low"] * 20, index=feat_df.index)
        model = _make_mock_model(np.array(["high"] * 20))
        result = build_prediction_frame(feat_df, model, y_true)
        assert result["correct"].sum() == 0

    def test_drops_target_columns_before_predict(self):
        """Model.predict should NOT receive demand_class or itcs_numberOfPassengers."""
        feat_df = _make_feat_df()
        feat_df["demand_class"] = "low"
        feat_df["itcs_numberOfPassengers"] = 5.0
        y_true = pd.Series(["low"] * len(feat_df), index=feat_df.index)
        model = _make_mock_model(np.array(["low"] * len(feat_df)))
        build_prediction_frame(feat_df, model, y_true)
        called_X = model.predict.call_args[0][0]
        assert "demand_class" not in called_X.columns
        assert "itcs_numberOfPassengers" not in called_X.columns

    def test_drops_rows_with_nat_timestamps(self):
        feat_df = _make_feat_df(n_rows=10)
        feat_df.loc[feat_df.index[0], "time_iso"] = pd.NaT
        y_true = pd.Series(["low"] * 10, index=feat_df.index)
        model = _make_mock_model(np.array(["low"] * 10))
        result = build_prediction_frame(feat_df, model, y_true)
        assert len(result) == 9

    def test_index_mismatch_handled(self):
        """y_true with a different integer index should still align correctly."""
        feat_df = _make_feat_df(n_rows=20)
        feat_df.index = range(100, 120)
        y_true = pd.Series(["low"] * 20, index=range(100, 120))
        model = _make_mock_model(np.array(["low"] * 20))
        result = build_prediction_frame(feat_df, model, y_true)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# TestMacroF1FromFrame
# ---------------------------------------------------------------------------

class TestMacroF1FromFrame:

    def test_perfect_predictions_return_one(self):
        df = _make_pred_frame(correct_frac=1.0)
        assert macro_f1_from_frame(df) == pytest.approx(1.0, abs=1e-6)

    def test_empty_frame_returns_nan(self):
        df = pd.DataFrame(columns=["y_true", "y_pred"])
        result = macro_f1_from_frame(df)
        assert np.isnan(result)

    def test_returns_float(self):
        df = _make_pred_frame()
        assert isinstance(macro_f1_from_frame(df), float)

    def test_in_unit_interval(self):
        df = _make_pred_frame()
        f1 = macro_f1_from_frame(df)
        assert 0.0 <= f1 <= 1.0

    def test_higher_correct_frac_gives_higher_f1(self):
        low_df  = _make_pred_frame(correct_frac=0.4, seed=0)
        high_df = _make_pred_frame(correct_frac=0.9, seed=0)
        assert macro_f1_from_frame(high_df) > macro_f1_from_frame(low_df)


# ---------------------------------------------------------------------------
# TestLabelCovidPeriod
# ---------------------------------------------------------------------------

class TestLabelCovidPeriod:

    def test_pre_covid_dates(self):
        ts = pd.Series(pd.to_datetime(["2019-06-15", "2020-01-10"]))
        result = label_covid_period(ts)
        assert (result == "pre_covid").all()

    def test_restriction_dates(self):
        ts = pd.Series(pd.to_datetime(["2020-04-01", "2021-01-15", "2022-02-01"]))
        result = label_covid_period(ts)
        assert (result == "restrictions").all()

    def test_post_covid_dates(self):
        ts = pd.Series(pd.to_datetime(["2022-05-01", "2022-11-30"]))
        result = label_covid_period(ts)
        assert (result == "post_covid").all()

    def test_outside_dataset(self):
        ts = pd.Series(pd.to_datetime(["2018-01-01", "2023-06-01"]))
        result = label_covid_period(ts)
        assert (result == "outside_dataset").all()

    def test_preserves_index(self):
        ts = pd.Series(pd.to_datetime(["2020-06-01", "2019-03-01"]), index=[10, 20])
        result = label_covid_period(ts)
        assert list(result.index) == [10, 20]

    def test_returns_series(self):
        ts = pd.Series(pd.to_datetime(["2021-01-01"]))
        assert isinstance(label_covid_period(ts), pd.Series)


# ---------------------------------------------------------------------------
# TestComputeMonthlyF1
# ---------------------------------------------------------------------------

class TestComputeMonthlyF1:

    def test_returns_dataframe(self):
        df = _make_pred_frame(start_date="2020-01-01", n_rows=200)
        result = compute_monthly_f1(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_pred_frame()
        result = compute_monthly_f1(df)
        for col in ["period", "macro_f1", "n_samples"]:
            assert col in result.columns

    def test_one_row_per_month(self):
        df = _make_pred_frame(n_rows=400, n_missions=4, start_date="2020-01-01")
        result = compute_monthly_f1(df)
        assert result["period"].nunique() == len(result)

    def test_sorted_by_period(self):
        df = _make_pred_frame(n_rows=400)
        result = compute_monthly_f1(df)
        periods = list(result["period"])
        assert periods == sorted(periods)

    def test_f1_in_unit_interval(self):
        df = _make_pred_frame()
        result = compute_monthly_f1(df)
        assert result["macro_f1"].between(0, 1).all()

    def test_n_samples_sums_to_total(self):
        df = _make_pred_frame(n_rows=200)
        result = compute_monthly_f1(df)
        assert result["n_samples"].sum() == len(df)


# ---------------------------------------------------------------------------
# TestComputeWithinMissionAccuracy
# ---------------------------------------------------------------------------

class TestComputeWithinMissionAccuracy:

    def test_returns_dataframe(self):
        df = _make_pred_frame()
        result = compute_within_mission_accuracy(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_pred_frame()
        result = compute_within_mission_accuracy(df)
        for col in ["bin", "accuracy", "n_samples"]:
            assert col in result.columns

    def test_bin_count_equals_trip_bins(self):
        df = _make_pred_frame(n_rows=400, n_missions=4)
        result = compute_within_mission_accuracy(df)
        assert len(result) == TRIP_BINS

    def test_accuracy_in_unit_interval(self):
        df = _make_pred_frame()
        result = compute_within_mission_accuracy(df)
        assert result["accuracy"].between(0, 1).all()

    def test_skips_short_missions(self):
        """Missions shorter than TRIP_BINS rows should be skipped entirely."""
        df = _make_pred_frame(n_rows=4, n_missions=1)
        result = compute_within_mission_accuracy(df)
        assert result.empty or len(result) == 0

    def test_perfect_model_all_ones(self):
        df = _make_pred_frame(correct_frac=1.0)
        result = compute_within_mission_accuracy(df)
        assert (result["accuracy"] == 1.0).all()


# ---------------------------------------------------------------------------
# TestComputeErrorAcf
# ---------------------------------------------------------------------------

class TestComputeErrorAcf:

    def test_returns_dict_with_required_keys(self):
        df = _make_pred_frame(n_rows=400, n_missions=2)
        result = compute_error_acf(df)
        for key in ["lags", "mean_acf", "std_acf"]:
            assert key in result

    def test_lags_length(self):
        df = _make_pred_frame(n_rows=400, n_missions=2)
        result = compute_error_acf(df)
        assert len(result["lags"]) == ACF_NLAGS + 1

    def test_mean_acf_length(self):
        df = _make_pred_frame(n_rows=400, n_missions=2)
        result = compute_error_acf(df)
        assert len(result["mean_acf"]) == ACF_NLAGS + 1

    def test_lag_zero_is_one(self):
        """ACF at lag 0 is always 1 by definition."""
        df = _make_pred_frame(n_rows=400, n_missions=2)
        result = compute_error_acf(df)
        assert result["mean_acf"][0] == pytest.approx(1.0, abs=0.01)

    def test_empty_frame_returns_zeros(self):
        """Missions too short to compute ACF should return zero fallback."""
        df = _make_pred_frame(n_rows=4, n_missions=1)
        result = compute_error_acf(df)
        assert result["mean_acf"][1] == 0.0

    def test_all_correct_returns_valid_acf(self):
        """Perfect predictions → error series is all zeros → ACF should be a valid number."""
        df = _make_pred_frame(correct_frac=1.0, n_rows=400, n_missions=2)
        result = compute_error_acf(df)
        assert np.isfinite(result["mean_acf"][1]) or result["mean_acf"][1] == 0.0


# ---------------------------------------------------------------------------
# TestComputeCovidBreakdown
# ---------------------------------------------------------------------------

class TestComputeCovidBreakdown:

    def test_returns_dataframe(self):
        df = _make_pred_frame(start_date="2020-01-01", n_rows=400, n_missions=4)
        result = compute_covid_breakdown(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_pred_frame(start_date="2020-03-01")
        result = compute_covid_breakdown(df)
        for col in ["period", "macro_f1", "accuracy", "n_samples"]:
            assert col in result.columns

    def test_only_known_periods(self):
        df = _make_pred_frame(start_date="2019-06-01", n_rows=400, n_missions=4)
        result = compute_covid_breakdown(df)
        valid = set(COVID_PERIODS.keys())
        assert set(result["period"]).issubset(valid)

    def test_f1_in_unit_interval(self):
        df = _make_pred_frame(start_date="2020-03-01")
        result = compute_covid_breakdown(df)
        if not result.empty:
            assert result["macro_f1"].between(0, 1).all()

    def test_empty_period_excluded(self):
        """If no data falls in a period, that period should not appear in results."""
        df = _make_pred_frame(start_date="2022-05-01", n_rows=100, n_missions=2)
        result = compute_covid_breakdown(df)
        assert "pre_covid" not in result["period"].values
        assert "restrictions" not in result["period"].values

    def test_n_samples_positive(self):
        df = _make_pred_frame(start_date="2020-03-01")
        result = compute_covid_breakdown(df)
        if not result.empty:
            assert (result["n_samples"] > 0).all()


# ---------------------------------------------------------------------------
# TestComputeTransitionAccuracy
# ---------------------------------------------------------------------------

class TestComputeTransitionAccuracy:

    def test_returns_tuple(self):
        df = _make_pred_frame(n_rows=200)
        result = compute_transition_accuracy(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_summary_has_two_rows(self):
        df = _make_pred_frame(n_rows=200)
        summary, _ = compute_transition_accuracy(df)
        if not summary.empty:
            assert len(summary) == 2

    def test_summary_has_required_columns(self):
        df = _make_pred_frame(n_rows=200)
        summary, _ = compute_transition_accuracy(df)
        if not summary.empty:
            for col in ["type", "accuracy", "n_samples"]:
                assert col in summary.columns

    def test_per_transition_has_required_columns(self):
        df = _make_pred_frame(n_rows=200)
        _, per_t = compute_transition_accuracy(df)
        if not per_t.empty:
            for col in ["transition", "accuracy", "n_samples"]:
                assert col in per_t.columns

    def test_accuracy_in_unit_interval(self):
        df = _make_pred_frame(n_rows=200)
        summary, per_t = compute_transition_accuracy(df)
        if not summary.empty:
            assert summary["accuracy"].between(0, 1).all()
        if not per_t.empty:
            assert per_t["accuracy"].between(0, 1).all()

    def test_empty_frame_returns_empty_tuple(self):
        df = pd.DataFrame(columns=["mission_name", "y_true", "y_pred", "correct",
                                    "time_iso", "ym"])
        summary, per_t = compute_transition_accuracy(df)
        assert summary.empty
        assert per_t.empty

    def test_stable_accuracy_higher_than_transition(self):
        """
        For a model that predicts the previous label (lagged prediction),
        stable periods should have higher accuracy than transitions.
        """
        rng = np.random.default_rng(99)
        n = 300
        true_seq = rng.choice(DEMAND_LABELS, size=n)
        # Predict lagged (always predicts previous label)
        pred_seq = np.roll(true_seq, 1)
        pred_seq[0] = true_seq[0]
        ts = pd.date_range("2021-01-01", periods=n, freq="s")
        df = pd.DataFrame({
            "time_iso":     ts,
            "mission_name": "mission_000",
            "y_true":       true_seq,
            "y_pred":       pred_seq,
            "correct":      (true_seq == pred_seq).astype(int),
            "ym":           ts.to_period("M"),
        })
        summary, _ = compute_transition_accuracy(df)
        if not summary.empty:
            stable_acc = summary[summary["type"] == "Stable periods"]["accuracy"].values[0]
            trans_acc  = summary[summary["type"] == "At transitions"]["accuracy"].values[0]
            assert stable_acc >= trans_acc


# ---------------------------------------------------------------------------
# TestComputeClassDistributionOverTime
# ---------------------------------------------------------------------------

class TestComputeClassDistributionOverTime:

    def test_returns_dataframe(self):
        df = _make_pred_frame()
        result = compute_class_distribution_over_time(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_pred_frame()
        result = compute_class_distribution_over_time(df)
        for col in ["ym", "source", "low", "medium", "high"]:
            assert col in result.columns

    def test_fractions_sum_to_one(self):
        df = _make_pred_frame(n_rows=400)
        result = compute_class_distribution_over_time(df)
        totals = result[["low", "medium", "high"]].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-6

    def test_source_values(self):
        df = _make_pred_frame()
        result = compute_class_distribution_over_time(df)
        assert set(result["source"].unique()).issubset({"actual", "predicted"})

    def test_two_rows_per_month(self):
        """Each month should produce one 'actual' and one 'predicted' row."""
        df = _make_pred_frame(n_rows=400, n_missions=4, start_date="2021-01-01")
        result = compute_class_distribution_over_time(df)
        counts = result.groupby("ym")["source"].count()
        assert (counts == 2).all()

    def test_fractions_in_unit_interval(self):
        df = _make_pred_frame()
        result = compute_class_distribution_over_time(df)
        for cls in ["low", "medium", "high"]:
            assert result[cls].between(0, 1).all()
