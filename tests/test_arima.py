"""
Tests for src/arima.py -- ARIMA fitting and forecasting utilities.
All tests use synthetic data (no real dataset access).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.arima import fit_arima, forecast_arima


class TestFitArima:
    """Tests for ARIMA model fitting."""

    def test_fit_returns_result_with_aic(self):
        """Fitted model should have an AIC score."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        result = fit_arima(series, d=0, max_p=3, max_q=2)
        assert hasattr(result, "aic")

    def test_fit_stationary_series_d0(self):
        """Stationary series should fit with d=0 without error."""
        rng = np.random.default_rng(42)
        series = rng.normal(10, 2, size=200)
        result = fit_arima(series, d=0, max_p=3, max_q=2)
        assert result is not None

    def test_fit_nonstationary_series_d1(self):
        """Non-stationary (random walk) series should fit with d=1."""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, size=200))
        result = fit_arima(series, d=1, max_p=3, max_q=2)
        assert result is not None

    def test_fit_selects_best_aic(self):
        """With multiple (p,q) candidates, should select the one with lowest AIC."""
        rng = np.random.default_rng(42)
        # AR(1) process: should favor p>=1
        ar_coef = 0.7
        series = np.zeros(300)
        for t in range(1, 300):
            series[t] = ar_coef * series[t - 1] + rng.normal(0, 1)
        result = fit_arima(series, d=0, max_p=5, max_q=2)
        # The selected order should have p >= 1 (AR component needed)
        order = result.model.order
        assert order[0] >= 1, f"Expected AR order >= 1 for AR(1) process, got {order}"

    def test_fit_with_small_max_p(self):
        """max_p=1 should still work, limiting search space."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        result = fit_arima(series, d=0, max_p=1, max_q=1)
        order = result.model.order
        assert order[0] <= 1
        assert order[2] <= 1

    def test_fit_returns_fitted_model(self):
        """Result should have forecast capability (fittedvalues attribute)."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        result = fit_arima(series, d=0, max_p=2, max_q=2)
        assert hasattr(result, "fittedvalues")


class TestForecastArima:
    """Tests for ARIMA forecasting."""

    def test_forecast_returns_correct_length(self):
        """Forecast should return exactly `steps` values."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        model = fit_arima(series, d=0, max_p=2, max_q=2)
        forecasts = forecast_arima(model, steps=10)
        assert len(forecasts) == 10

    def test_forecast_returns_numpy_array(self):
        """Forecast should be a numpy array of floats."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        model = fit_arima(series, d=0, max_p=2, max_q=2)
        forecasts = forecast_arima(model, steps=5)
        assert isinstance(forecasts, np.ndarray)
        assert forecasts.dtype in [np.float64, np.float32]

    def test_forecast_no_nans(self):
        """Forecast should not contain NaN values."""
        rng = np.random.default_rng(42)
        series = rng.normal(10, 2, size=200)
        model = fit_arima(series, d=0, max_p=2, max_q=2)
        forecasts = forecast_arima(model, steps=20)
        assert not np.any(np.isnan(forecasts))

    def test_forecast_single_step(self):
        """Forecasting a single step should work."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=200)
        model = fit_arima(series, d=0, max_p=2, max_q=2)
        forecasts = forecast_arima(model, steps=1)
        assert len(forecasts) == 1
