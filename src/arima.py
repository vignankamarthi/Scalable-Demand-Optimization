"""
ARIMA fitting and forecasting utilities for hourly ridership analysis.
"""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(
    series: np.ndarray,
    d: int = 1,
    max_p: int = 20,
    max_q: int = 5,
) -> object:
    """
    Fit ARIMA(p, d, q) with grid search over p and q, selecting by lowest AIC.

    Parameters:
        series: 1D array of time series values
        d: differencing order (default 1, confirmed by stationarity analysis)
        max_p: maximum AR order to search
        max_q: maximum MA order to search

    Returns:
        Fitted ARIMA model result (statsmodels ARIMAResultsWrapper)
    """
    best_aic = np.inf
    best_result = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
            except Exception:
                continue

    if best_result is None:
        # Fallback: fit simplest model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=(1, d, 0))
            best_result = model.fit()

    return best_result


def forecast_arima(model, steps: int) -> np.ndarray:
    """
    Forecast the next `steps` values from a fitted ARIMA model.

    Returns:
        1D numpy array of point forecasts
    """
    forecast = model.forecast(steps=steps)
    return np.asarray(forecast, dtype=np.float64)
