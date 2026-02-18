"""
Target variable construction: discretize passenger counts into demand classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import N_DEMAND_CLASSES, DEMAND_LABELS


def compute_tercile_boundaries(series: pd.Series) -> tuple[float, float]:
    """
    Compute the 33rd and 67th percentile boundaries for 3-class binning.
    Returns (low_high_boundary, medium_high_boundary).
    """
    clean = series.dropna()
    q1 = float(clean.quantile(1 / 3))
    q2 = float(clean.quantile(2 / 3))
    return q1, q2


def assign_demand_class(
    series: pd.Series,
    low_boundary: float,
    high_boundary: float,
) -> pd.Series:
    """
    Assign demand class labels based on precomputed boundaries.

    - low: value <= low_boundary
    - medium: low_boundary < value <= high_boundary
    - high: value > high_boundary

    NaN inputs produce NaN outputs.
    """
    conditions = [
        series <= low_boundary,
        (series > low_boundary) & (series <= high_boundary),
        series > high_boundary,
    ]
    choices = DEMAND_LABELS
    result = pd.Series(
        np.select(conditions, choices, default="__missing__"),
        index=series.index,
    )
    result[series.isna()] = np.nan
    return result


def compute_class_distribution(labels: pd.Series) -> dict[str, int]:
    """
    Count occurrences of each demand class.
    Returns dict like {"low": 1000, "medium": 800, "high": 600}.
    """
    counts = labels.value_counts()
    return {label: int(counts.get(label, 0)) for label in DEMAND_LABELS}
