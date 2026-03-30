"""
ARIMA baseline evaluation for ZTBus ridership demand classification.
Fits ARIMA on hourly aggregate ridership, forecasts test period, thresholds
to demand classes, and compares against ML baselines.

Runs locally (~2 min).
"""

import sys
import os
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FIGURES_DIR, RESULTS_DIR, RANDOM_SEED, DATA_DIR, DEMAND_LABELS
from src.data_loading import load_metadata
from src.timeseries import build_hourly_aggregate
from src.target import compute_tercile_boundaries
from src.arima import fit_arima, forecast_arima
from src.model_pipeline import evaluate_model

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01", tz="UTC")
STOP_COLS = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]
TS_RESULTS_DIR = os.path.join(RESULTS_DIR, "timeseries")
TS_FIGURES_DIR = os.path.join(FIGURES_DIR, "timeseries")
os.makedirs(TS_RESULTS_DIR, exist_ok=True)
os.makedirs(TS_FIGURES_DIR, exist_ok=True)

# ARIMA search bounds (d=1 confirmed by Task 1 stationarity analysis)
ARIMA_D = 1
ARIMA_MAX_P = 10
ARIMA_MAX_Q = 3


def save_fig(fig, name):
    path = os.path.join(TS_FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# [1/6] Build hourly aggregate
# -----------------------------------------------------------------------

print("=" * 60)
print("ARIMA Baseline Evaluation")
print("=" * 60)

print(f"\n[1/6] Building hourly aggregate ridership...")
meta = load_metadata()
t0 = time.time()
hourly_agg = build_hourly_aggregate(meta, DATA_DIR, STOP_COLS)
print(f"  Built in {time.time() - t0:.1f}s")
print(f"  Series: {len(hourly_agg)} hourly data points")
print(f"  Range: {hourly_agg.index.min()} to {hourly_agg.index.max()}")

# -----------------------------------------------------------------------
# [2/6] Temporal split
# -----------------------------------------------------------------------

print(f"\n[2/6] Temporal split (cutoff: {TEMPORAL_CUTOFF.date()})...")

# Handle timezone: if index is tz-naive, localize; if tz-aware, convert
idx = hourly_agg.index
if idx.tz is None:
    idx = idx.tz_localize("UTC")
else:
    idx = idx.tz_convert("UTC")
hourly_agg.index = idx

train_series = hourly_agg[hourly_agg.index < TEMPORAL_CUTOFF]
test_series = hourly_agg[hourly_agg.index >= TEMPORAL_CUTOFF]

print(f"  Train: {len(train_series)} hours ({train_series.index.min().date()} to {train_series.index.max().date()})")
print(f"  Test:  {len(test_series)} hours ({test_series.index.min().date()} to {test_series.index.max().date()})")

# -----------------------------------------------------------------------
# [3/6] Compute tercile boundaries from training data
# -----------------------------------------------------------------------

print(f"\n[3/6] Computing tercile boundaries from training hourly data...")
q1, q2 = compute_tercile_boundaries(train_series)
print(f"  q1={q1:.2f}, q2={q2:.2f}")
print(f"  low: <= {q1:.2f} | medium: ({q1:.2f}, {q2:.2f}] | high: > {q2:.2f}")

# -----------------------------------------------------------------------
# [4/6] Fit ARIMA
# -----------------------------------------------------------------------

print(f"\n[4/6] Fitting ARIMA(p, {ARIMA_D}, q) on training data (max_p={ARIMA_MAX_P}, max_q={ARIMA_MAX_Q})...")
print(f"  Grid search over {(ARIMA_MAX_P + 1) * (ARIMA_MAX_Q + 1) - 1} (p,q) combinations...")
t0 = time.time()
model = fit_arima(
    train_series.values,
    d=ARIMA_D,
    max_p=ARIMA_MAX_P,
    max_q=ARIMA_MAX_Q,
)
fit_time = time.time() - t0
order = model.model.order
print(f"  Selected order: ARIMA{order}")
print(f"  AIC: {model.aic:.2f}")
print(f"  Fit time: {fit_time:.1f}s")

# -----------------------------------------------------------------------
# [5/6] Forecast and evaluate
# -----------------------------------------------------------------------

print(f"\n[5/6] Forecasting {len(test_series)} test hours...")
forecasts = forecast_arima(model, steps=len(test_series))

# Threshold to demand classes
def threshold_to_class(values, q1, q2):
    labels = []
    for v in values:
        if v <= q1:
            labels.append("low")
        elif v <= q2:
            labels.append("medium")
        else:
            labels.append("high")
    return labels

y_pred = pd.Series(threshold_to_class(forecasts, q1, q2), name="y_pred")
y_true = pd.Series(threshold_to_class(test_series.values, q1, q2), name="y_true")

metrics = evaluate_model(y_true, y_pred)

print(f"\n  ARIMA{order} Results:")
print(f"    Macro F1:          {metrics['macro_f1']:.4f}")
print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
for cls in DEMAND_LABELS:
    pc = metrics["per_class"][cls]
    print(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} (n={pc['support']})")

# -----------------------------------------------------------------------
# [6/6] Plot and save
# -----------------------------------------------------------------------

print(f"\n[6/6] Plotting forecast vs actual...")

# Forecast vs actual scatter
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: time series overlay (first 200 test hours for readability)
n_plot = min(200, len(test_series))
ax = axes[0]
ax.plot(range(n_plot), test_series.values[:n_plot], label="Actual", color="#2196F3", linewidth=0.8)
ax.plot(range(n_plot), forecasts[:n_plot], label=f"ARIMA{order}", color="#E91E63", linewidth=0.8, alpha=0.8)
ax.axhline(q1, color="orange", linestyle="--", linewidth=0.6, label=f"q1={q1:.1f}")
ax.axhline(q2, color="red", linestyle="--", linewidth=0.6, label=f"q2={q2:.1f}")
ax.set_xlabel("Test Hour Index")
ax.set_ylabel("Mean Hourly Ridership")
ax.set_title(f"ARIMA{order} Forecast vs Actual (first {n_plot} test hours)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bottom: confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred, labels=DEMAND_LABELS)
disp = ConfusionMatrixDisplay(cm, display_labels=DEMAND_LABELS)
disp.plot(ax=axes[1], cmap="Purples", colorbar=False)
axes[1].set_title(f"ARIMA{order} Confusion Matrix (F1={metrics['macro_f1']:.4f})")

fig.tight_layout()
save_fig(fig, "06_arima_baseline.png")

# Save results JSON
results_out = {
    "model": f"ARIMA{order}",
    "order": list(order),
    "aic": model.aic,
    "fit_time_s": round(fit_time, 2),
    "macro_f1": metrics["macro_f1"],
    "balanced_accuracy": metrics["balanced_accuracy"],
    "confusion_matrix": metrics["confusion_matrix"].tolist(),
    "per_class": metrics["per_class"],
    "tercile_boundaries": {"q1": q1, "q2": q2},
    "train_hours": len(train_series),
    "test_hours": len(test_series),
}
results_path = os.path.join(TS_RESULTS_DIR, "arima_results.json")
with open(results_path, "w") as f:
    json.dump(results_out, f, indent=2)
print(f"  Saved: {results_path}")

# -----------------------------------------------------------------------
# Summary comparison
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("ARIMA BASELINE COMPARISON")
print("=" * 60)

# Load existing ML baselines if available
ext_temporal_path = os.path.join(RESULTS_DIR, "extended", "results_temporal.json")
if os.path.exists(ext_temporal_path):
    with open(ext_temporal_path) as f:
        ml_results = json.load(f)
    print(f"\n  {'Model':<25} {'Macro F1':>10} {'Bal Acc':>10}")
    print("  " + "-" * 50)
    for model_name, m in ml_results.items():
        print(f"  {model_name:<25} {m['macro_f1']:>10.4f} {m['balanced_accuracy']:>10.4f}")
    print(f"  {'ARIMA' + str(order):<25} {metrics['macro_f1']:>10.4f} {metrics['balanced_accuracy']:>10.4f}")
else:
    print("\n  No ML baseline results found at results/extended/results_temporal.json")
    print(f"  ARIMA{order}: F1={metrics['macro_f1']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}")

print(f"\nDone.")
