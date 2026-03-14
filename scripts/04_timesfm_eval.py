"""
TimesFM evaluation script for ZTBus ridership demand classification.
Designed to run on NEU Explorer cluster (GPU preferred, CPU fallback).

WHAT THIS SCRIPT DOES
---------------------
TimesFM is a forecasting foundation model, not a classifier. This script
adapts it for 3-class demand classification using a rolling-window strategy:

  1. For each test mission, build overlapping windows of context_len seconds
     of passenger count history.
  2. Feed each window into TimesFM to forecast the next horizon_len seconds.
  3. Take the first step of each forecast as the predicted passenger count.
  4. Threshold predictions using the training-set tercile boundaries (q1, q2)
     to produce low/medium/high demand labels.
  5. Evaluate with the same metrics as 02_train.py and 03_train_extended.py:
     macro F1, balanced accuracy, 3x3 confusion matrix.

Two modes are run:
  - Zero-shot:    raw TimesFM inference with no fine-tuning
  - Fine-tuned:   TimesFM weights updated on training missions (optional,
                  set FINETUNE=True below; adds ~30-60min on GPU)

IMPORTANT NOTES
---------------
- TimesFM does NOT run on Apple Silicon (ARM). Cluster only.
- Requires Python 3.11. NEU Explorer default may be 3.10 or 3.11.
  If you are on 3.12+, install pyenv and use 3.11.10 (see sbatch script).
- TimesFM is installed from GitHub source (not pip), see sbatch script.
- Recommended: GPU partition. CPU fallback is supported but slow (~4-8hrs).
- Context window: 512 seconds (~8.5 min of 1Hz data). Chosen to fit
  TimesFM-2.0's max context of 2048 while keeping per-mission batches
  manageable in RAM.

Usage:
    python scripts/04_timesfm_eval.py                  # zero-shot only
    python scripts/04_timesfm_eval.py --finetune       # zero-shot + finetune
    python scripts/04_timesfm_eval.py --cpu            # force CPU backend
"""

from __future__ import annotations

import sys
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, EDA_USE_COLS, DEMAND_LABELS,
)
from src.data_loading import load_metadata, load_sample_missions
from src.preprocessing import forward_fill_stop_columns
from src.target import compute_tercile_boundaries, assign_demand_class
from src.model_pipeline import evaluate_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_TFM_DIR = os.path.join(RESULTS_DIR, "timesfm")
FIGURES_TFM_DIR = os.path.join(FIGURES_DIR, "timesfm")
os.makedirs(RESULTS_TFM_DIR, exist_ok=True)
os.makedirs(FIGURES_TFM_DIR, exist_ok=True)

# Temporal split (same as 03_train_extended.py for fair comparison)
TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01", tz="UTC")

# TimesFM inference parameters
CONTEXT_LEN = 512       # seconds of history fed to model per window
HORIZON_LEN = 1         # forecast steps; we only need next-step prediction
STRIDE = 60             # step between windows (seconds); 60 = 1 per minute
MIN_CONTEXT = 64        # minimum history rows before a window is valid
BATCH_SIZE = 256        # number of context windows per model.forecast() call
                        # reduce if OOM; increase if GPU has headroom

# Fine-tuning config (only used if --finetune flag is passed)
FINETUNE = "--finetune" in sys.argv
FINETUNE_EPOCHS = 5
FINETUNE_LR = 1e-4
FINETUNE_BATCH = 32

# Backend: auto-detect GPU, override with --cpu flag
FORCE_CPU = "--cpu" in sys.argv


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Return 'gpu', 'cpu'. Respects --cpu flag."""
    if FORCE_CPU:
        log("  Backend: cpu (forced via --cpu flag)")
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            log(f"  Backend: gpu ({torch.cuda.get_device_name(0)})")
            return "gpu"
        else:
            log("  Backend: cpu (no CUDA device found)")
            return "cpu"
    except ImportError:
        log("  Backend: cpu (torch not available)")
        return "cpu"


# ---------------------------------------------------------------------------
# TimesFM loader
# ---------------------------------------------------------------------------

def load_timesfm_model(backend: str):
    """
    Load TimesFM 2.0-500m (PyTorch) from HuggingFace.

    Uses the 500m checkpoint (better accuracy, longer context than 200m).
    Falls back to 200m if 500m download fails (less common on HPC due to
    network restrictions).

    Returns a loaded timesfm.TimesFm instance ready for inference.
    """
    try:
        import timesfm
    except ImportError:
        raise ImportError(
            "timesfm not installed. Install with:\n"
            "  git clone https://github.com/google-research/timesfm.git\n"
            "  cd timesfm && pip install -e .[torch]\n"
            "See scripts/04_timesfm_eval.sbatch for full setup."
        )

    log("Loading TimesFM 2.0-500m checkpoint...")
    try:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=BATCH_SIZE,
                horizon_len=HORIZON_LEN,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
                context_len=CONTEXT_LEN,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        log("  Loaded: timesfm-2.0-500m-pytorch")
    except Exception as e:
        log(f"  500m load failed ({e}), falling back to 200m checkpoint...")
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=BATCH_SIZE,
                horizon_len=HORIZON_LEN,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                context_len=min(CONTEXT_LEN, 512),
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            ),
        )
        log("  Loaded: timesfm-1.0-200m-pytorch (fallback)")

    return tfm


# ---------------------------------------------------------------------------
# Windowing utilities
# ---------------------------------------------------------------------------

def build_windows(
    series: np.ndarray,
    context_len: int = CONTEXT_LEN,
    stride: int = STRIDE,
    min_context: int = MIN_CONTEXT,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Build overlapping context windows from a 1D time series.

    Returns:
        windows: list of 1D arrays, each of length <= context_len
        target_indices: list of int, the index in `series` that each
                        window is trying to predict (window end + 1)

    Windows shorter than min_context are skipped.
    The last valid target index is len(series) - 1.
    """
    n = len(series)
    windows = []
    target_indices = []

    for end in range(min_context, n, stride):
        start = max(0, end - context_len)
        window = series[start:end]
        if len(window) < min_context:
            continue
        # Clamp NaNs to linear interpolation within window
        window = _interpolate_nans(window)
        windows.append(window)
        target_indices.append(end)  # index to predict

    return windows, target_indices


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation for NaN values within a window."""
    arr = arr.copy().astype(float)
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans]) if (~nans).any() else 0.0
    return arr


# ---------------------------------------------------------------------------
# Mission-level inference
# ---------------------------------------------------------------------------

def run_inference_on_missions(
    missions_df: pd.DataFrame,
    tfm,
    q1: float,
    q2: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Run TimesFM inference on all missions in missions_df.

    For each mission:
      1. Extract passenger count series (forward-filled)
      2. Build overlapping context windows
      3. Run model.forecast() in batches
      4. Take first forecast step as predicted passenger count
      5. Threshold to demand class using q1, q2

    Returns (y_true, y_pred) as pd.Series of demand labels.
    """
    all_true = []
    all_pred = []

    mission_names = missions_df["mission_name"].unique()
    log(f"  Running inference on {len(mission_names)} missions...")

    for i, mission_name in enumerate(mission_names):
        if (i + 1) % 50 == 0:
            log(f"    Progress: {i+1}/{len(mission_names)} missions")

        m_df = missions_df[missions_df["mission_name"] == mission_name].copy()
        m_df = m_df.sort_values("time_unix") if "time_unix" in m_df.columns else m_df

        pax = m_df["itcs_numberOfPassengers"].values.astype(float)
        demand_labels = m_df["demand_class"].values

        if len(pax) < MIN_CONTEXT + 1:
            continue  # mission too short for any valid window

        windows, target_indices = build_windows(pax)
        if not windows:
            continue

        # Batch inference
        forecasts = []
        for batch_start in range(0, len(windows), BATCH_SIZE):
            batch = windows[batch_start: batch_start + BATCH_SIZE]
            # TimesFM expects list of arrays; frequency=0 (high-freq, default)
            point_forecast, _ = tfm.forecast(
                inputs=batch,
                freq=[0] * len(batch),
            )
            # point_forecast shape: (batch_size, horizon_len)
            # Take first horizon step
            forecasts.extend(point_forecast[:, 0].tolist())

        # Map forecast values to demand classes
        forecast_arr = np.array(forecasts)
        pred_labels = _threshold_to_demand(forecast_arr, q1, q2)

        # True labels at the predicted indices
        true_labels = demand_classes_at_indices(demand_labels, target_indices)

        # Filter out any None (missing label at boundary)
        valid = [(t, p) for t, p in zip(true_labels, pred_labels) if t is not None]
        if not valid:
            continue

        t_valid, p_valid = zip(*valid)
        all_true.extend(t_valid)
        all_pred.extend(p_valid)

    y_true = pd.Series(all_true, name="y_true")
    y_pred = pd.Series(all_pred, name="y_pred")
    return y_true, y_pred


def _threshold_to_demand(values: np.ndarray, q1: float, q2: float) -> list[str]:
    """Threshold continuous values to low/medium/high using tercile boundaries."""
    labels = []
    for v in values:
        if np.isnan(v) or v <= q1:
            labels.append("low")
        elif v <= q2:
            labels.append("medium")
        else:
            labels.append("high")
    return labels


def demand_classes_at_indices(
    demand_array: np.ndarray,
    indices: list[int],
) -> list[str | None]:
    """Return demand class labels at given indices, or None if out of bounds."""
    n = len(demand_array)
    result = []
    for idx in indices:
        if idx < n and demand_array[idx] not in (None, np.nan, "__missing__"):
            result.append(demand_array[idx])
        else:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# Fine-tuning (optional)
# ---------------------------------------------------------------------------

def finetune_timesfm(tfm, train_df: pd.DataFrame) -> None:
    """
    Fine-tune TimesFM on training missions' passenger count series.

    Uses the timesfm finetuning API introduced in v1.1+.
    Modifies tfm in-place.

    Each mission contributes one time series. Series are truncated to
    CONTEXT_LEN * 2 to keep memory manageable.
    """
    try:
        import timesfm.finetuning as ft
    except ImportError:
        log("  WARNING: timesfm.finetuning not available in this version.")
        log("           Skipping fine-tuning. Run: pip install timesfm --upgrade")
        return

    log("  Preparing fine-tuning data...")
    train_series = []
    for mission_name, group in train_df.groupby("mission_name"):
        pax = group["itcs_numberOfPassengers"].values.astype(float)
        pax = _interpolate_nans(pax)
        # Truncate to keep memory manageable
        if len(pax) > CONTEXT_LEN * 4:
            pax = pax[-(CONTEXT_LEN * 4):]
        if len(pax) >= MIN_CONTEXT:
            train_series.append(pax)

    if not train_series:
        log("  WARNING: No valid training series for fine-tuning.")
        return

    log(f"  Fine-tuning on {len(train_series)} mission series "
        f"({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR})...")

    try:
        ft.finetune(
            model=tfm,
            train_data=train_series,
            num_epochs=FINETUNE_EPOCHS,
            learning_rate=FINETUNE_LR,
            batch_size=FINETUNE_BATCH,
        )
        log("  Fine-tuning complete.")
    except Exception as e:
        log(f"  WARNING: Fine-tuning failed ({e}). Continuing with zero-shot model.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timesfm_confusion(cm, mode: str, macro_f1: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["low", "medium", "high"],
        yticklabels=["low", "medium", "high"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"TimesFM [{mode}] (F1={macro_f1:.4f})")
    fig.tight_layout()
    fname = os.path.join(FIGURES_TFM_DIR, f"cm_timesfm_{mode}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")


def plot_forecast_sample(
    pax_series: np.ndarray,
    forecast_value: float,
    true_value: float,
    q1: float,
    q2: float,
    mission_name: str,
):
    """
    Plot a sample context window + forecast vs. actual.
    Saves one representative figure for qualitative inspection.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pax_series, color="#2196F3", linewidth=0.8, label="Context (history)")
    ax.axhline(q1, color="orange", linestyle="--", linewidth=0.8, label=f"q1={q1:.1f}")
    ax.axhline(q2, color="red", linestyle="--", linewidth=0.8, label=f"q2={q2:.1f}")
    ax.scatter(len(pax_series), forecast_value, color="#E91E63", s=80,
               zorder=5, label=f"Forecast={forecast_value:.1f}")
    ax.scatter(len(pax_series), true_value, color="#4CAF50", s=80,
               zorder=5, label=f"Actual={true_value:.1f}", marker="^")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Passenger count")
    ax.set_title(f"TimesFM sample forecast — {mission_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fname = os.path.join(FIGURES_TFM_DIR, "sample_forecast.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log("=" * 60)
log("TimesFM Evaluation Pipeline (04_timesfm_eval.py)")
log("=" * 60)
log(f"  Mode: {'zero-shot + fine-tune' if FINETUNE else 'zero-shot only'}")
log(f"  Context length: {CONTEXT_LEN}s  |  Stride: {STRIDE}s  |  Horizon: {HORIZON_LEN}s")

# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------

log("\n[1] Loading data...")
meta = load_metadata()
t0 = time.time()
df_raw = load_sample_missions(meta, usecols=EDA_USE_COLS)
log(f"  Loaded {len(df_raw):,} rows in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------
# 2. Forward-fill
# -----------------------------------------------------------------------

log("\n[2] Forward-filling ITCS columns...")
filled_parts = []
for mission_name, group in df_raw.groupby("mission_name"):
    filled_parts.append(forward_fill_stop_columns(group))
df = pd.concat(filled_parts, ignore_index=True)
df = df[df["itcs_numberOfPassengers"].notna()].copy()
log(f"  Usable rows: {len(df):,}")

# -----------------------------------------------------------------------
# 3. Temporal split (same cutoff as 03_train_extended.py)
# -----------------------------------------------------------------------

log(f"\n[3] Temporal split (cutoff: {TEMPORAL_CUTOFF.date()})...")
meta_dates = meta[["name", "startTime_iso"]].copy()
meta_dates["startTime_iso"] = pd.to_datetime(meta_dates["startTime_iso"])

train_missions = set(meta_dates.loc[meta_dates["startTime_iso"] < TEMPORAL_CUTOFF, "name"])
test_missions = set(meta_dates.loc[meta_dates["startTime_iso"] >= TEMPORAL_CUTOFF, "name"])

train_df = df[df["mission_name"].isin(train_missions)].copy()
test_df = df[df["mission_name"].isin(test_missions)].copy()

log(f"  Train: {len(train_df):,} rows ({train_df['mission_name'].nunique()} missions)")
log(f"  Test:  {len(test_df):,} rows ({test_df['mission_name'].nunique()} missions)")

if len(test_df) == 0:
    log("ERROR: No test missions after temporal cutoff. Check TEMPORAL_CUTOFF.")
    sys.exit(1)

# -----------------------------------------------------------------------
# 4. Tercile boundaries (train only)
# -----------------------------------------------------------------------

log("\n[4] Computing tercile boundaries from training data...")
q1, q2 = compute_tercile_boundaries(train_df["itcs_numberOfPassengers"])
log(f"  q1={q1:.2f}, q2={q2:.2f}")

train_df = train_df.copy()
test_df = test_df.copy()
train_df["demand_class"] = assign_demand_class(
    train_df["itcs_numberOfPassengers"], q1, q2
)
test_df["demand_class"] = assign_demand_class(
    test_df["itcs_numberOfPassengers"], q1, q2
)

# -----------------------------------------------------------------------
# 5. Load TimesFM
# -----------------------------------------------------------------------

log("\n[5] Loading TimesFM model...")
backend = detect_backend()
tfm = load_timesfm_model(backend)

# -----------------------------------------------------------------------
# 6. Zero-shot inference
# -----------------------------------------------------------------------

log("\n[6] Zero-shot inference...")
t0 = time.time()
y_true_zs, y_pred_zs = run_inference_on_missions(test_df, tfm, q1, q2)
zs_time = time.time() - t0
log(f"  Inference complete in {zs_time:.1f}s ({len(y_true_zs):,} predictions)")

metrics_zs = evaluate_model(y_true_zs, y_pred_zs)
metrics_zs["inference_time_s"] = round(zs_time, 2)
metrics_zs["mode"] = "zero_shot"
metrics_zs["backend"] = backend
metrics_zs["context_len"] = CONTEXT_LEN
metrics_zs["stride"] = STRIDE

log(f"  Zero-shot Macro F1:          {metrics_zs['macro_f1']:.4f}")
log(f"  Zero-shot Balanced Accuracy: {metrics_zs['balanced_accuracy']:.4f}")
for cls in DEMAND_LABELS:
    pc = metrics_zs["per_class"][cls]
    log(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

plot_timesfm_confusion(metrics_zs["confusion_matrix"], "zero_shot", metrics_zs["macro_f1"])

# -----------------------------------------------------------------------
# 7. Fine-tuned inference (optional)
# -----------------------------------------------------------------------

metrics_ft = None
if FINETUNE:
    log("\n[7] Fine-tuning TimesFM on training missions...")
    t0 = time.time()
    finetune_timesfm(tfm, train_df)
    ft_train_time = time.time() - t0
    log(f"  Fine-tuning took {ft_train_time:.1f}s")

    log("  Running fine-tuned inference...")
    t0 = time.time()
    y_true_ft, y_pred_ft = run_inference_on_missions(test_df, tfm, q1, q2)
    ft_inf_time = time.time() - t0
    log(f"  Inference complete in {ft_inf_time:.1f}s")

    metrics_ft = evaluate_model(y_true_ft, y_pred_ft)
    metrics_ft["inference_time_s"] = round(ft_inf_time, 2)
    metrics_ft["finetune_time_s"] = round(ft_train_time, 2)
    metrics_ft["mode"] = "fine_tuned"
    metrics_ft["backend"] = backend

    log(f"  Fine-tuned Macro F1:          {metrics_ft['macro_f1']:.4f}")
    log(f"  Fine-tuned Balanced Accuracy: {metrics_ft['balanced_accuracy']:.4f}")
    for cls in DEMAND_LABELS:
        pc = metrics_ft["per_class"][cls]
        log(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

    plot_timesfm_confusion(metrics_ft["confusion_matrix"], "fine_tuned", metrics_ft["macro_f1"])

# -----------------------------------------------------------------------
# 8. Sample forecast plot
# -----------------------------------------------------------------------

log("\n[8] Generating sample forecast plot...")
try:
    sample_mission = list(test_missions)[0]
    sample_df = test_df[test_df["mission_name"] == sample_mission]
    pax_sample = sample_df["itcs_numberOfPassengers"].values.astype(float)
    if len(pax_sample) > MIN_CONTEXT:
        ctx = _interpolate_nans(pax_sample[:CONTEXT_LEN])
        forecast_val, _ = tfm.forecast(inputs=[ctx], freq=[0])
        plot_forecast_sample(
            ctx, forecast_val[0, 0], pax_sample[CONTEXT_LEN] if len(pax_sample) > CONTEXT_LEN else np.nan,
            q1, q2, sample_mission,
        )
except Exception as e:
    log(f"  Sample forecast plot failed (non-critical): {e}")

# -----------------------------------------------------------------------
# 9. Save results
# -----------------------------------------------------------------------

log("\n[9] Saving results...")

def _metrics_to_dict(m: dict) -> dict:
    return {
        "macro_f1": m["macro_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "confusion_matrix": m["confusion_matrix"].tolist(),
        "per_class": m["per_class"],
        "inference_time_s": m.get("inference_time_s"),
        "finetune_time_s": m.get("finetune_time_s"),
        "mode": m.get("mode"),
        "backend": m.get("backend"),
        "context_len": m.get("context_len", CONTEXT_LEN),
        "stride": m.get("stride", STRIDE),
    }

results_out = {"zero_shot": _metrics_to_dict(metrics_zs)}
if metrics_ft:
    results_out["fine_tuned"] = _metrics_to_dict(metrics_ft)

results_path = os.path.join(RESULTS_TFM_DIR, "timesfm_results.json")
with open(results_path, "w") as f:
    json.dump(results_out, f, indent=2)
log(f"  Saved: {results_path}")

# Append to extended comparison table if it exists
ext_comparison_path = os.path.join(RESULTS_DIR, "extended", "model_comparison_extended.csv")
if os.path.exists(ext_comparison_path):
    ext_df = pd.read_csv(ext_comparison_path)
    new_rows = []
    for mode, m in results_out.items():
        new_rows.append({
            "model": f"timesfm_{mode}",
            "framework": "temporal",
            "macro_f1": round(m["macro_f1"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "f1_low": round(m["per_class"]["low"]["f1"], 4),
            "f1_medium": round(m["per_class"]["medium"]["f1"], 4),
            "f1_high": round(m["per_class"]["high"]["f1"], 4),
            "train_time_s": m.get("finetune_time_s", 0),
        })
    ext_df = pd.concat([ext_df, pd.DataFrame(new_rows)], ignore_index=True)
    ext_df.to_csv(ext_comparison_path, index=False)
    log(f"  Updated: model_comparison_extended.csv with TimesFM rows")

# -----------------------------------------------------------------------
# 10. Summary
# -----------------------------------------------------------------------

log("\n" + "=" * 60)
log("TIMESFM EVALUATION COMPLETE")
log("=" * 60)
log(f"\n  Backend:         {backend}")
log(f"  Context length:  {CONTEXT_LEN}s")
log(f"  Stride:          {STRIDE}s")
log(f"  Test missions:   {test_df['mission_name'].nunique()}")
log(f"  Test predictions:{len(y_true_zs):,}")
log(f"\n  Zero-shot  — F1={metrics_zs['macro_f1']:.4f}  "
    f"BalAcc={metrics_zs['balanced_accuracy']:.4f}")
if metrics_ft:
    log(f"  Fine-tuned — F1={metrics_ft['macro_f1']:.4f}  "
        f"BalAcc={metrics_ft['balanced_accuracy']:.4f}")
log(f"\n  Results: {RESULTS_TFM_DIR}/")
log(f"  Figures: {FIGURES_TFM_DIR}/")
