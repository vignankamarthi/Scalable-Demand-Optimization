"""
Extract TimesFM penultimate-layer embeddings for all missions.
Saves per-mission 32-dim PCA-reduced embeddings to data/cache/embeddings/.

Runs on cluster (GPU partition). Requires Python 3.11 + timesfm.

Flow:
  1. Load TimesFM 2.5-200M
  2. Register forward hook on last transformer layer
  3. For each mission: build windows, run forward pass, capture embeddings
  4. Mean-pool per mission -> one 1280-dim vector per mission
  5. PCA reduce to 32 dims (fit on train missions only)
  6. Save per-mission NPY files + PCA model

Usage:
    python scripts/08_extract_embeddings.py
    python scripts/08_extract_embeddings.py --cpu   # force CPU backend
"""
from __future__ import annotations

import sys
import os
import json
import time
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED, EDA_USE_COLS, METADATA_PATH,
)
from src.data_loading import load_metadata, load_mission_csv
from src.preprocessing import forward_fill_stop_columns

from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01", tz="UTC")
CONTEXT_LEN = 512
STRIDE = 60
MIN_CONTEXT = 64
BATCH_SIZE = 256
EMBEDDING_DIM_RAW = 1280
EMBEDDING_DIM_PCA = 32

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cache", "embeddings",
)
os.makedirs(CACHE_DIR, exist_ok=True)

FORCE_CPU = "--cpu" in sys.argv


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reuse windowing from 04_timesfm_eval.py
# ---------------------------------------------------------------------------

def build_windows(series, context_len=CONTEXT_LEN, stride=STRIDE,
                  min_context=MIN_CONTEXT):
    """Build overlapping context windows from a 1D time series."""
    n = len(series)
    windows = []
    for end in range(min_context, n, stride):
        start = max(0, end - context_len)
        window = series[start:end].copy().astype(float)
        nans = np.isnan(window)
        if nans.any():
            idx = np.arange(len(window))
            if (~nans).any():
                window[nans] = np.interp(idx[nans], idx[~nans], window[~nans])
            else:
                window[:] = 0.0
        windows.append(window)
    return windows


# ---------------------------------------------------------------------------
# Model loading + hook
# ---------------------------------------------------------------------------

def load_model():
    """Load TimesFM and register embedding hook."""
    try:
        import timesfm
    except ImportError:
        raise ImportError(
            "timesfm not installed. Install with:\n"
            "  pip install git+https://github.com/google-research/timesfm.git"
        )

    log("Loading TimesFM 2.5-200M...")
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    config = timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=1,
        per_core_batch_size=BATCH_SIZE,
    )
    tfm.compile(config)
    log("  Model loaded and compiled.")

    # Register hook on last transformer layer
    captured = {"embeddings": None}

    def hook_fn(module, input, output):
        # output is the hidden state after the last transformer layer
        # Shape: [batch, seq_len, 1280]
        if isinstance(output, tuple):
            captured["embeddings"] = output[0].detach().cpu().numpy()
        else:
            captured["embeddings"] = output.detach().cpu().numpy()

    # Access the underlying module's transformer stack
    # Verified via probe: tfm.model.stacked_xf is a ModuleList of 20 Transformer layers
    model_module = tfm.model
    stacked_xf = model_module.stacked_xf
    stacked_xf[-1].register_forward_hook(hook_fn)
    log(f"  Hook registered on transformer layer {len(stacked_xf) - 1} (stacked_xf)")

    return tfm, captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log("=" * 60)
log("TimesFM Embedding Extraction (08_extract_embeddings.py)")
log("=" * 60)

# 1. Load metadata and determine train/test split
log("\n[1] Loading metadata...")
meta = load_metadata()
meta_dates = meta[["name", "startTime_iso"]].copy()
meta_dates["startTime_iso"] = pd.to_datetime(meta_dates["startTime_iso"])
train_missions = set(meta_dates.loc[meta_dates["startTime_iso"] < TEMPORAL_CUTOFF, "name"])
test_missions = set(meta_dates.loc[meta_dates["startTime_iso"] >= TEMPORAL_CUTOFF, "name"])
all_missions = list(meta["name"])
log(f"  Total missions: {len(all_missions)}")
log(f"  Train: {len(train_missions)}, Test: {len(test_missions)}")

# 2. Load model
log("\n[2] Loading TimesFM model...")
tfm, captured = load_model()

# 3. Extract raw embeddings for all missions
log("\n[3] Extracting embeddings for all missions...")

raw_embeddings = {}  # mission_name -> 1280-dim vector (mean-pooled)
cols = ["time_unix", "itcs_numberOfPassengers"]

for i, mission_name in enumerate(all_missions):
    if (i + 1) % 100 == 0:
        log(f"  Progress: {i+1}/{len(all_missions)} missions")

    df = load_mission_csv(mission_name, usecols=cols)
    if df is None:
        continue

    df = forward_fill_stop_columns(df)
    pax = df["itcs_numberOfPassengers"].values.astype(float)
    pax = pax[~np.isnan(pax)]

    if len(pax) < MIN_CONTEXT + 1:
        continue

    windows = build_windows(pax)
    if not windows:
        continue

    # Run inference in batches (hook captures embeddings)
    mission_embeddings = []
    for batch_start in range(0, len(windows), BATCH_SIZE):
        batch = windows[batch_start:batch_start + BATCH_SIZE]
        try:
            tfm.forecast(horizon=1, inputs=batch)
        except Exception as e:
            log(f"  WARNING: {mission_name} batch failed: {e}")
            continue

        if captured["embeddings"] is not None:
            # Mean-pool across sequence dimension: [batch, seq, 1280] -> [batch, 1280]
            batch_emb = captured["embeddings"].mean(axis=1)
            mission_embeddings.append(batch_emb)

    if not mission_embeddings:
        continue

    # Mean-pool across all windows for this mission -> [1280]
    all_batch_emb = np.concatenate(mission_embeddings, axis=0)
    mission_emb = all_batch_emb.mean(axis=0)
    raw_embeddings[mission_name] = mission_emb

log(f"  Extracted embeddings for {len(raw_embeddings)} missions")

# 4. PCA reduction (fit on train only)
log(f"\n[4] PCA reduction: {EMBEDDING_DIM_RAW} -> {EMBEDDING_DIM_PCA} dims...")

train_emb_names = [n for n in raw_embeddings if n in train_missions]
train_matrix = np.array([raw_embeddings[n] for n in train_emb_names])
log(f"  Fitting PCA on {len(train_emb_names)} training mission embeddings")

pca = PCA(n_components=EMBEDDING_DIM_PCA, random_state=RANDOM_SEED)
pca.fit(train_matrix)
explained = pca.explained_variance_ratio_.sum()
log(f"  Explained variance: {explained:.4f} ({EMBEDDING_DIM_PCA} components)")

# 5. Transform and save all embeddings
log(f"\n[5] Saving PCA-reduced embeddings to {CACHE_DIR}/...")

saved_count = 0
for mission_name, raw_emb in raw_embeddings.items():
    reduced = pca.transform(raw_emb.reshape(1, -1))[0]
    np.save(os.path.join(CACHE_DIR, f"{mission_name}.npy"), reduced)
    saved_count += 1

# Save PCA model
pca_path = os.path.join(CACHE_DIR, "pca_model.pkl")
with open(pca_path, "wb") as f:
    pickle.dump(pca, f)

log(f"  Saved {saved_count} mission embeddings + PCA model")

# 6. Summary
log("\n" + "=" * 60)
log("EMBEDDING EXTRACTION COMPLETE")
log("=" * 60)
log(f"  Missions processed: {len(raw_embeddings)} / {len(all_missions)}")
log(f"  PCA dimensions: {EMBEDDING_DIM_PCA}")
log(f"  Explained variance: {explained:.4f}")
log(f"  Cache dir: {CACHE_DIR}")
log(f"  PCA model: {pca_path}")
