"""
Train ML models with TimesFM embeddings as additional features.
Requires pre-computed embeddings from 08_extract_embeddings.py.

Tests four feature configurations:
  1. Baseline (~50 features, from 03_train_extended results)
  2. + Lag features (~52 features, from 07_train_with_lags results)
  3. + TimesFM embeddings (~82 features)
  4. + Both lags + embeddings (~84 features)

Runs on cluster: 16 CPUs, 256GB RAM.

Usage:
    python scripts/09_train_with_embeddings.py
    python scripts/09_train_with_embeddings.py --fresh
"""
from __future__ import annotations

import sys
import os
import json
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FIGURES_DIR, RESULTS_DIR, RANDOM_SEED, EDA_USE_COLS, DEMAND_LABELS,
)
from src.data_loading import load_metadata, load_sample_missions
from src.preprocessing import forward_fill_stop_columns
from src.feature_engineering import build_feature_set
from src.target import compute_tercile_boundaries, assign_demand_class
from src.model_pipeline import (
    mission_stratified_split,
    prepare_features_and_target,
    evaluate_model,
)
from src.timesfm_features import attach_embeddings
from src.checkpoint import load_checkpoint, save_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. Continuing with DT and RF only.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_EMB_DIR = os.path.join(RESULTS_DIR, "embeddings")
FIGURES_EMB_DIR = os.path.join(FIGURES_DIR, "embeddings")
os.makedirs(RESULTS_EMB_DIR, exist_ok=True)
os.makedirs(FIGURES_EMB_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_EMB_DIR, "checkpoint.json")
FRESH_RUN = "--fresh" in sys.argv

TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01", tz="UTC")

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cache", "embeddings",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_model_configs(seed=RANDOM_SEED):
    configs = {
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=seed),
            "needs_label_encoding": False,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=seed,
            ),
            "needs_label_encoding": False,
        },
    }
    if XGB_AVAILABLE:
        configs["xgboost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                n_jobs=-1, random_state=seed, eval_metric="mlogloss", verbosity=0,
            ),
            "needs_label_encoding": True,
        }
    return configs


def prepare_xy(train_feat, test_feat):
    """Prepare aligned X, y matrices."""
    X_tr, y_tr = prepare_features_and_target(
        train_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    X_te, y_te = prepare_features_and_target(
        test_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    common = sorted(set(X_tr.columns) & set(X_te.columns))
    X_tr, X_te = X_tr[common], X_te[common]
    tr_mask = X_tr.notna().all(axis=1)
    te_mask = X_te.notna().all(axis=1)
    return X_tr[tr_mask], y_tr[tr_mask], X_te[te_mask], y_te[te_mask]


def train_and_evaluate(X_train, y_train, X_test, y_test, config_name):
    """Train all models, return results dict."""
    configs = get_model_configs()
    le = LabelEncoder()
    le.fit(DEMAND_LABELS)
    results = {}

    for name, cfg in configs.items():
        model = cfg["model"]
        log(f"    {name}...")
        t0 = time.time()

        if cfg["needs_label_encoding"]:
            y_tr = le.transform(y_train)
        else:
            y_tr = y_train

        model.fit(X_train, y_tr)
        train_time = time.time() - t0

        y_pred_raw = model.predict(X_test)

        if cfg["needs_label_encoding"]:
            y_pred = pd.Series(le.inverse_transform(y_pred_raw), index=y_test.index)
        else:
            y_pred = pd.Series(y_pred_raw, index=y_test.index)

        metrics = evaluate_model(y_test, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        results[name] = metrics

        log(f"      F1={metrics['macro_f1']:.4f} BalAcc={metrics['balanced_accuracy']:.4f} ({train_time:.0f}s)")

        # Save confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Greens",
                    xticklabels=DEMAND_LABELS, yticklabels=DEMAND_LABELS, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} [{config_name}] (F1={metrics['macro_f1']:.4f})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_EMB_DIR, f"cm_{config_name}_{name}.png"),
                    bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)

    return results


# ===========================================================================
# Main
# ===========================================================================

log("=" * 60)
log("Training with TimesFM Embeddings (09_train_with_embeddings.py)")
log("=" * 60)

# Verify embeddings exist
if not os.path.exists(CACHE_DIR):
    log(f"ERROR: Embedding cache not found at {CACHE_DIR}")
    log("Run scripts/08_extract_embeddings.py first.")
    sys.exit(1)

npy_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
log(f"  Found {len(npy_files)} cached embeddings in {CACHE_DIR}")

# 1. Load data
log("\n[1] Loading data...")
meta = load_metadata()
t0 = time.time()
df_raw = load_sample_missions(meta, usecols=EDA_USE_COLS)
log(f"  Loaded {len(df_raw):,} rows in {time.time()-t0:.1f}s")

# 2. Forward-fill
log("\n[2] Forward-filling ITCS columns...")
parts = []
for mission_name, group in df_raw.groupby("mission_name"):
    parts.append(forward_fill_stop_columns(group))
df = pd.concat(parts, ignore_index=True)
df = df[df["itcs_numberOfPassengers"].notna()].copy()
log(f"  Usable rows: {len(df):,}")

# 3. Temporal split
log("\n[3] Temporal split...")
meta_dates = meta[["name", "startTime_iso"]].copy()
meta_dates["startTime_iso"] = pd.to_datetime(meta_dates["startTime_iso"])
train_missions = set(meta_dates.loc[meta_dates["startTime_iso"] < TEMPORAL_CUTOFF, "name"])
test_missions = set(meta_dates.loc[meta_dates["startTime_iso"] >= TEMPORAL_CUTOFF, "name"])
train_df = df[df["mission_name"].isin(train_missions)].copy()
test_df = df[df["mission_name"].isin(test_missions)].copy()
log(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

# 4. Tercile boundaries
log("\n[4] Tercile boundaries...")
q1, q2 = compute_tercile_boundaries(train_df["itcs_numberOfPassengers"])
train_df["demand_class"] = assign_demand_class(train_df["itcs_numberOfPassengers"], q1, q2)
test_df["demand_class"] = assign_demand_class(test_df["itcs_numberOfPassengers"], q1, q2)

# 5. Feature engineering (includes lag features)
log("\n[5] Feature engineering...")
t0 = time.time()
train_feat = build_feature_set(train_df)
test_feat = build_feature_set(test_df)
log(f"  Done in {time.time()-t0:.1f}s")

# 6. Attach embeddings
log("\n[6] Attaching TimesFM embeddings...")
train_feat_emb = attach_embeddings(train_feat, CACHE_DIR)
test_feat_emb = attach_embeddings(test_feat, CACHE_DIR)

emb_cols = [c for c in train_feat_emb.columns if c.startswith("tfm_emb_")]
log(f"  Embedding columns: {len(emb_cols)}")

# Count missions with embeddings
train_has_emb = train_feat_emb.groupby("mission_name")["tfm_emb_0"].first().notna().sum()
test_has_emb = test_feat_emb.groupby("mission_name")["tfm_emb_0"].first().notna().sum()
log(f"  Train missions with embeddings: {train_has_emb}")
log(f"  Test missions with embeddings: {test_has_emb}")

# 7. Train: lags + embeddings (temporal eval)
log("\n[7] Training: lags + embeddings (temporal)...")
X_tr, y_tr, X_te, y_te = prepare_xy(train_feat_emb, test_feat_emb)
log(f"  X_train: {X_tr.shape}, X_test: {X_te.shape}")
results_both = train_and_evaluate(X_tr, y_tr, X_te, y_te, "lags_emb_temporal")

# 8. Save results
log("\n[8] Saving results...")

def serialize(results):
    out = {}
    for name, m in results.items():
        out[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}
        if "confusion_matrix" in m:
            cm = m["confusion_matrix"]
            out[name]["confusion_matrix"] = cm.tolist() if hasattr(cm, "tolist") else cm
    return out

path = os.path.join(RESULTS_EMB_DIR, "results_temporal.json")
with open(path, "w") as f:
    json.dump(serialize(results_both), f, indent=2)
log(f"  Saved: {path}")

# 9. Summary comparison
log("\n" + "=" * 60)
log("FULL MODEL COMPARISON (Temporal Eval)")
log("=" * 60)

log(f"\n  {'Config':<30} {'Model':<20} {'Macro F1':>10}")
log("  " + "-" * 65)

# Load baselines
for config_name, results_path in [
    ("Baseline (no temporal)", os.path.join(RESULTS_DIR, "extended", "results_temporal.json")),
    ("+ Lag features", os.path.join(RESULTS_DIR, "lag_features", "results_temporal.json")),
]:
    if os.path.exists(results_path):
        with open(results_path) as f:
            baseline = json.load(f)
        for model_name, m in baseline.items():
            log(f"  {config_name:<30} {model_name:<20} {m['macro_f1']:>10.4f}")

for model_name, m in results_both.items():
    log(f"  {'+ Lags + Embeddings':<30} {model_name:<20} {m['macro_f1']:>10.4f}")

log("\nDone.")
