"""
Train ML models WITH lag features (pax_lag_60, pax_lag_300).
Lag features are now integrated into build_feature_set().
This script re-runs the same pipeline as 03_train_extended.py
but saves results to results/lag_features/ for comparison.

Runs on cluster: 16 CPUs, 256GB RAM.

Usage:
    python scripts/07_train_with_lags.py             # resume from checkpoint
    python scripts/07_train_with_lags.py --fresh     # fresh start
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
    FIGURES_DIR, RESULTS_DIR, RANDOM_SEED, EDA_USE_COLS,
    FEATURE_GROUP_MAP, FEATURE_GROUP_PREFIXES, METADATA_PATH, DEMAND_LABELS,
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
# Paths (separate from 03_train_extended to preserve baseline comparison)
# ---------------------------------------------------------------------------

RESULTS_LAG_DIR = os.path.join(RESULTS_DIR, "lag_features")
FIGURES_LAG_DIR = os.path.join(FIGURES_DIR, "lag_features")
os.makedirs(RESULTS_LAG_DIR, exist_ok=True)
os.makedirs(FIGURES_LAG_DIR, exist_ok=True)

CHECKPOINT_POOLED = os.path.join(RESULTS_LAG_DIR, "checkpoint_pooled.json")
CHECKPOINT_TEMPORAL = os.path.join(RESULTS_LAG_DIR, "checkpoint_temporal.json")
FRESH_RUN = "--fresh" in sys.argv

TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01", tz="UTC")


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
                n_estimators=300, n_jobs=-1, random_state=seed,
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
    """Prepare aligned X, y matrices from feature-engineered DataFrames."""
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


def run_training(X_train, y_train, X_test, y_test, framework, checkpoint_path):
    """Train all models, evaluate, return results dict."""
    configs = get_model_configs()
    le = LabelEncoder()
    le.fit(DEMAND_LABELS)

    ckpt = {} if FRESH_RUN else load_checkpoint(checkpoint_path)
    results = {}

    for name, cfg in configs.items():
        if name in ckpt:
            log(f"  {name}: loaded from checkpoint")
            results[name] = ckpt[name]
            continue

        model = cfg["model"]
        log(f"  Training {name}...")
        t0 = time.time()

        if cfg["needs_label_encoding"]:
            y_tr = le.transform(y_train)
        else:
            y_tr = y_train

        model.fit(X_train, y_tr)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred_raw = model.predict(X_test)
        predict_time = time.time() - t0

        if cfg["needs_label_encoding"]:
            y_pred = pd.Series(le.inverse_transform(y_pred_raw), index=y_test.index)
        else:
            y_pred = pd.Series(y_pred_raw, index=y_test.index)

        metrics = evaluate_model(y_test, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["predict_time_s"] = round(predict_time, 2)
        results[name] = metrics

        log(f"    F1={metrics['macro_f1']:.4f} BalAcc={metrics['balanced_accuracy']:.4f} "
            f"({train_time:.0f}s train, {predict_time:.1f}s predict)")

        # Save confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=DEMAND_LABELS, yticklabels=DEMAND_LABELS, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} [{framework}] (F1={metrics['macro_f1']:.4f}) -- with lags")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_LAG_DIR, f"cm_{framework}_{name}.png"),
                    bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)

        # Checkpoint after each model
        # save_checkpoint expects numpy CM + "params" key
        metrics["params"] = {}
        ckpt[name] = metrics
        save_checkpoint(ckpt, checkpoint_path)

    return results


# ===========================================================================
# Main
# ===========================================================================

log("=" * 60)
log("Training with Lag Features (07_train_with_lags.py)")
log("=" * 60)
log(f"Lag features: pax_stop_lag_1, pax_stop_lag_3, pax_stop_lag_5 (stop-level)")

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

# 3. Splits
log("\n[3] Building train/test splits...")

# Pooled
train_pooled, test_pooled = mission_stratified_split(
    df, test_size=0.2, seed=RANDOM_SEED)
log(f"  Pooled  -- train: {len(train_pooled):,} | test: {len(test_pooled):,}")

# Temporal
meta_dates = meta[["name", "startTime_iso"]].copy()
meta_dates["startTime_iso"] = pd.to_datetime(meta_dates["startTime_iso"])
train_missions = set(meta_dates.loc[meta_dates["startTime_iso"] < TEMPORAL_CUTOFF, "name"])
test_missions = set(meta_dates.loc[meta_dates["startTime_iso"] >= TEMPORAL_CUTOFF, "name"])
train_temporal = df[df["mission_name"].isin(train_missions)].copy()
test_temporal = df[df["mission_name"].isin(test_missions)].copy()
log(f"  Temporal -- train: {len(train_temporal):,} | test: {len(test_temporal):,}")

# 4. Tercile boundaries (train only)
log("\n[4] Computing tercile boundaries...")
q1_p, q2_p = compute_tercile_boundaries(train_pooled["itcs_numberOfPassengers"])
q1_t, q2_t = compute_tercile_boundaries(train_temporal["itcs_numberOfPassengers"])

for split_name, (tr, te, q1, q2) in [
    ("pooled", (train_pooled, test_pooled, q1_p, q2_p)),
    ("temporal", (train_temporal, test_temporal, q1_t, q2_t)),
]:
    tr["demand_class"] = assign_demand_class(tr["itcs_numberOfPassengers"], q1, q2)
    te["demand_class"] = assign_demand_class(te["itcs_numberOfPassengers"], q1, q2)

# 5. Feature engineering (now includes lag features)
log("\n[5] Feature engineering (with lag features)...")
t0 = time.time()
train_pooled_feat = build_feature_set(train_pooled)
test_pooled_feat = build_feature_set(test_pooled)
train_temporal_feat = build_feature_set(train_temporal)
test_temporal_feat = build_feature_set(test_temporal)
log(f"  Feature engineering done in {time.time()-t0:.1f}s")

# Verify lag columns exist
lag_cols = [c for c in train_pooled_feat.columns if c.startswith("pax_stop_lag_")]
log(f"  Lag features present: {lag_cols}")
assert len(lag_cols) >= 3, f"Expected 3 stop-level lag features, found: {lag_cols}"

log(f"  Total features: {len([c for c in train_pooled_feat.columns if c not in ['mission_name', 'itcs_numberOfPassengers', 'demand_class']])}")

# 6. Prepare X, y
log("\n[6] Preparing feature matrices...")
X_tr_p, y_tr_p, X_te_p, y_te_p = prepare_xy(train_pooled_feat, test_pooled_feat)
X_tr_t, y_tr_t, X_te_t, y_te_t = prepare_xy(train_temporal_feat, test_temporal_feat)
log(f"  Pooled  -- X_train: {X_tr_p.shape}, X_test: {X_te_p.shape}")
log(f"  Temporal -- X_train: {X_tr_t.shape}, X_test: {X_te_t.shape}")

# 7. Train -- pooled
log("\n[7] Training (pooled evaluation)...")
results_pooled = run_training(X_tr_p, y_tr_p, X_te_p, y_te_p, "pooled", CHECKPOINT_POOLED)

# 8. Train -- temporal
log("\n[8] Training (temporal evaluation)...")
results_temporal = run_training(X_tr_t, y_tr_t, X_te_t, y_te_t, "temporal", CHECKPOINT_TEMPORAL)

# 9. Save results
log("\n[9] Saving results...")

def serialize_results(results):
    out = {}
    for name, m in results.items():
        out[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}
        if "confusion_matrix" in m:
            cm = m["confusion_matrix"]
            out[name]["confusion_matrix"] = cm.tolist() if hasattr(cm, "tolist") else cm
    return out

for framework, results in [("pooled", results_pooled), ("temporal", results_temporal)]:
    path = os.path.join(RESULTS_LAG_DIR, f"results_{framework}.json")
    with open(path, "w") as f:
        json.dump(serialize_results(results), f, indent=2)
    log(f"  Saved: {path}")

# 10. Summary
log("\n" + "=" * 60)
log("RESULTS WITH LAG FEATURES")
log("=" * 60)

for framework, results in [("pooled", results_pooled), ("temporal", results_temporal)]:
    log(f"\n  {framework.upper()} evaluation:")
    log(f"  {'Model':<20} {'Macro F1':>10} {'Bal Acc':>10}")
    log("  " + "-" * 45)
    for name, m in results.items():
        log(f"  {name:<20} {m['macro_f1']:>10.4f} {m['balanced_accuracy']:>10.4f}")

# Compare against baseline
ext_temporal_path = os.path.join(RESULTS_DIR, "extended", "results_temporal.json")
if os.path.exists(ext_temporal_path):
    with open(ext_temporal_path) as f:
        baseline = json.load(f)
    log(f"\n  TEMPORAL: Baseline vs With-Lags")
    log(f"  {'Model':<20} {'Baseline F1':>12} {'Lag F1':>12} {'Delta':>10}")
    log("  " + "-" * 60)
    for name in results_temporal:
        if name in baseline:
            b_f1 = baseline[name]["macro_f1"]
            l_f1 = results_temporal[name]["macro_f1"]
            delta = l_f1 - b_f1
            sign = "+" if delta > 0 else ""
            log(f"  {name:<20} {b_f1:>12.4f} {l_f1:>12.4f} {sign}{delta:>9.4f}")

log("\nDone.")
