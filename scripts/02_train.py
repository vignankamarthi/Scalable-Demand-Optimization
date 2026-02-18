"""
Training script for ZTBus ridership demand classification.
Designed to run on NEU Explorer cluster (16 CPUs, 128GB RAM).

Trains Decision Tree and Random Forest on ~37M rows, evaluates on ~9M rows.
Supports per-model checkpointing: if the job is killed mid-training,
re-running the script resumes from the last completed model.
Use --fresh to discard checkpoint and start from scratch.

Usage:
    python scripts/02_train.py           # resume from checkpoint if present
    python scripts/02_train.py --fresh   # ignore checkpoint, start over
"""

import sys
import os
import json
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FIGURES_DIR, RESULTS_DIR, RANDOM_SEED, EDA_USE_COLS,
    FEATURE_GROUP_MAP, FEATURE_GROUP_PREFIXES,
)
from src.data_loading import load_metadata, load_sample_missions
from src.preprocessing import forward_fill_stop_columns, apply_unit_conversions
from src.feature_engineering import build_feature_set
from src.target import compute_tercile_boundaries, assign_demand_class
from src.model_pipeline import (
    mission_stratified_split,
    prepare_features_and_target,
    evaluate_model,
    get_model_configs,
)
from src.checkpoint import load_checkpoint, save_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.json")
FRESH_RUN = "--fresh" in sys.argv


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------
# 1. Load ALL missions
# -----------------------------------------------------------------------

log("=" * 60)
log("ZTBus Training Pipeline")
log("=" * 60)

log("Loading metadata...")
meta = load_metadata()
log(f"  Total missions: {len(meta)}")

log("Loading ALL mission CSVs...")
t0 = time.time()
df = load_sample_missions(meta, usecols=EDA_USE_COLS)
log(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------
# 2. Forward-fill and preprocess
# -----------------------------------------------------------------------

log("Forward-filling ITCS columns per mission...")
t0 = time.time()
filled_parts = []
for mission_name, group in df.groupby("mission_name"):
    filled = forward_fill_stop_columns(group)
    filled_parts.append(filled)
df = pd.concat(filled_parts, ignore_index=True)
# Drop rows before first stop (NaN passenger count)
df = df[df["itcs_numberOfPassengers"].notna()].copy()
log(f"  After forward-fill: {len(df):,} rows ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. Mission-level train/test split (BEFORE feature engineering)
# -----------------------------------------------------------------------

log("Mission-level train/test split (80/20)...")
train_df, test_df = mission_stratified_split(df, test_size=0.2, seed=RANDOM_SEED)
log(f"  Train: {len(train_df):,} rows ({train_df['mission_name'].nunique()} missions)")
log(f"  Test:  {len(test_df):,} rows ({test_df['mission_name'].nunique()} missions)")

# -----------------------------------------------------------------------
# 4. Compute tercile boundaries on TRAINING data only
# -----------------------------------------------------------------------

log("Computing tercile boundaries on training data...")
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

log("Train class distribution:")
for cls in ["low", "medium", "high"]:
    n = (train_df["demand_class"] == cls).sum()
    log(f"  {cls}: {n:,} ({n/len(train_df)*100:.1f}%)")

# -----------------------------------------------------------------------
# 5. Feature engineering
# -----------------------------------------------------------------------

log("Feature engineering...")
t0 = time.time()
train_feat = build_feature_set(train_df)
test_feat = build_feature_set(test_df)
log(f"  Feature columns: {len(train_feat.columns) - 2}")  # minus mission_name, demand_class
log(f"  Done in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------
# 6. Prepare X, y
# -----------------------------------------------------------------------

X_train, y_train = prepare_features_and_target(
    train_feat, target_col="demand_class",
    drop_cols=["mission_name", "itcs_numberOfPassengers"],
)
X_test, y_test = prepare_features_and_target(
    test_feat, target_col="demand_class",
    drop_cols=["mission_name", "itcs_numberOfPassengers"],
)

# Align columns (test may have different one-hot columns)
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Drop any remaining NaN rows (from rolling windows at mission boundaries)
train_mask = X_train.notna().all(axis=1)
test_mask = X_test.notna().all(axis=1)
X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

log(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
log(f"  Features: {list(X_train.columns)}")

# -----------------------------------------------------------------------
# 7. Train and evaluate all models (with per-model checkpointing)
# -----------------------------------------------------------------------

results = load_checkpoint(CHECKPOINT_PATH, fresh=FRESH_RUN)
if results:
    log(f"\nResuming from checkpoint: {len(results)} model(s) already completed")
    for name in results:
        log(f"  [checkpoint] {name}: F1={results[name]['macro_f1']:.4f}")

configs = get_model_configs(seed=RANDOM_SEED)
trained_models = {}  # retain fitted model objects for post-training analysis

for model_name, cfg in configs.items():
    if model_name in results:
        log(f"\n--- Skipping: {model_name} (already in checkpoint) ---")
        continue

    log(f"\n--- Training: {model_name} ---")
    model = cfg["model"]

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log(f"  Trained in {train_time:.1f}s")
    trained_models[model_name] = model

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    metrics = evaluate_model(y_test, pd.Series(y_pred, index=y_test.index))
    metrics["train_time_s"] = round(train_time, 2)
    metrics["predict_time_s"] = round(pred_time, 2)
    metrics["params"] = cfg["params"]

    results[model_name] = metrics
    save_checkpoint(results, CHECKPOINT_PATH)
    log(f"  [checkpoint saved: {len(results)}/{len(configs)} models]")

    log(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    log(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    log(f"  Predict time:      {pred_time:.2f}s")
    for cls in ["low", "medium", "high"]:
        pc = metrics["per_class"][cls]
        log(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

# -----------------------------------------------------------------------
# 8. Save results
# -----------------------------------------------------------------------

log("\n--- Saving results ---")

# Summary table
summary_rows = []
for model_name, m in results.items():
    summary_rows.append({
        "model": model_name,
        "macro_f1": round(m["macro_f1"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "train_time_s": m["train_time_s"],
        "predict_time_s": m["predict_time_s"],
    })
summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
summary_df.to_csv(summary_path, index=False)
log(f"  Summary: {summary_path}")

# Full results (JSON-serializable)
json_results = {}
for model_name, m in results.items():
    json_results[model_name] = {
        "macro_f1": m["macro_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "confusion_matrix": m["confusion_matrix"].tolist(),
        "per_class": m["per_class"],
        "train_time_s": m["train_time_s"],
        "predict_time_s": m["predict_time_s"],
        "params": m["params"],
    }
json_path = os.path.join(RESULTS_DIR, "model_results.json")
with open(json_path, "w") as f:
    json.dump(json_results, f, indent=2)
log(f"  Full results: {json_path}")

# Remove checkpoint now that final results are saved
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    log("  Checkpoint cleared (final results saved)")

# -----------------------------------------------------------------------
# 9. Confusion matrix plots
# -----------------------------------------------------------------------

log("\n--- Generating confusion matrix plots ---")
for model_name, m in results.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        m["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=["low", "medium", "high"],
        yticklabels=["low", "medium", "high"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} (F1={m['macro_f1']:.4f})")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, f"cm_{model_name}.png"),
        bbox_inches="tight", dpi=150, facecolor="white",
    )
    plt.close(fig)
    log(f"  Saved: cm_{model_name}.png")

# -----------------------------------------------------------------------
# 9b. Feature importance (Random Forest)
# -----------------------------------------------------------------------

log("\n--- Feature importance analysis (Random Forest) ---")


def _feature_to_group(feat_name):
    """Map a feature name to its group using exact match or prefix."""
    if feat_name in FEATURE_GROUP_MAP:
        return FEATURE_GROUP_MAP[feat_name]
    for prefix, group in FEATURE_GROUP_PREFIXES.items():
        if feat_name.startswith(prefix):
            return group
    return "Unknown"


if "random_forest" in trained_models:
    rf_model = trained_models["random_forest"]
    importances = rf_model.feature_importances_
    feature_names = X_train.columns.tolist()

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Top-20 feature importance bar chart
    top20 = imp_df.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="#2196F3")
    ax.set_xlabel("Gini Importance")
    ax.set_title("Random Forest -- Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "feature_importance.png"),
        bbox_inches="tight", dpi=150, facecolor="white",
    )
    plt.close(fig)
    log("  Saved: feature_importance.png")

    # Group-level importance
    imp_df["group"] = imp_df["feature"].apply(_feature_to_group)
    group_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Temporal": "#FF9800", "Spatial": "#4CAF50", "Operational": "#2196F3",
        "Sensor": "#9C27B0", "Status": "#F44336", "Categorical": "#607D8B",
        "Unknown": "#9E9E9E",
    }
    bar_colors = [colors.get(g, "#9E9E9E") for g in group_imp.index]
    ax.barh(group_imp.index, group_imp.values, color=bar_colors)
    ax.set_xlabel("Summed Gini Importance")
    ax.set_title("Feature Group Importance (Random Forest)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "feature_group_importance.png"),
        bbox_inches="tight", dpi=150, facecolor="white",
    )
    plt.close(fig)
    log("  Saved: feature_group_importance.png")

    # Save raw importances to CSV for reference
    imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)
    log("  Saved: results/feature_importances.csv")
else:
    log("  WARNING: random_forest not in trained_models (skipped or checkpointed)")
    log("  Feature importance figures not generated.")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
summary_sorted = summary_df.sort_values("macro_f1", ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(summary_sorted)))
ax.barh(summary_sorted["model"], summary_sorted["macro_f1"], color=colors)
ax.set_xlabel("Macro F1 Score")
ax.set_title("Model Comparison - Macro F1")
ax.set_xlim(0, 1)
for i, (_, row) in enumerate(summary_sorted.iterrows()):
    ax.text(row["macro_f1"] + 0.01, i, f"{row['macro_f1']:.4f}", va="center")
fig.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, "model_comparison.png"),
    bbox_inches="tight", dpi=150, facecolor="white",
)
plt.close(fig)
log("  Saved: model_comparison.png")

# -----------------------------------------------------------------------
# 10. Summary
# -----------------------------------------------------------------------

log("\n" + "=" * 60)
log("TRAINING COMPLETE")
log("=" * 60)
log(f"\nTercile boundaries: q1={q1:.2f}, q2={q2:.2f}")
log(f"Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")
log(f"Features: {X_train.shape[1]}")
log(f"\nModel Rankings (Macro F1):")
for _, row in summary_df.iterrows():
    log(f"  {row['model']:20s}  F1={row['macro_f1']:.4f}  "
        f"BalAcc={row['balanced_accuracy']:.4f}  "
        f"Train={row['train_time_s']:.1f}s")
log(f"\nResults: {RESULTS_DIR}/")
log(f"Figures: {FIGURES_DIR}/")
