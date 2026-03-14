"""
Extended training script: model exploration and final selection.
Designed to run on NEU Explorer cluster (16 CPUs, 128GB RAM).

Extends 02_train.py with:
  1. XGBoost as a third candidate model
  2. COVID features (covid_flag, covid_intensity) via src/covid.py
  3. Two evaluation frameworks:
       - Pooled (cross-sectional): GroupShuffleSplit on mission_name, same as 02_train.py
       - Temporal: hard date cutoff (train on pre-2022, test on 2022)
  4. Full comparison table across all models x both frameworks
  5. Per-model checkpointing (same atomic JSON pattern as 02_train.py)

Usage:
    python scripts/03_train_extended.py             # resume from checkpoint
    python scripts/03_train_extended.py --fresh     # ignore checkpoint, start over

Output files (in results/extended/):
    model_comparison_extended.csv     -- all models x both frameworks
    results_pooled.json               -- full metrics, pooled eval
    results_temporal.json             -- full metrics, temporal eval
    feature_importances_xgb.csv       -- XGBoost feature importances
    feature_importances_rf_ext.csv    -- RF importances (re-run with COVID features)

Output figures (in figures/extended/):
    cm_pooled_<model>.png             -- confusion matrices, pooled eval
    cm_temporal_<model>.png           -- confusion matrices, temporal eval
    model_comparison_pooled.png       -- bar chart, pooled macro F1
    model_comparison_temporal.png     -- bar chart, temporal macro F1
    feature_importance_xgb.png        -- XGBoost top-20
    feature_group_importance_xgb.png  -- XGBoost group-level
    covid_effect.png                  -- ridership distribution split by covid_flag
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
    FEATURE_GROUP_MAP, FEATURE_GROUP_PREFIXES, METADATA_PATH,
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
    print("WARNING: xgboost not installed. Install with: pip install xgboost")
    print("         Continuing with Decision Tree and Random Forest only.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_EXT_DIR = os.path.join(RESULTS_DIR, "extended")
FIGURES_EXT_DIR = os.path.join(FIGURES_DIR, "extended")
os.makedirs(RESULTS_EXT_DIR, exist_ok=True)
os.makedirs(FIGURES_EXT_DIR, exist_ok=True)

CHECKPOINT_POOLED = os.path.join(RESULTS_EXT_DIR, "checkpoint_pooled.json")
CHECKPOINT_TEMPORAL = os.path.join(RESULTS_EXT_DIR, "checkpoint_temporal.json")
FRESH_RUN = "--fresh" in sys.argv

# Temporal split: train on missions starting before this date, test on after
TEMPORAL_CUTOFF = pd.Timestamp("2022-01-01")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model configs (DT + RF + XGBoost)
# ---------------------------------------------------------------------------

def get_extended_model_configs(seed: int = RANDOM_SEED) -> dict:
    """
    Return all three model configurations.

    XGBoost uses the hist tree method (fastest on CPU, comparable to GPU hist).
    n_jobs=-1 for full CPU parallelism. scale_pos_weight not needed since
    tercile binning produces balanced classes.

    Labels must be integer-encoded for XGBoost (handled in train loop below).
    """
    configs = {
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=seed),
            "params": {"max_depth": None, "criterion": "gini"},
            "needs_label_encoding": False,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=seed,
            ),
            "params": {"n_estimators": 300, "max_depth": None, "n_jobs": -1},
            "needs_label_encoding": False,
        },
    }

    if XGB_AVAILABLE:
        configs["xgboost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",       # fastest CPU method
                n_jobs=-1,
                random_state=seed,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "params": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
            },
            "needs_label_encoding": True,   # XGBoost requires integer labels
        }

    return configs


# ---------------------------------------------------------------------------
# Temporal split utility
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    cutoff: pd.Timestamp = TEMPORAL_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split missions by start date: train on missions starting before cutoff,
    test on missions starting on/after cutoff.

    Uses metaData startTime_iso to determine each mission's date.
    Joins on mission_name == name column in metadata.

    This simulates a real deployment scenario: the model is trained on
    historical data and evaluated on future unseen missions.
    """
    meta_dates = meta[["name", "startTime_iso"]].copy()
    meta_dates["startTime_iso"] = pd.to_datetime(meta_dates["startTime_iso"])

    train_missions = set(
        meta_dates.loc[meta_dates["startTime_iso"] < cutoff, "name"]
    )
    test_missions = set(
        meta_dates.loc[meta_dates["startTime_iso"] >= cutoff, "name"]
    )

    train_df = df[df["mission_name"].isin(train_missions)].copy()
    test_df = df[df["mission_name"].isin(test_missions)].copy()

    return train_df, test_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _feature_to_group(feat_name: str) -> str:
    if feat_name in FEATURE_GROUP_MAP:
        return FEATURE_GROUP_MAP[feat_name]
    for prefix, group in FEATURE_GROUP_PREFIXES.items():
        if feat_name.startswith(prefix):
            return group
    # COVID features
    if feat_name in ("covid_flag", "covid_intensity"):
        return "COVID"
    return "Unknown"


GROUP_COLORS = {
    "Temporal": "#FF9800", "Spatial": "#4CAF50", "Operational": "#2196F3",
    "Sensor": "#9C27B0", "Status": "#F44336", "Categorical": "#607D8B",
    "COVID": "#E91E63", "Unknown": "#9E9E9E",
}


def plot_confusion_matrix(cm, model_name, macro_f1, framework, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["low", "medium", "high"],
        yticklabels=["low", "medium", "high"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} [{framework}] (F1={macro_f1:.4f})")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"cm_{framework}_{model_name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")


def plot_model_comparison(results: dict, framework: str, out_dir: str):
    rows = [
        {"model": name, "macro_f1": m["macro_f1"],
         "balanced_accuracy": m["balanced_accuracy"]}
        for name, m in results.items()
    ]
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df["model"], df["macro_f1"], color=colors)
    ax.set_xlabel("Macro F1 Score")
    ax.set_title(f"Model Comparison [{framework}] - Macro F1")
    ax.set_xlim(0, 1)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["macro_f1"] + 0.01, i, f"{row['macro_f1']:.4f}", va="center")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"model_comparison_{framework}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")


def plot_feature_importance(model, feature_names, model_name, out_dir):
    """Plot top-20 and group-level feature importance for tree models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Top-20 bar chart
    top20 = imp_df.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="#2196F3")
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} -- Top 20 Feature Importances")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"feature_importance_{model_name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")

    # Group-level bar chart
    imp_df["group"] = imp_df["feature"].apply(_feature_to_group)
    group_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=True)
    bar_colors = [GROUP_COLORS.get(g, "#9E9E9E") for g in group_imp.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(group_imp.index, group_imp.values, color=bar_colors)
    ax.set_xlabel("Summed Importance")
    ax.set_title(f"{model_name} -- Feature Group Importance")
    fig.tight_layout()
    fname = os.path.join(out_dir, f"feature_group_importance_{model_name}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")

    # Save raw importances
    csv_path = os.path.join(RESULTS_EXT_DIR, f"feature_importances_{model_name}.csv")
    imp_df.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")


def plot_covid_effect(df: pd.DataFrame, out_dir: str):
    """
    Show ridership distribution split by covid_flag.
    Helps visually confirm the COVID feature carries signal.
    """
    if "covid_flag" not in df.columns or "itcs_numberOfPassengers" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Passenger count distribution by covid period
    for flag, label, color in [(0, "No restrictions", "#2196F3"), (1, "COVID restrictions active", "#E91E63")]:
        subset = df[df["covid_flag"] == flag]["itcs_numberOfPassengers"].dropna()
        axes[0].hist(subset, bins=50, alpha=0.6, label=label, color=color, density=True)
    axes[0].set_xlabel("Passenger Count")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Ridership Distribution by COVID Period")
    axes[0].legend()

    # Mean ridership by covid_intensity decile
    df_copy = df[["covid_intensity", "itcs_numberOfPassengers"]].dropna().copy()
    df_copy["intensity_decile"] = pd.qcut(
        df_copy["covid_intensity"], q=10, duplicates="drop", labels=False
    )
    decile_means = df_copy.groupby("intensity_decile")["itcs_numberOfPassengers"].mean()
    axes[1].bar(decile_means.index, decile_means.values, color="#9C27B0", alpha=0.8)
    axes[1].set_xlabel("COVID Intensity Decile (0=lowest, 9=highest)")
    axes[1].set_ylabel("Mean Passenger Count")
    axes[1].set_title("Mean Ridership by COVID Intensity Decile")

    fig.suptitle("COVID Feature vs. Ridership Signal", fontsize=13)
    fig.tight_layout()
    fname = os.path.join(out_dir, "covid_effect.png")
    fig.savefig(fname, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Core training loop for one eval framework
# ---------------------------------------------------------------------------

def run_training_loop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    framework: str,
    checkpoint_path: str,
) -> tuple[dict, dict]:
    """
    Train all models, evaluate, checkpoint per model.

    Returns (results_dict, trained_models_dict).
    results_dict: model_name -> metrics dict
    trained_models_dict: model_name -> fitted estimator
    """
    results = load_checkpoint(checkpoint_path, fresh=FRESH_RUN)
    if results:
        log(f"  Resuming [{framework}]: {len(results)} model(s) already completed")
        for name in results:
            log(f"    [checkpoint] {name}: F1={results[name]['macro_f1']:.4f}")

    configs = get_extended_model_configs(seed=RANDOM_SEED)
    trained_models = {}

    # Label encoder for XGBoost (needs integer targets)
    le = LabelEncoder()
    le.fit(["low", "medium", "high"])

    for model_name, cfg in configs.items():
        if model_name in results:
            log(f"\n  --- Skipping: {model_name} (already checkpointed) ---")
            continue

        log(f"\n  --- Training [{framework}]: {model_name} ---")
        model = cfg["model"]

        # Encode labels for XGBoost
        if cfg["needs_label_encoding"]:
            y_tr = le.transform(y_train)
            y_te = le.transform(y_test)
        else:
            y_tr = y_train
            y_te = y_test

        t0 = time.time()
        model.fit(X_train, y_tr)
        train_time = time.time() - t0
        log(f"    Trained in {train_time:.1f}s")
        trained_models[model_name] = model

        t0 = time.time()
        y_pred_raw = model.predict(X_test)
        pred_time = time.time() - t0

        # Decode XGBoost predictions back to string labels
        if cfg["needs_label_encoding"]:
            y_pred = pd.Series(le.inverse_transform(y_pred_raw), index=y_test.index)
            y_eval = y_test  # already string labels
        else:
            y_pred = pd.Series(y_pred_raw, index=y_test.index)
            y_eval = y_test

        metrics = evaluate_model(y_eval, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["predict_time_s"] = round(pred_time, 2)
        metrics["params"] = cfg["params"]

        results[model_name] = metrics
        save_checkpoint(results, checkpoint_path)
        log(f"    [checkpoint saved: {len(results)}/{len(configs)} models]")

        log(f"    Macro F1:          {metrics['macro_f1']:.4f}")
        log(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        log(f"    Train time:        {train_time:.1f}s  |  Predict: {pred_time:.2f}s")
        for cls in ["low", "medium", "high"]:
            pc = metrics["per_class"][cls]
            log(f"      {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

    return results, trained_models


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

log("=" * 60)
log("ZTBus Extended Training Pipeline (03_train_extended.py)")
log("=" * 60)

# -----------------------------------------------------------------------
# 1. Load ALL missions + metadata
# -----------------------------------------------------------------------

log("\n[1] Loading data...")
meta = load_metadata()
log(f"  Total missions in metadata: {len(meta)}")

t0 = time.time()
df_raw = load_sample_missions(meta, usecols=EDA_USE_COLS)
log(f"  Loaded {len(df_raw):,} rows in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------
# 2. Forward-fill ITCS columns per mission
# -----------------------------------------------------------------------

log("\n[2] Forward-filling ITCS columns...")
t0 = time.time()
filled_parts = []
for mission_name, group in df_raw.groupby("mission_name"):
    filled_parts.append(forward_fill_stop_columns(group))
df = pd.concat(filled_parts, ignore_index=True)
df = df[df["itcs_numberOfPassengers"].notna()].copy()
log(f"  After forward-fill: {len(df):,} rows ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------------------
# 3. COVID feature visualization (uses raw passenger counts before split)
# -----------------------------------------------------------------------

log("\n[3] Generating COVID effect plot...")
# Add covid features temporarily to raw df for visualization
from src.covid import build_covid_features
df_covid_viz = build_covid_features(df[["time_iso", "itcs_numberOfPassengers"]].copy())
plot_covid_effect(df_covid_viz, FIGURES_EXT_DIR)
del df_covid_viz

# -----------------------------------------------------------------------
# 4. Build both splits BEFORE feature engineering
# -----------------------------------------------------------------------

log("\n[4] Building train/test splits...")

# --- Pooled (cross-sectional) split ---
log("  Pooled split (GroupShuffleSplit on mission_name, 80/20)...")
train_pooled, test_pooled = mission_stratified_split(df, test_size=0.2, seed=RANDOM_SEED)
log(f"    Train: {len(train_pooled):,} rows ({train_pooled['mission_name'].nunique()} missions)")
log(f"    Test:  {len(test_pooled):,} rows ({test_pooled['mission_name'].nunique()} missions)")

# --- Temporal split ---
log(f"  Temporal split (cutoff: {TEMPORAL_CUTOFF.date()})...")
train_temporal, test_temporal = temporal_split(df, meta, cutoff=TEMPORAL_CUTOFF)
log(f"    Train: {len(train_temporal):,} rows ({train_temporal['mission_name'].nunique()} missions)")
log(f"    Test:  {len(test_temporal):,} rows ({test_temporal['mission_name'].nunique()} missions)")

if len(test_temporal) == 0:
    log("  WARNING: No missions found after temporal cutoff. Check TEMPORAL_CUTOFF date.")
    log("           Skipping temporal evaluation framework.")
    RUN_TEMPORAL = False
else:
    RUN_TEMPORAL = True

# -----------------------------------------------------------------------
# 5. Compute tercile boundaries on TRAINING data only (per framework)
# -----------------------------------------------------------------------

log("\n[5] Computing tercile boundaries...")

q1_pooled, q2_pooled = compute_tercile_boundaries(train_pooled["itcs_numberOfPassengers"])
log(f"  Pooled:   q1={q1_pooled:.2f}, q2={q2_pooled:.2f}")

train_pooled = train_pooled.copy()
test_pooled = test_pooled.copy()
train_pooled["demand_class"] = assign_demand_class(
    train_pooled["itcs_numberOfPassengers"], q1_pooled, q2_pooled
)
test_pooled["demand_class"] = assign_demand_class(
    test_pooled["itcs_numberOfPassengers"], q1_pooled, q2_pooled
)

if RUN_TEMPORAL:
    q1_temp, q2_temp = compute_tercile_boundaries(train_temporal["itcs_numberOfPassengers"])
    log(f"  Temporal: q1={q1_temp:.2f}, q2={q2_temp:.2f}")

    train_temporal = train_temporal.copy()
    test_temporal = test_temporal.copy()
    train_temporal["demand_class"] = assign_demand_class(
        train_temporal["itcs_numberOfPassengers"], q1_temp, q2_temp
    )
    test_temporal["demand_class"] = assign_demand_class(
        test_temporal["itcs_numberOfPassengers"], q1_temp, q2_temp
    )

# -----------------------------------------------------------------------
# 6. Feature engineering (includes COVID features via build_feature_set)
# -----------------------------------------------------------------------

log("\n[6] Feature engineering (includes COVID features)...")
t0 = time.time()
train_pooled_feat = build_feature_set(train_pooled)
test_pooled_feat = build_feature_set(test_pooled)
log(f"  Pooled features: {len(train_pooled_feat.columns) - 2} columns")

if RUN_TEMPORAL:
    train_temporal_feat = build_feature_set(train_temporal)
    test_temporal_feat = build_feature_set(test_temporal)
    log(f"  Temporal features: {len(train_temporal_feat.columns) - 2} columns")

log(f"  Feature engineering done in {time.time()-t0:.1f}s")

# -----------------------------------------------------------------------
# 7. Prepare X, y matrices
# -----------------------------------------------------------------------

log("\n[7] Preparing feature matrices...")

def prepare_xy(train_feat, test_feat):
    """Prepare aligned X_train, y_train, X_test, y_test."""
    X_tr, y_tr = prepare_features_and_target(
        train_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    X_te, y_te = prepare_features_and_target(
        test_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    # Align columns (one-hot may differ between splits)
    common = sorted(set(X_tr.columns) & set(X_te.columns))
    X_tr = X_tr[common]
    X_te = X_te[common]

    # Drop NaN rows (rolling window boundaries)
    tr_mask = X_tr.notna().all(axis=1)
    te_mask = X_te.notna().all(axis=1)
    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
    X_te, y_te = X_te[te_mask], y_te[te_mask]

    return X_tr, y_tr, X_te, y_te

X_train_p, y_train_p, X_test_p, y_test_p = prepare_xy(train_pooled_feat, test_pooled_feat)
log(f"  Pooled   — X_train: {X_train_p.shape}, X_test: {X_test_p.shape}")

if RUN_TEMPORAL:
    X_train_t, y_train_t, X_test_t, y_test_t = prepare_xy(
        train_temporal_feat, test_temporal_feat
    )
    log(f"  Temporal — X_train: {X_train_t.shape}, X_test: {X_test_t.shape}")

# Confirm COVID features made it in
covid_cols = [c for c in X_train_p.columns if "covid" in c]
log(f"  COVID features present: {covid_cols}")

# -----------------------------------------------------------------------
# 8. Train and evaluate — Pooled framework
# -----------------------------------------------------------------------

log("\n" + "=" * 60)
log("POOLED (CROSS-SECTIONAL) EVALUATION")
log("=" * 60)

results_pooled, trained_pooled = run_training_loop(
    X_train_p, y_train_p, X_test_p, y_test_p,
    framework="pooled",
    checkpoint_path=CHECKPOINT_POOLED,
)

# -----------------------------------------------------------------------
# 9. Train and evaluate — Temporal framework
# -----------------------------------------------------------------------

results_temporal = {}
trained_temporal = {}

if RUN_TEMPORAL:
    log("\n" + "=" * 60)
    log("TEMPORAL EVALUATION (train pre-2022, test 2022)")
    log("=" * 60)

    results_temporal, trained_temporal = run_training_loop(
        X_train_t, y_train_t, X_test_t, y_test_t,
        framework="temporal",
        checkpoint_path=CHECKPOINT_TEMPORAL,
    )

# -----------------------------------------------------------------------
# 10. Save results
# -----------------------------------------------------------------------

log("\n[10] Saving results...")

def results_to_json_safe(results: dict) -> dict:
    out = {}
    for name, m in results.items():
        out[name] = {
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "confusion_matrix": m["confusion_matrix"].tolist(),
            "per_class": m["per_class"],
            "train_time_s": m["train_time_s"],
            "predict_time_s": m["predict_time_s"],
            "params": m["params"],
        }
    return out

with open(os.path.join(RESULTS_EXT_DIR, "results_pooled.json"), "w") as f:
    json.dump(results_to_json_safe(results_pooled), f, indent=2)
log("  Saved: results_pooled.json")

if results_temporal:
    with open(os.path.join(RESULTS_EXT_DIR, "results_temporal.json"), "w") as f:
        json.dump(results_to_json_safe(results_temporal), f, indent=2)
    log("  Saved: results_temporal.json")

# Combined comparison table
comparison_rows = []
for model_name, m in results_pooled.items():
    comparison_rows.append({
        "model": model_name,
        "framework": "pooled",
        "macro_f1": round(m["macro_f1"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "f1_low": round(m["per_class"]["low"]["f1"], 4),
        "f1_medium": round(m["per_class"]["medium"]["f1"], 4),
        "f1_high": round(m["per_class"]["high"]["f1"], 4),
        "train_time_s": m["train_time_s"],
    })
for model_name, m in results_temporal.items():
    comparison_rows.append({
        "model": model_name,
        "framework": "temporal",
        "macro_f1": round(m["macro_f1"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "f1_low": round(m["per_class"]["low"]["f1"], 4),
        "f1_medium": round(m["per_class"]["medium"]["f1"], 4),
        "f1_high": round(m["per_class"]["high"]["f1"], 4),
        "train_time_s": m["train_time_s"],
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values(
    ["framework", "macro_f1"], ascending=[True, False]
)
comparison_path = os.path.join(RESULTS_EXT_DIR, "model_comparison_extended.csv")
comparison_df.to_csv(comparison_path, index=False)
log(f"  Saved: model_comparison_extended.csv")

# -----------------------------------------------------------------------
# 11. Figures
# -----------------------------------------------------------------------

log("\n[11] Generating figures...")

# Confusion matrices
for model_name, m in results_pooled.items():
    plot_confusion_matrix(m["confusion_matrix"], model_name, m["macro_f1"],
                          "pooled", FIGURES_EXT_DIR)

for model_name, m in results_temporal.items():
    plot_confusion_matrix(m["confusion_matrix"], model_name, m["macro_f1"],
                          "temporal", FIGURES_EXT_DIR)

# Model comparison bar charts
if results_pooled:
    plot_model_comparison(results_pooled, "pooled", FIGURES_EXT_DIR)
if results_temporal:
    plot_model_comparison(results_temporal, "temporal", FIGURES_EXT_DIR)

# Feature importance (for all tree models that expose feature_importances_)
importance_models = {**trained_pooled}
for model_name, model in importance_models.items():
    plot_feature_importance(model, X_train_p.columns.tolist(), model_name, FIGURES_EXT_DIR)

# -----------------------------------------------------------------------
# 12. Clean up checkpoints
# -----------------------------------------------------------------------

for cp in [CHECKPOINT_POOLED, CHECKPOINT_TEMPORAL]:
    if os.path.exists(cp):
        os.remove(cp)
        log(f"  Checkpoint cleared: {cp}")

# -----------------------------------------------------------------------
# 13. Final summary
# -----------------------------------------------------------------------

log("\n" + "=" * 60)
log("EXTENDED TRAINING COMPLETE")
log("=" * 60)

log("\nPooled (cross-sectional) results:")
for _, row in comparison_df[comparison_df["framework"] == "pooled"].iterrows():
    log(f"  {row['model']:20s}  F1={row['macro_f1']:.4f}  "
        f"BalAcc={row['balanced_accuracy']:.4f}  "
        f"Train={row['train_time_s']:.1f}s")

if results_temporal:
    log(f"\nTemporal results (train pre-{TEMPORAL_CUTOFF.year}, "
        f"test {TEMPORAL_CUTOFF.year}+):")
    for _, row in comparison_df[comparison_df["framework"] == "temporal"].iterrows():
        log(f"  {row['model']:20s}  F1={row['macro_f1']:.4f}  "
            f"BalAcc={row['balanced_accuracy']:.4f}  "
            f"Train={row['train_time_s']:.1f}s")

log(f"\nResults: {RESULTS_EXT_DIR}/")
log(f"Figures: {FIGURES_EXT_DIR}/")
log("\nNext step: review model_comparison_extended.csv for final model selection.")
