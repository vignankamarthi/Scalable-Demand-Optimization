"""
Time-series evaluation entry point for ZTBus ridership demand classification.

Runs all 6 temporal diagnostics across DT, RF, and XGBoost:
  1. Monthly F1 over time (with COVID shading)
  2. Within-mission accuracy by trip position
  3. Error autocorrelation (ACF at lags 1-60)
  4. COVID period breakdown (pre/during/post restrictions)
  5. Class transition accuracy (stable vs demand-change points)
  6. Predicted vs actual class distribution over time (calibration)

Usage:
    python scripts/05_timeseries_eval.py

Output:
    results/timeseries/ts_metrics_pooled.json
    results/timeseries/ts_summary_pooled.csv
    figures/timeseries/pooled/*.png (6 diagnostic plots)
"""
from __future__ import annotations

import sys
import os
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RANDOM_SEED, EDA_USE_COLS
from src.data_loading import load_metadata, load_sample_missions
from src.preprocessing import forward_fill_stop_columns
from src.feature_engineering import build_feature_set
from src.target import compute_tercile_boundaries, assign_demand_class
from src.model_pipeline import (
    mission_stratified_split,
    prepare_features_and_target,
    get_model_configs,
)
from src.ts_evaluation import run_timeseries_evaluation

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder


class XGBStringLabelWrapper:
    """Wraps XGBoost to accept/return string labels like sklearn classifiers.

    run_timeseries_evaluation calls model.predict(X) and expects string labels
    back ("low", "medium", "high"). XGBoost returns integer labels (0, 1, 2).
    This wrapper handles the encoding/decoding transparently.
    """

    def __init__(self, model, label_encoder: LabelEncoder):
        self._model = model
        self._le = label_encoder

    def fit(self, X, y):
        self._model.fit(X, self._le.transform(y))
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._model.predict(X))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_extended_model_configs(seed: int) -> dict:
    """Return DT + RF + XGBoost model configs for time-series evaluation."""
    configs = get_model_configs(seed=seed)

    if XGB_AVAILABLE:
        configs["xgboost"] = {
            "model": xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        }
    else:
        log("WARNING: xgboost not installed. Running DT + RF only.")

    return configs


def main() -> None:
    log("=" * 60)
    log("ZTBus Time-Series Evaluation (05_timeseries_eval.py)")
    log("Models: DT + RF + XGBoost")
    log("=" * 60)

    # 1. Load data
    log("\n[1] Loading data...")
    meta = load_metadata()
    df_raw = load_sample_missions(meta, usecols=EDA_USE_COLS)

    # 2. Forward-fill per mission
    log("[2] Forward-filling...")
    filled_parts = []
    for _, group in df_raw.groupby("mission_name"):
        filled_parts.append(forward_fill_stop_columns(group))
    df = pd.concat(filled_parts, ignore_index=True)
    df = df[df["itcs_numberOfPassengers"].notna()].copy()

    # 3. Mission-level split (same seed as training for reproducibility)
    log("[3] Mission-level split (same seed as training)...")
    train_df, test_df = mission_stratified_split(df, test_size=0.2, seed=RANDOM_SEED)

    # 4. Tercile boundaries from training data only (no leakage)
    log("[4] Tercile boundaries from training data...")
    q1, q2 = compute_tercile_boundaries(train_df["itcs_numberOfPassengers"])
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["demand_class"] = assign_demand_class(
        train_df["itcs_numberOfPassengers"], q1, q2
    )
    test_df["demand_class"] = assign_demand_class(
        test_df["itcs_numberOfPassengers"], q1, q2
    )

    # 5. Feature engineering (includes COVID features)
    log("[5] Feature engineering...")
    train_feat = build_feature_set(train_df)
    test_feat = build_feature_set(test_df)

    # Preserve time_iso and mission_name on test_feat for temporal evaluation
    # (run_timeseries_evaluation drops them before predict())
    if "time_iso" not in test_feat.columns and "time_iso" in test_df.columns:
        test_feat["time_iso"] = test_df["time_iso"].values
    if "mission_name" not in test_feat.columns and "mission_name" in test_df.columns:
        test_feat["mission_name"] = test_df["mission_name"].values

    X_train, y_train = prepare_features_and_target(
        train_feat,
        target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    X_test, y_test = prepare_features_and_target(
        test_feat,
        target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )

    # Align columns (test may have different one-hot columns)
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Drop rows with NaN from rolling window boundaries
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    test_feat_aligned = test_feat[test_mask].copy()

    # Drop one-hot columns from test_feat_aligned that weren't in training
    # (build_prediction_frame uses all non-metadata columns for predict())
    extra_cols = set(test_feat_aligned.columns) - set(common_cols) - {
        "demand_class", "itcs_numberOfPassengers", "mission_name",
        "time_iso", "time_unix",
    }
    if extra_cols:
        log(f"  Dropping {len(extra_cols)} test-only feature columns")
        test_feat_aligned = test_feat_aligned.drop(columns=list(extra_cols))

    log(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 6. Train all models
    log("\n[6] Training models...")
    configs = get_extended_model_configs(seed=RANDOM_SEED)
    trained_models = {}

    # Label encoder for XGBoost (needs integer targets)
    le = LabelEncoder()
    le.fit(["low", "medium", "high"])

    for model_name, cfg in configs.items():
        log(f"  Fitting {model_name}...")
        t0 = time.time()

        if model_name == "xgboost":
            # Wrap XGBoost so run_timeseries_evaluation gets string labels back
            wrapper = XGBStringLabelWrapper(cfg["model"], le)
            wrapper.fit(X_train, y_train)
            trained_models[model_name] = wrapper
        else:
            cfg["model"].fit(X_train, y_train)
            trained_models[model_name] = cfg["model"]

        elapsed = time.time() - t0
        log(f"    Done in {elapsed:.1f}s")

    # 7. Run time-series evaluation
    log("\n[7] Running time-series evaluation...")
    run_timeseries_evaluation(
        test_feat=test_feat_aligned,
        y_test=y_test,
        trained_models=trained_models,
        framework="pooled",
    )

    log("\nDone. Results in results/timeseries/ and figures/timeseries/")


if __name__ == "__main__":
    main()
