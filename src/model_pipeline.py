"""
Model pipeline: train/test splitting, model configs, training, and evaluation.
All scikit-learn based. MLPs use sklearn MLPClassifier (GPU via PyTorch
can be added later if needed, but sklearn handles 5M rows on CPU fine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from src.config import RANDOM_SEED, DEMAND_LABELS


def mission_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    mission_col: str = "mission_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data so no mission appears in both train and test.
    Uses GroupShuffleSplit with mission_name as the group key.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[mission_col]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "itcs_numberOfPassengers",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y), dropping non-feature columns.
    """
    if drop_cols is None:
        drop_cols = ["mission_name"]

    y = df[target_col]
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    return X, y


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute evaluation metrics for a classification model.

    Returns dict with:
    - macro_f1: Macro-averaged F1 score
    - balanced_accuracy: Balanced accuracy
    - confusion_matrix: 3x3 numpy array
    - per_class: dict of per-class precision, recall, f1
    """
    labels = DEMAND_LABELS

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        else:
            per_class[label] = {"precision": 0, "recall": 0, "f1": 0, "support": 0}

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def get_model_configs(seed: int = RANDOM_SEED) -> dict:
    """
    Return model configurations for all experiments.

    Each config has:
    - 'model': instantiated sklearn estimator
    - 'params': dict of hyperparameters (for logging)
    - 'needs_scaling': whether StandardScaler should be applied
    """
    return {
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=seed),
            "params": {"max_depth": None, "criterion": "gini"},
            "needs_scaling": False,
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=seed,
            ),
            "params": {"n_estimators": 300, "max_depth": None, "n_jobs": -1},
            "needs_scaling": False,
        },
        "knn": {
            "model": KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
            "params": {"n_neighbors": 11, "weights": "uniform", "n_jobs": -1},
            "needs_scaling": True,
        },
        "mlp_small": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=200, random_state=seed, early_stopping=True,
            ),
            "params": {"layers": "(64,)", "max_iter": 200, "early_stopping": True},
            "needs_scaling": True,
        },
        "mlp_medium": {
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=200, random_state=seed, early_stopping=True,
            ),
            "params": {"layers": "(128, 64)", "max_iter": 200, "early_stopping": True},
            "needs_scaling": True,
        },
        "mlp_large": {
            "model": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=200, random_state=seed, early_stopping=True,
            ),
            "params": {"layers": "(256, 128, 64)", "max_iter": 200, "early_stopping": True},
            "needs_scaling": True,
        },
    }
