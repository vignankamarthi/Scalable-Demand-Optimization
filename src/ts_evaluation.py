"""
Time-series evaluation for ZTBus ridership demand classification.

Adds temporal structure to the standard cross-sectional evaluation.
Designed to run AFTER 02_train.py or 03_train_extended.py have produced
trained model objects and predictions.

Can be run standalone:
    python scripts/05_timeseries_eval.py

Or imported and called from within a training script:
    from src.ts_evaluation import run_timeseries_evaluation

Produces:
    results/timeseries/ts_metrics.json       -- all numeric metrics
    results/timeseries/ts_summary.csv        -- model x period breakdown
    figures/timeseries/01_f1_over_time.png   -- monthly F1 line chart
    figures/timeseries/02_within_mission.png -- accuracy by trip position
    figures/timeseries/03_error_acf.png      -- error autocorrelation
    figures/timeseries/04_covid_breakdown.png-- F1 by COVID period
    figures/timeseries/05_class_transitions.png -- accuracy at demand shifts
    figures/timeseries/06_calibration.png    -- predicted vs actual dist over time
"""
from __future__ import annotations

import sys
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as compute_acf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RESULTS_DIR, FIGURES_DIR, RANDOM_SEED, EDA_USE_COLS,
    FEATURE_GROUP_MAP, FEATURE_GROUP_PREFIXES, DEMAND_LABELS,
)
from src.data_loading import load_metadata, load_sample_missions
from src.preprocessing import forward_fill_stop_columns
from src.feature_engineering import build_feature_set
from src.target import compute_tercile_boundaries, assign_demand_class
from src.model_pipeline import (
    mission_stratified_split,
    prepare_features_and_target,
    get_model_configs,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_TS_DIR = os.path.join(RESULTS_DIR, "timeseries")
FIGURES_TS_DIR = os.path.join(FIGURES_DIR, "timeseries")
os.makedirs(RESULTS_TS_DIR, exist_ok=True)
os.makedirs(FIGURES_TS_DIR, exist_ok=True)

# COVID period definitions (matches src/covid.py logic)
COVID_PERIODS = {
    "pre_covid":    (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2020-02-29", tz="UTC")),
    "restrictions": (pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp("2022-03-31", tz="UTC")),
    "post_covid":   (pd.Timestamp("2022-04-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")),
}

ACF_NLAGS = 60      # seconds of autocorrelation to compute
TRIP_BINS = 10      # number of equal-width position bins per mission
ROLLING_WINDOW_DAYS = 30   # days for rolling F1 smoothing


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Core utility: attach predictions to a timestamped DataFrame
# ---------------------------------------------------------------------------

def build_prediction_frame(
    feat_df: pd.DataFrame,
    model,
    y_true: pd.Series,
    time_col: str = "time_iso",
) -> pd.DataFrame:
    """
    Run model.predict() and return a DataFrame with timestamps,
    true labels, predicted labels, and a binary correct column.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with time_iso and mission_name columns still attached.
        (Pass the full featured df before calling prepare_features_and_target.)
    model : fitted sklearn estimator
    y_true : pd.Series
        True demand class labels, aligned to feat_df index.
    time_col : str

    Returns
    -------
    pd.DataFrame with columns:
        time_iso, mission_name, y_true, y_pred, correct, year, month, date
    """
    X = feat_df.drop(
        columns=[c for c in ["demand_class", "itcs_numberOfPassengers",
                             "mission_name", "time_iso", "time_unix"]
                 if c in feat_df.columns],
        errors="ignore",
    )
    y_pred = pd.Series(model.predict(X), index=feat_df.index, name="y_pred")

    # Align y_true index to feat_df index in case they diverged after masking
    y_true_aligned = pd.Series(y_true.values, index=feat_df.index, name="y_true")

    out = pd.DataFrame({
        "time_iso":     pd.to_datetime(feat_df[time_col]) if time_col in feat_df.columns
                        else pd.NaT,
        "mission_name": feat_df["mission_name"] if "mission_name" in feat_df.columns
                        else "unknown",
        "y_true":  y_true_aligned,
        "y_pred":  y_pred,
    })
    out["correct"] = (out["y_true"] == out["y_pred"]).astype(int)
    out["year"]  = out["time_iso"].dt.year
    out["month"] = out["time_iso"].dt.month
    out["date"]  = out["time_iso"].dt.normalize()
    out["ym"]    = out["time_iso"].dt.to_period("M")
    return out.dropna(subset=["time_iso"])


def macro_f1_from_frame(df: pd.DataFrame) -> float:
    """Compute macro F1 from a prediction frame."""
    from sklearn.metrics import f1_score
    if len(df) == 0:
        return np.nan
    return f1_score(
        df["y_true"], df["y_pred"],
        labels=DEMAND_LABELS, average="macro", zero_division=0,
    )


def label_covid_period(ts: pd.Series) -> pd.Series:
    """Map a datetime Series to COVID period labels."""
    result = pd.Series("outside_dataset", index=ts.index)
    for period, (start, end) in COVID_PERIODS.items():
        mask = (ts >= start) & (ts <= end)
        result[mask] = period
    return result


# ---------------------------------------------------------------------------
# 1. Monthly F1 over time
# ---------------------------------------------------------------------------

def compute_monthly_f1(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro F1 per calendar month.

    Returns DataFrame with columns: period (Period[M]), macro_f1, n_samples.
    """
    rows = []
    for period, group in pred_frame.groupby("ym"):
        f1 = macro_f1_from_frame(group)
        rows.append({"period": period, "macro_f1": f1, "n_samples": len(group)})
    return pd.DataFrame(rows).sort_values("period")


def plot_f1_over_time(
    monthly_results: dict[str, pd.DataFrame],
    out_dir: str,
):
    """
    Line chart of monthly macro F1 for all models.
    Shaded bands show COVID restriction periods.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # COVID shading
    for period, (start, end) in COVID_PERIODS.items():
        if period == "restrictions":
            ax.axvspan(start, end, alpha=0.08, color="#E91E63", label="COVID restrictions")

    colors = {"decision_tree": "#7F77DD", "random_forest": "#1D9E75",
              "xgboost": "#EF9F27", "timesfm_zero_shot": "#D85A30"}

    for model_name, monthly_df in monthly_results.items():
        if monthly_df.empty:
            continue
        dates = monthly_df["period"].dt.to_timestamp()
        color = colors.get(model_name, "#888780")
        ax.plot(dates, monthly_df["macro_f1"], marker="o", markersize=3,
                linewidth=1.5, label=model_name.replace("_", " "), color=color)

        # 30-day rolling smooth
        if len(monthly_df) >= 3:
            smoothed = monthly_df["macro_f1"].rolling(3, center=True, min_periods=1).mean()
            ax.plot(dates, smoothed, linewidth=2.5, alpha=0.4, color=color)

    ax.axhline(1/3, color="#888780", linestyle=":", linewidth=0.8, label="random baseline")
    ax.set_xlabel("Month")
    ax.set_ylabel("Macro F1")
    ax.set_title("Model performance over time (monthly macro F1)")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    path = os.path.join(out_dir, "01_f1_over_time.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Within-mission drift
# ---------------------------------------------------------------------------

def compute_within_mission_accuracy(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bin each mission into TRIP_BINS equal-width position bins.
    Returns DataFrame: bin (0..TRIP_BINS-1), accuracy, n_samples.
    """
    rows = []
    for _, group in pred_frame.groupby("mission_name"):
        n = len(group)
        if n < TRIP_BINS:
            continue
        group = group.reset_index(drop=True)
        group["bin"] = pd.cut(group.index, bins=TRIP_BINS, labels=False)
        for b, bgroup in group.groupby("bin"):
            rows.append({
                "bin": int(b),
                "accuracy": bgroup["correct"].mean(),
                "n_samples": len(bgroup),
            })
    if not rows:
        return pd.DataFrame(columns=["bin", "accuracy", "n_samples"])

    result = (
        pd.DataFrame(rows)
        .groupby("bin")
        .agg(accuracy=("accuracy", "mean"), n_samples=("n_samples", "sum"))
        .reset_index()
    )
    return result


def plot_within_mission(
    within_results: dict[str, pd.DataFrame],
    out_dir: str,
):
    """
    Line chart: accuracy by normalized trip position (0% = trip start, 100% = end).
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"decision_tree": "#7F77DD", "random_forest": "#1D9E75",
              "xgboost": "#EF9F27"}

    x_labels = [f"{int(i * 100 / TRIP_BINS)}–{int((i+1) * 100 / TRIP_BINS)}%"
                for i in range(TRIP_BINS)]

    for model_name, df in within_results.items():
        if df.empty:
            continue
        color = colors.get(model_name, "#888780")
        ax.plot(df["bin"], df["accuracy"], marker="o", markersize=4,
                linewidth=1.8, label=model_name.replace("_", " "), color=color)

    ax.set_xticks(range(TRIP_BINS))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Position in trip")
    ax.set_ylabel("Accuracy")
    ax.set_title("Within-mission accuracy by trip position")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_within_mission.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Error autocorrelation
# ---------------------------------------------------------------------------

def compute_error_acf(pred_frame: pd.DataFrame) -> dict:
    """
    Compute ACF of the binary error signal (1 = wrong, 0 = correct)
    for each mission, then average across missions.

    Returns dict with keys: lags, mean_acf, std_acf.
    """
    all_acfs = []
    for _, group in pred_frame.groupby("mission_name"):
        errors = (1 - group["correct"]).values.astype(float)
        if len(errors) < ACF_NLAGS + 10:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acf_vals = compute_acf(errors, nlags=ACF_NLAGS, fft=True)
            all_acfs.append(acf_vals)
        except Exception:
            continue

    if not all_acfs:
        return {"lags": list(range(ACF_NLAGS + 1)),
                "mean_acf": [0.0] * (ACF_NLAGS + 1),
                "std_acf": [0.0] * (ACF_NLAGS + 1)}

    stacked = np.array(all_acfs)
    return {
        "lags": list(range(ACF_NLAGS + 1)),
        "mean_acf": stacked.mean(axis=0).tolist(),
        "std_acf": stacked.std(axis=0).tolist(),
    }


def plot_error_acf(
    acf_results: dict[str, dict],
    out_dir: str,
):
    """
    ACF plot for each model's error signal.
    Strong positive ACF at short lags = errors cluster in time.
    """
    n_models = len(acf_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {"decision_tree": "#7F77DD", "random_forest": "#1D9E75",
              "xgboost": "#EF9F27"}

    for ax, (model_name, acf_data) in zip(axes, acf_results.items()):
        lags = acf_data["lags"]
        mean_acf = np.array(acf_data["mean_acf"])
        std_acf = np.array(acf_data["std_acf"])
        color = colors.get(model_name, "#888780")

        ax.bar(lags[1:], mean_acf[1:], color=color, alpha=0.7, width=0.8)
        ax.fill_between(lags[1:],
                        mean_acf[1:] - std_acf[1:],
                        mean_acf[1:] + std_acf[1:],
                        alpha=0.2, color=color)
        # 95% confidence interval for white noise
        ci = 1.96 / np.sqrt(1000)
        ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (seconds)")
        ax.set_title(f"{model_name.replace('_', ' ')}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean ACF of errors")
    fig.suptitle("Error autocorrelation — do mistakes cluster in time?", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_error_acf.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 4. COVID period breakdown
# ---------------------------------------------------------------------------

def compute_covid_breakdown(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro F1 and accuracy per COVID period.
    Returns DataFrame: period, macro_f1, accuracy, n_samples.
    """
    pred_frame = pred_frame.copy()
    pred_frame["covid_period"] = label_covid_period(pred_frame["time_iso"])

    rows = []
    for period in ["pre_covid", "restrictions", "post_covid"]:
        group = pred_frame[pred_frame["covid_period"] == period]
        if len(group) == 0:
            continue
        rows.append({
            "period": period,
            "macro_f1": macro_f1_from_frame(group),
            "accuracy": group["correct"].mean(),
            "n_samples": len(group),
        })
    return pd.DataFrame(rows)


def plot_covid_breakdown(
    covid_results: dict[str, pd.DataFrame],
    out_dir: str,
):
    """
    Grouped bar chart: macro F1 by COVID period for each model.
    """
    period_labels = {
        "pre_covid": "Pre-COVID\n(2019–Feb 2020)",
        "restrictions": "Restrictions\n(Mar 2020–Mar 2022)",
        "post_covid": "Post-restriction\n(Apr–Dec 2022)",
    }
    colors = {"decision_tree": "#7F77DD", "random_forest": "#1D9E75",
              "xgboost": "#EF9F27"}

    models = list(covid_results.keys())
    periods = list(period_labels.keys())
    x = np.arange(len(periods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        df = covid_results[model_name]
        if df.empty:
            continue
        f1_vals = []
        for p in periods:
            row = df[df["period"] == p]
            f1_vals.append(row["macro_f1"].values[0] if len(row) > 0 else 0.0)
        color = colors.get(model_name, "#888780")
        bars = ax.bar(x + i * width, f1_vals, width, label=model_name.replace("_", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, f1_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([period_labels[p] for p in periods], fontsize=10)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model performance by COVID period")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_covid_breakdown.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Class transition accuracy
# ---------------------------------------------------------------------------

def compute_transition_accuracy(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    For each mission, find rows where the true label changes (a transition).
    Compute accuracy at transition rows vs non-transition rows.

    Returns DataFrame: type (transition/stable), accuracy, n_samples.
    Also returns per-transition-type breakdown:
        low->medium, medium->high, high->medium, etc.
    """
    transition_rows = []
    stable_rows = []

    for _, group in pred_frame.groupby("mission_name"):
        group = group.reset_index(drop=True)
        labels = group["y_true"].values
        shifted = np.roll(labels, 1)
        shifted[0] = labels[0]
        is_transition = labels != shifted

        for i, (is_t, row) in enumerate(zip(is_transition, group.itertuples())):
            if i == 0:
                continue
            entry = {
                "correct": row.correct,
                "transition": f"{shifted[i]}->{labels[i]}" if is_t else "stable",
                "is_transition": is_t,
            }
            if is_t:
                transition_rows.append(entry)
            else:
                stable_rows.append(entry)

    all_rows = transition_rows + stable_rows
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_rows)

    summary = pd.DataFrame([
        {
            "type": "At transitions",
            "accuracy": df[df["is_transition"]]["correct"].mean(),
            "n_samples": df["is_transition"].sum(),
        },
        {
            "type": "Stable periods",
            "accuracy": df[~df["is_transition"]]["correct"].mean(),
            "n_samples": (~df["is_transition"]).sum(),
        },
    ])

    per_transition = (
        df[df["is_transition"]]
        .groupby("transition")["correct"]
        .agg(accuracy="mean", n_samples="count")
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )

    return summary, per_transition


def plot_class_transitions(
    transition_results: dict[str, tuple],
    out_dir: str,
):
    """
    Two-panel plot:
    Left: accuracy at transitions vs stable periods per model.
    Right: per-transition-type accuracy heatmap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"decision_tree": "#7F77DD", "random_forest": "#1D9E75",
              "xgboost": "#EF9F27"}

    # Left panel: transition vs stable
    models = list(transition_results.keys())
    x = np.arange(2)
    width = 0.25

    for i, model_name in enumerate(models):
        summary, _ = transition_results[model_name]
        if summary is None or summary.empty:
            continue
        vals = summary["accuracy"].values
        color = colors.get(model_name, "#888780")
        bars = ax1.bar(x + i * width, vals, width,
                       label=model_name.replace("_", " "), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(["At transitions", "Stable periods"], fontsize=11)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy: transitions vs stable periods")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)

    # Right panel: per-transition heatmap (best model only)
    best_model = max(
        transition_results.keys(),
        key=lambda m: (
            transition_results[m][0]["accuracy"].mean()
            if transition_results[m][0] is not None and not transition_results[m][0].empty
            else 0.0
        ),
    )
    _, per_t = transition_results[best_model]
    if per_t is not None and not per_t.empty:
        top_transitions = per_t.head(9)
        ax2.barh(top_transitions["transition"],
                 top_transitions["accuracy"],
                 color=colors.get(best_model, "#888780"), alpha=0.8)
        ax2.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="0.5 baseline")
        ax2.set_xlabel("Accuracy")
        ax2.set_title(f"Per-transition accuracy ({best_model.replace('_', ' ')})")
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)

    fig.tight_layout()
    path = os.path.join(out_dir, "05_class_transitions.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Predicted vs actual class distribution over time
# ---------------------------------------------------------------------------

def compute_class_distribution_over_time(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly fraction of predictions in each class vs actual fractions.
    Returns DataFrame: ym, source (actual/predicted), low_frac, medium_frac, high_frac.
    """
    rows = []
    for ym, group in pred_frame.groupby("ym"):
        n = len(group)
        if n == 0:
            continue
        for source, col in [("actual", "y_true"), ("predicted", "y_pred")]:
            counts = group[col].value_counts()
            rows.append({
                "ym": ym,
                "source": source,
                "low":    counts.get("low", 0) / n,
                "medium": counts.get("medium", 0) / n,
                "high":   counts.get("high", 0) / n,
            })
    return pd.DataFrame(rows).sort_values("ym")


def plot_calibration_over_time(
    calibration_results: dict[str, pd.DataFrame],
    out_dir: str,
):
    """
    For each model: stacked area chart of predicted class fractions over time,
    overlaid with actual class fractions as dotted lines.
    """
    n_models = len(calibration_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(13, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    class_colors = {"low": "#378ADD", "medium": "#EF9F27", "high": "#E24B4A"}

    for ax, (model_name, df) in zip(axes, calibration_results.items()):
        if df.empty:
            ax.set_title(f"{model_name} — no data")
            continue

        pred_df = df[df["source"] == "predicted"].copy()
        act_df  = df[df["source"] == "actual"].copy()

        dates = pred_df["ym"].dt.to_timestamp()

        for cls in ["low", "medium", "high"]:
            ax.fill_between(dates, pred_df[cls], alpha=0.3,
                            color=class_colors[cls], label=f"predicted {cls}")
            if not act_df.empty:
                act_dates = act_df["ym"].dt.to_timestamp()
                ax.plot(act_dates, act_df[cls], linestyle=":",
                        linewidth=1.5, color=class_colors[cls], label=f"actual {cls}")

        ax.set_ylabel("Fraction of predictions")
        ax.set_title(f"{model_name.replace('_', ' ')} — predicted vs actual class distribution")
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.legend(fontsize=8, ncol=6, loc="upper right")

    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    path = os.path.join(out_dir, "06_calibration.png")
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_timeseries_evaluation(
    test_feat: pd.DataFrame,
    y_test: pd.Series,
    trained_models: dict,
    framework: str = "pooled",
) -> dict:
    """
    Run all time-series evaluations for a set of trained models.

    Parameters
    ----------
    test_feat : pd.DataFrame
        Full featured test DataFrame with time_iso and mission_name retained.
    y_test : pd.Series
        True demand class labels.
    trained_models : dict
        model_name -> fitted sklearn estimator.
    framework : str
        Label for output files ('pooled' or 'temporal').

    Returns
    -------
    dict of all computed metrics (also saved to disk).
    """
    log(f"\n{'='*60}")
    log(f"Time-series evaluation [{framework}]")
    log(f"{'='*60}")

    out_dir = os.path.join(FIGURES_TS_DIR, framework)
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = {}
    monthly_results = {}
    within_results = {}
    acf_results = {}
    covid_results = {}
    transition_results = {}
    calibration_results = {}

    for model_name, model in trained_models.items():
        log(f"\n  Processing: {model_name}")

        # Build prediction frame (needs time_iso + mission_name)
        pred_frame = build_prediction_frame(test_feat, model, y_test)
        log(f"    Prediction frame: {len(pred_frame):,} rows")

        # 1. Monthly F1
        log("    Computing monthly F1...")
        monthly_df = compute_monthly_f1(pred_frame)
        monthly_results[model_name] = monthly_df

        # 2. Within-mission accuracy
        log("    Computing within-mission accuracy...")
        within_df = compute_within_mission_accuracy(pred_frame)
        within_results[model_name] = within_df

        # 3. Error ACF
        log("    Computing error ACF...")
        acf_data = compute_error_acf(pred_frame)
        acf_results[model_name] = acf_data

        # 4. COVID breakdown
        log("    Computing COVID period breakdown...")
        covid_df = compute_covid_breakdown(pred_frame)
        covid_results[model_name] = covid_df

        # 5. Class transitions
        log("    Computing transition accuracy...")
        try:
            summary, per_t = compute_transition_accuracy(pred_frame)
            transition_results[model_name] = (summary, per_t)
        except Exception as e:
            log(f"    WARNING: transition accuracy failed ({e})")
            transition_results[model_name] = (None, None)

        # 6. Calibration
        log("    Computing calibration over time...")
        calib_df = compute_class_distribution_over_time(pred_frame)
        calibration_results[model_name] = calib_df

        # Collect scalar metrics
        all_metrics[model_name] = {
            "monthly_f1_mean": float(monthly_df["macro_f1"].mean()) if not monthly_df.empty else None,
            "monthly_f1_std":  float(monthly_df["macro_f1"].std())  if not monthly_df.empty else None,
            "monthly_f1_min":  float(monthly_df["macro_f1"].min())  if not monthly_df.empty else None,
            "covid_f1": {
                row["period"]: round(row["macro_f1"], 4)
                for _, row in covid_df.iterrows()
            } if not covid_df.empty else {},
            "acf_lag1":  acf_data["mean_acf"][1]  if len(acf_data["mean_acf"]) > 1 else None,
            "acf_lag10": acf_data["mean_acf"][10] if len(acf_data["mean_acf"]) > 10 else None,
            "acf_lag60": acf_data["mean_acf"][60] if len(acf_data["mean_acf"]) > 60 else None,
        }

        log(f"    Monthly F1 mean={all_metrics[model_name]['monthly_f1_mean'] or 0:.4f} "
            f"std={all_metrics[model_name]['monthly_f1_std'] or 0:.4f}")

    # Generate all plots
    log("\n  Generating figures...")
    plot_f1_over_time(monthly_results, out_dir)
    plot_within_mission(within_results, out_dir)
    plot_error_acf(acf_results, out_dir)
    plot_covid_breakdown(covid_results, out_dir)
    if any(v[0] is not None for v in transition_results.values()):
        plot_class_transitions(transition_results, out_dir)
    plot_calibration_over_time(calibration_results, out_dir)

    # Save metrics JSON
    metrics_path = os.path.join(RESULTS_TS_DIR, f"ts_metrics_{framework}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log(f"\n  Saved: {metrics_path}")

    # Save summary CSV
    summary_rows = []
    for model_name, m in all_metrics.items():
        row = {"model": model_name, "framework": framework}
        row["monthly_f1_mean"] = round(m["monthly_f1_mean"], 4) if m["monthly_f1_mean"] else None
        row["monthly_f1_std"]  = round(m["monthly_f1_std"], 4)  if m["monthly_f1_std"]  else None
        for period in ["pre_covid", "restrictions", "post_covid"]:
            row[f"f1_{period}"] = m["covid_f1"].get(period)
        row["acf_lag1"]  = round(m["acf_lag1"],  4) if m["acf_lag1"]  else None
        row["acf_lag10"] = round(m["acf_lag10"], 4) if m["acf_lag10"] else None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_TS_DIR, f"ts_summary_{framework}.csv")
    summary_df.to_csv(summary_path, index=False)
    log(f"  Saved: {summary_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    log("=" * 60)
    log("ZTBus Time-Series Evaluation (05_timeseries_eval.py)")
    log("=" * 60)

    # -----------------------------------------------------------------------
    # Load data (mirrors 02_train.py setup)
    # -----------------------------------------------------------------------

    log("\n[1] Loading data...")
    meta = load_metadata()
    df_raw = load_sample_missions(meta, usecols=EDA_USE_COLS)

    log("[2] Forward-filling...")
    filled_parts = []
    for mission_name, group in df_raw.groupby("mission_name"):
        filled_parts.append(forward_fill_stop_columns(group))
    df = pd.concat(filled_parts, ignore_index=True)
    df = df[df["itcs_numberOfPassengers"].notna()].copy()

    log("[3] Mission-level split (same seed as training)...")
    train_df, test_df = mission_stratified_split(df, test_size=0.2, seed=RANDOM_SEED)

    log("[4] Tercile boundaries from training data...")
    q1, q2 = compute_tercile_boundaries(train_df["itcs_numberOfPassengers"])
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["demand_class"] = assign_demand_class(train_df["itcs_numberOfPassengers"], q1, q2)
    test_df["demand_class"]  = assign_demand_class(test_df["itcs_numberOfPassengers"],  q1, q2)

    log("[5] Feature engineering...")
    train_feat = build_feature_set(train_df)
    test_feat  = build_feature_set(test_df)

    # Keep time_iso + mission_name on test_feat for temporal evaluation
    # (they get dropped inside run_timeseries_evaluation before predict())
    if "time_iso" not in test_feat.columns and "time_iso" in test_df.columns:
        test_feat["time_iso"] = test_df["time_iso"].values
    if "mission_name" not in test_feat.columns and "mission_name" in test_df.columns:
        test_feat["mission_name"] = test_df["mission_name"].values

    X_train, y_train = prepare_features_and_target(
        train_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    X_test, y_test = prepare_features_and_target(
        test_feat, target_col="demand_class",
        drop_cols=["mission_name", "itcs_numberOfPassengers"],
    )
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test  = X_test[common_cols]
    train_mask = X_train.notna().all(axis=1)
    test_mask  = X_test.notna().all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test,  y_test  = X_test[test_mask],   y_test[test_mask]
    test_feat_aligned = test_feat[test_mask].copy()

    log(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # -----------------------------------------------------------------------
    # Train models (or load from checkpoint)
    # -----------------------------------------------------------------------

    log("\n[6] Training models...")
    configs = get_model_configs(seed=RANDOM_SEED)
    trained_models = {}

    for model_name, cfg in configs.items():
        log(f"  Fitting {model_name}...")
        t0 = time.time()
        cfg["model"].fit(X_train, y_train)
        log(f"    Done in {time.time()-t0:.1f}s")
        trained_models[model_name] = cfg["model"]

    # -----------------------------------------------------------------------
    # Run time-series evaluation
    # -----------------------------------------------------------------------

    run_timeseries_evaluation(
        test_feat=test_feat_aligned,
        y_test=y_test,
        trained_models=trained_models,
        framework="pooled",
    )

    log("\nDone. Results in results/timeseries/ and figures/timeseries/")
