# ML Experiment Plan

## Problem

3-class ridership demand classification (low / medium / high) at stop-time granularity on ZTBus trolleybus data (1,409 missions, 1-second resolution, Zurich 2019-2022).

## Pipeline Status

| Stage | Status | Module | Tests |
|-------|--------|--------|-------|
| Data loading | DONE | src/data_loading.py | 11/11 |
| Preprocessing (units, temporal, ffill) | DONE | src/preprocessing.py | 48/48 |
| Target construction (tercile binning) | DONE | src/target.py | 14/14 |
| Feature engineering (rolling, encoding) | DONE | src/feature_engineering.py | 34/34 |
| Model pipeline (split, train, eval) | DONE | src/model_pipeline.py | 18/18 |
| EDA (11 figures) | DONE | scripts/01_eda.py | visual |
| Training script | READY | scripts/02_train.py | -- |
| **TOTAL TESTS** | **125/125 PASSING** | | 1.30s |

## The 6 Models

| # | Model | Config | Scaling | Why |
|---|-------|--------|---------|-----|
| 1 | **Decision Tree** | gini, no max_depth | No | Interpretable baseline, human-readable rules |
| 2 | **Random Forest** | 300 trees, all CPUs | No | Ensemble strength, feature importance (RQ2) |
| 3 | **k-NN** | k=11, uniform weights | Yes | Local similarity, non-parametric |
| 4 | **MLP Small** | (64,), ReLU, early stop | Yes | Lightweight neural baseline |
| 5 | **MLP Medium** | (128, 64), ReLU, early stop | Yes | Intermediate capacity |
| 6 | **MLP Large** | (256, 128, 64), ReLU, early stop | Yes | Depth-performance tradeoff |

## Feature Groups

- **Temporal**: hour, dayofweek, month, year, is_weekend, is_rush_hour
- **Spatial**: lat_deg, lon_deg (converted from radians)
- **Operational**: speed_kmh, temp_C, acceleration, rolling mean/std (speed + power, 60s + 300s windows)
- **Categorical**: route (one-hot, drop_first), stop_name (top-20, rest bucketed as __other__)
- **Sensor**: door state, HVAC power, odometry

## Evaluation Protocol

- **Primary metrics**: Macro F1, Balanced Accuracy
- **Diagnostics**: 3x3 confusion matrix per model, per-class precision/recall/F1
- **Splits**: Mission-level 80/20 train/test (GroupShuffleSplit) -- no mission in both sets
- **Leakage prevention**: Tercile boundaries from train only, scaler fitted on train only
- **Reproducibility**: RANDOM_SEED = 42 everywhere

## EDA Rationale (11 Figures)

Each EDA figure answers a specific question required before modeling. Source: proposal Section 5.

| Fig | What | Why It Matters for Modeling |
|-----|------|-----------------------------|
| 01 | Passenger count distribution + tercile lines | Validates tercile binning produces balanced classes. If heavily skewed, we'd need quantile-based or domain-driven thresholds instead. Result: 37/32/32% -- balanced, terciles work. |
| 02 | Temporal distributions (hour, day, month) | Detects sampling bias. If data only covers weekdays or certain months, temporal features become unreliable. Result: good coverage across all periods. |
| 03 | Ridership by hour (weekday vs weekend) | Confirms rush-hour patterns exist in the data, justifying the `is_rush_hour` feature. Result: clear weekday peak at hour 15, weekend shifted to 13 -- temporal features carry signal. |
| 04 | Seasonal trends (monthly mean) | Tests whether month/year features are worth including. If ridership is flat across months, they add noise. Result: visible seasonal variation + COVID disruption in 2020 -- month is informative. |
| 05 | Stop-level demand (top 25 stops) | Identifies spatial demand clustering. If demand is uniform across stops, stop encoding wastes dimensions. Result: high variance across stops -- stop_name is a strong feature. |
| 06 | Route comparison (violin plots) | Tests whether routes have different demand profiles. If identical, route encoding adds nothing. Result: distinct distributions per route -- route is informative. |
| 07 | Correlation matrix (9 numeric features) | Identifies collinear features (redundant) and features with predictive signal for passenger count. Guides feature selection. |
| 08 | Temperature vs ridership | Directly tests whether ambient temperature predicts demand. Result: r = -0.011 -- no relationship. Informed decision to exclude temp from features if needed. |
| 09 | Door events (mean pax + speed by door state) | Tests whether door state correlates with boarding activity. Result: NULL -- mean pax nearly identical (~16.5 vs ~16) due to forward-fill artifact. Speed difference is trivially obvious (bus stops to open doors). Door state as a raw binary feature carries minimal predictive signal for demand class. |
| 10 | Missing data assessment | Quantifies NaN/dash prevalence per column. Guides imputation strategy. Result: GNSS 1.3% NaN, everything else negligible -- forward-fill is sufficient. |
| 11 | Target class distribution | Final validation that the 3-class target is balanced after tercile binning. If one class dominates, we'd need resampling or class weights. Result: well-balanced. |

## Data Integrity Decisions (Quantitative Evidence)

| Decision | Evidence |
|----------|----------|
| Tercile binning | Class balance: 37/32/32% (well-balanced) |
| Forward-fill ITCS columns | Physically correct: pax count constant between stops. Recovers 0.9% -> 100% usable |
| Top-20 stop encoding | 147 unique stops; top-20 covers majority, rest bucketed to avoid sparse explosion |
| Temperature NOT a feature | Correlation with ridership: -0.011 (essentially zero) |
| Mission-level splits | Prevents temporal leakage from same-mission observations |

## Execution

```bash
# Local
python3 scripts/02_train.py

# Cluster (NEU Explorer, 1x H100)
sbatch scripts/train.sbatch
```

## Expected Outputs

```
results/model_summary.csv      -- 6 rows, macro_f1 + balanced_acc per model
results/model_results.json     -- full metrics + confusion matrices
figures/cm_<model>.png         -- confusion matrix heatmap per model (x6)
figures/model_comparison.png   -- bar chart comparing all 6 models
```

## What Happens After Training

1. Review model_summary.csv -- rank models by macro F1
2. Inspect confusion matrices -- check for systematic misclassification patterns
3. If any model has < 0.5 macro F1, diagnose (class collapse? feature issues?)
4. Feature importance from Random Forest -- answers RQ2 (which feature groups matter)
5. Write results section for final report
6. Submit deliverables (report + code + figures via Teams)
