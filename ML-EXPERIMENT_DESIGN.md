# ML Experiment Plan

## Problem

3-class ridership demand classification (low / medium / high) at stop-time granularity on ZTBus trolleybus data (1,409 missions, 1-second resolution, Zurich 2019-2022).

## Pipeline Status

| Stage | Status | Module | Tests |
|-------|--------|--------|-------|
| Data loading | DONE | src/data_loading.py | 11/11 |
| Preprocessing (units, temporal, ffill) | DONE | src/preprocessing.py | 48/48 |
| Target construction (tercile binning) | DONE | src/target.py | 14/14 |
| Feature engineering (rolling, encoding) | DONE | src/feature_engineering.py | 42/42 |
| Model pipeline (split, train, eval) | DONE | src/model_pipeline.py | 18/18 |
| EDA (11 figures) | DONE | scripts/01_eda.py | visual |
| Training script | READY | scripts/02_train.py | -- |
| **TOTAL TESTS** | **150/150 PASSING** | | 3.08s |

## The 2 Models

| # | Model | Config | Scaling | Why |
|---|-------|--------|---------|-----|
| 1 | **Decision Tree** | gini, no max_depth | No | Interpretable baseline, human-readable rules |
| 2 | **Random Forest** | 300 trees, all CPUs | No | Ensemble strength, feature importance (RQ2) |

## Feature Specification (Source of Truth)

This table is exhaustive. Every feature entering the model is listed below.
Code MUST match this spec. See `src/feature_engineering.py:build_feature_set()`.

### Excluded Features

| Feature | Raw Source | Why Excluded |
|---------|-----------|-------------|
| `busNumber` | metaData.csv | Bus identifier (183 or 208), not a predictor. Introduces spurious correlation. |
| `temp_C` | `temperature_ambient` (K->C) | EDA Figure 08: r = -0.011 with ridership. No signal. |
| `time_iso`, `time_unix` | raw columns | Raw timestamps. Replaced by engineered temporal features. |
| `gnss_latitude`, `gnss_longitude` | raw columns | Raw radians. Replaced by `lat_deg`, `lon_deg`. |
| `odometry_vehicleSpeed` | raw column | Raw m/s. Replaced by `speed_kmh` + rolling features. |
| `temperature_ambient` | raw column | Raw Kelvin. temp excluded entirely per EDA. |
| `gnss_course` | raw column | Not loaded (EDA_USE_COLS uses GNSS_COLS[:3], skipping course). Heading not relevant to demand. |
| Individual wheel speeds (6 cols) | raw columns | Not loaded. Redundant with `odometry_vehicleSpeed`. |
| `odometry_articulationAngle` | raw column | Not loaded. Articulation angle not relevant to demand. |
| `odometry_steeringAngle` | raw column | Not loaded. Steering angle not relevant to demand. |

### Included Features (~53 total)

| # | Engineered Name | Source Column | Type | Unit | Category | Justification |
|---|----------------|---------------|------|------|----------|---------------|
| 1 | `hour` | `time_iso` | int | 0-23 | Temporal | EDA Fig 03: clear hourly ridership patterns |
| 2 | `dayofweek` | `time_iso` | int | 0-6 (Mon-Sun) | Temporal | EDA Fig 03: weekday vs weekend demand differs |
| 3 | `month` | `time_iso` | int | 1-12 | Temporal | EDA Fig 04: seasonal variation visible |
| 4 | `year` | `time_iso` | int | 2019-2022 | Temporal | EDA Fig 04: COVID disruption in 2020 |
| 5 | `is_weekend` | `time_iso` | bool | - | Temporal | EDA Fig 03: weekend demand pattern shift |
| 6 | `is_rush_hour` | `time_iso` | bool | - | Temporal | EDA Fig 03: weekday peak at hour 15 |
| 7 | `lat_deg` | `gnss_latitude` | float | degrees | Spatial | Spatial demand clustering across Zurich |
| 8 | `lon_deg` | `gnss_longitude` | float | degrees | Spatial | Spatial demand clustering across Zurich |
| 9 | `gnss_altitude` | `gnss_altitude` | float | meters | Spatial | Elevation variation in hilly Zurich routes |
| 10 | `speed_kmh` | `odometry_vehicleSpeed` | float | km/h | Operational | Vehicle operational state indicator |
| 11 | `acceleration` | `odometry_vehicleSpeed` (diff) | float | m/s^2 | Operational | Stop-and-go dynamics; computed per-mission |
| 12 | `speed_roll_mean_60` | `odometry_vehicleSpeed` | float | m/s | Operational | 1-min smoothed speed trend |
| 13 | `speed_roll_std_60` | `odometry_vehicleSpeed` | float | m/s | Operational | 1-min speed variability (traffic) |
| 14 | `speed_roll_mean_300` | `odometry_vehicleSpeed` | float | m/s | Operational | 5-min smoothed speed trend |
| 15 | `speed_roll_std_300` | `odometry_vehicleSpeed` | float | m/s | Operational | 5-min speed variability |
| 16 | `electric_powerDemand` | `electric_powerDemand` | float | W | Sensor | Total vehicle power draw; increases with HVAC + passenger load |
| 17 | `power_roll_mean_60` | `electric_powerDemand` | float | W | Sensor | 1-min smoothed power trend |
| 18 | `power_roll_std_60` | `electric_powerDemand` | float | W | Sensor | 1-min power variability |
| 19 | `power_roll_mean_300` | `electric_powerDemand` | float | W | Sensor | 5-min smoothed power trend |
| 20 | `power_roll_std_300` | `electric_powerDemand` | float | W | Sensor | 5-min power variability |
| 21 | `traction_tractionForce` | `traction_tractionForce` | float | N | Sensor | Motor force estimate; load-dependent |
| 22 | `traction_brakePressure` | `traction_brakePressure` | float | Pa | Sensor | Brake line pressure; stop pattern indicator |
| 23 | `status_doorIsOpen` | `status_doorIsOpen` | bool | - | Status | Door state; boarding indicator (EDA Fig 09: weak but keep for feature importance) |
| 24 | `status_gridIsAvailable` | `status_gridIsAvailable` | bool | - | Status | Overhead grid connection; route segment type |
| 25 | `status_haltBrakeIsActive` | `status_haltBrakeIsActive` | bool | - | Status | Halt brake = standing still; stop dwell proxy |
| 26 | `status_parkBrakeIsActive` | `status_parkBrakeIsActive` | bool | - | Status | Park brake = extended stop; terminus detection |
| 27-34 | `route_*` (8 dummies) | `itcs_busRoute` | int | 0/1 | Categorical | One-hot, drop_first. EDA Fig 06: distinct demand per route. |
| 35-53 | `stop_*` (19 dummies) | `itcs_stopName` | int | 0/1 | Categorical | Top-20 + __other__, drop_first. EDA Fig 05: high stop-level variance. |

Total: ~53 features (exact count depends on route/stop cardinality in training data).

### Feature Groups Summary

- **Temporal** (6): hour, dayofweek, month, year, is_weekend, is_rush_hour
- **Spatial** (3): lat_deg, lon_deg, gnss_altitude
- **Operational** (5): speed_kmh, acceleration, speed_roll_mean/std at 60s and 300s
- **Sensor** (7): electric_powerDemand, power_roll_mean/std at 60s and 300s, traction_tractionForce, traction_brakePressure
- **Status** (4): status_doorIsOpen, status_gridIsAvailable, status_haltBrakeIsActive, status_parkBrakeIsActive
- **Categorical** (~27): route dummies (8) + stop dummies (19)

## Evaluation Protocol

- **Primary metrics**: Macro F1, Balanced Accuracy
- **Diagnostics**: 3x3 confusion matrix per model, per-class precision/recall/F1
- **Splits**: Mission-level 80/20 train/test (GroupShuffleSplit) -- no mission in both sets
- **Leakage prevention**: Tercile boundaries from train only
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

## Compute Environment

- **Cluster**: NEU Explorer (SLURM)
- **Partition**: `short` (4-hour wall time limit)
- **Default allocation**: 16 CPUs, 128GB RAM, 4 hours
- **Why short partition**: DT + RF total training time is ~2.2 hours, well within the 4-hour limit.
- **Why 128GB RAM**: First run at 64GB was killed ~25 min into Random Forest training. X_train alone is ~15.9GB (37M rows x 53 features x 8 bytes). With DataFrame copies during feature engineering and RF building 300 trees concurrently on 16 CPUs (`n_jobs=-1`), peak memory exceeds 64GB. 128GB provides adequate headroom.
- **All models are sklearn (CPU-only)**: Decision Tree, Random Forest. No GPU needed.

## Execution

```bash
# Local
python3 scripts/02_train.py

# Cluster (NEU Explorer, 16 CPUs, 128GB RAM)
sbatch scripts/train.sbatch
```

## Expected Outputs

```
results/model_summary.csv          -- 2 rows, macro_f1 + balanced_acc per model
results/model_results.json         -- full metrics + confusion matrices
results/feature_importances.csv    -- per-feature Gini importance from RF
figures/cm_<model>.png             -- confusion matrix heatmap per model (x2)
figures/model_comparison.png       -- bar chart comparing both models
figures/feature_importance.png     -- top-20 feature importance bar chart (RF)
figures/feature_group_importance.png -- group-level importance bar chart (RF)
```

## What Happens After Training

1. Review model_summary.csv -- rank models by macro F1
2. Inspect confusion matrices -- check for systematic misclassification patterns
3. If any model has < 0.5 macro F1, diagnose (class collapse? feature issues?)
4. **Feature importance analysis (RQ2)**:
   - Extract Random Forest `.feature_importances_` (Gini importance, built-in)
   - Generate `figures/feature_importance.png` -- bar chart of top-20 features by importance
   - Aggregate importance by feature group (Temporal, Spatial, Operational, Sensor, Status, Categorical)
   - Generate `figures/feature_group_importance.png` -- group-level bar chart
   - Empirically validate EDA exclusion decisions:
     - Confirm temp_C exclusion was justified (compare to included features of similar type)
     - Check which status flags carry real signal vs noise (door, grid, halt brake, park brake)
   - Compute permutation importance on test set for both models (cross-validate Gini findings)
5. Write results section for final report
6. Submit deliverables (report + code + figures via Teams)
