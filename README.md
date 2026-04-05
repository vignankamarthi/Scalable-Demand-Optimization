# Scalable Demand Optimization

A ground-up ML pipeline for predicting urban transit ridership demand using the ZTBus dataset from ETH Zurich.

**Dataset**: ZTBus -- Second-Resolution Trolleybus Dataset from Zurich -- [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c)
**Course**: IE 7275 Data Mining in Engineering (Northeastern University, Spring 2026)

All credit for the ZTBus dataset belongs to the original authors at ETH Zurich.

## What This Repository Does

The ZTBus dataset contains 1,409 trolleybus missions recorded at 1-second resolution across 26 sensor channels in Zurich (2019--2022). Passenger counts are reported as sparse stop events (~1% of rows).

**This repository builds a full classification pipeline**:
- **Task**: 3-class ridership demand classification (low / medium / high)
- **Key Insight**: Forward-filling passenger counts between stops (physically valid -- count is constant between boarding events) transforms 0.9% usable data into 100%
- **Models**: Decision Tree, Random Forest, XGBoost, ARIMA baseline, TimesFM (foundation model)
- **Temporal Analysis**: Stationarity testing (ADF/KPSS), autocorrelation diagnostics (ACF/PACF), stop-level lag feature engineering, foundation model embedding extraction
- **Evaluation**: Macro F1, balanced accuracy, per-class confusion matrices; pooled and temporal (pre-2022 train, 2022 test) frameworks

## Final Results (Temporal Evaluation)

| Config | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|
| Baseline (cross-sectional features only) | 0.5253 | 0.6066 | 0.6013 |
| + Stop-level lag features | 0.7551 | **0.8287** | 0.8247 |
| + Stop-level lags + TimesFM embeddings | 0.7398 | 0.8241 | 0.8254 |

ARIMA on hourly aggregate (pure temporal, no features): F1 = 0.1795 -- confirms temporal signal alone is insufficient for classification.

**Winner: Random Forest with stop-level lag features, F1 = 0.8287** (+0.2221 over baseline).

**Null result**: TimesFM embeddings (32-dim PCA of penultimate transformer layer) did not improve over stop-level lags. This validates that the temporal signal in bus ridership is fundamentally short-range and well-captured by recent stop history.

## The Problem: Urban Transit Demand Prediction

Given second-resolution sensor telemetry from a trolleybus (GPS, speed, power demand, temperature, door state), classify the current ridership demand level:

| Class | Passenger Count | Distribution |
|-------|----------------|--------------|
| Low | <= q1 (~9) | ~37% |
| Medium | q1 < x <= q2 (~19) | ~32% |
| High | > q2 (~19) | ~32% |

Tercile boundaries computed on training data only. Mission-level train/test splits prevent data leakage.

## Repository Structure

```
Scalable-Demand-Optimization/
├── src/
│   ├── config.py               # Centralized paths, constants, hyperparameters
│   ├── data_loading.py         # Metadata parsing, stratified sampling, CSV loading
│   ├── preprocessing.py        # Unit conversions, temporal features, forward-fill
│   ├── feature_engineering.py  # Categorical encoding, rolling windows, acceleration, lag features
│   ├── target.py               # Tercile binning, demand class assignment
│   ├── model_pipeline.py       # Train/test split, model configs, evaluation
│   ├── timeseries.py           # Stationarity tests (ADF/KPSS), ACF/PACF, hourly aggregation
│   ├── arima.py                # ARIMA fitting (AIC grid search) and forecasting
│   ├── timesfm_features.py     # TimesFM embedding loading and feature integration
│   ├── covid.py                # COVID restriction features (Oxford Stringency Index)
│   └── checkpoint.py           # Atomic JSON checkpointing for long-running jobs
├── tests/                      # 243 tests (TDD -- all written before implementation)
├── scripts/
│   ├── 01_eda.py               # Exploratory data analysis (11 figures)
│   ├── 02_train.py             # Baseline training pipeline (DT + RF)
│   ├── 03_train_extended.py    # Extended training (DT + RF + XGBoost, pooled + temporal eval)
│   ├── 04_timesfm_eval.py      # TimesFM zero-shot/fine-tuned demand classification
│   ├── 05_timeseries_eda.py    # Time series EDA (stationarity, ACF/PACF at stop + hourly level)
│   ├── 06_arima_baseline.py    # ARIMA baseline on hourly aggregate ridership
│   ├── 07_train_with_lags.py   # ML training with passenger count lag features
│   ├── 08_extract_embeddings.py # TimesFM penultimate-layer embedding extraction (GPU)
│   └── 09_train_with_embeddings.py # ML training with TimesFM embedding features
├── Final-Project-Proposal-Markdown/  # 10-section project proposal
├── data/                       # Dataset (gitignored)
├── figures/                    # EDA and evaluation plots
├── results/                    # Model metrics and summaries
├── logs/                       # Training logs (gitignored)
├── ML-EXPERIMENT_DESIGN.md     # Full experiment plan, model configs, EDA rationale
├── DATASET_README.txt          # Authoritative column definitions from dataset authors
├── TEST_VALIDATION.md          # TDD methodology and test coverage strategy
└── PLAN.md                     # Milestone 3 execution log and final results
```

## Feature Engineering

Features are extracted from the dense (forward-filled) telemetry stream:

- **Temporal**: hour, day of week, month, year, weekend flag, rush hour flag
- **Sensor**: altitude, power demand, traction force, brake pressure, door state
- **Kinematic**: speed (km/h), acceleration (m/s^2), rolling mean/std of speed and power (60s, 300s windows)
- **Spatial**: latitude/longitude (degrees), route (one-hot), stop name (top-20 + other bucket)
- **Stop-level lag**: passenger count at the previous 1, 3, and 5 stop events (computed from actual stop transitions, not forward-fill staircase -- prevents target leakage)
- **COVID**: restriction flag and intensity from Oxford Stringency Index
- **TimesFM Embeddings**: 32-dim PCA-reduced representations from TimesFM 2.5-200M penultimate transformer layer

## Training

Training runs on the NEU Explorer cluster (16 CPUs, 256GB RAM). GPU partition for TimesFM embedding extraction. All stochastic operations seeded for reproducibility.

```bash
# Baseline (DT + RF + XGBoost, pooled + temporal eval)
sbatch scripts/03_train_extended.sbatch

# With lag features
sbatch scripts/07_train_with_lags.sbatch

# TimesFM embedding extraction (GPU, run first)
sbatch scripts/08_extract_embeddings.sbatch

# With embeddings (after 08 completes)
sbatch scripts/09_train_with_embeddings.sbatch
```

## Time Series Analysis

Stationarity and autocorrelation diagnostics run locally:

```bash
# ACF/PACF plots, ADF/KPSS tests
python scripts/05_timeseries_eda.py

# ARIMA baseline (pure temporal, no features)
python scripts/06_arima_baseline.py
```

## Testing

TDD methodology -- all 243 tests written before implementation. Mock data throughout, no dataset dependency.

```bash
python -m pytest tests/ -v
```

## Pipeline Design Decisions

| Decision | Rationale |
|----------|-----------|
| Forward-fill passenger counts | Physically valid: count constant between stops. Transforms 0.9% to 100% usable data |
| Mission-level train/test split | Prevents temporal leakage from same-mission observations appearing in both sets |
| Temporal evaluation framework | Train pre-2022, test 2022+. Simulates real deployment scenario |
| Tercile boundaries from train only | Prevents information leakage from test distribution |
| No feature scaling needed | Tree-based models; scaling is irrelevant for split-based classifiers |
| Top-20 stop encoding | Avoids 147-column explosion; rare stops bucketed as "other" |
| Rolling windows (60s, 300s) | Captures short-term and medium-term kinematic context |
| Stop-level lag features (not 1Hz) | Naive 1Hz lags on forward-filled data leak the target (~95% correlation). Stop-level lags capture real autocorrelation across stop events |
| PCA on TimesFM embeddings | Reduces 1280-dim to 32-dim; fitted on training missions only |
| Merge-based embedding attachment | `pd.merge()` on a 1409-row lookup table replaced an iterative `.loc` loop (45K ops on 47M rows), cutting attach time from 55 min to 15 seconds |

## License

This project is for educational purposes. See the [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c) for dataset terms.
