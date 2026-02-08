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
- **Models**: Decision Tree, Random Forest, k-NN, MLP (Small / Medium / Large)
- **Evaluation**: Macro F1, balanced accuracy, per-class confusion matrices

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
│   ├── feature_engineering.py  # Categorical encoding, rolling windows, acceleration
│   ├── target.py               # Tercile binning, demand class assignment
│   └── model_pipeline.py       # Train/test split, model configs, evaluation
├── tests/                      # 125 tests (TDD -- all written before implementation)
├── scripts/
│   ├── 01_eda.py               # Exploratory data analysis (11 figures)
│   ├── 02_train.py             # Full training pipeline (6 models)
│   └── train.sbatch            # SLURM batch script for GPU cluster
├── Final-Project-Proposal-Markdown/  # 10-section project proposal
├── data/                       # Dataset (gitignored)
├── figures/                    # EDA and evaluation plots (gitignored)
├── results/                    # Model metrics and summaries (gitignored)
├── logs/                       # Training logs (gitignored)
├── ML-EXPERIMENT_DESIGN.md     # Full experiment plan, model configs, EDA rationale
├── DATASET_README.txt          # Authoritative column definitions from dataset authors
├── TEST_VALIDATION.md          # TDD methodology and test coverage strategy
└── TASK.md                     # Reproducible execution guide
```

## Feature Engineering

Features are extracted from the dense (forward-filled) telemetry stream:

- **Temporal**: hour, day of week, month, year, weekend flag, rush hour flag
- **Sensor**: temperature (C), altitude, power demand, traction force, brake pressure, door state
- **Kinematic**: speed (km/h), acceleration (m/s^2), rolling mean/std of speed (60s, 300s windows)
- **Spatial**: latitude/longitude (degrees), route (one-hot), stop name (top-20 + other bucket)

## Training

Training runs on a GPU cluster (1x H100, 12 CPUs). All stochastic operations seeded for reproducibility.

```bash
# Full pipeline (all 6 models)
python scripts/02_train.py

# SLURM submission
sbatch scripts/train.sbatch
```

## Testing

TDD methodology -- all 125 tests written before implementation. Mock data throughout, no dataset dependency.

```bash
python -m pytest tests/ -v
```

## Pipeline Design Decisions

| Decision | Rationale |
|----------|-----------|
| Forward-fill passenger counts | Physically valid: count constant between stops. Transforms 0.9% to 100% usable data |
| Mission-level train/test split | Prevents temporal leakage from same-mission observations appearing in both sets |
| Tercile boundaries from train only | Prevents information leakage from test distribution |
| StandardScaler on train only | Applied to k-NN and MLP; tree models use raw features |
| Top-20 stop encoding | Avoids 147-column explosion; rare stops bucketed as "other" |
| Rolling windows (60s, 300s) | Captures short-term and medium-term kinematic context |

## License

This project is for educational purposes. See the [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c) for dataset terms.
