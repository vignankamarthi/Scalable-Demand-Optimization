# Task: Execute Training Pipeline on Cluster

Reproducible step-by-step guide for running the full ZTBus demand classification experiment on the NEU Explorer GPU cluster.

## Prerequisites

- SSH access to NEU Explorer cluster
- SLURM job submission privileges
- GPU partition access (1x H100)
- Python 3.11+ available on the cluster

## Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/vignankamarthi/Scalable-Demand-Optimization.git
cd Scalable-Demand-Optimization
```

## Step 2: Download the Dataset

Download the ZTBus compressed dataset from ETH Zurich Research Collection:

**Link**: https://www.research-collection.ethz.ch/entities/researchdata/61ac2f6e-2ca9-4229-8242-aed3b0c0d47c

Download `ZTBus_compressed.zip` and `metaData.csv` to the cluster. If direct download is not available via `wget`, download locally and `scp` to the cluster.

```bash
# Create data directory
mkdir -p data/processed/raw

# Transfer files (from local machine)
scp ZTBus_compressed.zip <username>@explorer.northeastern.edu:~/Scalable-Demand-Optimization/data/processed/
scp metaData.csv <username>@explorer.northeastern.edu:~/Scalable-Demand-Optimization/data/processed/raw/

# On cluster: extract
cd data/processed
unzip ZTBus_compressed.zip -d raw/
cd ../..
```

**Verify**: `ls data/processed/raw/ | wc -l` should return `1410` (1,409 mission CSVs + 1 metaData.csv).

## Step 3: Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn scipy pytest
```

## Step 4: Run Tests

Verify all modules work correctly before training.

```bash
python -m pytest tests/ -v
```

**Expected**: 125 tests passed. Do not proceed if any tests fail.

## Step 5: Run EDA (Optional)

Generate exploratory visualizations. Not required for training, but useful for understanding the data.

```bash
python scripts/01_eda.py
```

Outputs 11 figures to `figures/`.

## Step 6: Run Training

### Option A: SLURM Batch Job (Recommended)

```bash
mkdir -p logs
sbatch scripts/train.sbatch
```

Monitor progress:
```bash
tail -f logs/train_<job_id>.out
```

### Option B: Interactive Session

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=08:00:00 --pty bash
source .venv/bin/activate
python scripts/02_train.py
```

## Step 7: Verify Results

After training completes, verify outputs exist:

```bash
# Model summary
cat results/model_summary.csv

# Full results
cat results/model_results.json

# Confusion matrix figures
ls figures/cm_*.png

# Model comparison chart
ls figures/model_comparison.png
```

## Step 8: Deliver Results

Results are gitignored and should NOT be pushed to the public repository.

**Send the following files privately via Microsoft Teams**:
- `results/model_summary.csv`
- `results/model_results.json`
- All files in `figures/` (EDA + confusion matrices + model comparison)

Do not commit or push any files from `results/`, `figures/`, or `logs/`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: src.*` | Run from repo root, not from `scripts/` |
| Tests fail | Check Python version (`python3 --version` >= 3.11) and all dependencies installed |
| SLURM job killed | Check `logs/train_<id>.err` -- likely OOM. Reduce batch size or request more memory |
| Dataset CSV count != 1410 | Re-extract the zip. Ensure `metaData.csv` is in `data/processed/raw/` |
| `FileNotFoundError` on data | Verify `data/processed/raw/` contains the CSVs, not a nested subdirectory |

## Expected Timeline

- Steps 1-4: ~15 minutes (setup + test verification)
- Step 5 (EDA): ~5 minutes
- Step 6 (Training): ~1-4 hours depending on cluster queue and data volume
- Step 7-8: ~5 minutes

Total: Under 5 hours including cluster queue wait.
