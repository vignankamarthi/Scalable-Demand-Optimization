#!/bin/bash
#SBATCH --job-name=ztbus_extended
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extended_%j.log
#SBATCH --error=logs/extended_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus extended training: DT + RF + XGBoost, pooled + temporal eval
# Run from repo root: sbatch scripts/03_train_extended.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Work from project root
cd ~/ondemand/Scalable-Demand-Optimization

# Activate virtual environment
source .venv/bin/activate

# Install xgboost if not already present (safe to re-run)
pip install xgboost --quiet

# Confirm environment
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
python -c "import sklearn; print('sklearn version:', sklearn.__version__)"

# Create logs dir if it doesn't exist
mkdir -p logs
mkdir -p results/extended
mkdir -p figures/extended

echo "Starting extended training pipeline..."
python scripts/03_train_extended.py

echo "Job finished: $(date)"
