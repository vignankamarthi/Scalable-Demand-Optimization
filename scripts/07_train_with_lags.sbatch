#!/bin/bash
#SBATCH --job-name=ztbus_lags
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lags_%j.log
#SBATCH --error=logs/lags_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus training with lag features: DT + RF + XGBoost, pooled + temporal
# Run from repo root: sbatch scripts/07_train_with_lags.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization
source .venv/bin/activate

python -c "import sklearn; print('sklearn:', sklearn.__version__)"

mkdir -p logs
mkdir -p results/lag_features
mkdir -p figures/lag_features

echo "Starting lag-augmented training pipeline..."
python scripts/07_train_with_lags.py --fresh

echo "Job finished: $(date)"
