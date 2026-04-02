#!/bin/bash
#SBATCH --job-name=ztbus_lags
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lags_%j.log
#SBATCH --error=logs/lags_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus training with stop-level lag features: DT + RF + XGBoost
# Run from repo root: sbatch scripts/07_train_with_lags.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization
source .venv/bin/activate

# ─── Dependency check: install if missing, die if install fails ───
check_dep() {
    local module="$1"
    local package="$2"
    python -c "import $module" 2>/dev/null || {
        echo "Installing missing dependency: $package"
        pip install "$package" || { echo "FATAL: Failed to install $package"; exit 1; }
    }
}

check_dep sklearn scikit-learn
check_dep xgboost xgboost
check_dep statsmodels statsmodels
check_dep pandas pandas
check_dep numpy numpy
check_dep matplotlib matplotlib
check_dep seaborn seaborn
check_dep joblib joblib

echo "All dependencies verified."

mkdir -p logs results/lag_features figures/lag_features

echo "Starting stop-level lag training pipeline..."
python scripts/07_train_with_lags.py --fresh

echo "Job finished: $(date)"
