#!/bin/bash
#SBATCH --job-name=ztbus_emb_train
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/emb_train_%j.log
#SBATCH --error=logs/emb_train_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus training with TimesFM embeddings
# Requires: 08_extract_embeddings.py completed first (data/cache/embeddings/)
# Run from repo root: sbatch scripts/09_train_with_embeddings.sbatch
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
check_dep pandas pandas
check_dep numpy numpy
check_dep matplotlib matplotlib
check_dep seaborn seaborn
check_dep joblib joblib

echo "All dependencies verified."

# Verify embeddings exist
EMBED_COUNT=$(ls data/cache/embeddings/*.npy 2>/dev/null | wc -l)
echo "Cached embeddings: $EMBED_COUNT"
if [ "$EMBED_COUNT" -eq 0 ]; then
    echo "FATAL: No embeddings found at data/cache/embeddings/. Run 08_extract_embeddings.py first."
    exit 1
fi

mkdir -p logs results/embeddings figures/embeddings

echo "Starting embedding-augmented training..."
python scripts/09_train_with_embeddings.py --fresh

echo "Job finished: $(date)"
