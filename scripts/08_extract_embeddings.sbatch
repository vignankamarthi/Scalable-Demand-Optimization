#!/bin/bash
#SBATCH --job-name=ztbus_em
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embeddings_%j.log
#SBATCH --error=logs/embeddings_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# TimesFM embedding extraction (GPU required)
# Uses .venv-timesfm (Python 3.11 + TimesFM, separate from main .venv)
# Run from repo root: sbatch scripts/08_extract_embeddings.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"

cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization
source .venv-timesfm/bin/activate

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "$HF_HOME"

# ─── Dependency check: install if missing, die if install fails ───
check_dep() {
    local module="$1"
    local package="$2"
    python -c "import $module" 2>/dev/null || {
        echo "Installing missing dependency: $package"
        pip install "$package" || { echo "FATAL: Failed to install $package"; exit 1; }
    }
}

check_dep torch "torch --index-url https://download.pytorch.org/whl/cu128"
check_dep timesfm "git+https://github.com/google-research/timesfm.git"
check_dep sklearn scikit-learn
check_dep pandas pandas
check_dep numpy numpy

echo "All dependencies verified."

# Verify CUDA
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
echo "CUDA available: $CUDA_OK"
if [ "$CUDA_OK" != "True" ]; then
    echo "WARNING: CUDA not available. Embedding extraction will be extremely slow on CPU."
fi

echo "Active Python: $(python --version)"

mkdir -p logs data/cache/embeddings

echo "Starting embedding extraction..."
python scripts/08_extract_embeddings.py

echo "Job finished: $(date)"
