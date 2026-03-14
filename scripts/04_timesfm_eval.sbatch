#!/bin/bash
#SBATCH --job-name=ztbus_timesfm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/timesfm_%j.log
#SBATCH --error=logs/timesfm_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus TimesFM evaluation — zero-shot (+ optional fine-tuning)
# Run from repo root: sbatch scripts/04_timesfm_eval.sbatch
#
# If no GPU partition is available on the cluster, change to:
#   #SBATCH --partition=short
#   #SBATCH --mem=128G
# and add --cpu flag to the python call below.
# CPU-only inference will take significantly longer (~4-8hrs).
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (CPU node)"

# -----------------------------------------------------------------------
# Python version check
# TimesFM requires Python 3.11. NEU Explorer may default to 3.10 or 3.12.
# If your venv is already Python 3.11, skip the pyenv block below.
# -----------------------------------------------------------------------

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Default Python: $PYTHON_VERSION"

# Check if .venv is already Python 3.11
if [[ -f .venv/bin/python ]]; then
    VENV_PY_VERSION=$(.venv/bin/python --version 2>&1 | awk '{print $2}')
    echo "venv Python: $VENV_PY_VERSION"
fi

# Work from project root
cd ~/ondemand/Scalable-Demand-Optimization

# Activate Python 3.11 venv (separate from main .venv which is Python 3.9)
source .venv-timesfm/bin/activate

echo "Active Python: $(python --version)"

# -----------------------------------------------------------------------
# Install TimesFM from source (required — not on pip as stable package)
# This is cached after first run; safe to re-run.
# -----------------------------------------------------------------------

TIMESFM_DIR="$HOME/timesfm"

if [[ ! -d "$TIMESFM_DIR" ]]; then
    echo "Cloning TimesFM repository..."
    git clone https://github.com/google-research/timesfm.git "$TIMESFM_DIR"
fi

echo "Installing TimesFM (torch backend)..."
cd "$TIMESFM_DIR"
git pull origin main --quiet
pip install -e .[torch] --quiet
cd -

# Confirm install
python -c "import timesfm; print('TimesFM version:', timesfm.__version__)" \
    || echo "WARNING: timesfm import failed — check install logs"

# -----------------------------------------------------------------------
# HuggingFace cache setup
# Point HF_HOME to scratch space to avoid filling home quota
# Adjust path to match your cluster's scratch/work filesystem
# -----------------------------------------------------------------------

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# -----------------------------------------------------------------------
# Create output directories
# -----------------------------------------------------------------------

mkdir -p logs
mkdir -p results/timesfm
mkdir -p figures/timesfm

# -----------------------------------------------------------------------
# Run zero-shot evaluation
# Add --finetune to also run fine-tuned evaluation (~30-60min extra)
# Add --cpu to force CPU inference (if no GPU available)
# -----------------------------------------------------------------------

echo "Starting TimesFM zero-shot evaluation..."
python scripts/04_timesfm_eval.py

# Uncomment to also run fine-tuned:
# python scripts/04_timesfm_eval.py --finetune

echo "Job finished: $(date)"
