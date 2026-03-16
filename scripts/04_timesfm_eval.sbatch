#!/bin/bash
#SBATCH --job-name=ztbus_timesfm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/timesfm_%j.log
#SBATCH --error=logs/timesfm_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus TimesFM evaluation — zero-shot (+ optional fine-tuning)
# Run from repo root: sbatch scripts/04_timesfm_eval.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# Work from project root
cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization

# Activate Python 3.11 venv with TimesFM pre-installed
source .venv-timesfm/bin/activate

echo "Active Python: $(python --version)"

# Confirm TimesFM is importable
python -c "import timesfm; print('TimesFM OK:', list(vars(timesfm).keys())[:5])" \
    || { echo "ERROR: timesfm import failed"; exit 1; }

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
