#!/bin/bash
#SBATCH --job-name=ztbus_em
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

echo "Active Python: $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import timesfm; print('TimesFM loaded')"

mkdir -p logs data/cache/embeddings

echo "Starting embedding extraction..."
python scripts/08_extract_embeddings.py

echo "Job finished: $(date)"
