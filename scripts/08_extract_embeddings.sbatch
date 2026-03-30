#!/bin/bash
#SBATCH --job-name=ztbus_embeddings
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
# Run from repo root: sbatch scripts/08_extract_embeddings.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization
source .venv/bin/activate

python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import timesfm; print('TimesFM loaded')"

mkdir -p logs
mkdir -p data/cache/embeddings

echo "Starting embedding extraction..."
python scripts/08_extract_embeddings.py

echo "Job finished: $(date)"
