#!/bin/bash
#SBATCH --job-name=ztbus_embeddings_train
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
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

python -c "import sklearn; print('sklearn:', sklearn.__version__)"

# Verify embeddings exist
EMBED_COUNT=$(ls data/cache/embeddings/*.npy 2>/dev/null | wc -l)
echo "Cached embeddings: $EMBED_COUNT"
if [ "$EMBED_COUNT" -eq 0 ]; then
    echo "ERROR: No embeddings found. Run 08_extract_embeddings.py first."
    exit 1
fi

mkdir -p logs
mkdir -p results/embeddings
mkdir -p figures/embeddings

echo "Starting embedding-augmented training..."
python scripts/09_train_with_embeddings.py --fresh

echo "Job finished: $(date)"
