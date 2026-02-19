#!/bin/bash
#SBATCH --job-name=ztbus-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Work from project root
cd ~/ondemand/Scalable-Demand-Optimization

# Activate environment
source .venv/bin/activate

# Create logs directory
mkdir -p logs

# Run training (fresh: discard old checkpoint, retrain DT + RF cleanly)
python3 scripts/02_train.py --fresh

echo "Job finished: $(date)"
