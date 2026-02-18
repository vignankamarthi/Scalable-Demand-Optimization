#!/bin/bash
#SBATCH --job-name=ztbus-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# pip --user packages
export PATH="$HOME/.local/bin:$PATH"

# Work from project root
cd ~/ondemand/Scalable-Demand-Optimization

# Create logs directory
mkdir -p logs

# Run training
python3 scripts/02_train.py

echo "Job finished: $(date)"
