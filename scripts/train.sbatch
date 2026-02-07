#!/bin/bash
#SBATCH --job-name=ztbus-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Activate environment
source .venv/bin/activate

# Create logs directory
mkdir -p logs

# Run training
python scripts/02_train.py

echo "Job finished: $(date)"
