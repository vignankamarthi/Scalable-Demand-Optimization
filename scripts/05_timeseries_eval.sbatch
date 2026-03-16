#!/bin/bash
#SBATCH --job-name=ztbus_ts_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ts_eval_%j.log
#SBATCH --error=logs/ts_eval_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus time-series evaluation (Explorer GPU node)
# Trains DT + RF (CPU) + XGBoost (GPU via gpu_hist), then runs 6 temporal
# diagnostics. Checkpoints trained models to skip retraining on resubmit.
# Run from repo root: sbatch scripts/05_timeseries_eval.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# Work from project root
cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization

source .venv/bin/activate

mkdir -p logs
mkdir -p results/timeseries
mkdir -p figures/timeseries

echo "Starting time-series evaluation..."
python scripts/05_timeseries_eval.py

echo "Job finished: $(date)"
