#!/bin/bash
#SBATCH --job-name=ztbus_ts_eval
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ts_eval_%j.log
#SBATCH --error=logs/ts_eval_%j.err
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------------------
# ZTBus time-series evaluation
# Runs standalone: re-trains models then runs all temporal metrics/plots.
# Run from repo root: sbatch scripts/05_timeseries_eval.sbatch
#
# Note: if you want to run this AFTER 03_train_extended.py on the same
# job allocation, you can chain them:
#   sbatch --dependency=afterok:<jobid_of_03> scripts/05_timeseries_eval.sbatch
# -----------------------------------------------------------------------

echo "Job started: $(date)"
echo "Node: $(hostname)"

# Work from project root
cd ~/ondemand/Scalable-Demand-Optimization

source .venv/bin/activate

pip install statsmodels --quiet

mkdir -p logs
mkdir -p results/timeseries
mkdir -p figures/timeseries

echo "Starting time-series evaluation..."
python scripts/05_timeseries_eval.py

echo "Job finished: $(date)"
