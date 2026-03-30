#!/bin/bash
#SBATCH --job-name=ztbus_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe_%j.log
#SBATCH --error=logs/probe_%j.err
#SBATCH --mail-type=END,FAIL

echo "Job started: $(date)"
echo "Node: $(hostname)"

cd /projects/SensingandInnovationLab/vignankamarthi/Scalable-Demand-Optimization
source .venv-timesfm/bin/activate

export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
mkdir -p "$HF_HOME" logs results/timeseries

python scripts/08_probe_timesfm.py

echo "Job finished: $(date)"
