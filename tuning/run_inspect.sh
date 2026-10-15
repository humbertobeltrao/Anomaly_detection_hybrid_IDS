#!/bin/bash
#SBATCH --job-name=inspect_optuna
#SBATCH --partition=short-complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

#SBATCH --output=inspect_optuna_%j.out
#SBATCH --error=inspect_optuna_%j.err

source ~/Anomaly_detection_hybrid_IDS/venv/bin/activate

cd ~/Anomaly_detection_hybrid_IDS

echo "========================================"
echo "OPTUNA STUDY INSPECTION"
echo "========================================"

hostname
date

python -m tuning.inspect_study

echo "========================================"
echo "FINISHED"
echo "========================================"

date
