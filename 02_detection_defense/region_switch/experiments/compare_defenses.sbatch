#!/bin/bash
#SBATCH --job-name=region-switch-ablation
#SBATCH --partition=base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
set -Eeuo pipefail
cd /home/jooyumm/ViT_robust
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/vit-region-comparison-${SLURM_JOB_ID}"
python -u 02_detection_defense/region_switch/experiments/compare_defenses.py
