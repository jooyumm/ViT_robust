#!/bin/bash
#SBATCH --job-name=vitguard_diversity
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/diversity_diagnostic/08_diversity_run_%j.txt
cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python experiments/diversity_diagnostic/vitguard_diversity_test.py --seed 456 --num_samples 50 --chunk 20
