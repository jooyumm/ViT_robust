#!/bin/bash
#SBATCH --job-name=vitguard_protocolc
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=09_protocol_c/results/09_protocolc_run_%j.txt
cd /home/jooyumm/ViT_robust/ViT_tradeoff
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python 09_protocol_c/protocol_c_tokenmatch.py --seed 789 --num_samples 50 --num_patch 4 --chunk 25
