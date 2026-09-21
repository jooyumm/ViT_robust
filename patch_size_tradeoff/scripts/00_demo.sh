#!/bin/bash
# 공격 데모 시각화 (P=8 vs P=16 vs P=32 비교, 이미지 1장) — visualize/00_attack_demo.py
#SBATCH --job-name=vit_00_demo
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=results/job_logs/nohup_00_demo_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

python visualize/00_attack_demo.py --seed 42