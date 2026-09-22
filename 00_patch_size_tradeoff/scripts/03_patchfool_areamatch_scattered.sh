#!/bin/bash
# 실험 4: PatchFool, P=8, num_patch=4 (attention 상위 4개 독립 선택 — 흩어질 수 있음)
# P=16 baseline(1토큰=256px)과 공격 면적(4*64=256px)을 맞춤 — tag=areamatch16
# "P=8이 강건한 게 순수 토큰화 효과인가, 그냥 공격 면적이 작아서인가?"를 확인하는 첫 단계.
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/03_patchfool_areamatch_scattered.sh
#SBATCH --job-name=vit_03_pf_am_scattered
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/job_logs/nohup_03_pf_am_scattered_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

python experiments/main.py \
  --attacks patch_fool \
  --patch_sizes 8 \
  --num_samples 1000 \
  --batch_size 8 \
  --pf_attack_mode CE_loss \
  --pf_iters 250 \
  --pf_num_patch 4 \
  --tag areamatch16 \
  --seed "$SEED"