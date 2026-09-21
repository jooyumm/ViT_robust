#!/bin/bash
# 실험 6: PatchFool, patch_select=Rand (attention 대신 랜덤 1개 토큰 선택) x P=8/16 — tag=randsel
# baseline(01_baseline.sh)의 Attn 선택 결과와 비교해서, "똑똑하게 토큰을 고르는 것"과
# "토큰 단위로 공격하는 것 자체" 중 뭐가 더 중요한지 분리.
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/05_patchfool_randsel.sh
#SBATCH --job-name=vit_05_pf_randsel
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=results/job_logs/nohup_05_pf_randsel_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

for P in 8 16; do
  BS=8; if [ "$P" != "8" ]; then BS=64; fi
  python experiments/main.py \
    --attacks patch_fool \
    --patch_sizes "$P" \
    --num_samples 1000 \
    --batch_size "$BS" \
    --pf_attack_mode CE_loss \
    --pf_iters 250 \
    --pf_patch_select Rand \
    --tag randsel \
    --seed "$SEED"
done