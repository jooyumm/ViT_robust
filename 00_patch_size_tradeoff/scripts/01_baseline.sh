#!/bin/bash
# 실험 1~3 (baseline): PGD / LaVAN / PatchFool x P=8/16/32
# 3개 공격 x 3개 patch size = 9개 조합을 한 GPU에서 순차 실행 (예전엔 스크립트 9개로 나눠서
# 병렬 제출했지만, 관리 편의를 위해 하나로 합침 — 대신 시간이 오래 걸려서 --time을 넉넉히 잡음).
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/01_baseline.sh
#SBATCH --job-name=vit_01_baseline
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --output=results/job_logs/nohup_01_baseline_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

for P in 8 16 32; do
  BS=16; PF_BS=8
  if [ "$P" != "8" ]; then BS=64; PF_BS=64; fi

  python experiments/main.py --attacks pgd --patch_sizes "$P" \
    --num_samples 1000 --batch_size "$BS" --seed "$SEED"

  python experiments/main.py --attacks lavan --patch_sizes "$P" \
    --num_samples 1000 --batch_size "$BS" --seed "$SEED"

  python experiments/main.py --attacks patch_fool --patch_sizes "$P" \
    --num_samples 1000 --batch_size "$PF_BS" \
    --pf_attack_mode CE_loss --pf_iters 250 --seed "$SEED"
done