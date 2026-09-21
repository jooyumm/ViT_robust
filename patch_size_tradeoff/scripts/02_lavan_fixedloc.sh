#!/bin/bash
# 실험 7: LaVAN 위치 고정(항상 이미지 중앙) x P=8/16 — tag=fixedloc
# baseline(01_baseline.sh)의 LaVAN 결과(위치 랜덤)와 비교해서, "P=16이 P=8보다
# 강건하다"는 결과가 공격 위치의 변동성 때문에 생긴 착시는 아닌지 확인.
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/02_lavan_fixedloc.sh
#SBATCH --job-name=vit_02_lavan_fixedloc
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=results/job_logs/nohup_02_lavan_fixedloc_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

for P in 8 16; do
  BS=16; if [ "$P" != "8" ]; then BS=64; fi
  python experiments/main.py --attacks lavan --patch_sizes "$P" \
    --num_samples 1000 --batch_size "$BS" \
    --lavan_loc center --tag fixedloc --seed "$SEED"
done