#!/bin/bash
# 실험 5: PatchFool, P=8, num_patch=4, patch_select=Contiguous
# 03번(scattered)과 달리 4개 패치가 P=16과 정확히 같은 2x2 인접 블록(같은 16x16 영역)을 이룸
# — tag=areamatch16contig
# "같은 면적이라도 공격이 뭉쳐있는지 흩어져있는지가 중요한가?"를 03번과 비교해서 확인.
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/04_patchfool_areamatch_contiguous.sh
#SBATCH --job-name=vit_04_pf_am_contig
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/job_logs/nohup_04_pf_am_contig_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

python experiments/main.py \
  --attacks patch_fool \
  --patch_sizes 8 \
  --num_samples 1000 \
  --batch_size 8 \
  --pf_attack_mode CE_loss \
  --pf_iters 250 \
  --pf_num_patch 4 \
  --pf_patch_select Contiguous \
  --tag areamatch16contig \
  --seed "$SEED"