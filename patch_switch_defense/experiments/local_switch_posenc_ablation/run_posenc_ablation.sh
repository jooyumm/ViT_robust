#!/bin/bash
# [탐색적, 옵션 A] local_switch에 고정 quadrant posenc를 더했을 때 복원율이 89.4%에서 개선되는지 확인
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_posenc_ablation
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/local_switch_posenc_ablation/posenc_ablation_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python experiments/local_switch_posenc_ablation/posenc_ablation_test.py \
  --num_samples 250 --cal_frac 0.4 --seed 42 --attn_layer_idx 4 --chunk 20
