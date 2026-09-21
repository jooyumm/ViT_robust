#!/bin/bash
# [신규] all_switch vs local_switch를 같은 이미지·같은 탐지 결과로 직접 비교 (§6+§16 대체)
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_system_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/system_comparison/system_comparison_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python experiments/system_comparison/system_comparison_test.py \
  --num_samples 250 --cal_frac 0.4 --seed 42 --attn_layer_idx 4 --chunk 20

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[경고] 이 노드에서도 CUDA를 못 씀 — cs-gpu-01 말고 다른 노드도 문제 있을 수 있음"
fi
