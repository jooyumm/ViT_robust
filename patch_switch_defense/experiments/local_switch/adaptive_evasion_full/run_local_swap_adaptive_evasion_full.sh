#!/bin/bash
# [탐색적, 롤백 가능] §18. §16(국소 토큰 세분화)에 §14급 완전판 adaptive evasion 테스트
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_local_swap_evasive
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=results/local_switch/adaptive_evasion_full/18_local_swap_adaptive_evasion_full_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/local_switch/adaptive_evasion_full/local_swap_adaptive_evasion_full_test.py \
  --num_samples 50 --seed 123 --num_calib 100 --calib_seed 42 --attn_layer_idx 4 --chunk 20
