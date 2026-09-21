#!/bin/bash
# [탐색적 검증, 롤백 가능] 완전판 adaptive attack(joint + 탐지 회피 제약) — 방어의 진짜 worst-case
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_evasive_joint
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/all_switch/adaptive_evasion_full/14_evasivejoint_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/all_switch/adaptive_evasion_full/vitguard_adaptive_evasion_full_test.py --seed 123 --num_samples 50 --chunk 20 --iters 250
