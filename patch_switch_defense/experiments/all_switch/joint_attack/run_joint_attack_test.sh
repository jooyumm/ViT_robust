#!/bin/bash
# [탐색적, 롤백 가능] joint attack(P16+P8 동시 공격) stress test — 방어를 아는 adaptive attacker 흉내
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_jointattack
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/all_switch/joint_attack/07_jointattack_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/all_switch/joint_attack/vitguard_joint_attack_test.py --seed 123 --num_samples 50 --chunk 20
