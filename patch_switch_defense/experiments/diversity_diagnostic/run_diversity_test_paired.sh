#!/bin/bash
# [샘플링 감사 후속] diversity diagnostic을 joint attack test(seed=123)와 같은 이미지 집합으로 재실행
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_divpaired
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/diversity_diagnostic/08_diversity_paired_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# seed=123: vitguard_joint_attack_test.py와 정확히 같은 50장 (원래 이 스크립트의 기본값은 456)
python experiments/diversity_diagnostic/vitguard_diversity_test.py --seed 123 --num_samples 50 --chunk 20
