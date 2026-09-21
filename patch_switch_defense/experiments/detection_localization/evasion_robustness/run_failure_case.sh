#!/bin/bash
# [탐색적, 롤백 가능] localization 실패 샘플 진단 (공격 성공 여부 + 이미지 저장)
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_failcase
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=results/detection_localization/evasion_robustness/03_failcase_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/detection_localization/evasion_robustness/vitguard_failure_case.py --patch_size 16 --num_samples 30 --seed 42
