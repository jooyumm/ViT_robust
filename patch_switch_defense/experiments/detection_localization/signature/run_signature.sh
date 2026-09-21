#!/bin/bash
# [탐색적, 롤백 가능] LaVAN vs PatchFool attention-rollout 시그니처 구분 여부 검증
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_probe
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=results/detection_localization/signature/01_signature_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/detection_localization/signature/vitguard_signature.py --patch_size 16 --num_samples 30 --seed 42
