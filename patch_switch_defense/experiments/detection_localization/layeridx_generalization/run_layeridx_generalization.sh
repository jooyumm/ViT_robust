#!/bin/bash
# [탐색적, 롤백 가능] attn_layer_idx를 바꿔가며 L=12 raw attention 탐지기의 일반화 여부 확인
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
#SBATCH --job-name=vitguard_layeridx
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=results/detection_localization/layeridx_generalization/nohup_layeridx_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/detection_localization/layeridx_generalization/vitguard_layeridx_generalization.py --patch_size 16 --num_samples 30 --seed 42
