#!/bin/bash
# [신규] all_switch vs local_switch 실제 파이프라인 비용 비교 (§10 + expected_cost_analysis.py 대체)
# ViT_patchSwitch/는 ViT_tradeoff/의 src/를 복사해온 완전히 독립된 프로젝트 -- 여기서 뭘 지우거나 바꿔도 ViT_tradeoff/에는 영향 없음
# system_comparison을 먼저 돌려서 results/system_comparison/에 npz가 있어야 pi-sweep까지 계산됨
#SBATCH --job-name=vitguard_cost_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclude=cs-gpu-01
#SBATCH --output=results/cost_comparison/cost_comparison_run_%j.txt

cd /home/jooyumm/ViT_robust/ViT_patchSwitch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python experiments/cost_comparison/cost_comparison_test.py \
  --n_warmup 20 --n_iters 100 --calib_seed 42 --num_calib 100
