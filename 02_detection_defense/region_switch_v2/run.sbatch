#!/bin/bash
# 보조 무작위 응답 실험: 병합도 포함한다. P16 단일 공격→고정 P8 분할 기본 실험은 상위 README 참고.
# region_switch_v2 500장 전체 평가. 지금 떠 있는 vit-direct-p8-p16(2629825),
# vit-attention-01-02(2629845) job이 끝난 뒤 자동 시작하도록 --dependency=afterany로 제출한다.
#SBATCH --job-name=vit-region-switch-v2
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
set -Eeuo pipefail
cd /home/jooyumm/ViT_robust
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/vit-region-switch-v2-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUT=/home/jooyumm/ViT_robust/02_detection_defense/region_switch_v2
trap 'rc=$?; echo "FAILED rc=$rc line=$LINENO $(date -Is)" > "$OUT/status.txt"; exit "$rc"' ERR
echo "RUNNING job=$SLURM_JOB_ID host=$(hostname) started=$(date -Is)" > "$OUT/status.txt"
python 02_detection_defense/region_switch_v2/run.py --output "$OUT/results/comparison.png"
echo "COMPLETED job=$SLURM_JOB_ID finished=$(date -Is)" > "$OUT/status.txt"
