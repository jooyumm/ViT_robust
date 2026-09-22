# scripts/*/*.sh 가 공통으로 source하는 설정
# SBATCH 지시어는 파일마다 달라서 공유할 수 없으니, 본문(cd/환경변수/SEED 기본값)만 여기 모아둔다.
cd /home/jooyumm/ViT_robust/ViT_tradeoff
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=${SEED:-42}