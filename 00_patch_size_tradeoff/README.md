# 패치 크기별 강건성 비교

ViT-B/8·ViT-B/16에서 PGD·LaVAN·PatchFool을 비교한다. `experiments/main.py` 한 파일에서 실험 5개를 번호로 선택한다. 모델·전처리·공격 구현은 루트 `src/`를 공유한다.

## 실험 5개

| 선택 옵션 | 실험 조건 | 비교 목적 |
|---|---|---|
| `--experiment 1` | PGD·LaVAN·PatchFool × P8/P16 | 원본 모델의 공격별 강건성 |
| `--experiment 2` | LaVAN 중앙 고정 × P8/P16 | 랜덤 위치 baseline과 비교 |
| `--experiment 3` | P8 attention 상위 4토큰 독립 선택 | P16 1토큰과 총 면적 맞춤 |
| `--experiment 4` | P8 인접 2×2 토큰 | 같은 면적에서 분산/인접 배치 비교 |
| `--experiment 5` | PatchFool 랜덤 1토큰 × P8/P16 | attention 기반 위치 선택의 효과 |

baseline에 공격 3개를 묶어 실행 단위는 5개다. `experiments/main.py`의 `EXPERIMENTS`를 01~05 주석으로 나눠 조건을 정의한다. 공통 처리는 00 루트의 `runner.py`에 둔다. 과거 결과의 시각화 번호는 유지했다.

## 실행

저장소 루트에서 실행한다. 기본은 seed=42, 1000장이다.

```bash
python 00_patch_size_tradeoff/experiments/main.py --experiment 1 --seed 42
python 00_patch_size_tradeoff/experiments/main.py --experiment 2 --seed 42
python 00_patch_size_tradeoff/experiments/main.py --experiment 3 --seed 42
python 00_patch_size_tradeoff/experiments/main.py --experiment 4 --seed 42
python 00_patch_size_tradeoff/experiments/main.py --experiment 5 --seed 42
```

다중 seed는 같은 명령의 `--seed`를 42/123/2024로 바꾼다. batch 기본값은 P8 PatchFool=8, P8 PGD/LaVAN=16, P16=64다. `--batch_size`로 통일할 수 있고 마지막 작은 batch도 평가한다. 공격은 별도 seed·실행 조건에 따라 달라질 수 있다.

```bash
# 작은 표본으로 P8 PatchFool만 확인
python 00_patch_size_tradeoff/experiments/main.py --experiment 1 --attacks patch_fool --patch_sizes 8 --num_samples 20 --batch_size 1

# 옵션 목록 및 반복 수 변경
python 00_patch_size_tradeoff/experiments/main.py --experiment 4 --help
python 00_patch_size_tradeoff/experiments/main.py --experiment 4 --pf_iters 500 --tag areamatch16contig_500
```

CLI 값은 선택한 실험의 기본 조건을 덮어쓴다. 정의를 바꾸는 실험에는 별도 `--tag`를 지정해 기존 결과와 구분한다. PatchFool 기본값은 실제 공격 함수에서 가져오며 최종 JSON에 펼쳐서 기록한다. 기존 조건은 `Attention`, 유효 픽셀 범위, incoming 위치 점수, replacement 초기화다. 더 오래된 로그는 당시 설정을 따로 확인해야 한다.

## SSH와 독립적으로 실행

Slurm 환경에서는 별도 `.sh` 파일 없이 제출할 수 있다. 아래 자원 설정은 현재 클러스터의 예시다.

```bash
sbatch --job-name=vit-baseline --partition=base_suma_rtx3090 --qos=base_qos --gres=gpu:1 --cpus-per-task=16 --mem=32G --time=24:00:00 --output=/dev/null --wrap="python -u $PWD/00_patch_size_tradeoff/experiments/main.py --experiment 1 --seed 42"
```

Slurm 없이 GPU 머신에서 실행하려면:

```bash
nohup python -u 00_patch_size_tradeoff/experiments/main.py --experiment 1 --seed 42 > /dev/null 2>&1 < /dev/null &
```

필요할 때 명령 앞에 `HF_HUB_OFFLINE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 지정한다. 이미지 경로·전처리는 `src/dataset.py`를 따른다. 데모 그림은 `python 00_patch_size_tradeoff/visualize/attack_demo.py --seed 42`로 따로 실행한다.

## 코드와 결과

- `experiments/main.py`: 01~05 조건, 공통 CLI, 모델·공격 조합 선택.
- `runner.py`: 단일 조합 실행, 공격별 옵션 전달, 배치 평가·최종 수치 저장. 00에서만 쓰므로 이 프로젝트 안에 둔다.
- `metrics.py`: 00 전용 CA/RA/ASR. 공격은 PGD까지 모두 루트 `src/attacks/`에서 가져온다.
- `results/metrics/`: 조건·모델·seed별 최종 JSON. `visualize/`: 기존 결과를 읽는 그림 코드.

CA는 원본 clean 정확도, RA는 전체 공격본 정확도, ASR은 원래 정답인 표본 중 공격 후 오답 비율이다. 서로 다른 checkpoint·면적·토큰 수·seed·표본 수의 결과를 같은 조건으로 간주하지 않는다. 국소 P8 방어는 `02_detection_defense/region_switch/`에서 별도로 평가한다.

## Fig 생성

표(tbl) 코드와 `results/tables/`는 제거했다. 저장된 최종 수치의 그래프는 단일 진입점으로 만든다.

```bash
python 00_patch_size_tradeoff/visualize/main.py
python 00_patch_size_tradeoff/visualize/main.py --fig baseline
python 00_patch_size_tradeoff/visualize/main.py --fig baseline --attacks pgd lavan patch_fool
python 00_patch_size_tradeoff/visualize/main.py --fig areamatch --output-dir /tmp/vit-figs
```

`--fig`는 all/ca/baseline/fixedloc/areamatch/randsel 중 선택한다. baseline 기본값은 LaVAN/PatchFool이다. fig 조건은 `visualize/main.py`에, 공통 결과 읽기와 그래프 스타일은 `fig_common/{results_io,plotting}.py`에 둔다. 개별 이미지 공격 예시는 `visualize/attack_demo.py`에 별도로 둔다. 결과는 `results/figs/`에 저장한다.

## 결과 보관

진행·배치 로그와 재실행 이력은 저장하지 않는다. `results/metrics/`에 **공격·모델·tag·seed별 최종 JSON 하나**만 원자적으로 교체한다. 최종 지표·A/B/C/D 집계·표본 수·배치 크기·공격 옵션을 저장하고, seed별 결과로 평균·표준편차를 다시 그린다. 실행이 실패하면 이전 정상 JSON은 유지한다.

진행 상황은 터미널에서만 출력한다. 결과 저장 위치는 `--result-dir`로 지정할 수 있다. Slurm 실행 시 별도 기록이 필요 없으면 `--output=/dev/null`을 사용한다.
