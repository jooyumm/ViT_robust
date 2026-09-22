# 00_patch_size_tradeoff

ViT-Base(timm)의 patch size(P=8/16/32)에 따른 적대적 공격(PGD / LaVAN / PatchFool) 강건성 실험.
(2026-09-15: 이 프로젝트는 원래 `ViT_robust`였다가, 방어 기법 개발이 별도 프로젝트
[`02_patch_switch_defense`](../02_patch_switch_defense/)로 분리되면서 이름을 바꿨다 — 여기는 "patch size 특성 분석"에
집중하는 원래 8개 실험 중 7개를 남긴다 — 실험 8(LaVAN 공격 면적 스윕)은 2026-09-17에
분석 가치가 낮다고 판단해 제거했다.)

## 실험 목록

번호는 `scripts/`, `visualize/`, `results/figures,tables/` 전체에서 공통으로 쓰는 **실행 순서**다
(실험 4~6보다 7을 먼저 도는 게 계획이라 번호가 토픽 순서와 다르게 매겨져 있음에 주의).

| # | 실험 | 비교 대상 | 목적 (RQ) |
|---|------|-----------|-----------|
| 1 | PGD baseline | P=8 vs 16 vs 32 | 전역 L∞ 공격에서 patch size가 강건성에 영향을 주는가 (RA=0% 전부 포화 → 분석 무의미) |
| 2 | LaVAN baseline | P=8 vs 16 vs 32 | 고정 면적 랜덤 위치 패치 공격에서 patch size 영향은? |
| 3 | PatchFool baseline | P=8 vs 16 vs 32 | 토큰 단위 attention 기반 공격에서 patch size 영향은? |
| 4 | PatchFool area-matched (scattered) | P8(4토큰, attention 상위 4개 독립 선택) vs P8/P16 baseline | P8이 강건한 게 순수 토큰화 효과인가, 공격 면적이 작아서인가? |
| 5 | PatchFool area-matched (contiguous) | P8(4토큰, P16과 같은 2×2 블록) vs #4(scattered) | 같은 면적이라도 공격이 뭉쳐있는지 흩어져있는지가 중요한가? |
| 6 | PatchFool 토큰 선택 (Attn vs Rand) | P8/16 attention 선택(baseline) vs 랜덤 선택 | 똑똑하게 토큰을 고르는 게 중요한가, 토큰 단위 공격 자체가 중요한가? |
| 7 | LaVAN 위치 고정 | P8/16 랜덤 위치(baseline) vs 중앙 고정 | #2의 결과가 위치 변동성 때문에 생긴 착시는 아닌가? |

(전부 ViT-Base, ImageNet-1k val, 3 seed(42/123/2024) × 1000 샘플)

`LaVAN 토큰 개수 고정`(오염 토큰을 P별로 동일하게 맞추는 실험)은 한 번 돌려봤지만 제외했다 —
"오염 토큰 수 = 면적 / P²"라 토큰 수를 고정하면 면적이 P²배씩 벌어지는데, 이건 baseline(#2)이
이미 다루는 (면적, P, RA) 관계를 다른 각도로 한 번 더 자른 것일 뿐 독립적인 정보를 안 줘서 뺐다.

**2026-09-17 제거**: 실험 8(LaVAN 공격 면적 스윕, `06_lavan_areasweep.*`)을 분석 가치가 낮다고
판단해 코드·시각화·결과 로그를 전부 지웠다. 필요하면 git 히스토리(`git log -- scripts/06_lavan_areasweep.sh` 등)에서 복구 가능.

## 관련 실험: Protocol C (`09_protocol_c/`)

공격 토큰 개수를 P8/P16/P32에서 동일하게 고정하는 실험 — 위 7개 실험이 면적을 통제 변수로
쓴 것과 대조되는 각도. 자세한 내용은 [`09_protocol_c/`](09_protocol_c/) 참고.

## 적응형 방어 프로젝트 (`02_patch_switch_defense/`)

위 실험들(패치 크기 vs 강건성 특성 분석)과는 별도로, 이 발견("PatchFool에 대해 P8이 P16보다
압도적으로 강건함")을 실제 방어로 발전시키는 후속 연구는 **독립된 프로젝트**
[`../02_patch_switch_defense/`](../02_patch_switch_defense/)로 분리했다(2026-09-15) — "P16으로 추론하다가 공격이
의심되면 P8로 통째 재분류하는 적응형 방어"의 탐지·위치특정·복원율·adaptive attacker 대응까지
전부 그쪽에 있다. 모델/데이터/공격 코드(`src/`)는 2026-09-21부터
[`../src/`](../src/)(`ViT_robust` 최상위)로 세 프로젝트가 공유한다 — 아래 "디렉토리 구조"와
[`../README.md`](../README.md) 참고. 자세한 내용은
[`02_patch_switch_defense/README.md`](../02_patch_switch_defense/README.md) 참고.

## 디렉토리 구조

**2026-09-21**: `src/`(models.py/dataset.py/attacks/{lavan,patch_fool}.py)는
`02_patch_switch_defense`/`01_patch_attack_detector`와 완전히 같은 내용이라 [`../src/`](../src/)(`ViT_robust`
최상위)로 세 프로젝트가 공유하도록 합쳤다(`ViT_robust/README.md` 참고). `metrics.py`와
`attacks/pgd.py`는 이 프로젝트에만 있는 고유 코드라 공유 위치로 옮기지 않고 프로젝트
루트에 바로 남겼다(`pgd.py`는 더 이상 `attacks/` 하위가 아님 — 공유 `attacks/`
패키지 안에는 이 프로젝트만의 파일을 둘 수 없기 때문).

```
metrics.py                 CA/RA/ASR 계산 (A/B/C/D 케이스 정의는 파일 상단 주석 참고) — 이
                            프로젝트 전용, 공유 안 됨
pgd.py                      PGD (전역 L∞) — epsilon/alpha 계산에 쓰는 정규화 상수는
                            ../src/dataset.py와 반드시 동일해야 함(현재 mean=std=0.5).
                            이 프로젝트 전용, 공유 안 됨(PGD는 02_patch_switch_defense/01_patch_attack_detector
                            범위 밖)

../src/                    공유 라이브러리 코드 (02_patch_switch_defense/01_patch_attack_detector와 공유,
                            ViT_robust/README.md 참고)
  models.py                  timm ViT 모델 로드 (patch size별 체크포인트)
  dataset.py                  ImageNet-1k val 데이터로더 (seed로 재현 가능한 샘플링)
                               전처리(mean/std/crop_pct/interpolation)는 timm의
                               pretrained_cfg에서 직접 읽어옴 — 우리 체크포인트는 표준
                               ImageNet 정규화가 아니라 mean=std=0.5(inception 스타일)를 씀에 주의
  attacks/
    lavan.py                    LaVAN (고정 픽셀 면적의 패치, 랜덤 위치 또는 fixed_loc='center' 고정)
    patch_fool.py                PatchFool (토큰 단위, patch_select: Attn/Rand/Contiguous)

experiments/               실험 실행기 (main.py는 얇은 진입점, 나머지가 각자 역할 담당)
  main.py                     진입점 — args/logfile/attack_experiment_runner를 순서대로 호출만 함
  args.py                      CLI 인자 정의(get_args) + 공격별 kwargs 매핑(build_attack_kwargs)
  logfile.py                    Tee(stdout을 파일에도 동시 기록) + 로그 파일명 생성(make_log_path)
  attack_experiment_runner.py                 배치 단위 공격 실행(run_attack_on_batch) + 여러 배치 누적 실행(run_experiment)

scripts/                   SLURM 배치 제출 스크립트. 파일명 번호 = 실행 순서(위 실험 목록과 동일)
  common.sh                   모든 스크립트가 source하는 공통 설정 (cd, CUDA 환경변수, SEED 기본값)
  00_demo.sh                    patch size 무관 — visualize/00_attack_demo.py 실행
  01_baseline.sh                 실험 1+2+3 통합: PGD/LaVAN/PatchFool x P=8/16/32 (한 job에서 순차 실행)
  02_lavan_fixedloc.sh            실험 7: LaVAN 위치 고정 x P=8/16 — tag=fixedloc
  03_patchfool_areamatch_scattered.sh   실험 4: PatchFool area-matched(scattered) x P=8 — tag=areamatch16
  04_patchfool_areamatch_contiguous.sh  실험 5: PatchFool area-matched(contiguous) x P=8 — tag=areamatch16contig
  05_patchfool_randsel.sh          실험 6: PatchFool Rand 토큰 선택 x P=8/16 — tag=randsel
                             예: sbatch --export=SEED=42 scripts/01_baseline.sh
                             (반드시 저장소 루트에서 제출할 것 — 로그 경로가 제출 위치 기준 상대경로임)
                             01_baseline.sh는 9개 조합을 한 GPU에서 순차 실행하므로 --time=15:00:00로 넉넉히 잡음

visualize/                 결과 시각화. 파일명 번호는 어느 scripts/*.sh 결과를 시각화하는지를 가리킴.
                           공용 모듈(results_io/plotting/tables/conditions)은 번호 없음
                           — 파이썬 모듈명이 숫자로 시작하면 다른 파일이 import를 못 해서 분리해둠
  results_io.py                results/logs/*.txt 파싱 + 다중 시드 집계
  plotting.py                   그래프 그리기 (plot_results, plot_single, plot_condition_comparison,
                                 plot_sweep) — P별 색상은 p_color_shades()로 공격 기본색(ATK_COLOR)의
                                 밝기만 다르게 자동 생성 (같은 공격이면 항상 같은 색 계열).
                                 plot_sweep/gather_sweep_data/make_sweep_table은 연속값 스윕 실험용
                                 공용 유틸(현재 이걸 쓰는 실험은 없지만 다음에 스윕형 ablation
                                 추가할 때 재사용하도록 남겨둠)
  tables.py                     표 그리기 (make_baseline_table, make_condition_table, make_sweep_table)
                                 — PNG만, LaTeX는 안 뽑음
  conditions.py                 ablation 비교 조건(CONDITIONS, 스윕 스펙) 정의 — 실험별 fig/tbl 스크립트가 공유
  00_attack_demo.py             실험 무관 — 공격 1건 시각화 (발표용, 이미지 1장 기준 결과표)
  00_fig_ca.py                   실험 무관 — Clean Accuracy만 단독 그래프 (공격과 무관, P에만 의존)
  01_fig_baseline.py             실험 1-3 그래프 — RA/ASR 메인 비교 (PGD/LaVAN/PatchFool x P=8/16/32)
  01_fig_baseline_p8p16.py        위와 같은 로그, PGD·P=32 제외한 축약판 (LaVAN/PatchFool x P=8/16,
                                 막대마다 P=8/P=16 라벨도 표시 — P값이 2개뿐일 때만 자동으로 켜짐)
  01_tbl_baseline.py             실험 1-3 표
  02_fig_lavan_fixedloc.py        실험 7 그래프
  02_tbl_lavan_fixedloc.py        실험 7 표
  03_fig_patchfool_areamatch.py   실험 4+5 그래프 (scattered/contiguous를 한 그래프에서 비교)
  03_tbl_patchfool_areamatch.py   실험 4+5 표
  05_fig_patchfool_randsel.py     실험 6 그래프
  05_tbl_patchfool_randsel.py     실험 6 표

results/                  전부 .gitignore 대상 (재현 가능한 산출물이라 버전관리 안 함)
  logs/                       실험 로그 원본 텍스트 (파일명에 attack/patch_size/seed/[tag]/timestamp 인코딩)
  job_logs/                    SLURM stdout 캡처 — 정상 종료된 job은 results/logs/와 내용이 겹쳐서
                                주기적으로 비워도 무방하지만, job이 결과 파일도 못 남기고 죽는 경우
                                (예: 노드 CUDA 오류로 조용히 CPU 폴백 후 타임아웃) 유일한 진단 단서이므로
                                #SBATCH --output 자체는 계속 유지할 것
  figures/                     그래프 전부 (하위 폴더 없이 평평하게) — 파일명 번호로 어느 실험인지 구분
  tables/                      표 전부 (PNG만, .tex 없음) — 파일명 번호로 어느 실험인지 구분
```

## 실험 로그 파일명 규칙

`results/logs/result_{attack}_P{patch_sizes}[_tag-{tag}]_seed{seed}_{timestamp}.txt`

- `tag`가 없으면 baseline. ablation(면적 통제, 랜덤 토큰 선택 등)은 반드시 `--tag`를 지정해서
  baseline과 섞이지 않게 한다 (`visualize/results_io.py`가 tag별로 따로 집계함).
- 같은 (attack, 개별 patch size, tag, seed) 조합의 로그가 여러 개면(재시도 등) **가장 최신 것만** 집계에 쓰인다
  — 한 로그 파일이 여러 patch size를 담고 있어도 patch size 단위로 개별 비교한다.

## 실험 돌리기

```bash
# 단발 실행
python experiments/main.py --attacks pgd --patch_sizes 8 --num_samples 1000 --batch_size 16 --seed 42

# 클러스터 제출 (반드시 저장소 루트에서)
sbatch --export=SEED=42 scripts/01_baseline.sh

# 다중 시드
for seed in 42 123 2024; do sbatch --export=SEED=$seed scripts/01_baseline.sh; done
```

새 ablation을 추가하려면: `scripts/05_patchfool_randsel.sh`를 복사해서 `main.py` 옵션(`--pf_num_patch`,
`--pf_patch_select` 등)과 `--tag`만 바꾸고, `visualize/conditions.py`에 조건 리스트를 추가한 뒤
`02_fig_lavan_fixedloc.py`/`02_tbl_lavan_fixedloc.py`처럼 `plot_condition_comparison()` /
`make_condition_table()`을 호출하는 그래프·표 스크립트를 몇 줄로 추가하면 된다
(결과 파일은 `results_io.FIG_DIR`/`TABLE_DIR`에 저장하면 다른 그래프/표들과 같은 자리에 모인다).
새 스크립트/시각화 파일 번호는 다음으로 이어서 붙이면 된다 (예: 실험 9 추가 시 `06_...`).
연속값을 스윕하는 실험이면 `plot_condition_comparison`/`make_condition_table` 대신
`plot_sweep`/`make_sweep_table` + `results_io.gather_sweep_data()`를 쓰면 된다.

## 시각화 뽑기

```bash
python visualize/00_fig_ca.py                   # results/figures/00_ca.png
python visualize/01_fig_baseline.py             # results/figures/01_baseline_all.png
python visualize/01_fig_baseline_p8p16.py       # results/figures/01_baseline.png
python visualize/01_tbl_baseline.py             # results/tables/01_baseline_table.png
python visualize/02_fig_lavan_fixedloc.py       # results/figures/02_lavan_fixedloc.png
python visualize/02_tbl_lavan_fixedloc.py       # results/tables/02_lavan_fixedloc_table.png
python visualize/03_fig_patchfool_areamatch.py  # results/figures/03_patchfool_areamatch.png
python visualize/03_tbl_patchfool_areamatch.py  # results/tables/03_patchfool_areamatch_table.png
python visualize/05_fig_patchfool_randsel.py    # results/figures/05_patchfool_randsel.png
python visualize/05_tbl_patchfool_randsel.py    # results/tables/05_patchfool_randsel_table.png
```