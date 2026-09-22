# 01_patch_attack_detector — ViT 패치 공격 탐지기 (공유 라이브러리)

ViT 방어 프로젝트들(`../02_patch_switch_defense/`의 all_switch/local_switch 등)이 공통으로 쓰는
**탐지기 자체**를 방어 로직(재분류·affine bridge)과 분리해서 독립적으로 발전시키기 위한
프로젝트다. 2026-09-21에 `02_patch_switch_defense/defense/detector.py`를 코드 변경 없이 그대로
옮겨와 만들었다(아래 "이력" 참고).

## 메커니즘 (Top-K Mass, `detector/topk_mass_v1.py`)

12번째 transformer block의 attention을 head 평균 낸 뒤, CLS→patch 행을 정규화한 벡터에서
top-4 원소의 합("top4_mass")을 이미지 단위 이상치 점수로 쓰고, 그 argmax 1개를 "의심 패치"
위치로 잡는다(recall@1 96.7%, `02_patch_switch_defense`의 §2). 점수를 calibration에서 정한 임계값과
비교해 escalate 여부를 결정한다 — 임계값 자체는 이 모듈이 만들지 않고, 매 실험이 자기
calibration 표본으로 정한다. 자세한 배경·한계(예: LaVAN 비탐지)는
[`../02_patch_switch_defense/README.md`](../02_patch_switch_defense/README.md)의 "공유 인프라: 탐지 + 위치특정"
절 참고.

**파일명이 `detector.py`가 아니라 `detector/topk_mass_v1.py`인 이유**: 이 프로젝트의 존재
이유 자체가 탐지기를 여러 버전으로 독립적으로 발전시키는 것이라(레이어별 sweep, LaVAN 보완용
Dual-Gate 등), 두 번째 버전이 생기기 전에 미리 버전이 드러나는 이름으로 옮겨뒀다. 앞으로
나올 개선된 탐지기는 `detector/dual_gate_v2.py`처럼 이 파일과 나란히 추가되고, 이 파일
자체는 바뀌지 않는다(`02_patch_switch_defense`가 실제 배포에 쓰는 버전이 바로 이것이므로).

## 핵심 결과 요약

**레이어별 AUROC/FPR/recall sweep** (`experiments/layer_sweep/`, PatchFool, n=250/seed=42,
`02_patch_switch_defense`의 system_comparison과 동일 표본·공격): AUROC-vs-layer 곡선은 **단조증가가
아니다** — L=1(0.530)에서 L=5(0.734)까지 오르다 L=7~9(0.522~0.554, chance 근처)로 다시
가라앉은 뒤 L=11(0.820)/L=12(0.846)에서 급등한다. "L=12까지 봐야 신호가 분리된다"는 주장은
**부분적으로만 맞다** — 정확히는 "L=11~12 두 레이어에서만 신호가 뚜렷하고, 중간 레이어(7~10)는
오히려 chance 수준으로 꺼진다"가 더 정확한 설명이다. 자세한 내용은 아래 "레이어별 Top-K Mass
sweep" 절 참고.

**Raw attention 특성 관찰** (`experiments/characterization/`, 탐지기 설계 아님 — 순수 관찰):
PatchFool vs LaVAN vs Clean의 raw attention을 레이어x헤드x전체 토큰 단위로 직접 비교한 결과,
**LaVAN은 모든 레이어에서 attention 분포가 clean과 거의 완전히 겹친다** — "LaVAN은 attention
구조를 안 건드릴 가능성"이 데이터로 확인됨. PatchFool은 L=10까지 clean/LaVAN과 거의 같은
궤적을 따르다가 L=11~12에서만 뚜렷이, 그리고 **거의 모든 헤드에서 동시에**(특정 헤드 하나가
아님) 급등한다. n=100 전체로 sink 위치까지 추적한 결과 "기존 sink를 증폭한다"는 가설은
L=11~12에서만 다수이고, 중간 레이어(L=6~9)에서는 오히려 "새 위치로 옮긴다"가 맞는
설명이었다. 자세한 내용은 아래 "Raw attention 특성 관찰"·"Sink 위치 일관성" 절 참고.

## 디렉토리 구조

```
ViT_robust/
  src/                         모델/데이터셋/공격 코드 — 00_patch_size_tradeoff/02_patch_switch_defense/
                                01_patch_attack_detector 세 프로젝트가 공유(아래 "공유 src/" 참고)

01_patch_attack_detector/
  detector/                    탐지기 구현들 — 방어 로직(02_patch_switch_defense)과 분리해서 이
                                프로젝트에서만 독립적으로 발전시킨다
    topk_mass_v1.py              현재 02_patch_switch_defense가 실배포에 쓰는 v1(raw attention
                                  L=12, top4_mass, top-1 argmax) — 위 "메커니즘" 참고

  experiments/
    layer_sweep/
      layer_sweep_test.py         L=1~12 전체의 AUROC/FPR/recall sweep + 얕은 레이어 히트맵
    characterization/            레이어x헤드x전체토큰 raw attention 특성 관찰(탐지기 설계
                                  아님) — "테스트하고 싶은 것"별로 파일을 나눴다:
      attention_lib.py             저수준: 헤드별 attention hook + 분포/집중도 계산 함수
      gt_lib.py                    저수준: 공격이 실제로 건드린 patch를 clean/adv 픽셀
                                    diff로 계산하는 ground truth 함수(attention 추정 아님)
      00_extract_attention.py         GPU 필요한 유일한 단계. clean/PatchFool/LaVAN 생성 +
                                    attention 추출 + 지표 계산 + GT(대표 이미지 원본/공격
                                    이미지 픽셀, gt_coverage) -> npz 하나로 저장
      data_io.py                   그 npz를 다시 불러오는 로더 (아래 두 스크립트가 공유)
      plotting.py                  두 스크립트가 공유하는 그림 함수
      01_concentration.py        "얼마나 몰리는가" — 히스토그램/헤드별 편차/레이어별 비교
                                    (npz만 읽음, GPU 불필요)
      02_sink_position.py        "어디로 몰리는가, 고정 위치인가, 공격이 위치를 바꾸는가"
                                    (npz만 읽음, GPU 불필요)
      03_token_bars.py            "정말 튀는 토큰이 있는가" — 레이어 1개 x 이미지 1개
                                    예시로, 197개 토큰(0=CLS, 1~196=patch) 순서 그대로
                                    선 그래프(정렬 안 함), 튀는 지점엔 몇 번 patch인지
                                    자동 라벨. 02의 정렬된 그래프와 상호보완적
                                    (npz만 읽음, GPU 불필요)
      04_column_check.py          "CLS 말고 다른 토큰들도 그 지점을 보는가" — 03에서 찾은
                                    스파이크(예: L12/img0의 patch 17)가 진짜 sink라면
                                    197개 쿼리 전부가 거길 봐야 한다는 가설을 검증. 03과
                                    반대 방향(고정 키, 가변 쿼리) (npz만 읽음, GPU 불필요)
      05_gt_check.py              "attention이 정말 공격당한 자리로 몰리는가" — 04까지는
                                    맞는지 한 장씩 손으로 diff해서 확인했는데, 이제 그걸
                                    자동화: GT(gt_lib.py, 픽셀 diff)와 관측된 attention
                                    argmax를 대표 이미지 전부(기본 5장)에 대해 자동 대조,
                                    이미지별 원본/공격/diff + GT-vs-관측 오버레이 그림도
                                    저장 (npz만 읽음, GPU 불필요)

  results/
    layer_sweep/                위 실험의 결과물 (npz/png/md)
    characterization/           위 실험들의 결과물 (npz/png/md)
```

## 이력

`detector/topk_mass_v1.py`는 `02_patch_switch_defense/defense/detector.py`에서 코드 한 글자도
바꾸지 않고 옮겨왔다 — 원본의 두 커밋(`92f4389`, `ebf8a89`) 이력을 그대로 재구성해 이
저장소의 git 로그에 남겼다(`git log -- detector/topk_mass_v1.py`로 확인 가능, 각 커밋
메시지에 원본 커밋 해시 명시). 이후 `src/detector.py` → `detector/topk_mass_v1.py`로 한 번
더 옮겼는데, 이건 같은 저장소 안의 이동이라 `git mv` 이력이 자동으로 이어진다.

## 공유 src/ (`ViT_robust/src/`)

**2026-09-21 이전**: 이 프로젝트의 `experiments/` 스크립트들은 표본/공격 생성을 위해
`02_patch_switch_defense`의 `src/`(models.py, dataset.py, attacks/)를 cross-import했었다.

**2026-09-21**: `00_patch_size_tradeoff`/`02_patch_switch_defense`/`01_patch_attack_detector` 세 프로젝트가 완전히 같은
`src/` 내용을 각자 사본으로 들고 있는 게(편집할 때 셋 다 따로 고쳐야 함) 비효율적이라
판단해, `ViT_robust/src/`(이 저장소들의 공통 상위 폴더)로 합쳤다 — 세 프로젝트 모두 이제
직접 이 위치를 import한다(자세한 배경은 `ViT_robust/README.md`). 이 프로젝트의
`experiments/` 스크립트는 `sys.path`에 `ViT_robust`를 추가해 `from src.models import ...`
식으로 쓴다.

`detector/`는 다르다 — 원래부터 여러 방어 프로젝트가 공유해야 하는 단일 소스이길 원해서
이 프로젝트에만 있는 것이 애초에 이 프로젝트가 생긴 이유다(위 "메커니즘" 참고). `src/`가
공유로 바뀐 지금도 `detector/`는 여전히 이 프로젝트 소유이고, `02_patch_switch_defense`가
cross-import하는 유일한 대상이다 — 아래 절 참고.

## 쓰는 곳 (cross-project import, `detector/`만 해당)

`02_patch_switch_defense`의 실험 스크립트는 `sys.path`에 `01_patch_attack_detector` ROOT를 추가해
`from detector.topk_mass_v1 import ...`로 직접 import한다. 현재 이 방식을 쓰는 곳은
`02_patch_switch_defense/experiments/{system_comparison,cost_comparison,
local_switch_posenc_ablation}/*_test.py` 3곳이다. `defense/all_switch.py`/
`defense/local_switch.py` 자체는 탐지기를 호출하지 않는다(이미 계산된 `flag_idx`를
파라미터로만 받음). `src/`가 이제 진짜로 공유되므로(위 절 참고), 예전에 있었던 "`src`
패키지 이름 충돌 방지를 위한 sys.path 순서" 문제는 더 이상 없다 — `02_patch_switch_defense`와
`01_patch_attack_detector` 둘 다 같은 `ViT_robust/src/`를 가리키므로 순서 무관하게 항상 같은 곳으로
resolve된다.

## 레이어별 Top-K Mass sweep (`experiments/layer_sweep/`)

지금까지 `02_patch_switch_defense`에서 확인된 지점은 L=6(AUROC 0.639)과 L=12(0.879, §1, n=30
clean/LaVAN/PatchFool 각 30장 기준) 두 개뿐이었다. 이 실험은 `detector/topk_mass_v1.py`의
`collect_layer_attn`/`raw_at_layer`/`top4_mass`를 레이어 인덱스만 바꿔가며 그대로 재사용해
L=1~12 전체 곡선을 채운다(새 탐지 로직 없음). `system_comparison`과 동일한 표본·공격
(n=250, seed=42, calibration 100/eval 150, PatchFool attn_layer_idx=4, chunk=20)을 써서
L=12 행이 system_comparison의 발표치와 직접 대조되는 내장 sanity check 역할을 하게 했다.

**방법**: clean/adv 이미지 각각 `collect_layer_attn`을 **한 번씩만** 호출(forward 1회로
12개 block 전부의 attention이 이미 나오므로 레이어마다 다시 forward하지 않음). 레이어별로
AUROC는 eval 150개(held-out)만으로, threshold는 calibration 100개로만 Youden's J로 정하고,
FPR/recall은 그 threshold로 eval에서 계산(system_comparison과 동일한 순환평가 방지 원칙).

**결과** (job 2302720, RTX 4090):

| L | AUROC | threshold | FPR | recall |
|---|---|---|---|---|
| 1 | 0.530 | 0.0482 | 0.600 (90/150) | 0.627 (94/150) |
| 2 | 0.637 | 0.0562 | 0.413 (62/150) | 0.580 (87/150) |
| 3 | 0.666 | 0.0492 | 0.547 (82/150) | 0.753 (113/150) |
| 4 | 0.695 | 0.0561 | 0.273 (41/150) | 0.567 (85/150) |
| 5 | 0.734 | 0.0643 | 0.360 (54/150) | 0.667 (100/150) |
| 6 | 0.646 | 0.1164 | 0.320 (48/150) | 0.560 (84/150) |
| 7 | 0.554 | 0.2587 | 0.427 (64/150) | 0.513 (77/150) |
| 8 | 0.522 | 0.3756 | 0.620 (93/150) | 0.680 (102/150) |
| 9 | 0.544 | 0.4798 | 0.480 (72/150) | 0.573 (86/150) |
| 10 | 0.647 | 0.4773 | 0.313 (47/150) | 0.553 (83/150) |
| 11 | 0.820 | 0.3874 | 0.133 (20/150) | 0.700 (105/150) |
| 12 | 0.846 | 0.5116 | **0.133 (20/150)** | **0.727 (109/150)** |

95% Wilson CI를 포함한 전체 표는
[`results/layer_sweep/auroc_fpr_recall_table_n250.md`](results/layer_sweep/auroc_fpr_recall_table_n250.md),
그래프는 [`results/layer_sweep/auroc_vs_layer_n250.png`](results/layer_sweep/auroc_vs_layer_n250.png)
참고. **L=12 sanity check**: threshold=0.5116, FPR=0.133, recall=0.727 —
`02_patch_switch_defense`의 system_comparison 발표치(threshold=0.5116, FPR=13.3%, recall=72.7%)와
완전히 일치(같은 표본·같은 공격이므로 예상된 결과, PatchFool 공격 생성이 두 스크립트에서
결정론적으로 재현됨을 추가로 확인해줌 — `src/`를 자체 사본으로, 그다음 공유
`ViT_robust/src/`로 바꾼 뒤에도 매번 다시 확인했다).

**해석**: 곡선은 단조증가가 아니라 두 구간으로 나뉜다 — L=1→5에서 완만히 오르다(0.530→0.734)
L=6→9에서 chance 근처까지 꺼지고(최저 L=8=0.522), L=10부터 다시 올라 L=11(0.820)/L=12(0.846)
에서 급등한다. PatchFool 공격이 `attn_layer_idx=4`를 직접 타겟하는 걸 감안하면 L=4~5 부근의
국소적 상승은 공격이 그 레이어의 attention을 직접 왜곡하기 때문일 가능성이 있고, L=11~12의
급등은 그와는 다른(더 늦은 레이어에 누적된) 신호일 가능성이 있다 — 두 메커니즘이 같은지는
이 sweep만으로는 판단 불가, 후속 분석 필요. **결론: "L=12까지 봐야 신호가 분리된다"는 절반만
맞다** — 정확히는 "L=11~12 두 레이어에서만 신호가 뚜렷하고 중간 레이어는 chance 수준으로
꺼진다"이다.

**얕은 레이어 정성적 히트맵** (clean 이미지 2장, L=1~3):
[`results/layer_sweep/shallow_layer_heatmaps_n250.png`](results/layer_sweep/shallow_layer_heatmaps_n250.png).
"얕은 층은 인접 토큰에 국소적으로 집중된다"는 예상과 달리, 이 2장에서는 L=1이 오히려 이미지
가장자리(모서리) 패치에 집중되는 경향을 보였고, L=2/L=3에서는 두 이미지 모두 특정 고정
위치(우측 가장자리 근처)에 유난히 밝은 단일 패치가 나타났다 — ViT의 attention sink류 현상과
일관된 패턴으로 보이나, 표본이 2장뿐이라 일반화하려면 더 봐야 한다.

**범위 제한 (다음 단계로 미룸)**: 레이어를 어떻게 결합할지(단일 채택/OR/누적합)나 Dual-Gate
설계는 이번 sweep 결과를 보고 나서 논의한다.

## Raw attention 특성 관찰 (`experiments/characterization/`)

**성격 규정**: 이 실험은 위 layer_sweep과 달리 AUROC/threshold/flag를 전혀 계산하지 않는다.
PatchFool/LaVAN이 attention 구조를 실제로 어떻게 바꾸는지 있는 그대로 관찰하는 게 목적이고,
탐지기 설계는 이 결과를 보고 다음 단계에서 논의한다. `detector/topk_mass_v1.py`의
CLS-row·헤드평균 전용 함수는 건드리지 않고, 헤드별·전체 197×197 attention을 그대로 뽑는
별도 코드(`attention_lib.py`의 `_full_attn_hook`/`collect_full_layer_attn`/
`row_distribution`/`col_distribution`)를 이 실험 전용으로 새로 작성했다.

**코드 구조**: "테스트하고 싶은 것"이 두 가지라 파일도 그렇게 나눴다 — 데이터 추출
(`00_extract_attention.py`, GPU 필요한 유일한 단계, 결과를 `00_attention_data_n{N}.npz`
하나로 저장)과 그걸 읽어서 분석하는 두 스크립트(`01_concentration.py`="얼마나 몰리는가",
`02_sink_position.py`="어디로 몰리는가") — 뒤의 두 스크립트는 GPU도, 공격 재생성도
필요 없다. 저수준 함수(`attention_lib.py`)와 npz 로더(`data_io.py`)/그림 함수
(`plotting.py`)는 셋이 공유한다. 디렉토리 구조 위 참고.

**방법**: 같은 기본 이미지 n=100장(seed=42)에서 clean/PatchFool(attn_layer_idx=4, 250
iters)/LaVAN(patch_ratio=0.02, 40 steps) 세 그룹을 만들고, 그룹마다 forward 1회로 12개
block 전부의 헤드별 attention(B,12 heads,197,197)을 수집. 레이어마다 (a) CLS→patch
행(row, detector의 대상과 동일)과 (b) 전체 쿼리가 각 patch에 주는 attention의 합
(column) 두 분포를 뽑고, 각각 헤드별로/헤드 평균 후로 top-1 mass·정규화 entropy·Gini를
계산했다(`00_extract_attention.py`, job 2302721, RTX 4090).

**결과**: 레이어별 top-1 mass(CLS row, 헤드평균) 중앙값 —

| L | clean | PatchFool | LaVAN | PF−clean | LaVAN−clean |
|---|---|---|---|---|---|
| 1 | 0.0136 | 0.0138 | 0.0136 | +0.0002 | +0.0000 |
| 2 | 0.0156 | 0.0186 | 0.0156 | +0.0030 | +0.0000 |
| 3 | 0.0151 | 0.0194 | 0.0149 | +0.0044 | −0.0001 |
| 4 | 0.0142 | 0.0203 | 0.0149 | +0.0061 | +0.0007 |
| 5 | 0.0208 | 0.0249 | 0.0210 | +0.0041 | +0.0002 |
| 6 | 0.0495 | 0.0467 | 0.0489 | −0.0028 | −0.0006 |
| 7 | 0.1154 | 0.1097 | 0.1179 | −0.0057 | +0.0025 |
| 8 | 0.1743 | 0.1630 | 0.1784 | −0.0113 | +0.0040 |
| 9 | 0.2176 | 0.1966 | 0.2105 | −0.0210 | −0.0072 |
| 10 | 0.1851 | 0.1778 | 0.1719 | −0.0073 | −0.0131 |
| 11 | 0.1202 | 0.2358 | 0.1060 | **+0.1156** | −0.0142 |
| 12 | 0.1536 | 0.4157 | 0.1273 | **+0.2621** | −0.0263 |

전체 표(5×균등분포 초과 비율 포함, 설명용 기준선이지 탐지 임계값 아님)는
[`results/characterization/01_concentration_summary_n100.md`](results/characterization/01_concentration_summary_n100.md)
참고(`01_concentration.py` 산출물).

**핵심 발견 4가지**:

1. **LaVAN은 attention 구조를 거의 안 건드린다.** 모든 레이어에서 LaVAN의 중앙값이 clean과
   거의 겹치고(최대 편차 L=12에서 −0.026), [`results/characterization/
   01_layer_comparison_top1_row.png`](results/characterization/01_layer_comparison_top1_row.png)
   에서 clean/LaVAN 선이 12개 레이어 내내 거의 포개진다. `02_patch_switch_defense` README가 이미
   지적한 "raw-attention 탐지기는 LaVAN을 원리적으로 못 잡는다"는 한계를, attention 분포
   자체가 안 변한다는 직접 증거로 뒷받침한다.
2. **PatchFool의 신호는 L=11~12에서만 뚜렷하고, 그마저도 이미지마다 다르다.** L=10까지는
   PatchFool도 clean/LaVAN과 거의 같은 궤적을 따른다(중간 레이어의 큰 상승·하강 자체는 세
   그룹 모두에게 공통된 ViT 자체의 성질이지 공격 특유의 현상이 아님). [`results/
   characterization/01_hist_top1_row_by_layer.png`](results/characterization/01_hist_top1_row_by_layer.png)
   의 L=11/L=12 패널을 보면 PatchFool 분포가 대부분 clean과 겹치되 **긴 꼬리**(최대
   0.6~0.8)를 만든다 — 이게 layer_sweep에서 recall이 72.7%에 그치고 100%가 아닌 이유와
   같은 그림이다(공격이 성공적으로 attention을 왜곡한 이미지와 아닌 이미지가 섞여 있음).
3. **L=12의 쏠림은 특정 헤드 하나의 이상치가 아니라 대부분의 헤드에서 동시에 나타난다.**
   [`results/characterization/01_head_variability_L5_L12.png`](results/characterization/01_head_variability_L5_L12.png)
   에서 L=12는 12개 헤드 중 대다수가 PatchFool에서 clean/LaVAN보다 높게 나오고(헤드 5개
   이상이 0.6 이상), L=5는 반대로 세 그룹이 헤드별로도 거의 구분되지 않는다 — L=5의 집계
   상승(layer_sweep에서 봤던 국소 peak)은 공격 신호가 아니라 자연적인 레이어 특성일
   가능성이 높다는 뜻.
4. **"5×균등분포 초과" 같은 고정 컷오프는 L=6부터 무의미해진다.** clean 이미지조차 L=6부터
   거의 100%가 그 기준을 넘는다(표 참고) — 중간~늦은 레이어에서는 절대적인 집중도 자체가
   이미 높아서, 레이어별 baseline을 고려하지 않는 고정 임계값은 못 쓴다는 뜻(다음 단계
   탐지기 설계에서 레이어별 calibration이 필요한 이유).

**정성적 히트맵**: [`results/characterization/02_attention_heatmap_grid_img0.png`](results/characterization/02_attention_heatmap_grid_img0.png),
[`img1`](results/characterization/02_attention_heatmap_grid_img1.png) — 같은 원본
사진 기준 clean/PatchFool/LaVAN 3×12(레이어) 그리드. image 1에서는 clean/LaVAN이 같은
위치(우상단)에 이미 "sink"처럼 보이는 밝은 패치를 갖고 있고, PatchFool은 그 자리를 훨씬
더 밝게 만드는 것처럼 보였다(표본 2장뿐인 관찰) — 아래 "Sink 위치 일관성" 절에서 n=100
전체로 이 관찰을 정량화한 결과, **완전히 맞지는 않았다**(레이어에 따라 다름, 자세한 내용은
해당 절 참고).
[`results/characterization/02_sorted_mass_by_layer_img0.png`](results/characterization/02_sorted_mass_by_layer_img0.png)
는 "슬라이드 막대그래프"(정상=분산, 공격=소수 토큰 집중)를 레이어별 정렬-질량 곡선으로
재현한 것 — image 0에서는 L=10~12에서만 PatchFool의 "왼쪽으로 치우친 급경사" 모양이
뚜렷해진다. (둘 다 `02_sink_position.py` 산출물.)

**"진짜 튀는 토큰이 있는가" 예시 확인**: [`results/characterization/03_token_bars_L12_img0.png`](results/characterization/03_token_bars_L12_img0.png)
(`03_token_bars.py`) — 위 그래프들은 값을 정렬하거나 히트맵 색으로 뭉뚱그려서 "특정 토큰
인덱스"가 눈에 안 들어온다. 여기서는 정렬 없이 197개 토큰(0=CLS, 1~196=patch) 순서
그대로 선 그래프를 그려서, image 0/L=12에서 PatchFool만 **patch 17**에서 0.63까지 치솟고
(같은 자리에서 clean은 0.11, LaVAN은 거의 0으로 평범한 수준) 나머지는 전부 평평함을 직접
확인했다 — top1_mass 같은 집계 지표가 아니라 "정말 그 위치 하나가 튀는 것"을 눈으로 보여주는
가장 직접적인 증거. (patch 17이 우연히 튄 게 아니라 **실제로 공격이 픽셀을 바꾼 바로 그
patch**인지도 clean/adv 이미지를 직접 diff해서 확인함 — 정확히 patch 17(0-indexed 16)에서만
픽셀이 바뀌었다. `patch_fool_attack`은 `attn_layer_idx=4`에서 CLS가 가장 많이 보는 patch
하나를 고른 뒤 그 patch 안에만 마스크를 씌워 perturbation을 넣으므로, "L=4에서 고른 그
patch가 L=12에서도 계속, 그리고 이제는 거의 모든 토큰에게 sink로 남는다"는 뜻.)

**CLS 말고 다른 토큰들도 patch 17을 보는가**: [`results/characterization/04_column_check_L12_img0_tok17.png`](results/characterization/04_column_check_L12_img0_tok17.png)
(`04_column_check.py`) — 03의 발견은 "CLS 하나"의 관점이다. PatchFool 논문 주장(공격이
attention을 특정 patch로 강하게 끌어당긴다)이 맞다면, CLS뿐 아니라 나머지 196개 patch
쿼리도 patch 17을 봐야 한다. 실제로 확인해보니:

| | CLS -> 17 | patch-쿼리 196개 평균 -> 17 | >0.3인 patch-쿼리 비율 |
|---|---|---|---|
| Clean | 0.113 | 0.110 | 0% |
| **PatchFool** | **0.630** | **0.515** | **94.9%** |
| LaVAN | 0.084 | 0.079 | 0% |

균등분포 기준선은 1/197=0.005. PatchFool에서는 CLS뿐 아니라 거의 모든 쿼리 토큰(196개 중
94.9%)이 patch 17에 0.3 이상의 attention을 준다 — 그림에서도 PatchFool 곡선(빨강)만 전체
쿼리 인덱스에 걸쳐 0.4~0.7 사이에 머무르고, clean/LaVAN(파랑/초록)은 쿼리 인덱스와 무관하게
0.05~0.15에 붙어있다. 즉 patch 17은 CLS만의 특이 현상이 아니라 **레이어 12에서 거의 모든
토큰이 공유하는 진짜 attention sink**다 — PatchFool의 주장과 일치. (clean도 baseline보다는
높은 0.11 정도를 보이는데, 이는 attention sink가 공격과 무관하게 후반 레이어에 자연적으로
존재하는 현상이고, PatchFool은 그 자연 sink를 4.7배(0.11→0.51)까지 증폭시켜 거의 모든
토큰을 하나로 결집시킨다고 해석할 수 있다.)

## GT 검증 시스템 (`gt_lib.py`, `05_gt_check.py`) — "정말 그런지" 눈대중이 아니라 자동 대조

위 patch 17 결과까지는 image 0 한 장을 놓고 clean/adv를 직접 diff하는 일회성 스크립트로
확인했다. 이걸 매번 손으로 하지 않도록, **공격이 실제로 어느 patch를 얼마나 건드렸는지를
pixel diff로 계산한 ground truth**를 파이프라인에 내장했다:

- `gt_lib.gt_coverage(clean_img, adv_img)`: 두 이미지를 직접 빼서(추정 아님) 196개 patch
  각각에서 실제로 바뀐 픽셀 비율(0~1)을 계산. PatchFool은 정확히 patch 1개만 1.0, 나머지는
  0. LaVAN은 16px 그리드에 정렬 안 된 위치에 놓이므로 여러 patch에 걸쳐 부분 비율로 나뉜다.
- `00_extract_attention.py`가 대표 이미지 **5장**(기존 2장에서 늘림)에 대해 원본 픽셀
  (`repr_images`)과 `gt_coverage`를 npz에 같이 저장 — 재실행 없이 언제든 "원본 vs 공격
  이미지"를 다시 볼 수 있는 재료.
- `05_gt_check.py`: 이 GT와 그 레이어의 관측된 attention argmax(`argmax_col_avg`)를 대표
  이미지 5장 x 2개 공격(PatchFool/LaVAN) 전체에 대해 자동 대조하고, 이미지마다 [clean |
  adv | diff 히트맵]에 GT patch(노란 박스)와 관측 argmax(하늘색 박스)를 겹쳐 그린
  `05_gt_overlay_{group}_img{i}_L{layer}.png` 10장 + 대조표
  `05_gt_check_L12_n100.md`를 저장.

**결과 (L=12)**: PatchFool 5장 중 4장 일치([`05_gt_overlay_patchfool_img0_L12.png`](results/characterization/05_gt_overlay_patchfool_img0_L12.png)처럼 GT와 관측이 정확히 같은 patch), 나머지 1장은
[`05_gt_overlay_patchfool_img1_L12.png`](results/characterization/05_gt_overlay_patchfool_img1_L12.png)에서 보듯 GT(25)와 관측(23)이 같은 행에서 2칸 차이 나는 "근접
미스"였다. **LaVAN은 5장 중 1장만 일치** — [`05_gt_overlay_lavan_img2_L12.png`](results/characterization/05_gt_overlay_lavan_img2_L12.png)를 보면 LaVAN이
건드린 3x3 블록(노란 박스, 물속 배경이라 정보량이 적은 영역)과 실제 attention이 몰리는
곳(하늘색 박스)이 완전히 다른 위치다.

**중요한 재해석**: n=100 전체에서 `argmax_col_avg`(L=12)의 최빈 patch를 세어보면 clean조차
patch 25/170/16/179가 각각 16/14/10/10회(100장 중)로 반복 등장한다 — **이미지 내용과
무관하게 특정 patch 위치로 attention이 쏠리는 "고정 sink"가 이 ViT에 이미 존재**하고,
PatchFool이 `attn_layer_idx=4`에서 고르는 patch(16, 25 등)가 우연히 이 고정 sink 후보와
자주 겹친다. 즉 PatchFool의 4/5 일치율은 "공격이 없던 곳에 완전히 새로운 sink를 만든다"
보다는 "이미 sink가 될 가능성이 있는 자리를 골라(레이어 4 CLS attention 기준) 그 sink를
극단적으로 증폭시킨다"(위 column_check의 0.11→0.51, 4.7배)에 더 가깝다. LaVAN은 attention을
전혀 신경 쓰지 않고 위치를 랜덤으로 골라 공격하므로 이 고정 sink와 겹칠 확률이 낮고, 그
결과 attention을 자기 위치로 끌어오지 못한다(1/5) — patch_select='Attn'(PatchFool)과
랜덤 위치(LaVAN)의 차이가 "공격이 attention을 실제로 지배하는가"를 가르는 핵심 변수라는 뜻.

**범위 제한**: 탐지 임계값/AUROC/flag 판정 없음(위 4번 항목의 "5×균등분포"도 설명용).
레이어 결합·Dual-Gate 설계는 이 결과를 보고 다음 단계에서 논의한다.

## Sink 위치 일관성 (n=100 정량화, `02_sink_position.py`)

위 2장짜리 정성적 관찰("PatchFool이 기존 sink를 증폭하는 것 같다")을 n=100 전체로
검증했다. CLS-row, 헤드평균 attention의 argmax 패치 위치(`detector/topk_mass_v1.py`의
`localize_top1`과 같은 정의, 전 레이어로 확장)를 이미지마다 기록해서 (1) 레이어별로 얼마나
소수의 고정 위치에 몰리는지, (2) 같은 이미지에서 clean일 때와 공격 후의 위치가 실제로
일치하는지 계산했다(탐지 임계값 아님, 순수 관찰).

**(1) 위치 집중도** — 그 레이어에서 가장 흔한 top-3 위치가 전체 100장의 몇 %를 차지하는가
(clean 기준, [`results/characterization/02_sink_position_concentration_n100.md`](results/characterization/02_sink_position_concentration_n100.md) 발췌):

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean top-3 비율 | 0.59 | 0.25 | 0.14 | 0.12 | 0.37 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |

**L=6~12에서 clean/LaVAN 모두 top-3 위치가 거의 똑같다**(`[25, 170, 179]`류가 계속
반복됨) — 이 ViT 체크포인트에 레이어와 무관하게 반복되는 고정 sink 위치가 실제로 존재한다는
직접 증거다(공격과 무관한 모델 자체의 성질). PatchFool의 top-3 비율은 오히려 이 구간에서
**더 낮다**(예: L=8 PF=0.25 vs clean=0.40, L=9 PF=0.26 vs 0.40) — PatchFool이 걸리면
top-1 위치가 이미지마다 더 다양해진다(= 고정 sink에서 벗어나 이미지별로 다른 위치로
튄다)는 뜻.

**(2) clean→공격 위치 일치율** — 같은 이미지에서 clean일 때의 top-1과 공격 후 top-1이
같은 patch인지 ([`results/characterization/02_sink_position_match_rate_n100.md`](results/characterization/02_sink_position_match_rate_n100.md) 전체):

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean vs PatchFool | 0.94 | 0.50 | 0.28 | 0.27 | 0.70 | 0.23 | 0.19 | 0.19 | 0.23 | 0.34 | 0.62 | 0.77 |
| clean vs LaVAN | 1.00 | 1.00 | 0.84 | 0.75 | 0.85 | 0.99 | 1.00 | 1.00 | 0.99 | 0.98 | 0.68 | 0.72 |

**해석 — "증폭" 가설은 절반만 맞았다**:
- **LaVAN은 예상대로 거의 항상(대부분 ≥0.84, L=6~10은 0.98~1.00) clean과 같은 위치를
  가리킨다** — attention 구조를 안 건드린다는 앞선 결과와 정확히 들어맞는, 위치 수준의
  직접 확인.
- **PatchFool은 레이어에 따라 반대되는 두 그림을 보인다.** L=6~9(정확히 top-1 mass
  자체는 chance 근처였던 그 구간)에서는 일치율이 **0.19~0.23으로 매우 낮다** — 이 구간
  에서는 "기존 sink 증폭"이 아니라 **"공격이 대부분 이미지에서 새 위치로 옮긴다"**가
  맞는 설명이다. 반면 L=11~12(실제 탐지 신호가 있는 구간)에서는 일치율이 0.62~0.77로
  올라간다 — **여기서는 "기존 sink 증폭"이 다수(그러나 전부는 아님, 23~38%는 여전히
  새 위치)**를 차지한다. 즉 2장짜리 정성적 관찰(image 1)은 우연히 L=11~12 스타일의
  사례를 봤던 것이고, 일반화하면 **"L=11~12에서는 증폭이 다수, 중간 레이어에서는 새
  위치 생성이 다수"**로 고쳐 써야 한다.
- L=1(0.94)은 예외적으로 매우 높은 일치율을 보이는데, top-3 집중도 자체도 0.59로 가장
  높아 애초에 고를 수 있는 "그럴듯한" 위치가 많지 않은(이미지 경계 등 저수준 특징에 좌우되는)
  레이어라는 뜻으로 보인다.

**범위 제한**: 여기서도 탐지 임계값/AUROC/flag 판정은 하지 않았다. 위 수치는 다음 단계
탐지기 설계(예: L=11/12만 볼지, 위치 일치 여부를 보조 신호로 쓸지)에 참고 자료로 남긴다.
