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
질문은 딱 하나 — "공격이 정말 토큰 하나로 쏠리는가, 모든 레이어에서 그런가". 답은 **"L=11~12
에서만 그렇다, 나머지 레이어는 아니다"**. LaVAN은 어느 레이어에서도 안 쏠린다. 자세한 내용은
아래 "Raw attention 특성 관찰" 절 참고.

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
    characterization/            "공격이 토큰 하나로 쏠리는가, 모든 레이어에서 그런가"만
                                  본다(탐지기 설계 아님):
      attention_lib.py             저수준: 헤드별 attention hook + 분포/집중도 계산 함수
      attack_gen.py                 PatchFool/LaVAN 공격 생성(00/05가 공유, 재현성용)
      00_extract_attention.py         GPU 필요한 유일한 단계. clean/PatchFool/LaVAN 생성 +
                                    attention 추출 + GT(gt_coverage) -> npz 하나로 저장
      data_io.py                   그 npz를 다시 불러오는 로더
      plotting.py                  01/03이 공유하는 그림 함수
      01_concentration.py        레이어별 top-1 mass(한 토큰이 가져가는 비율) — 핵심 질문에
                                    대한 정량적 답 (npz만 읽음, GPU 불필요)
      03_token_bars.py            대표 이미지 1장 x 레이어 1개로 "진짜 토큰 하나가 튀는지"
                                    눈으로 확인 (npz만 읽음, GPU 불필요)
      05_attack_outcome_compare.py  공격 성공(오분류)/실패(여전히 정답) 이미지를 하나씩
                                    뽑아 attention을 비교 — attention 왜곡이 실제 오분류와
                                    같이 가는지 확인 (GPU 필요, 공격 재생성)

  ground_truth/                  "공격이 실제로 어디를 건드렸는가"(픽셀 diff)만 확정하는
                                  시스템 — attention이나 탐지기 출력과 비교하는 건 여기 안
                                  한다(그건 이 GT를 갖다 쓰는 별도의 나중 단계). characterization
                                  전용이 아니라 이 프로젝트 최상위에 둬서 앞으로 detector/
                                  자체를 테스트할 때도 그대로 재사용한다.
    gt_lib.py                      저수준: clean/adv 픽셀 diff로 "어느 patch를 얼마나
                                    건드렸는지"(196차원 coverage)를 계산하는 ground truth 함수
    gt_plot.py                      GT로 확정된 patch(노란 박스, 겹침 %)를 원본/공격/diff
                                    이미지 위에 그리는 그림 함수 — attention 정보 없음
    visualize_gt.py                실행 스크립트. characterization의 00_extract_attention.py가
                                    저장한 npz(gt_coverage/repr_images)를 읽어 대표 이미지
                                    5장 x 2개 공격 전부의 GT를 표+그림으로 저장
                                    (npz만 읽음, GPU 불필요)
    results/
      gt_summary_n100.md            대표 이미지별 GT patch(겹침 %) 표
      patchfool/img{0..4}.png       공격별로 나눈 GT 그림
      lavan/img{0..4}.png

  results/
    layer_sweep/                위 실험의 결과물 (npz/png/md)
    characterization/           위 실험들의 결과물 (npz/png/md, GT 관련 제외 — ground_truth/results/ 참고)
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

**질문**: 공격이 정말 토큰 하나로 쏠리는가, 그 경향이 모든 레이어에서 같은가. AUROC/threshold/
flag 계산 없음 — PatchFool/LaVAN이 attention을 실제로 어떻게 바꾸는지만 본다.

**방법**: 같은 기본 이미지 n=100장(seed=42)에서 clean/PatchFool(attn_layer_idx=4,
attack_mode='Attention', 250 iters)/LaVAN(patch_ratio=0.02, 40 steps) 세 그룹을 만들고,
forward 1회로 12개 block 전부의 헤드별 attention을 수집. 레이어마다 CLS→patch 행(row,
detector의 대상과 동일)에서 top-1 mass(가장 많이 보는 토큰 하나가 가져가는 비율)를
계산했다(`00_extract_attention.py`, RTX 4090).

**attack_mode 버그**: `chunked_patch_fool`이 원래 `attack_mode='CE_loss'`로 덮어써서
`patch_fool_attack`의 실제 기본값(`'Attention'`: CE loss와 attention loss를 PCGrad로
결합)을 안 쓰고 있었다. 기본값으로 되돌리려다 `src/attacks/patch_fool.py`의 진짜 버그를
발견했다 — attention을 가로채는 forward hook(`_make_attn_hook`)이 `torch.no_grad()` +
`.detach()`로 짜여 있어서, patch 선택(1회, gradient 불필요)에는 맞지만 공격 루프 안에서
attention loss의 gradient를 `delta`까지 역전파하는 데는 못 쓴다 — 실제로 시도하면
"element 0 of tensors does not require grad" 에러로 즉시 죽는다. 즉 `attack_mode=
'Attention'`은 이 코드베이스에서 **한 번도 정상 동작한 적이 없었다**. 원본 논문 저장소
([RICE-EIC/Patch-Fool](https://github.com/RICE-EIC/Patch-Fool))는 모델 forward가
attention을 직접 반환해서 이 문제가 없는데, 여기서는 timm 모델을 그대로 쓰면서 hook으로
가로채다 보니 생긴 문제였다. `_make_attn_hook_grad`/`_collect_attn_grad`(no_grad/detach
없는 버전)를 추가해서 공격 루프 안에서만 쓰도록 고쳤다 — 아래 결과는 이 수정 이후, 진짜
`attack_mode='Attention'`으로 뽑은 것이다.

**결과**: 레이어별 top-1 mass(CLS row, 헤드평균) 중앙값 —

| L | clean | PatchFool | LaVAN | PF−clean |
|---|---|---|---|---|
| 1 | 0.0136 | 0.0137 | 0.0136 | +0.0001 |
| 2 | 0.0156 | 0.0187 | 0.0156 | +0.0031 |
| 3 | 0.0151 | 0.0209 | 0.0149 | +0.0059 |
| 4 | 0.0142 | 0.0224 | 0.0149 | +0.0082 |
| 5 | 0.0208 | 0.0291 | 0.0210 | +0.0083 |
| 6 | 0.0495 | 0.0471 | 0.0489 | −0.0025 |
| 7 | 0.1154 | 0.1093 | 0.1179 | −0.0061 |
| 8 | 0.1743 | 0.1632 | 0.1784 | −0.0111 |
| 9 | 0.2176 | 0.1951 | 0.2105 | −0.0225 |
| 10 | 0.1851 | 0.1848 | 0.1719 | −0.0003 |
| 11 | 0.1202 | 0.2899 | 0.1060 | **+0.1697** |
| 12 | 0.1536 | 0.4365 | 0.1273 | **+0.2829** |

전체 표는 [`results/characterization/01_concentration_summary_n100.md`](results/characterization/01_concentration_summary_n100.md)
(`01_concentration.py` 산출물).

**답 — 교수님 질문("토큰 하나로 쏠리는가, 모든 레이어에서 그런가")**:

1. **아니다, 모든 레이어에서 그런 게 아니다.** L=1~10은 PatchFool도 clean/LaVAN과 top-1
   mass가 거의 같다. **L=11~12에서만** PatchFool이 뚜렷이 튄다(0.12→0.29, 0.15→0.44) —
   "쏠림"은 레이어 전체가 아니라 후반 2개 레이어에서만 나타나는 현상이다.
   [`results/characterization/01_layer_comparison_top1_row.png`](results/characterization/01_layer_comparison_top1_row.png)
2. **LaVAN은 어느 레이어에서도 안 쏠린다** — 모든 레이어에서 중앙값이 clean과 거의 겹친다
   (최대 편차 L=12에서 −0.026). raw attention만으로 LaVAN을 못 잡는 이유다.
   [`results/characterization/01_hist_top1_row_by_layer.png`](results/characterization/01_hist_top1_row_by_layer.png)의
   L=11/L=12 패널을 보면 PatchFool이 긴 꼬리(최대 0.6~0.8)를 만드는데, 이게 layer_sweep에서
   recall이 100%가 아니라 72.7%인 이유와 같은 그림이다(쏠리는 이미지와 안 쏠리는 이미지가
   섞여 있음).

**진짜 토큰 하나가 튀는지 예시로 확인**:
[`results/characterization/03_token_bars_L12_img0.png`](results/characterization/03_token_bars_L12_img0.png) —
197개 토큰을 정렬 없이 그대로 그려서, image 0/L=12에서 PatchFool만 **token 99**(patch 98)
에서 0.453까지 치솟고(clean 0.180, LaVAN 0.107) 나머지는 평평함을 확인했다. 이 자리가
진짜 공격당한 patch인지도 `ground_truth/`로 확정해 뒀다 — 정확히 patch 98(0-indexed)에서만
픽셀이 바뀌었고, 이 스파이크 위치와 정확히 일치한다.

**범위 제한**: 탐지 임계값/AUROC/flag 판정 없음. 레이어 결합·Dual-Gate 설계는 이 결과를
보고 다음 단계에서 논의한다.

## GT 시스템 (`ground_truth/`) — "정말 그런지" 눈대중이 아니라 픽셀로 확정

**공격이 실제로 어느 patch를 얼마나 건드렸는지를 pixel diff로 계산한 ground truth**를
파이프라인에 내장했다. `ground_truth/`의 역할은 딱 이것뿐이다 — attention이나 탐지기
출력과 비교하는 로직은 여기 없다(그건 이 GT를 가져다 쓰는 별도의 나중 단계).
characterization 전용이 아니라 이 프로젝트 최상위에 둬서, 앞으로 탐지기(`detector/`)
자체를 테스트할 때도("탐지가 가리키는 위치 vs 실제 공격 위치") 같은 GT를 재사용한다:

- `ground_truth/gt_lib.py`의 `gt_coverage(clean_img, adv_img)`: 두 이미지를 직접 빼서
  (추정 아님) 196개 patch 각각에서 실제로 바뀐 픽셀 비율(0~1)을 계산. PatchFool은 정확히
  patch 1개만 1.0, 나머지는 0. LaVAN은 16px 그리드에 정렬 안 된 위치에 놓이므로 여러
  patch에 걸쳐 부분 비율로 나뉜다.
- `experiments/characterization/00_extract_attention.py`가 대표 이미지 5장에 대해 원본
  픽셀(`repr_images`)과 `gt_coverage`를 npz에 같이 저장 — 재실행 없이 언제든 "원본 vs
  공격 이미지"를 다시 볼 수 있는 재료.
- `ground_truth/visualize_gt.py`: 5장 x 2개 공격(PatchFool/LaVAN) 전부에 대해 GT
  patch(노란 박스, 겹침 %)를 [clean | adv | diff 히트맵]에 그린 그림을 공격별 폴더
  (`results/patchfool/`, `results/lavan/`)에, 요약표를 `results/gt_summary_n100.md`에
  저장한다. attention/레이어 인자 없음 — 순전히 GT만 본다.
