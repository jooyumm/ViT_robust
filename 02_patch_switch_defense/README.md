# 02_patch_switch_defense — ViT 적응형 방어(P16→P8 폴백) 프로젝트

**"P16으로 기본 추론하다가, 공격이 의심되는 패치만 국소적으로 P8 강건성으로 전환하는 적응형
ViT 방어(local switch)"**를 설계·검증하는 독립 프로젝트다. 원래 [`00_patch_size_tradeoff/`](../00_patch_size_tradeoff/)
(패치 크기 vs 강건성 7개 정식 실험 — PGD/LaVAN/PatchFool × P8/P16/P32)에서 나온 발견("PatchFool에
대해 P8이 P16보다 압도적으로 강건함")을 실제 방어로 발전시키기 위해 2026-09-15에 분리했다.

**범위 — 00_patch_size_tradeoff와 다른 점**: 이 프로젝트는 방어 메커니즘 자체(토큰화 격자 불일치)에
집중하므로 **PGD는 제외, patch size는 P8/P16만, 공격은 LaVAN·PatchFool만** 다룬다(P32는
방어 로직과 무관, PGD는 원래 실험에서도 전역 L∞라 RA가 전부 포화돼 분석 무의미했음) — 이건
코드로 강제한 제약이 아니라 이 프로젝트의 실험 스크립트들이 그냥 P32/PGD를 안 부르는
것뿐이다. 모델/데이터/공격 코드는 00_patch_size_tradeoff와 **완전히 동일**하다 — 처음엔(2026-09-15)
이 프로젝트 안에 새로 복사한 사본으로 뒀지만, 2026-09-21에 `01_patch_attack_detector`까지 포함해 세
프로젝트가 완전히 같은 내용을 각자 사본으로 들고 있는 게 비효율적이라 판단해
[`../src/`](../src/)(`ViT_robust` 최상위, timm 체크포인트 이름까지 동일한
[`../src/models.py`](../src/models.py))로 합쳤다(`ViT_robust/README.md` 참고). 이 프로젝트
고유 코드(`defense/`, `experiments/`)는 여전히 이 프로젝트 안에만 있다.

## 지금 상태: defense/만 다듬는 중, experiments/는 비워둠

이 프로젝트는 지금 **일시 정지 상태**다 — 방어 메커니즘 코드(`defense/`)는 계속
다듬지만, 그걸 평가하는 실험(`experiments/`)은 전부 지웠다. 이유: 지금까지의 평가는 전부
`01_patch_attack_detector`의 탐지기(raw-attention Top-K Mass)가 flag한 패치를 넘겨받는
전제로 짜여 있었는데, 그 탐지기 쪽이 아직 레이어별 sweep·특성 분석 단계라 어떤 레이어/
방식을 실배포 기준으로 쓸지 확정되지 않았다. 탐지기가 안 굳은 상태에서 나온 방어 평가
숫자는 탐지기가 바뀌면 다시 재현해야 하므로, 지금 단계에서 붙들고 있는 게 낭비라고
판단했다. 자세한 내용과 지운 목록은 [`experiments/README.md`](experiments/README.md) —
git 이력에서 전부 복구 가능하다.

**탐지기가 확정되면**: 이전에 쓰던 방법론(같은 calibration, 같은 eval 이미지, 같은 탐지
판정으로 all_switch/local_switch를 직접 비교) 그대로 재개하면 된다 — 그 설계 자체가
문제였던 적은 없고, 재검증이 필요했던 건 항상 "어떤 탐지기냐"였다.

## 목표: Local Switch — 왜 국소 전환인가

이 프로젝트의 목표는 처음부터 **국소 전환(local switch)**이었다: 탐지기가 공격을 의심하면
이미지 전체가 아니라 **그 패치 1개만** 4개의 P8 서브패치로 국소 교체하고, 나머지 195개 토큰과
P16의 12개 transformer block은 전혀 건드리지 않고 그대로 재사용한다. 이미지 전체를 P8로 다시
돌리는 "전체 전환(all switch)"은 그 자체가 목표가 아니라, **local switch가 얼마나 저렴하면서도
동급의 강건성을 내는지 보여줄 비교 기준선**이다. all switch는 훨씬 단순한 설계라(별도 모델을
통째로 다시 돌릴 뿐, 두 표현을 이어붙일 정렬 메커니즘이 필요 없음) "naive하지만 확실히
작동하는 상한선" 역할을 한다.

| | **Local Switch** (`defense/local_switch.py`) — 목표 | **All Switch** (`defense/all_switch.py`) — 비교 기준선 |
|---|---|---|
| 전환 방식 | 의심되는 **패치 1개만** 4개의 P8 서브패치로 국소 교체 (196→200토큰), 나머지 195토큰·12개 block은 P16 그대로 재사용 | 의심되면 이미지 **전체**를 P8로 다시 분류 |
| 정렬 메커니즘 | closed-form 최소제곱 아핀 변환(768×768+bias)으로 patch_embed 레벨에서 P8 서브패치를 P16 좌표계에 투영 | 없음 (완전히 별도 모델을 그대로 돌림) |

두 함수 다 탐지 여부를 스스로 판단하지 않는다 — "이 패치가 의심된다"는 판단(과, local_switch의
경우 "어느 패치인지")은 호출부(탐지기)가 이미 끝낸 뒤 `flag_idx`/이미 걸러진 `images`로
넘어온다고 가정한다. 지금은 `01_patch_attack_detector/detector/topk_mass_v1.py`가 그 판단을
맡지만, 이 두 함수 자체는 어떤 탐지기를 쓰든 그대로 재사용 가능하도록 설계했다.

## defense/local_switch.py 최적화: P8 patch_embed를 4개만 계산

**문제**: local_switch는 원래 P8의 patch_embed를 이미지 **전체**(784개 서브패치)에 대해
계산하고 그중 flag된 4개만 썼다 — 실제 필요한 것보다 훨씬 더 계산하는 구현이었다(예전
cost_comparison에서 이 4개가 local_switch 추가 비용의 "보수적 상한"이라고 명시했던 항목).

**최적화**: `model8.patch_embed`는 padding 없는 stride=kernel_size Conv2d다(겹치는 영역이
없음) — 그래서 flag된 P16 패치 하나에 대응하는 16×16 픽셀 crop **하나에만** conv를 돌려도,
전체 224×224 이미지에 conv를 돌리고 같은 위치를 슬라이스한 것과 수학적으로 완전히 동일한
값이 나온다(non-overlapping conv는 각 출력 위치가 자기 receptive field만 보기 때문). 이
사실을 이용해 `_p8_subpatch_embed_only()`가 4개 서브패치의 patch_embed만 직접 계산한다.

**검증**: 전용 스크립트로 (1) 이 함수의 출력이 "전체 784개 계산 후 4개 슬라이스"와 **최대
절대오차 0.0**(완전히 동일)임을 확인했고, (2) 미분 가능 경로(`local_swap_logits`, joint/
adaptive-evasion 공격 스크립트가 gradient를 구하는 데 쓰는 함수)도 crop을 거쳐 정상적으로
gradient가 흐름을 확인했다 — autograd가 슬라이싱/crop을 그대로 추적하므로 기존과 다른 동작이
아니다.

## 이전 실험 결과 (git 이력, 재현 불가 — 위 "지금 상태" 참고)

`experiments/`를 비우기 전, 마지막으로 확인됐던 핵심 수치 요약(탐지기가 확정되지 않은
상태에서 나온 숫자이므로 참고용):

- **system_comparison** (n=250, seed=42): 복원율 all_switch 96.5% vs local_switch 89.4%
  (95% CI 겹침), 시스템 정확도 66.0% vs 65.3%(사실상 동률), escalate 추가 비용 6.91ms vs
  **2.58ms**(local_switch가 2.7배 저렴 — 위 최적화 이후로는 이 2.58ms도 낮아질 여지가 있음).
- **naive/adaptive 공격 강건성**: joint attack 완전 무력화 18.4% vs 16.7%, adaptive evasion
  worst-case 15.8% vs 21.4% — 전부 신뢰구간이 겹쳐 "local switch가 더 취약하다는 근거 없음".
- **탐지기 자체**: raw attention L=12, AUROC 0.879(clean 대비)~0.891(LaVAN 대비), 위치특정
  recall@1 96.7% — 단, `01_patch_attack_detector`의 후속 관찰(레이어별 sweep, attention
  특성 분석)에서 이 L=12 단일 레이어 채택 자체가 재검토 대상이 됐다(그 프로젝트 README 참고).
  **탐지기가 LaVAN을 원리적으로 못 잡는다**는 한계도 그대로 유효하다.

이 숫자들이 나온 실험 코드·그림·원자료는 전부 git 이력에 남아있다
(`git log --diff-filter=D -- experiments/` 로 삭제 커밋을 찾아 그 직전 커밋에서 복구).
