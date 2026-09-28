# PatchFool: CLS와 column attention 탐지 비교

## 무엇을 확인하는 실험인가?

공격 패치의 attention이 다른 패치보다 크게 튄다는 가정으로, 정상 오탐을 제한하면서 PatchFool을 탐지할 수 있는지 확인하는 baseline이다. CLS 대신 모든 query의 평균을 쓰면 개선되는지도 비교한다. LaVAN은 이 test에서 제외한다.

ARMRO의 순위 간격 아이디어를 참고한 비교이며, 원 논문 전체 재현이나 마스킹 방어 성능 평가가 아니다. attention sink나 적응형 공격에 대한 강건성을 입증하는 실험도 아니다.

## 입력과 동일 조건

- 원본: `results/characterization/attention_comparison/run_s42`의 저장된 clean/PatchFool 이미지 100쌍. 공격을 재생성하지 않는다.
- 모델: `vit_base_patch16_224.augreg2_in21k_ft_in1k`, 12개 레이어와 12개 헤드.
- 원본 청크·모델 해시, 표본 ID, 재계산 CLS attention·예측·실제 변경 픽셀 GT·성공/실패 라벨을 검증한다.
- CLS 점수는 헤드를 평균한 CLS query attention이다. Column 점수는 모든 헤드와 모든 query(CLS 포함)를 평균한 attention이다. 둘 다 CLS key를 후보에서 제외하고 196개 이미지 패치를 평가한다.
- 정상 attention과의 차이를 탐지 입력으로 쓰지 않는다. 각 입력 한 장에서 점수를 계산하며 GT는 평가에만 사용한다.

## 탐지 규칙과 평가

1. 각 레이어의 패치 점수를 내림차순으로 정렬하고 상위 5개를 가져온다.
2. 인접 점수 비율 `s1/s2, s2/s3, s3/s4, s4/s5`가 임계값을 넘으면 높은 쪽의 토큰 하나를 선택한다. 상위 토큰 전부를 선택하는 규칙은 아니다.
3. 모든 레이어에서 선택한 토큰의 합집합을 의심 위치로 삼는다. 하나라도 선택하면 이미지 경보다.
4. seed=2026으로 이미지 ID를 무작위 분할한다. 정상 50장은 보정, 별도 50장은 평가에만 사용한다. 같은 원본의 공격 이미지도 같은 분할에 속한다.
5. 정상 보정 이미지에서 전체 레이어의 최대 비율로 방법별 임계값을 정한다. 목표 FPR=5%지만 50장에서는 최대 2장(4%) 경보를 허용한다. 평가 FPR이 같아지거나 5% 이하로 보장되는 것은 아니다.
6. 평가 50장 중 clean 정답 43장만 공격 성공/실패로 비교한다. PatchFool 성공 38장, 실패 5장이다. 원래 오답 7장은 공격 그룹 비교에서 제외하고 정상 FPR 평가에는 포함한다.

경보율은 의심 위치를 하나라도 찾은 이미지 비율이다. GT 적중률은 실제 변경 토큰을 하나라도 찾은 이미지 비율이다. GT recall은 변경 토큰 중 선택한 비율, precision은 선택 토큰 중 GT 비율이며 무선택 이미지의 precision은 0으로 계산해 전체 이미지 평균에 포함한다. 레이어별 곡선은 해당 레이어의 GT 적중률과 그 레이어까지의 누적 적중률을 구분한다.

## 현재 결과와 해석

| 항목 | CLS | Column |
|---|---:|---:|
| 임계값 | 14.7568 | 12.3670 |
| 정상 평가 오탐 | 1/50 (2%) | 0/50 (0%) |
| 공격 성공 이미지 경보 | 8/38 (21.1%) | 4/38 (10.5%) |
| 공격 성공 이미지 GT 적중 | 7/38 (18.4%) | 3/38 (7.9%) |
| 공격 실패 이미지 경보 | 1/5 (20%) | 0/5 (0%) |
| 공격 실패 이미지 GT 적중 | 0/5 | 0/5 |

최초 GT 탐지는 CLS에서 L11 1장·L12 6장, column에서 L12 3장이다. 단순 순위 격차만으로는 탐지율이 낮았고 column의 개선 근거는 나오지 않았다. 다만 실제 평가 FPR이 다르고 실패 표본이 5장뿐이므로 일반적인 우열을 확정할 수 없다. 레이어별 하나의 공통 임계값, query/head 평균, 한 번의 작은 표본 분할이라는 한계가 있다. 전체 attention 기반 탐지의 불가능성을 뜻하지 않는다.

## 출력

`results/characterization/test/cls_column_s42/`에는 다음 PNG만 남긴다.

- `detection_and_localization.png`: 정상 오탐과 공격 성공/실패별 경보율·GT 적중률.
- `layer_gt_localization.png`: 레이어별 GT 적중률(점선)과 누적 적중률(실선).
- `summary.png`: 표본 수, 수치 요약, 임계값과 평가 조건을 담은 표.

CSV·JSON·NPZ·PDF·로그 파일은 생성하지 않는다. 재현에 필요한 분할 ID와 임계값, 원본 manifest 및 스크립트 해시는 PNG의 Description 메타데이터에 기록한다. 이미지 뷰어에서는 표와 그래프를 보면 된다. 중간 점수는 메모리에서만 사용하므로 재실행에는 저장된 이미지의 모델 forward가 다시 필요하다. 원본 paired run의 데이터는 유지한다.

현재 PNG는 기존 검증된 점수로 PatchFool만 다시 집계한 결과다. 새 공격이나 GPU 추론은 수행하지 않았다.

## 실행 방법

저장소 루트에서 GPU 노드로 실행한다. output은 새 폴더를 지정한다.

```bash
python 01_patch_attack_detector/experiments/characterization/test/run_attention_baseline.py \
  --run 01_patch_attack_detector/results/characterization/attention_comparison/run_s42 \
  --output 01_patch_attack_detector/results/characterization/test/cls_column_patchfool_rerun
```

참고: https://proceedings.mlr.press/v202/liu23n/liu23n.pdf
