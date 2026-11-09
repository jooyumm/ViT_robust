# 공통 평가 지표 v2 (00–02)

`src/evaluation.py`가 유일한 계산 원본이다. JSON의 모든 공통 비율은 [0,1]이고 00의
표시용 CA/ERA/ASR/AUA만 백분율이다. metric_version=2를 명시한다.

| 우선순위 | JSON key | 정의 | 분모 |
|---|---|---|---|
| 1 | empirical_robust_accuracy | 정상·공격 후보에서 모두 정답인 비율 | 전체 N |
| 2 | clean_accuracy | 정상 입력 정확도 | 전체 N |
| 3 | attack_success_rate | 정상 정답에서 공격 후 오답으로 바뀐 비율 | 해당 시스템 정상 정답 수 |
| 보조 | accuracy_under_attack | 생성 공격 이미지의 정확도 | 전체 N |

평가 대상 시스템은 00의 각 모델, 01의 분석 대상 모델, 02의 각 방어 정책이다.
02는 방어를 적용한 정상 입력으로 clean-correct 집단을 정한다. 방어 전 모델의
clean-correct 기준 복구율/손상률은 별도 보조 지표이며 공통 ASR과 혼동하지 않는다.
탐지·방어의 선별은 이미지로만 수행하고 label/GT는 평가에서만 쓴다. GT 기반 영역 밖 대조군은 예외적인 평가용 정책이다.

분모가 0이면 null/N/A이며 0%로 대신하지 않는다. 수치와 함께 N과 clean_correct를 기록한다.
여러 후보는 동일 위협모델·예산 안에서 하나라도 오분류하면 실패로 보존한다.
현재 저장된 최종 iterate 하나의 평가는 후보 집합이 작으므로 공격 강도 충분성을 보증하지 않는다.
경험적 지표는 인증 보장이 아니다. ASR이 높을수록 공격이 강하며 방어 관점에서는 낮을수록 좋다.

A/B/C/D 집계는 내부 검산용일 뿐 발표 지표가 아니다. 원본 오답→공격 정답 표본은
accuracy_under_attack에는 포함하고 empirical_robust_accuracy에는 포함하지 않는다.
기존 RA=(A+C)/N를 ERA=A/N로 이름만 바꾸면 안 된다. 00의 legacy JSON은
네 집계가 모두 있을 때만 메모리에서 재계산하고 없으면 오류를 낸다. 원본 파일은 덮어쓰지 않는다.

무작위 후보의 기대값은 paired clean/attack의 동일 후보별로 joint correctness를 계산한다.
`mean(clean_correct)*mean(attack_correct)`처럼 주변확률을 곱하지 않는다.

## 기존 결과 이전

- 00: 과거 PNG는 전체 재실행 완료 전까지 원래 위치에 보존한다. 고정 test 2,000장·3개 seed로 00 자체를 재실행해 새 수치와 그림을 생성한다. 02 파일럿을 대신 사용하지 않는다.
- 면적 파일럿: 원본 NPZ의 label/ID/예측으로 `classification_metrics.json` sidecar를 재계산한다.
  (a)(b)(c) 그림은 ERA를 표시하며 색·구성 등 기존 스타일을 유지한다. AUA는 sidecar에 남긴다.
- 01: `classification_metrics.json/png`는 동일 분석 집합의 분류 성능 맥락이다. attention 탐지 성능이 아니다.
- 02: baseline/defended의 각각 paired 예측으로 재계산한다. 영역 밖 정책의 정상 예측이 없으면
  미완료로 표시하고 기존 baseline 정상 예측으로 대체하지 않는다.
- 분류 지표와 별도로 02의 TPR/FPR·위치 적중·복구·정상 손상·비용은 유지한다.

## 근거

- [AutoAttack, ICML 2020: robust_flags 집계](https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py#L114)
- [RobustBench, NeurIPS 2021: clean/robust 평가](https://github.com/RobustBench/robustbench/blob/master/robustbench/eval.py)
- [PatchFool, ICLR 2022: 전체 공격본 정확도와 False→True](https://github.com/GATECH-EIC/Patch-Fool/blob/main/main.py#L301)
- [PatchCleanser, USENIX Security 2022: 방어 전후 정상 성능 및 인증 지표 분리](https://github.com/inspire-group/PatchCleanser/blob/main/pc_certification.py#L96)

ASR은 clean-correct 조건부 실패 비율로 명시적으로 정의한다. AutoAttack의 Lp 공격을 실행했다거나
본 실험이 RobustBench 공식 패치 벤치마크라는 의미는 아니다.
