# 연구 제안: 패치 제외 영향량으로 ViT 공격 탐지

작성일: 2026-09-26. 상태: 문헌 조사와 수식 참조 구현 검증 완료, 실제 탐지 성능·속도·독창성은 미검증.

## 범위와 목표

모든 분야의 모든 탐지기를 망라했다는 주장은 하지 않는다. 고정된 사전학습 ViT의 디지털 패치 공격 탐지에 직접 관련된 attention, 내부 특징, 재구성, 학습 기반, 의미 일관성, 마스킹·보증 방어와 인접 attribution 연구를 조사했다. 아래 문헌 중 일부는 초록·방법 설명 수준으로 확인했으며, 원 구현과 세부 설정까지 재현한 상태가 아니다.

목표는 clean 원본이나 GT 없이 입력 한 장으로 의심 패치를 찾는 것이다. 기존 방법 전체를 모든 조건에서 이긴다는 요구 대신, 동일 위협 모델·오탐률에서 탐지와 위치 추정이 개선되고 지연시간 예산도 만족하는지 검증한다. 현재 test에서는 PatchFool만 평가한다. 다른 공격은 향후 일반화 검증 계획이며 이번 요청으로 실행하지 않는다.

## 조사한 비교군과 중복 위험

| 방법 | 확인한 핵심 | 제안과 비교할 사항 |
|---|---|---|
| ARMRO (ICML 2023) | 비정상 attention으로 패치 위치를 찾고 마스킹 | 가장 직접적인 비교군. 현 baseline은 원 논문 전체 재현이 아님 |
| Robust Self-Attention / RSA (ICML workshop 2021) | value 벡터와 평균의 거리로 이상 토큰을 찾아 robust aggregation | value 이상치 자체는 이미 제안됨. 필수 대조군 |
| Token Attention Restriction (ICASSP 2023) | 얕은 token norm과 attention을 제한하는 방어 | norm 추가만으로 신규성 주장 불가 |
| ViTGuard (2024 preprint) | MAE 재구성 전후 attention 및 CLS 표현 차이 | clean 원본은 필요 없지만 재구성 모델과 추가 추론이 존재 |
| PatchZero (WACV 2023) | 학습한 픽셀 탐지기로 위치 추정 후 제거 | 별도 학습과 공격 분포 의존성, 실제 비용 비교 |
| Z-Mask (AAAI 2023) | 내부 특징의 Z-score와 공간적 과활성 분석 | 레이어별 정규화와 feature anomaly는 기존 아이디어 |
| Jujutsu (2021 preprint) | 영향력이 큰 국소 특징으로 탐지하고 생성 모델로 복원 | 영향력 기반 탐지 자체가 새롭지는 않음 |
| AdvPatchXAI (CVPR 2025) | prototype을 활용하는 설명 가능한 패치 탐지 | 최근 학습 기반 탐지 비교군 |
| PATCHOUT (2025) | 예측 클래스의 의미 일관성, SAM 분할로 위치 추정 | 의미 일관성만 추가하는 설계와 구분 필요 |
| PatchGuard / PatchGuard++ | 작은 receptive field와 특징 마스킹, 보증된 강건성·탐지 | 구조가 다른 보증 방법. empirical TPR만으로 보증보다 우수하다고 할 수 없음 |
| PatchCleanser (USENIX Security 2022) | 두 단계 입력 마스킹과 조건부 보증 | 탐지와 분류 복원을 구분해 비용·정확도 비교 |
| CertMask (AAAI 2026) | 한 단계 마스킹으로 보증 방어 비용 개선 | 최근 비용 개선을 빠뜨리고 무조건 더 빠르다고 주장하지 않음 |
| Concept-Based Masking (2025 preprint) | 영향력 있는 concept을 억제하는 방어 | 특징 수준 억제의 인접 선행연구 |
| Attention is Not Only a Weight (2020), ALTI (EMNLP 2022) | attention 이외 value·projection·residual·LayerNorm 등을 고려한 기여도 해석 | attention×value 및 token interaction 자체는 신규성 없음 |
| Token-Based Detection of Spurious Correlations (2025 preprint) | 토큰 제외 영향으로 spurious correlation 분석 | token removal 자체도 기존 원리 |
| Adversarial Decoys (2026 preprint) | 악성 패치와 attention 순위를 유도하는 decoy를 분리 | attention ranking 이외 점수도 적응형 공격으로 검증해야 함 |

물체 검출기용 방법은 이미지 분류기와 바로 동일 순위로 비교하지 않는다. 이식한 feature baseline과 원 논문 재현을 구별한다. IC CV 2025의 patch defense benchmark는 object detector 대상이라 ViT 분류기에서의 우열 증거로 사용하지 않는다.

## 제안의 핵심 가설

attention을 얼마나 받는지보다, 특정 패치의 메시지가 현재 attention 출력에 얼마나 고유하게 기여하는지를 측정한다. 가설은 성공적인 패치 공격 일부가 정상 패치보다 비정상적으로 큰 국소 메시지 영향력을 만든다는 것이다. 작은 정상 물체나 중요한 질감도 영향력이 클 수 있으므로 악성의 필요충분조건이 아니다.

attention 값이 큰 모든 토큰을 공격으로 보지 않는다. value가 다른 토큰의 출력과 유사하면 attention이 높아도 제외 영향은 작을 수 있다. 반대로 attention이 최고가 아니어도 value 차이가 크면 영향이 클 수 있다. 이 가설이 sink를 자동으로 구별한다는 뜻은 아니다. sink도 정상화 분모를 통해 실제 영향을 미칠 수 있고 decoy 역시 value까지 조작할 수 있다.

## 수식과 정확한 의미

한 레이어·한 head에서 query i의 attention 출력을 o_i = Σ_k a_ik v_k 라 하자. 현재 Q/K/V를 고정한 채 key j를 제외하고 나머지 확률을 재정규화하면:

    o_i_without_j = (o_i - a_ij v_j) / (1 - a_ij)
    o_i - o_i_without_j = a_ij / (1 - a_ij) * (v_j - o_i)

후보 점수는 이 변화 벡터의 제곱 노름이다. 모든 head를 평균하기 전에 각각 계산한다. 확률 1 근처에서는 원래 logits의 masked softmax 또는 명시적 제외로 안정적으로 계산해야 한다. epsilon으로 분모를 임의로 바꾼 값을 정확한 개입 결과라고 부르지 않는다.

이 식은 단일 head 내부의 정확한 key 제외 영향이다. 원본 이미지에서 패치를 지우고 전체 모델을 재실행한 결과가 아니다. 후속 레이어, MLP, residual 경로의 전체 인과 영향이나 최종 logit 기여도를 보장하지 않는다. head별 제곱 노름 평균은 output projection과 head 사이 상쇄를 반영하지 않으므로 full projection 버전과 반드시 비교한다.

국소 영향량 수식 그 자체를 새로운 수학적 발견으로 주장하지 않는다. 후보 연구 기여는 이를 고정된 ViT의 제한된 query에서 계산하고 정상 기준으로 보정해, 단순 attention·value 이상치보다 좋은 탐지-비용 균형을 얻는지다. 현재 조사만으로 이 조합의 최초 제안 여부도 확정하지 않는다.

## 구체적인 1차 detector 설계

1. ViT 가중치와 예측 계산은 고정한다. hook에서 이미 계산된 Q/K/V를 재사용한다. QKV projection을 hook에서 중복 계산하는 기존 방식의 비용을 숨기지 않는다.
2. 12개 레이어에서 CLS와 공간적으로 고정된 3×3 격자의 9개 image query를 관측한다. 이미지 query 위치는 attack attention 순위로 선택하지 않는다. key 후보는 모든 이미지 토큰이다. 자기 자신 query=key 항은 공간 전파 점수에서 제외한다.
3. 각 query-key-head의 제외 영향량을 계산한다. CLS 채널과 spatial query 평균 채널을 분리한다. head별 에너지를 계산한 뒤 합산하므로 벡터를 먼저 평균해 상쇄하는 것과 다르다. 토큰 개수나 차원이 다른 모델은 별도로 보정한다.
4. 별도 정상 reference set에서 log(영향량+작은 상수)의 레이어·채널별 중앙값과 MAD를 구한다. 후보당 이상 점수 z를 얻는다. 위치별 보정이 필요한지는 정상 위치 편향 진단으로 결정하고 개발 단계에서 고정한다.
5. 각 패치의 점수는 모든 레이어·채널의 z 최댓값, 이미지 점수는 모든 패치 점수의 최댓값으로 정의한다. 최초 버전은 추가 feature나 임의 가중합을 넣지 않는다. 한 레이어만 이상해도 검출할 수 있으며 여러 비교로 오탐이 커지는 문제는 다음 단계에서 보정한다.
6. reference와 분리한 정상 calibration set에서 최종 이미지 최댓값의 임계값을 정한다. GT나 공격 성공 여부로 임계값을 고르지 않는다. evaluation set에서 목표 FPR뿐 아니라 실제 FPR과 신뢰구간을 보고한다.
7. 이미지 경보와 패치별 z 맵을 출력한다. 패치 임계값의 합집합은 이미지 경보와 일치하도록 한다. 위치 지도는 현재 레이어 token 좌표의 점수이므로 후기 token의 정보 혼합이 GT 위치 추정을 흐릴 수 있다.

관측 query를 줄여도 모든 key 후보를 평가한다. 따라서 attention 상위 몇 개 key만 후보로 삼는 선별 손실은 피하지만, 관측하지 않은 query를 통한 공격은 놓칠 수 있다. Q=10과 full-query 버전의 차이를 측정해야 한다.

## 구조적으로 설명할 수 있는 장점과 미검증 주장

설계상: 입력 한 장, clean 원본 없음, GT 없음, 별도 복원·분할 모델 없음, backbone 재학습 없음, 역전파 없음, 전체 모델 재추론 없음. 탐지용 정상 reference/calibration 데이터는 필요하므로 데이터가 전혀 필요 없는 방법이라고 부르지 않는다.

미검증: ARMRO보다 높은 탐지율, sink 오탐 감소, decoy 저항성, 특정 GPU에서 더 빠른 실행. 한 번의 forward라도 hook/attention 재계산·메모리 접근 비용이 크면 느릴 수 있다. 보증 방어의 certificate는 제공하지 않는다. input 전체 forward를 완료하는 설계이므로 조기 종료에 의한 연산 절감을 주장하지 않는다.

## 실험 계획과 채택 기준

현재 100쌍은 이미 여러 차례 분석했으므로 개발용으로만 사용한다. 이를 최종 독립 평가로 다시 부르지 않는다. 신규 정상 reference 2,000장, 정상 calibration 2,000장, 별도 평가 최소 2,000장을 제안한다. 이는 실행한 수가 아니라 계획이다. 최종 sample size는 목표 FPR과 신뢰구간 폭에 맞춰 정한다. 같은 원본과 그 공격본은 항상 같은 split에 둔다.

첫 단계 비교군:
- 현재 CLS/column rank-gap baseline.
- 원 구현/설정을 확인한 ARMRO 및 공정하게 보정한 변형(둘을 명시적으로 구별).
- RSA 방식 value 거리, token norm, attention×value norm, 레이어별 정규화한 attention.
- 제안 영향량: CLS만 / spatial만 / 둘 다; 10 queries / full queries; head 에너지 / output projection 반영.

다음 단계 외부 비교군: ViTGuard, PatchZero, AdvPatchXAI 중 실제 재현 가능한 분류용 방법. 인증 방식은 탐지·분류·보증·시간을 별도 열로 비교한다.

평가 공격: 우선 PatchFool의 seed·위치·크기 변화. 일반화는 Masked-PGD, Attention-Fool, decoy 및 제안 점수를 직접 낮추는 white-box 공격으로 검증한다. 이번 실험 범위에서 LaVAN은 제외한다. 공격 예산(면적, 픽셀 범위, 반복 수, restart, 모델)을 맞추고 official PatchFool과 로컬 구현 차이를 먼저 감사한다. 현재 로컬 PatchFool의 unbounded 입력을 그대로 유일한 평가로 쓰면 입력 범위 이상을 잡은 것일 수 있으므로 유효 픽셀 범위 조건을 별도로 평가한다.

적응형 평가에서는 분류 공격과 탐지 회피를 함께 최적화하고 여러 손실 가중치·restart·iteration 수를 사용한다. max/선택 연산에는 smooth surrogate나 적절한 gradient 방법을 쓰되, 최종 성공은 실제 detector의 hard rule로 판정한다. 조작으로 탐지 점수만 낮췄으나 분류 공격도 실패한 경우를 회피 성공으로 세지 않는다. gradient masking 진단과 query 기반 우회도 포함한다.

필수 정상 대조: 이미지 내 작은 물체, 텍스트, 고대비 배경, 반복 질감, 정상 sticker, 자연스러운 occlusion, 비공격 random patch. 같은 정상 데이터에서 임계값을 내리기만 해 얻은 TPR 개선을 채택하지 않는다.

측정: ROC/저FPR 구간, 실제 FPR, TPR, 성공한 미탐지 공격 비율, GT hit/recall/precision, 최초 GT 탐지 레이어, clean 정답·오답별 경보율. 타이밍은 warmup·CUDA 동기화 후 batch=1 및 batch=10의 median/p95와 peak GPU memory, fused attention 설정을 보고한다.

개발 전에 고정할 목표 예시(측정된 성능 아님): 동일 FPR 1% 또는 5%에서 재현한 가장 강한 저비용 비교군 대비 TPR +10pp, paired bootstrap 차이의 95% 구간 하한 >0; baseline backbone 대비 median 지연 증가 20% 이내. 이 기준은 조정한다면 평가 데이터를 열기 전에 문서에 남긴다. 공격별 실패를 평균으로 숨기지 않는다.

중단 기준: attention×value/RSA와 비슷하면 신규성을 주장하지 않는다. 정상 중요 패치 때문에 저FPR TPR이 낮으면 단순 영향량 가설을 기각한다. 적응형 공격에서 무너지면 decoy-resistant라고 부르지 않는다. 속도가 기준을 넘으면 경량 detector라고 부르지 않는다. 더 복잡한 특징을 계속 덧붙여 현재 100장에 맞추지 않는다.

## 현재 완료한 것

`key_removal_influence.py`는 NumPy float64 수식 참조 구현이다. 명시적으로 key를 제외한 계산과 효율적 항등식의 결과를 합성 텐서에서 비교했고 최대 오차는 약 3.11e-15였다. 공통 value 이동 불변성, 모든 value가 같을 때 영향량 0, 극단 attention 확률 대체 경로도 확인했다.

이 검사는 수식과 구현의 일치만 확인한다. 이미지 공격 탐지율·GPU 속도 검사가 아니며 학습/추론 파이프라인까지 완성한 detector가 아니다. 신규 GPU 작업은 제출하지 않았고, 결과 폴더에 새로운 수치나 파일을 생성하지 않았다.

## 출처

- ARMRO: https://proceedings.mlr.press/v202/liu23n/liu23n.pdf
- RSA: https://people.eecs.berkeley.edu/~daw/papers/patchsa-udl21.pdf
- Token Attention Restriction: https://doi.org/10.1109/ICASSP49357.2023.10096862
- ViTGuard: https://arxiv.org/abs/2409.13828
- PatchZero: https://openaccess.thecvf.com/content/WACV2023/papers/Xu_PatchZero_Defending_Against_Adversarial_Patch_Attacks_by_Detecting_and_Zeroing_WACV_2023_paper.pdf
- Z-Mask: https://ojs.aaai.org/index.php/AAAI/article/view/26758
- Jujutsu: https://arxiv.org/abs/2108.05075
- AdvPatchXAI: https://openaccess.thecvf.com/content/CVPR2025/papers/Kumar_A_Unified_Resilient_and_Explainable_Adversarial_Patch_Detector_CVPR_2025_paper.pdf
- PATCHOUT: https://link.springer.com/article/10.1007/s11063-025-11775-5
- PatchGuard: https://www.usenix.org/system/files/sec21-xiang.pdf
- PatchGuard++: https://arxiv.org/abs/2104.12609
- PatchCleanser: https://www.usenix.org/conference/usenixsecurity22/presentation/xiang
- CertMask: https://ojs.aaai.org/index.php/AAAI/article/view/37716
- Concept-Based Masking: https://arxiv.org/abs/2510.04245
- Attention is Not Only a Weight: https://arxiv.org/abs/2004.10102
- ALTI: https://aclanthology.org/2022.emnlp-main.595/
- Token-Based Detection of Spurious Correlations: https://arxiv.org/abs/2509.04009
- Adversarial Decoys: https://arxiv.org/abs/2607.07922
- Object-detector defense benchmark (범위 참고): https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_Revisiting_Adversarial_Patch_Defenses_on_Object_Detectors_Unified_Evaluation_Large-Scale_ICCV_2025_paper.pdf
