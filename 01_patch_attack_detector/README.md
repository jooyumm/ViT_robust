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

## 현재 디렉토리 구조

```text
experiments/characterization/
  01_extract_paired_attention.py          동일 쌍의 이미지·예측·attention·GT 추출
  02_compare_attention_success_failure.py 성공/실패 비교, 전체 레이어 및 특이 레이어 분석

detector/topk_mass_v1.py                  다른 방어 프로젝트에서 사용하는 공유 탐지기
ground_truth/                            GT 계산·표시 도구
results/characterization/
  attention_comparison/<run>/            현재 실험의 실행별 데이터와 분석 결과
  ...                                   과거 결과 (현재 실험의 입력으로 사용하지 않음)
```

모델·데이터셋·공격 구현은 저장소 루트의 `src/`를 사용한다. 탐지기와 GT 도구는
`characterization` 정리 범위 밖의 별도 기능이다. `ground_truth/visualize_gt.py`는 과거
NPZ 예시를 직접 읽으며 현재 실험 코드에 의존하지 않는다.

## 공격 전후 attention과 성공/실패 비교: 동일 이미지의 공격 전후 attention과 공격 성공/실패 비교

**질문**: 같은 이미지에서 attention 맵이 어떻게 변하며, 그 변화는 공격 성공과 실패에서
어떻게 다른가? 먼저 모든 레이어를 관찰하고, 변화가 두드러진 레이어와 앞뒤 레이어·헤드를 확인한다.

코드는 두 단계로 통일했다.

| 파일 | 역할 |
|---|---|
| `experiments/characterization/01_extract_paired_attention.py` | 같은 공격 실행에서 이미지·예측·헤드별 CLS attention·GT를 함께 추출 |
| `experiments/characterization/02_compare_attention_success_failure.py` | 해당 실행 폴더만 검증해서 읽고 전체 레이어, 성공/실패, 특이 레이어 후보를 분석 |

### 동일 값 비교의 조건

- clean과 공격 후 각각 한 번의 forward에서 logits와 attention을 함께 수집한다.
- 표본 ID와 원본 경로, 정답, 예측, 공격 이미지, 모든 레이어·헤드의 CLS attention을 저장한다.
- 실제 clean–attack 픽셀 차이로 GT coverage를 계산한다. clean과 attack에 **같은 GT 위치**를 적용한다.
- 성공은 clean 정답 → 공격 후 오답, 실패는 clean 정답 → 공격 후 정답이다.
  clean부터 오답인 표본도 저장하지만 두 집단의 비교에서는 제외한다.
- 실행 ID와 파일 SHA256, 표본 순서, 이미지로 재계산한 GT, logits의 argmax와 저장 예측,
  성공/실패 판정 및 확률 질량을 검증한다. 다른 실행이나 미완료 실행은 분석하지 않는다.
- 모델 파라미터 지문, 코드 지문, 공격 인자, 라이브러리 버전을 manifest에 남긴다.
  배치 크기는 공격 난수와 LaVAN 위치에 영향을 주므로 재현 시 동일하게 유지한다.

**기존 자료와 관계**: 과거 실험 스크립트와 사용하지 않는 보조 모듈은 삭제했다.
과거 결과 파일은 보존하지만, 예측·GT 연결이 없는 자료를 현재 성공/실패 비교에 섞지 않는다.
현재 분석 입력은 새 추출기가 저장한 `manifest.json`과 `data/chunk_*.npz`뿐이다.

### 실행 순서 (사용자가 실행)

저장소 루트에서, 추출은 GPU 노드에서 실행한다. 아래 경로는 아직 존재하지 않는 새 폴더여야 한다.

```bash
python 01_patch_attack_detector/experiments/characterization/01_extract_paired_attention.py \
  --num_samples 100 --batch_size 10 --seed 42 \
  --output 01_patch_attack_detector/results/characterization/attention_comparison/run_s42

python 01_patch_attack_detector/experiments/characterization/02_compare_attention_success_failure.py \
  --run 01_patch_attack_detector/results/characterization/attention_comparison/run_s42
```

추출기의 `--output`을 생략하면 시각과 실행 ID로 새 폴더를 만든다. 재시작 이어받기는 지원하지 않으며,
중단된 실행에는 `complete=false`가 남는다. 분석도 기본 `analysis/`가 이미 있으면 덮어쓰지 않는다.
재분석은 `--output .../analysis_v2`처럼 새 폴더를 지정한다. 분석에는 GPU가 필요 없다.

결과는 한 실행 폴더 안에 모인다.

```text
attention_comparison/<run>/
  manifest.json                 실행 설정, 표본 ID/경로, 모델·코드·청크 지문, 완료 여부
  data/chunk_*.npz              같은 실행의 전체 이미지·logits·예측·헤드별 attention·GT
  analysis/
    *_success_failure_by_layer.png  12개 레이어의 성공/실패 중앙값과 IQR
    maps/                          선정 사례의 전체 12레이어 clean / attack / 차이 맵
    layer_details/                 후보 레이어와 앞뒤 맵, 사례별 레이어×헤드 지도
    samples.csv                    전체 표본의 정답·예측·성공/실패·GT 크기
    per_sample_layer.csv           전체 표본의 레이어별 TV·GT/비GT/CLS 증감
    per_sample_head.csv            전체 표본의 레이어·헤드별 GT 증감과 TV
    success_failure_summary.csv    성공/실패별 표본 수·사분위 통계
    layer_candidates.csv           레이어 후보와 선정 근거
    selected_examples.csv          사례별 선정 이유
    analysis_settings.json         입력/분석 코드 지문과 공통 색 범위
    README.md                      결과 해석 기준
```

### 맵과 수치의 정의

맵은 **CLS query가 각 patch key에 주는 attention**이다. softmax 이후 헤드 평균을 사용하고,
CLS key의 질량을 제거하거나 patch 부분을 재정규화하지 않는다. 맵에서 CLS 칸은 생략하지만
CLS 변화량은 별도로 표시한다. 모든 공간 맵의 선형 색 범위는 공격·성공/실패·표본·레이어에 걸쳐
동일하다. 차이 맵은 0을 중심으로 증가(빨강)와 감소(파랑)를 표시한다. 사진은 첫 행에만 있고,
아래에는 사진을 겹치지 않은 맵을 배치한다. 청록색 테두리는 GT 토큰이다.

- **TV**: CLS를 포함한 `0.5 × Σ|attack − clean|`. 맵 전체의 변화 크기.
- **GT 증감**: coverage > 0인 토큰으로 들어가는 raw attention의 변화량 합.
  부분적으로 변경된 토큰도 전체 토큰으로 포함한다. coverage로 가중한 값은 별도 보조 지표다.
- **보존 확인**: GT 영역 + 나머지 patch + CLS 증감 합은 0이어야 한다(`atol=1e-4`, `rtol=0`).
- 집단 그림은 중앙값/IQR이며 신뢰구간이나 유의성 검정이 아니다. 빈 성공/실패 집단도 n=0으로 표시한다.

### 특이 레이어를 발견한 뒤 확인하는 방법

전체 clean 정답 표본에서 TV 중앙값이 가장 큰 레이어, 이전 레이어보다 TV 중앙값이 가장 많이
증가한 레이어, 성공/실패 GT 증감 중앙값 차이가 가장 큰 레이어를 탐색 후보로 표시한다.
표본이 없는 집단의 비교는 생략한다. 후보마다 각 집단의 TV가 가장 큰 사례를 골라 앞뒤 레이어
맵과 모든 헤드의 GT 증감 지도를 저장한다. 이는 탐색용 순위이며 통계적 이상 판정이나 인과 증거가 아니다.

전체 레이어 맵의 기본 사례는 성공/실패별 평균 TV가 집단 중앙에 가까운 사례와 최대 사례다.
마지막 레이어 GT 증가 없이 성공한 사례, GT가 증가했지만 실패한 사례도 있으면 추가한다.
선정 이유는 CSV에 남기며, `--all_maps`를 붙이면 **모든 clean 정답 표본**의 전체 레이어 맵을 저장한다.
전체 표본의 수치는 옵션과 무관하게 CSV에 저장된다.

공유 PatchFool은 unbounded 픽셀 공격이고 LaVAN과 면적·최적화 예산이 다르다. 공격별 성공/실패를
비교하며 서로의 차이를 공격 목적함수 하나의 효과로 해석하지 않는다. attention 변화와 오분류의
연관성이 곧 원인이라는 주장은 하지 않는다. 출력 그림은 PNG만 사용하고 PDF·로그 파일은 만들지 않는다.
