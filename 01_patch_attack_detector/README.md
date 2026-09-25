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
중단된 실행에는 `complete=false`가 남는다. 분석에는 GPU가 필요 없다.

추출 결과(재현에 필요한 원본)와 분석 결과(사람이 보는 요약)는 폴더를 분리한다 — 실행마다
`analysis/` 폴더가 새로 쌓이면 결과 폴더가 계속 깊어지기 때문에, 분석 출력은 실행 폴더 밖의
`results/characterization/`에 공격별로 하나씩만 모은다(재실행하면 그 자리를 덮어쓴다).

```text
results/characterization/
  attention_comparison/<run>/
    manifest.json                실행 설정, 표본 ID/경로, 모델·코드·청크 지문, 완료 여부
    data/chunk_*.npz             같은 실행의 전체 이미지·logits·예측·헤드별 attention·GT
  README.md                      결과 해석 기준 + 공격별 레이어 요약표(마크다운, CSV 없음)
  <attack>/                      patchfool/, lavan/ 각각 — 하위 폴더 없이 파일 몇 개뿐
    layer_comparison.png         12개 레이어의 attack_success/attack_fail 중앙값과 IQR
    example_attack_success_sample<id>.png  공격 성공(오분류) 대표 사례 1장 — 12개 레이어 전부
    example_attack_fail_sample<id>.png     공격 실패(여전히 정답) 대표 사례 1장 — 12개 레이어 전부
```

파일 이름에 `attack_success`/`attack_fail`을 넣은 이유: "success/failure"만 쓰면 공격이
성공했다는 건지 분류가 실패했다는 건지 헷갈린다는 지적을 반영해, 무엇이 성공·실패했는지 이름
자체에 남겼다 — 폴더를 따로 나누지 않고 파일명만으로 구분되게 했다. 대표 사례 그림도 공격당
2장(성공/실패 각 1장)만 남기고 CSV는 만들지 않는다. 전체 표본·레이어·헤드의 원값이 다시
필요하면 `attention_comparison/<run>/data/`와 이 스크립트의 `compute_metrics()`로 그때그때
다시 계산한다.

### 맵과 수치의 정의

맵은 **CLS query가 각 patch key에 주는 attention**이다. softmax 이후 헤드 평균을 사용하고,
CLS key의 질량을 제거하거나 patch 부분을 재정규화하지 않는다. 맵에서 CLS 칸은 생략하지만
CLS 변화량은 별도로 표시한다. 사진 행을 포함한 모든 행이 같은 3열 grid를 쓰고, 컬러바는
사진 행까지 포함한 열 전체에서 폭을 빌려온다 — 컬러바가 붙는 마지막 행만 좁아져서 다른 행과
어긋나 보이는 문제를 막기 위해서다. raw attention 히트맵은 흰색→순수 파랑, attack−clean은
연초록→초록(감소), 흰색(변화 없음), 연빨강→빨강(attention이 더 몰림)이다. 작은 변화도 흰색과
구분되도록 |값|^0.5로 펼친 스케일이라 컬러바 눈금이 비선형이다(작은 변화가 실제보다 커 보일 수 있다). 양 끝이 검정으로 가면 패널끼리 구분이
안 되어서 끝 색을 원색으로 뒀다. 색 범위는 같은 공격의 12레이어 전체를 모아 계산한
99.5퍼센타일 상한을 쓴다 — 극단적인 표본 하나가 스케일을 지배하지 않게 하면서도, 초반은 옅고
후반으로 갈수록 진해지는 실제 변화는 그대로 드러난다. 상한을 넘는 값은 컬러바의 화살표로
표시하고 맨 끝 색으로 그린다(값 자체를 바꾸지 않는다). GT 토큰 테두리는 `ground_truth/draw_boxes.py`의
규약을 그대로 따른다 — clean 패널은 파랑(`CLEAN_BOX_COLOR`), 공격 후/차이 패널은 노랑
(`ATTACK_BOX_COLOR`). 박스 그리는 `draw_patch_box`와 색 상수를 거기서 import해서 쓴다.

- **TV**: CLS를 포함한 `0.5 × Σ|attack − clean|`. 맵 전체의 변화 크기.
- **GT 증감**: coverage > 0인 토큰으로 들어가는 raw attention의 변화량 합.
  부분적으로 변경된 토큰도 전체 토큰으로 포함한다.
- **보존 확인**: GT 영역 + 나머지 patch + CLS 증감 합은 0이어야 한다(`atol=1e-4`, `rtol=0`).
- 집단 그림은 중앙값/IQR이며 신뢰구간이나 유의성 검정이 아니다. 빈 집단도 n=0으로 표시한다.

### 특이 레이어를 발견한 뒤 확인하는 방법

전체 clean 정답 표본에서 TV 중앙값이 가장 큰 레이어, 이전 레이어보다 TV 중앙값이 가장 많이
증가한 레이어, 성공/실패 GT 증감 중앙값 차이가 가장 큰 레이어를 탐색 후보로 표시한다(`results/characterization/README.md`의
표에 ⚠로 표시). 표본이 없는 집단의 비교는 생략한다. 이는 탐색용 순위이며 통계적 이상 판정이나
인과 증거가 아니다. `example_attack_success_*`, `example_attack_fail_*`의 대표 사례는 각 집단의
평균 TV가 중앙값에 가장 가까운 표본이며, 12개 레이어 전부에서 clean/attack/차이 맵을 보여준다 —
표시된 ⚠ 레이어는 어디부터 눈여겨봐야 하는지 알려주는 표시일 뿐, 다른 레이어를 생략하지 않는다.

공유 PatchFool은 unbounded 픽셀 공격이고 LaVAN과 면적·최적화 예산이 다르다. 공격별 성공/실패를
비교하며 서로의 차이를 공격 목적함수 하나의 효과로 해석하지 않는다. attention 변화와 오분류의
연관성이 곧 원인이라는 주장은 하지 않는다. 출력 그림은 PNG만 사용하고 PDF·로그 파일은 만들지 않는다.
