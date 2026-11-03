# 01_attention_characterization

PatchFool 공격이 ViT-B/16의 attention에 남기는 흔적을 레이어·영역 단위로 살펴보는 탐색 실험 모음. 모든 하위 실험은 루트 `../data/gt/imagenet500.json`이 고정한 같은 500장 clean/공격 쌍을 입력으로 쓴다.

## 공유 인프라

- `src/cohort.py`: 고정 500장 ID·원본·모델 계약과 로컬 원본 loader.
- `src/data/paths.py`: 원본·공격본·GT·어텐션의 공통 경로.
- `src/data/paired.py`: 저장된 이미지·어텐션·GT 읽기 및 해시 검증.
- `src/data/extract_attention.py`: **저장된 공격본에서 어텐션만 추출**. 공격을 새로 만들지 않는다.
- `src/patches.py`, `src/ground_truth/`: 공통 위치 GT 계산·요약·표시. 01 전용이 아니다.
- `src/visualization/`: 공통 격자·박스·색상·fig 저장. 01 전용 어텐션 배치는 `vis/attention_maps.py`에 둔다.
- `src/attention.py`: timm 공식 `AttentionExtract`로 실제 어텐션 추출.

01과 02는 `data/attacks/patchfool_p16_official/`의 **동일한 공격본 500장**을 사용한다. 01의 기본 어텐션 캐시는 `data/attention/p16_patchfool_official/`이며, 원본·공격 픽셀과 GT를 한 값도 바꾸지 않고 어텐션만 추가한다. 로더는 원본 공격 manifest SHA256 및 공식 PCGrad 설정을 검증해 다른 공격본을 기본 분석에 사용할 수 없게 한다. 기존 projected 캐시는 삭제했다.

```bash
# SSH 종료와 관계없는 스케줄러 작업에서 이 명령을 실행할 수 있다.
python 01_attention_characterization/refresh.py --device cuda
```

새 공격 생성 없이 어텐션 추출 → 입력·정답·예측·GT 일치 검증 → 01/02/03 분석을 수행한다. 모든 단계가 성공하면 results/를 한 번에 교체한다. 진행 상태는 `data/attention/.official_refresh/status.json`에 기록하고 텍스트 로그는 남기지 않는다.

## 01_attention_maps

입력은 `src/data/paired.py`로 검증한 공통 paired run이다. `compare_attention.py`가 공격별·결과별(성공/실패)로 clean 대비 attention 변화를 레이어별로 비교하고, `patchfool_success_split.py`가 PatchFool 성공 사례를 "공격 패치로 CLS attention이 실제로 몰린 경우"와 "안 몰린 경우"로 나눠 추가로 비교한다.

맵은 CLS-query attention을 헤드 평균해서 보여주며 CLS key로 가는 질량은 재정규화하지 않고 유지한다. GT 패치는 `src/ground_truth/`의 박스 색을 그대로 써서 표시한다. 결과는 공격별 폴더로 `results/01_attention_maps/`에 저장되며, outcome(`attack_success*`/`attack_fail*`)은 파일명으로 구분한다.

공격 성공·실패 수와 레이어별 수치는 갱신된 `results/01_attention_maps/README.md`에서 확인한다. 다른 공격본에서 얻은 이전 수치 표는 재사용하지 않는다.

TV는 0.5 × sum(|공격 - clean|)(197개 토큰, CLS 포함). GT Δ는 GT 패치에 대한 attention 변화 합(raw, 미재정규화). raw attention 맵은 흰색→파랑, 차이 맵은 초록(감소)→흰색(무변화)→빨강(증가)이며, 둘 다 같은 공격의 모든 블록이 공유하는 99.5퍼센타일 상한을 쓴다(최댓값이 아님 — 극단값 하나가 나머지를 다 묻어버리지 않도록).

## 02_layer_profiles

`final_peak.py`가 clean/PatchFool 각 조건에서 "자기 자신의 마지막 레이어(12)에서 CLS attention이 가장 높은 patch"를 조건별로 독립적으로 고르고(두 조건의 위치를 맞추지 않음), 그 patch 하나의 CLS attention 값을 레이어 1~12에 걸쳐 추적한다. raw attention은 patch 합으로 재정규화하지 않는다. 결과는 `results/02_layer_profiles/`에 있다.

- `clean_layer.png`, `patchfool_layer.png`: 각 조건의 레이어별 궤적 분포.
- `comparison.png`: 두 조건의 궤적을 겹쳐 비교.
- `examples.png`: clean 궤적의 중앙값 모양에 가장 가까운 예시 1장 + 가장 먼 예시 1장을 골라, 선택된 레이어의 attention 맵을 보여준다.

## 03_region_attention

clean과 PatchFool의 영역 간 attention을 비교한다. 224×224 이미지를 16×16 패치로 나눈 14×14 격자를 네 영역(2×2)으로 묶는다. 각 영역에는 49개 패치가 있다.

`extract_region_flow.py`는 softmax attention을 레이어·헤드별로 집계한다. `F[i,j]`는 출발 영역 i의 query 패치가 도착 영역 j의 key 패치에 주는 attention을 합하고, 출발 패치 49개에 대해 평균한 값이다. 자기 영역도 포함한다. CLS로 향하는 확률은 별도 저장하므로 F의 행 합은 1보다 작을 수 있다.

결과는 `results/03_region_attention/patchfool/`에 있다.

- `00_outcome_breakdown.png`: `outcome_breakdown.py`가 만든다. 고정 500장을 clean 정답 여부 x 공격 후 정답 여부로 나눈 배타적 Case 1-4를 행으로, Clean/Attack/Count를 열로 나열한 표다.
- `01_summary_examples.png`: 공격 성공·실패·원본 오분류 사례의 clean/공격 쌍. 왼쪽은 이미지와 영역/공격 위치, 가운데는 전체 patch query에서 각 영역으로 들어오는 attention 평균(헤드 평균), 오른쪽은 CLS가 각 영역에 주는 attention 합(헤드 평균).
- `02_layer_flow_heatmaps.png`: 12개 레이어별 4×4 행렬. 세로는 query 영역, 가로는 key 영역이며 이미지 500장과 12개 헤드를 평균한다. 첫 줄은 clean, 둘째 줄은 공격, 셋째 줄은 공격 평균−clean 평균의 원래 값 차이다. 차이 행은 별도의 색 범위를 쓴다.

시각화에는 탐지 특징·정상 기준·이상 점수·임계값을 쓰지 않는다. GT는 이미지에 공격 위치를 표시할 때만 쓴다.

### 실행

기존 추출 데이터로 그림만 생성한다. 기존 출력 폴더는 덮어쓰지 않으므로 재실행할 때 새 경로를 지정한다.

```bash
MPLCONFIGDIR=/tmp/vit-mpl python 01_attention_characterization/experiments/03_region_attention/analyze_regions.py --output /tmp/region_visualization
```

기본 입력은 `extracted/flow.npz`이며 `--data`로 바꿀 수 있다. GPU 재추론은 필요 없다. `flow_io.py`가 압축 흐름 배열을 임시 파일로 해제하고 디스크 매핑하며 종료 시 정리한다.

새 흐름을 추출할 때만 아래 명령을 쓴다.

```bash
HF_HUB_OFFLINE=1 python 01_attention_characterization/experiments/03_region_attention/verify_data.py
HF_HUB_OFFLINE=1 python 01_attention_characterization/experiments/03_region_attention/extract_region_flow.py --output-root /tmp/new_region_flow
```

`--device cpu`로 CPU를 선택할 수 있으며 기본은 CUDA다. `--save-pilot`은 첫 clean 이미지의 원시 attention/hidden states를 추가 저장한다. 원본 이미지와 공격 데이터는 보존한다.

패치 탐지·방어 코드는 [`02_detection_defense/`](../02_detection_defense/)에 있으며, 03 시각화와 별도로 실행한다.
