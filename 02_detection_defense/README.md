# 02: 패치 공격 탐지와 방어

파일은 공통 평가, 방어별 구현, 추가 실험, 결과로 나눈다. 공격 데이터 준비는 공통 src/data/에서 관리한다. 한 방어에만 필요한 코드는 그 방어 폴더 안에 두고, 두 방어가 함께 쓰는 평가는 `defense_common/`에서 관리한다.

```text
02_detection_defense/
├── defense_common/             # v1·v2 공통
│   ├── evaluation.py           # 지표·혼합 입력 평가
│   └── plotting.py             # 기본 정확도 막대그래프
├── region_switch/              # 버전 1: 탐지 사분면만 고정 P8 분할
│   ├── run.py                  # 기본 방어 평가 CLI
│   ├── visualize.py            # 저장된 수치 → 영어 막대그래프
│   ├── core/
│   │   ├── attention.py        # 어텐션 특징·TV 기반 탐지
│   │   ├── switch.py           # P16→P8 커널·토큰·면적 가중치
│   │   └── inference.py        # 이미지 전용 추론·고정 변환 검증
│   ├── experiments/
│   │   ├── compare_defenses.py # 무작위 위치·적응형 비교
│   │   ├── compare_defenses.sbatch
│   │   ├── compare_models.py   # P8/P16 직접 공격 비교
│   │   ├── model_support.py    # 모델 비교 전용 설정·그리기
│   │   └── verify_cases.py     # 기본 실험 archive의 개별 사례 확인
│   └── results/                # PNG와 최종 JSON
└── region_switch_v2/            # 별도 무작위 응답 방어
    ├── run.py
    ├── randomize.py
    ├── run.sbatch
    └── results/
```

## 기본 실험

1. 방어 없는 원본 ViT-B/16에 PatchFool을 적용한다. P16 격자에서 incoming attention으로 **16×16 패치 하나**를 선택한다.
2. 고정 500장의 공격본과 원본 출력·설정·해시를 루트 `data/attacks/`에 저장하고 공격 프로세스를 종료한다.
3. P16 모델을 새로 로드해 이미지에서만 탐지한다. 정답·공격 위치·공격 여부·대응 정상 이미지는 추론에 전달하지 않는다.
4. 탐지한 112×112 사분면의 P16 토큰 49개를 P8 토큰 196개로 바꾼다. 나머지 147개 P16은 유지하고, 미탐지 시 원본 P16 예측을 유지한다.

학습·보정은 없다. P8 임베딩은 P16 가중치의 고정 변환이며 별도 사전학습 ViT-B/8이 아니다. 새 키의 면적 가중치는 1/4이다.

공격은 공식 vanilla 흐름을 timm ViT에 적용한 설정이다: Attn, 1 patch, CE+attention(weight .002), layer 4(0-based), 250회, Adam lr .22, StepLR(10, .95), replacement 초기화, 정상 픽셀 범위, Sparse/Mild 없음, 공식 norm 분모 PCGrad. 공식 DeiT 모델·전처리까지 동일한 논문 수치 재현은 아니다. 00/01의 PCGrad 기본값은 변경하지 않는다.

## 실행: 저장된 공격본 재사용

02에는 공격 생성 코드가 없다. 기본 평가 경로는 `src/data/paths.py`의 `data/attacks/patchfool_p16_official/`이다. 공격 이미지와 GT는 공통 `data/`에서 가져오며 GT를 탐지기에 전달하지 않는다.

```bash
python 02_detection_defense/region_switch/run.py --device cuda --output /tmp/defense.png
# 다른 저장 데이터셋을 명시하려면 --run 사용
python 02_detection_defense/region_switch/run.py --run data/attacks/patchfool_p16_official --output /tmp/defense.png
```

공격본이 없거나 설정을 바꿀 때만 공통 준비 코드 `python src/data/generate_patchfool.py --output data/attacks/patchfool_p16_new`를 별도로 실행한다. 평소 방어 실행에서는 공격을 생성하지 않는다. 데이터 구조·GT 의미·01과의 공격 설정 차이는 [data/README.md](../data/README.md)를 참고한다.

## 결과와 그래프

기본 500장 결과는 정상 83.8% / 공격 후 32.0% / 방어 후 78.0%다. Clean은 방어 없는 원본 모델이다. 결과 JSON은 사용자 요청으로 현재 삭제한 상태이며, 새 평가를 실행하면 다시 기록한다. 공격본은 `data/attacks/patchfool_p16_official/`이며 중복 `latest_run.json`은 두지 않는다.

```bash
python 02_detection_defense/region_switch/visualize.py
```

`results/`에 영어 막대그래프를 출력한다.

- `comparison.png`: Clean → Attacked → Defended.
- `localization_ablation.png`: 같은 탐지 발동 조건에서 무작위 영역 58.0%와 탐지 영역 78.0% 비교.
- `detector_metrics.png`: TPR / FPR / 영역 적중률. 분모는 전체 공격본 / 정상 입력 / 탐지된 공격본.
- `detection_pipeline.png`: 설명용으로 수동 생성한 탐지 과정 그림. 생성 코드는 보관하지 않는다.

텍스트 실행 로그·PPT 생성 코드·표 이미지는 보관하지 않는다. 그래프 재생성은 새 공격을 실행하지 않는다.

## 추가 실험

`experiments/`는 기본 실행에서 자동 호출하지 않는다. `compare_defenses.py`의 기본 실행은 기존 공격본의 무작위 위치 비교만 한다. 적응형 공격은 `--adaptive --adaptive-run 새경로`를 둘 다 지정할 때만 생성하며 기본 그래프에는 포함하지 않는다.

```bash
# 현재 기본 실험의 실제 이미지 사례
python 02_detection_defense/region_switch/experiments/verify_cases.py --output /tmp/case_studies.png
# 별도 모델별 공격 비교
python 02_detection_defense/region_switch/experiments/compare_models.py --output /tmp/model_comparison.png
# 무작위 위치 및 적응형 비교: 코드/설정이 달라졌으면 별도 새 archive 사용
python 02_detection_defense/region_switch/experiments/compare_defenses.py --adaptive --adaptive-run data/attacks/patchfool_p16_adaptive_fresh
```

공격·코드 지문이 다른 과거 archive를 새 실행 결과로 재사용하지 않는다. 기존 archive의 지문은 당시 실행의 기록으로 유지한다. v2는 무작위 세분화·병합을 포함하므로 기본 버전 1의 고정 P8 분할 실험과 구분한다.
