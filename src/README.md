# 공유 모델·데이터·공격 코드

`src/`는 사전학습 모델 로드, 입력 전처리, 공격 이미지 생성을 제공한다. 공격의 gradient 계산은 **입력 이미지를 최적화**하는 과정이며 모델 가중치 학습과 다르다. 공통 GT 표시·데이터 준비도 제공하며, 탐지기 구현은 해당 실험 폴더에 둔다.

## 구성

| 파일 | 역할 |
|---|---|
| `models.py` | P8/P16 체크포인트, 장치 선택, eval 모델 로드 |
| `dataset.py` | ImageFolder, 고정 전처리, seed별 표본 선택 |
| `data/` | 공통 데이터 경로·이미지/어텐션 읽기·공격본 준비·어텐션 전용 추출 |
| `ground_truth/` | 공통 위치 GT 요약·이미지 표시 (탐지 입력에 전달하지 않음) |
| `visualization/` | 공통 격자·박스·색상·fig 저장 |
| `cohort.py` | 01/02 공통 500장 ID·경로·모델 계약과 loader (`../data/gt/imagenet500.json`) |
| `provenance.py` | 파일 SHA256·모델 state_dict 지문 |
| `patches.py` | 패치/영역 좌표와 실제 변경 coverage |
| `attention.py` | timm 공식 AttentionExtract, 한 forward의 logits·attention, hook/설정 복원 |
| `attack_utils.py` | 모델 모드/RNG, 마스크·정규화·손실, PGD, 미분 가능한 Q/K hook |
| `attacks/pgd.py` | 공유 전역 L∞ PGD, 픽셀 공간 epsilon/alpha |
| `attacks/patch_fool.py` | attention 기반 위치 선택, Adam + CE/attention 손실 |
| `attacks/lavan.py` | 정사각 패치 하나의 sign-gradient 최적화 |
| `attacks/attention_fool.py` | pre-softmax Q/K 집중 목적함수 + 분류 손실 |
| `attacks/__init__.py` | 세 공격의 공개 import와 반환 규약 |

실행 흐름은 `모델 로드 → 데이터 로드 → 공격 함수 → 호출부 평가/시각화`다. 고정 paired 표본은 `src/cohort.py`와 루트 `data/gt/imagenet500.json`, 영역 탐지·방어는 `02_detection_defense/region_switch/`에서 관리한다.

## 모델과 데이터 계약

입력은 정규화된 `B×3×H×W`, 정답은 `B` 크기 클래스 인덱스다. 모델은 224×224 입력을 쓴다. 전처리는 mean/std=0.5, bicubic Resize 248, CenterCrop 224이며 `_TRANSFORM`에 정의돼 있다.

```python
from src.models import load_vit_model
from src.dataset import get_dataloader

model = load_vit_model(16, 'cpu', strict=True)
loader, dataset = get_dataloader(batch_size=4, num_samples=100, seed=42,
                                 dataset_root='/dataset/imagenet/val', num_workers=0)
```

`load_vit_model`의 기본 동작은 오류 메시지와 None 반환이다. `strict=True`는 원래 예외를 전달한다. 지원 크기는 P8/P16다.

표본 선택은 **이미지 단위 무작위 비복원 추출**이며 클래스별 같은 수를 선택하지 않는다. 같은 seed이면 batch 크기가 달라도 같은 순서다. 데이터 경로·worker·pin_memory를 keyword로 지정할 수 있다.

## 공격별 차이

| 공격 | 변경 허용 영역 | 예산/step 단위 | 반환 |
|---|---|---|---|
| PGD | 전체 이미지 | `epsilon`, `alpha`: **[0,1] 픽셀 공간** | `adv` |
| PatchFool | 모델 token 단위, 다중·고정 sparse 지원 | `attack_lr`, `mild_l_inf`: **정규화 공간** | `(adv, 공격 전 attention 목록)` |
| LaVAN | 면적 비율의 정사각 패치 하나, 배치 내 위치 공유 | `alpha`: **[0,1] 픽셀 공간** | `adv` |
| Attention-Fool | 공통 엔진의 픽셀 마스크/정사각 영역 | `eps`, `alpha`: **[0,1] 픽셀 공간** | `adv` 또는 `(adv, info)` |

std=0.5에서 PatchFool `mild_l_inf=0.2`는 픽셀 eps=0.1이다. PatchFool 기본 `clip_valid_range=True`는 유효 픽셀 범위를 유지한다. 무제한 실험은 `clip_valid_range=False`를 명시한다. 다른 공격과 비교할 때 면적·픽셀 범위·반복 수도 확인해야 한다.

호출 중에는 eval로 계산하고 반환 시 원래 모델 모드를 복원한다. 모델 가중치·기존 `.grad`를 바꾸지 않고 입력 gradient만 계산한다. 명시적 seed를 받는 공통 엔진은 CPU/CUDA RNG를 격리한다. seed=None이면 전역 RNG가 진행하며 MPS의 명시적 seed 격리는 지원하지 않는다.

`info['mask']`는 변경 **허용** 영역이다. 실제 변경 위치는 원본-공격본 차이로 계산한다. 정상부터 오답인 표본은 일반적인 공격 성공률의 분모에서 제외한다.

## PatchFool 실행 파라미터

기본은 공식 README의 vanilla 예시처럼 `patch_select='Attn', num_patch=1, attack_mode='Attention'`이다. 위치 점수는 전체 query에서 들어오는 attention 평균(`attention_source='incoming'`), 초기화는 유효 범위의 무작위 패치 교체(`initialization='replacement'`)다. 모델은 `src.models.load_vit_model`에서 로드한다(공식 저장소의 DeiT 구현을 쓰지 않는다).

| 바꿀 내용 | 함수 옵션 | 실험 CLI |
|---|---|---|
| 위치 선택 | `patch_select='Attn'/'Rand'/'Saliency'/'Contiguous'` | `--pf_patch_select` |
| 변경 token 수 | `num_patch=4` | `--pf_num_patch 4` |
| CE만 사용 | `attack_mode='CE_loss'` | `--pf_attack_mode CE_loss` |
| Mild L∞ | `mild_l_inf=0.2` | `--pf_mild_l_inf 0.2` |
| 고정 Sparse | `sparse_pixel_num=250` | `--pf_sparse_pixel_num 250` |
| 픽셀 범위 해제 | `clip_valid_range=False` | `--no-pf_clip_valid_range` |
| CLS 기반 위치 점수 | `attention_source='cls'` | `--pf_attention_source cls` |
| 작은 Gaussian 섭동 초기화 | `initialization='delta'` | `--pf_initialization delta` |

`Contiguous`는 고정 16×16px 면적으로 token 수를 정하므로 num_patch보다 실제 영역이 우선한다. paired 데이터 추출 스크립트는 `--pf_attention_source`, `--pf_initialization`, `--pf_clip_valid`/`--no-pf_clip_valid`로 기록한다.

## 설계상 한계

- **체크포인트 혼입 변수:** P8/P16는 서로 다른 사전학습 가중치이다. 성능 차이를 patch 크기 하나의 효과로 단정할 수 없다.
- **PatchFool 다중 패치:** attention 손실의 target은 첫 선택 token 하나다. 나머지 영역도 CE로 최적화되지만 모든 token에 같은 attention 유도를 적용하지는 않는다.
- **PatchFool Rand:** 중복 token을 선택할 수 있어 실제 허용 면적이 num_patch보다 작아질 수 있다. 면적 통제 실험에서는 실제 mask/변경 면적을 확인해야 한다.
- **공격별 의미론:** LaVAN은 배치 내 위치를 공유한다. PatchFool·LaVAN은 최종 iterate를 반환하고, 공통 엔진 기반인 Attention-Fool은 성공 여부를 우선한 최선 후보를 반환한다.
- **메모리:** PatchFool은 모든 레이어의 full attention을 수집하고 선택용은 CPU로 복사한다. P8·큰 batch에서 RAM·VRAM 사용량이 크다.
- **지원 범위:** PatchFool·LaVAN의 유효 범위는 mean/std=0.5 모델용이다. 공통 엔진은 mean/std를 받지만 attention 공격은 CLS 1개인 일반 ViT를 전제로 한다. Swin·register token·동적 격자는 지원하지 않는다.
- PatchFool은 [공식 main.py](https://github.com/GATECH-EIC/Patch-Fool/blob/main/main.py)의 위치 선택·초기화·기본 최적화 흐름을 따른다. timm 체크포인트, PCGrad, 정규화 공간의 Mild 예산, 고정 Sparse/Contiguous 확장을 쓰므로 공식 실험 전체의 재현은 아니다.

## 검증

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_src_contracts.py' -v
```

작은 무작위 ViT로 PCGrad, 실제 attention과 Q/K hook 일치, 예산·수정 영역, 모드·gradient·가중치 보존, 실패 시 hook 해제, seed 재현성과 데이터 표본 순서를 검사한다. 가중치 다운로드·대규모 공격 생성은 하지 않는다. CPU 검증이며 CUDA/MPS는 별도 확인이 필요하다.

실험별 그림·탐지 지표·방어 평가 루프는 `src/`로 올리지 않는다. 02에서만 공유하는 평가는 `02_detection_defense/defense_common/`에 둔다. 새 PatchFool 실행 설정은 `patch_fool_config(**overrides)`로 기본값까지 기록한다. 과거 실행의 설정을 현재 기본값으로 추정하지 않는다.

00·01·02의 공격 호출부는 `src.attacks`의 공개 함수만 가져온다. 02의 기본 방어 평가는 공통 archive를 읽고, 직접 공격 비교는 이 공개 PatchFool을 호출한다. 같은 함수 사용과 동일 공격 파라미터 사용은 구분하며, 과거 archive가 현재 코드로 재생성됐다는 뜻은 아니다.

`pcgrad_mode="official"`은 공식 Patch-Fool 저장소의 norm 분모를 선택한다. 기본 `projected`는 기존 norm² 동작을 유지한다. 02의 분리 P16 공격 프로토콜은 공식 모드를 명시하며 이전 00 결과와 설정 차이를 기록한다.
