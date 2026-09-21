# ViT_robust — ViT 패치 공격 강건성 연구 (프로젝트 모음)

`patch_size_tradeoff`/`patch_switch_defense`/`patch_attack_detector` 세 프로젝트가 사는 상위 폴더. 각자 독립된
git 저장소이고 계보로 이어져 있다:

```
patch_size_tradeoff/      원조 — 패치 크기(P8/P16/P32) vs 강건성 정식 실험 (PGD/LaVAN/PatchFool)
      │  "PatchFool에 대해 P8이 P16보다 압도적으로 강건함"이라는 발견
      ▼
patch_switch_defense/   그 발견을 실제 방어(P16→P8 적응형 전환, local switch)로 발전 (2026-09-15 분리)
      │  raw-attention 기반 Top-K Mass 탐지기 개발
      ▼
patch_attack_detector/        그 탐지기를 방어 로직과 분리해 독립적으로 발전 (2026-09-21 분리)
```

각 프로젝트의 배경·실험·결과는 각자의 README(`patch_size_tradeoff/README.md` 등) 참고. 이 문서는
세 프로젝트에 걸친 것만 다룬다.

## 공유 `src/` (모델/데이터셋/공격)

`models.py`(timm ViT 로드), `dataset.py`(ImageNet-1k val 로더), `attacks/{patch_fool.py,
lavan.py}`는 세 프로젝트가 완전히 동일한 코드를 쓴다. 2026-09-21 이전에는 각 프로젝트가
이 코드의 사본을 자기 안에 갖고 있었다(`patch_size_tradeoff`가 원본, `patch_switch_defense`/
`patch_attack_detector`는 "형제 프로젝트에 의존하지 않는다"는 원칙에 따라 매번 새로 복사). 세 사본이
정말로 완전히 같은 내용이라 한쪽을 고치면 나머지 둘도 일일이 따라 고쳐야 하는 게
비효율적이라 판단해, 이 위치(`ViT_robust/src/`)로 합쳤다. 세 프로젝트 모두 이제
`sys.path`에 `ViT_robust`(이 폴더)를 추가해 `from src.models import ...` 식으로 직접
import한다.

**PGD/`metrics.py`는 공유하지 않는다** — `patch_size_tradeoff`만 PGD 공격과 CA/RA/ASR 지표
계산을 쓰고(`patch_switch_defense`/`patch_attack_detector`는 범위 밖), 여러 프로젝트가 실제로 공유하는
코드가 아니기 때문이다. 그래서 `patch_size_tradeoff/pgd.py`, `patch_size_tradeoff/metrics.py`로 그
프로젝트 루트에 그대로 남아있다(예전엔 `patch_size_tradeoff/src/` 밑에 있었지만, 공유 `src/`가
된 이상 이 프로젝트만의 파일을 그 안에 둘 수 없어서 밖으로 뺐다).

`MODEL_NAMES`에 P32가 남아있는 건 `patch_size_tradeoff`가 P8/P16/P32를 다 쓰기 때문이다 —
`patch_switch_defense`/`patch_attack_detector`는 P8/P16만 쓰고 P32를 그냥 안 부른다(코드로 막아둔 제약이
아니라 그 프로젝트들의 실험 스크립트가 32를 요청하지 않는 것뿐).

## `detector/`는 공유하지 않는다 (patch_attack_detector 소유, 의도적인 예외)

탐지기(`patch_attack_detector/detector/topk_mass_v1.py`)는 `src/`와 달리 세 프로젝트가 나란히 갖는
게 아니라 `patch_attack_detector` 하나가 소유하고 `patch_switch_defense`만 cross-import한다(`patch_size_tradeoff`는
탐지기 개념 자체가 없음). `src/`를 합친 것과는 반대 방향의 결정처럼 보이지만 이유가
다르다 — `src/`는 "완전히 같은 내용을 세 번 유지하는 게 낭비"라서 합친 것이고, `detector/`는
애초부터 "탐지기 자체를 독립적으로 여러 버전으로 발전시키기 위해" `patch_attack_detector`라는 프로젝트를
따로 만든 것이다(`patch_attack_detector/README.md` 참고). 자세한 import 방식은
`patch_switch_defense/README.md`의 "탐지기가 별도 프로젝트로 분리됨" 절 참고.

## 회귀 테스트

`src/`를 공유로 합친 리팩터링은 29개 파일(각 프로젝트가 자기 프로젝트 루트 기준으로
`from src.X import ...`하던 것 전부)의 import 경로를 직접 고치는 방식으로 했다(심볼릭 링크
대신 — 명시적인 게 낫다는 판단). 검증은 두 단계로 했다:
- **전체 파일**: `python <script>.py --help`(또는 --help 없이 바로 실행되는 스크립트는
  전체 실행)로 import 단계까지 통과하는지 확인 — 실제로 `ViT_robust/src/attacks/__init__.py`
  가 여전히 (더 이상 거기 없는) `pgd.py`를 참조하던 버그를 이 단계에서 잡았다.
- **알려진 수치가 있는 3개 실험**은 GPU로 실제로 재실행해 리팩터링 전후 수치가 완전히
  동일한지 확인: `patch_switch_defense`의 `system_comparison`(n=250), `patch_attack_detector`의
  `layer_sweep`(n=250)과 `extract_attention`(n=100). 나머지(§1~§18 등 이미 검증 끝난
  patch_switch_defense 실험들)는 GPU 재실행 비용이 커서 import 검증까지만 했다 — 코드 로직
  자체는 안 바뀌었으므로(파일 위치만 이동) 위험은 대부분 "import 경로를 잘못 고쳤는가"이고,
  그건 위 --help 검증이 잡아낸다.
