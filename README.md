# ViT_robust — ViT 패치 공격 강건성 연구 (프로젝트 모음)

`00_patch_size_tradeoff`/`01_patch_attack_detector`/`02_patch_switch_defense` 세 프로젝트가
사는 상위 폴더. 폴더 번호는 **지금 시점에 진행하는 순서**를 뜻한다 — 역사적으로 만들어진
순서(tradeoff → switch_defense → detector가 switch_defense에서 분리)와는 다르다:

```
00_patch_size_tradeoff/    이미 끝난 기초 연구 — 패치 크기(P8/P16/P32) vs 강건성
                            정식 실험(PGD/LaVAN/PatchFool). "PatchFool에 대해 P8이
                            P16보다 압도적으로 강건함"이라는 발견이 다음 단계의 출발점.
01_patch_attack_detector/  지금 진행 중 — raw-attention 기반 탐지기(Top-K Mass)를
                            레이어별로 특성 분석하며 독립적으로 발전시키는 중.
02_patch_switch_defense/   탐지기가 붙는 실제 방어(P16→P8 국소/전체 전환) — 탐지기가
                            안정되기 전까지 평가 실험은 보류(아래 "각 프로젝트 현재 상태"
                            참고), 방어 메커니즘 자체(defense/)는 계속 다듬는 중.
```

원래 만들어진 순서는 `00_patch_size_tradeoff`(2026-09-15 이전) → `02_patch_switch_defense`
(2026-09-15, tradeoff의 발견을 방어로 발전) → `01_patch_attack_detector`(2026-09-21,
patch_switch_defense의 탐지기 부분만 분리)다. 각 프로젝트의 배경·실험·결과는 각자의
README 참고. 이 문서는 세 프로젝트에 걸친 것만 다룬다.

## 각 프로젝트 현재 상태 (2026-09-\*)

- **`00_patch_size_tradeoff`**: 완결. 7개 정식 실험 + Protocol C, 재실행 계획 없음.
- **`01_patch_attack_detector`**: 활성 개발 중. `detector/topk_mass_v1.py`(raw attention
  L=12, top4_mass)가 현재 실배포판이고, `experiments/`에서 레이어별 sweep·attention 특성
  관찰을 계속하고 있다. 자세한 내용은 `01_patch_attack_detector/README.md`.
- **`02_patch_switch_defense`**: 방어 메커니즘(`defense/all_switch.py`,
  `defense/local_switch.py`) 코드는 계속 다듬는 중(예: local_switch의 P8 patch_embed
  최적화, `defense/local_switch.py` 참고). **평가 실험(`experiments/`)은 전부 비웠다** —
  system_comparison/cost_comparison 등 이전 실험들이 전부 탐지기(`01_patch_attack_detector`)
  와 엮여 있었는데, 그 탐지기가 아직 안정되지 않아서 지금 나온 숫자는 다시 재현해야 할 게
  뻔하기 때문이다(`02_patch_switch_defense/experiments/README.md`에 자세한 설명, 지운 실험은
  git 이력에서 복구 가능). 탐지기가 자리 잡으면 같은 방법론(같은 calibration/eval/탐지
  판정으로 두 방어 직접 비교)으로 재개할 예정.

## 공유 `src/` (모델/데이터셋/공격)

`models.py`(timm ViT 로드), `dataset.py`(ImageNet-1k val 로더), `attacks/{patch_fool.py,
lavan.py}`는 세 프로젝트가 완전히 동일한 코드를 쓴다. 예전에는 각 프로젝트가 이 코드의
사본을 자기 안에 갖고 있었다(`00_patch_size_tradeoff`가 원본, 나머지 둘은 "형제 프로젝트에
의존하지 않는다"는 원칙에 따라 매번 새로 복사). 세 사본이 정말로 완전히 같은 내용이라
한쪽을 고치면 나머지 둘도 일일이 따라 고쳐야 하는 게 비효율적이라 판단해, 이 위치
(`ViT_robust/src/`)로 합쳤다. 세 프로젝트 모두 이제 `sys.path`에 `ViT_robust`(이 폴더)를
추가해 `from src.models import ...` 식으로 직접 import한다.

**PGD/`metrics.py`는 공유하지 않는다** — `00_patch_size_tradeoff`만 PGD 공격과 CA/RA/ASR
지표 계산을 쓰고(나머지 둘은 범위 밖) 여러 프로젝트가 실제로 공유하는 코드가 아니기
때문이다. `00_patch_size_tradeoff/pgd.py`, `00_patch_size_tradeoff/metrics.py`로 그 프로젝트
루트에 그대로 남아있다(공유 `src/`가 된 이상 이 프로젝트만의 파일을 그 안에 둘 수 없어서
밖으로 뺐다).

`MODEL_NAMES`에 P32가 남아있는 건 `00_patch_size_tradeoff`가 P8/P16/P32를 다 쓰기 때문이다
— 나머지 두 프로젝트는 P8/P16만 쓰고 P32를 그냥 안 부른다(코드로 막아둔 제약이 아니라 그
프로젝트들의 실험 스크립트가 32를 요청하지 않는 것뿐).

## `detector/`는 공유하지 않는다 (01_patch_attack_detector 소유, 의도적인 예외)

탐지기(`01_patch_attack_detector/detector/topk_mass_v1.py`)는 `src/`와 달리 세 프로젝트가
나란히 갖는 게 아니라 `01_patch_attack_detector` 하나가 소유하고 `02_patch_switch_defense`
만 cross-import한다(`00_patch_size_tradeoff`는 탐지기 개념 자체가 없음). `src/`를 합친 것과는
반대 방향의 결정처럼 보이지만 이유가 다르다 — `src/`는 "완전히 같은 내용을 세 번 유지하는
게 낭비"라서 합친 것이고, `detector/`는 애초부터 "탐지기 자체를 독립적으로 여러 버전으로
발전시키기 위해" `01_patch_attack_detector`라는 프로젝트를 따로 만든 것이다
(`01_patch_attack_detector/README.md` 참고). 자세한 import 방식은
`02_patch_switch_defense/README.md`의 "탐지기가 별도 프로젝트로 분리됨" 절 참고.

