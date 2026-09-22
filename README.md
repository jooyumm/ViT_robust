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

## 폴더 이름 이력

`ViT_tradeoff`/`ViT_patchSwitch`/`ViT_detect` → `patch_size_tradeoff`/`patch_switch_defense`/
`patch_attack_detector`(각 프로젝트가 실제로 하는 일을 이름에 반영, "ViT_" 접두어 제거) →
`00_patch_size_tradeoff`/`01_patch_attack_detector`/`02_patch_switch_defense`(지금 진행
순서를 번호로 반영) — 두 차례 이름을 바꿨다. 코드/문서 안의 하드코딩된 경로 문자열(예:
`os.path.join(SHARED_SRC_ROOT, '01_patch_attack_detector')`)도 매번 같이 고쳤고, 바꿀
때마다 영향받는 스크립트를 `--help`(또는 전체 실행)로 재검증했다.

## git 이력

`00_patch_size_tradeoff`/`01_patch_attack_detector`/`02_patch_switch_defense`/`src`는 원래
각자 독립된 git 저장소였다. 2026-09-\*에 GitHub에 단일 저장소로 올리기 위해 이 폴더
(`ViT_robust`) 자체를 새 git 저장소로 통합했다 — 네 저장소의 커밋 이력을 병합하는 대신
(이 환경에 git-subtree/git-filter-repo가 없어서) 새 이력으로 시작했다. 각 프로젝트의 이전
이력은 `<프로젝트>/.git.bak/`에 저장소 형태 그대로 로컬 백업돼 있다(이 저장소에서는
`.gitignore`로 제외, 삭제 안 함 — 필요하면 `.git.bak`를 `.git`으로 다시 이름 바꾸면 그
프로젝트만 예전 이력으로 되돌릴 수 있다).

## 회귀 테스트

`src/`를 공유로 합친 리팩터링은 29개 파일(각 프로젝트가 자기 프로젝트 루트 기준으로
`from src.X import ...`하던 것 전부)의 import 경로를 직접 고치는 방식으로 했다(심볼릭 링크
대신 — 명시적인 게 낫다는 판단). 검증은 두 단계로 했다:
- **전체 파일**: `python <script>.py --help`(또는 --help 없이 바로 실행되는 스크립트는
  전체 실행)로 import 단계까지 통과하는지 확인 — 실제로 `src/attacks/__init__.py`가 여전히
  (더 이상 거기 없는) `pgd.py`를 참조하던 버그를 이 단계에서 잡았다.
- **알려진 수치가 있는 실험**은 GPU로 실제로 재실행해 리팩터링 전후 수치가 완전히 동일한지
  확인: `system_comparison`(n=250, 이후 02_patch_switch_defense의 experiments/ 정리로
  삭제됨 — 삭제 전 마지막으로 재확인한 수치는 git 이력에 남아있음), `01_patch_attack_detector`
  의 `layer_sweep`(n=250)과 `00_extract_attention`(n=100). 폴더 이름을 두 번 바꾸는 동안에도
  매번 다시 확인해서 항상 리팩터링 전과 완전히 동일한 수치가 나왔다.
- `defense/local_switch.py`의 P8 patch_embed 최적화(784개 전부 계산 → flag된 4개만 계산)는
  전용 검증 스크립트로 최적화 전후 출력이 **완전히 동일**(최대 절대오차 0.0)함과, 미분
  가능 경로(`local_swap_logits`)의 gradient가 여전히 정상적으로 흐름을 확인했다.
