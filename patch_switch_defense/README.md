# patch_switch_defense — ViT 적응형 방어(P16→P8 폴백) 프로젝트

**"P16으로 기본 추론하다가, 공격이 의심되는 패치만 국소적으로 P8 강건성으로 전환하는 적응형
ViT 방어(local switch)"**를 설계·검증하는 독립 프로젝트다. 원래 [`patch_size_tradeoff/`](../patch_size_tradeoff/)
(패치 크기 vs 강건성 7개 정식 실험 — PGD/LaVAN/PatchFool × P8/P16/P32)에서 나온 발견("PatchFool에
대해 P8이 P16보다 압도적으로 강건함")을 실제 방어로 발전시키기 위해 2026-09-15에 분리했다.

**범위 — patch_size_tradeoff와 다른 점**: 이 프로젝트는 방어 메커니즘 자체(토큰화 격자 불일치)에
집중하므로 **PGD는 제외, patch size는 P8/P16만, 공격은 LaVAN·PatchFool만** 다룬다(P32는
방어 로직과 무관, PGD는 원래 실험에서도 전역 L∞라 RA가 전부 포화돼 분석 무의미했음) — 이건
코드로 강제한 제약이 아니라 이 프로젝트의 실험 스크립트들이 그냥 P32/PGD를 안 부르는
것뿐이다. 모델/데이터/공격 코드는 patch_size_tradeoff와 **완전히 동일**하다 — 처음엔(2026-09-15)
이 프로젝트 안에 새로 복사한 사본으로 뒀지만, 2026-09-21에 `patch_attack_detector`까지 포함해 세
프로젝트가 완전히 같은 내용을 각자 사본으로 들고 있는 게 비효율적이라 판단해
[`../src/`](../src/)(`ViT_robust` 최상위, timm 체크포인트 이름까지 동일한
[`../src/models.py`](../src/models.py))로 합쳤다(`ViT_robust/README.md` 참고). 이 프로젝트
고유 코드(`defense/`, `experiments/`)는 여전히 이 프로젝트 안에만 있다.

## 목표: Local Switch — 왜 국소 전환인가

이 프로젝트의 목표는 처음부터 **국소 전환(local switch)**이었다: 탐지기가 공격을 의심하면
이미지 전체가 아니라 **그 패치 1개만** 4개의 P8 서브패치로 국소 교체하고, 나머지 195개 토큰과
P16의 12개 transformer block은 전혀 건드리지 않고 그대로 재사용한다. 이미지 전체를 P8로 다시
돌리는 "전체 전환(all switch)"은 그 자체가 목표가 아니라, **local switch가 얼마나 저렴하면서도
동급의 강건성을 내는지 보여줄 비교 기준선**으로 검증했다. all switch는 훨씬 단순한 설계라
(별도 모델을 통째로 다시 돌릴 뿐, 두 표현을 이어붙일 정렬 메커니즘이 필요 없음) 먼저 완결적으로
검증해 "naive하지만 확실히 작동하는 상한선" 역할을 하도록 했고, local switch는 **같은 이미지·
같은 탐지 결과**로 그 기준선과 직접 맞대어 검증했다(아래 "System comparison" 참고).

| | **Local Switch** (`defense/local_switch.py`) — 목표 | **All Switch** (`defense/all_switch.py`) — 비교 기준선 |
|---|---|---|
| 전환 방식 | 의심되는 **패치 1개만** 4개의 P8 서브패치로 국소 교체 (196→200토큰), 나머지 195토큰·12개 block은 P16 그대로 재사용 | 의심되면 이미지 **전체**를 P8로 다시 분류 |
| 정렬 메커니즘 | closed-form 최소제곱 아핀 변환(768×768+bias)으로 patch_embed 레벨에서 P8 서브패치를 P16 좌표계에 투영 | 없음 (완전히 별도 모델을 그대로 돌림) |

## 핵심 성능 요약

**system_comparison** (n=250, seed=42, calibration 100/eval 150 분리 — [아래](#system-comparison-두-방어를-같은-이미지같은-탐지-결과로) 참고)에서
**같은 113개 공격-성공 이미지, 같은 탐지 결과(FPR 13.3%/recall 72.7%)**로 두 방어를 직접 비교:

| 지표 | Local Switch | All Switch |
|---|---|---|
| 복원율 (같은 113개 공격 성공 이미지 기준) | 89.4% (101/113, 95% CI [82.4%, 93.8%]) | 96.5% (109/113, 95% CI [91.3%, 98.6%]) |
| 시스템 정확도 (eval 150개 전체, 공격 상황) | **65.3%** | **66.0%** |
| clean 오탐 비용 (P16 clean 84.0% 대비) | +0.7%p (84.7%) | +4.0%p (88.0%, P8의 원래 더 높은 clean acc가 반영됨) |
| 파이프라인 추가 비용 (escalate 시, batch=1) | **2.58ms** | 6.91ms |
| 총 latency (baseline 3.73ms 포함) | 6.31ms (1.69x) | 10.64ms (2.85x) |

**naive/adaptive 공격 강건성** (§7/§8/§14/§17/§18, 아래 각 섹션 참고, seed=123/n=50로 서로 매칭됨):

| 지표 | Local Switch | All Switch |
|---|---|---|
| naive joint attack 완전 무력화 | 16.7% [8.3%, 30.6%] | 18.4% [9.2%, 33.4%] |
| **완전판 adaptive evasion worst-case** | **21.4%** [11.7%, 35.9%] (calibrated_threshold 기준; clean_max 기준 참고치 11.9%) | **15.8%** [7.4%, 30.4%] |

**결론**: 시스템 정확도(65.3% vs 66.0%)는 사실상 동률이고, naive/adaptive 공격 강건성도 모든
신뢰구간이 겹친다 — **local switch가 all switch보다 통계적으로 더 취약하다는 근거는 없다.**
그러면서도 escalate 시 추가 비용은 **2.7배 저렴하다**(2.58ms vs 6.91ms). 두 방어가 공유하는
탐지·위치특정(raw attention L=12) 성능: AUROC 0.879(clean 대비)~0.891(LaVAN 대비), 위치특정
recall@1 96.7%. 논문/PPT에 바로 쓸 수 있는 합쳐진 표·그림은
[`experiments/paper_summary/paper_summary_table.md`](experiments/paper_summary/paper_summary_table.md),
[`experiments/paper_summary/paper_summary_headline.png`](experiments/paper_summary/paper_summary_headline.png) 참고.

## 디렉토리 구조

```
ViT_robust/
  src/                              모델/데이터셋/공격 — patch_size_tradeoff/patch_switch_defense/
                                    patch_attack_detector 세 프로젝트가 공유(2026-09-21, ViT_robust/
                                    README.md 참고). PGD는 없음(patch_size_tradeoff 전용, 범위 밖)

patch_switch_defense/
  defense/                          방어(switch) 메커니즘 구현 2개뿐 — 평가 루프·결과물 없음
    local_switch.py                   AffineAdapter + 국소 교체 + P16 재통과 (미분 가능 버전 포함)
    all_switch.py                     apply_all_switch(model8, images) — P8 전체 재분류 한 줄
                                       (공유 탐지·위치특정 detector.py는 2026-09-21에 별도
                                       형제 프로젝트 ../patch_attack_detector/detector/topk_mass_v1.py로
                                       이동 — 아래 "탐지기가 별도 프로젝트로 분리됨" 참고)

  experiments/                      모든 평가/공격/비교 실험 — defense/+공유 src/를 import해서 씀
    system_comparison/                신규 — 두 방어를 같은 이미지·같은 탐지 결과로 직접 비교
    cost_comparison/                  신규 — 두 방어의 실제 파이프라인 비용(latency/FLOPs) 비교
    paper_summary/                    신규 — 위 둘 + 기존 §7/§14/§17/§18을 합친 표·그림 (GPU 실험 아님)
    local_switch/
      joint_attack/                     §17 joint attack stress test (§8 정렬 트랩 회피 검증)
      adaptive_evasion_full/            §18 완전판 adaptive evasion (all switch와 동급 검증)
    all_switch/
      joint_attack/                     §7  naive joint attack
      adaptive_evasion_full/            §14 완전판 adaptive evasion (탐지 회피 제약 포함)
    detection_localization/
      signature/                        §1  탐지 시그니처 + 레이어 스윕
      localization/                      §2  위치 특정
      evasion_robustness/                 §3  탐지기 회피 시도 견고성
      layeridx_generalization/             §4  attn_layer_idx 일반화
    diversity_diagnostic/             §8  patch size 차이 vs "다른 모델" (정렬 트랩 진단)

```

**2026-09-\* 재구성**: 결과물(그림 .png, 원자료 .npz, 로그 .txt)은 `experiments/.../*.py`
바로 옆에 저장한다 — 예전엔 `experiments/`와 완전히 같은 구조로 대응하는 별도의
최상위 `results/` 트리가 있었지만, 코드와 결과물을 오가며 찾아야 해서 오히려 헷갈린다는
피드백에 따라 실험 폴더 하나에 코드+결과가 같이 있도록 합쳤다(`defense/`는 여전히 결과물이
없는 순수 라이브러리).

`defense/`는 이제 순수 라이브러리다 — 두 파일 다 `import defense.local_switch`/
`defense.all_switch`로 가져다 쓰기 위한 것이고, 실행해서 뭔가 저장하는 스크립트가 아니다.
새로 짜는 실험(`system_comparison`, `cost_comparison`)은 이 모듈들을 직접 import해서 쓴다.
이미 검증이 끝난 기존 실험(`local_switch/joint_attack` 등)은 재검증 리스크를 피하려고 자기
안에 복사된 인라인 코드를 그대로 유지했다 — 공유 모듈로의 마이그레이션은 하지 않았다(둘 다
원본 탐지기 코드에서 나온, 결과가 완전히 동일함을 직접 검증한 코드다).

### 탐지기가 별도 프로젝트로 분리됨 (2026-09-21)

공유 탐지·위치특정 로직(`_attn_hook`/`collect_layer_attn`/`raw_at_layer`/`top4_mass`,
raw attention L=12 기반)은 `defense/detector.py`에서 형제 프로젝트
[`../patch_attack_detector/`](../patch_attack_detector/)의 `detector/topk_mass_v1.py`로 옮겼다(처음엔
`src/detector.py`였다가, `patch_attack_detector`가 탐지기를 여러 버전으로 발전시킬 걸 반영해 버전이
드러나는 이름의 전용 `detector/` 폴더로 한 번 더 옮김 — `patch_attack_detector/README.md` 참고).
목적은 레이어별 실험·Dual-Gate(LaVAN 보완 탐지, 위 "로드맵" 참고) 등 탐지기 자체의 발전을
이 프로젝트의 방어 로직(local switch/all switch)과 분리해서 독립적으로 진행하기 위함이다.
코드는 한 글자도 바뀌지 않았다 — git 이력(`git show <commit>:defense/detector.py`로 이전
두 커밋에서 확인 가능)과 함께 그대로 옮겨졌고, `patch_attack_detector`의 커밋 이력에 원본 커밋 해시가
남아있다.

**원칙이 바뀐 이력**: 2026-09-21 이전까지 이 프로젝트는 "다른 형제 프로젝트에 의존하지
않는다"는 원칙을 모델/데이터/공격 코드에도 적용해서 `src/`를 자기 사본으로 갖고 있었다.
같은 날 그 원칙은 두 갈래로 갈렸다 — (1) 모델/데이터/공격(`src/`)은 세 프로젝트가 완전히
동일한 내용을 각자 사본으로 들던 게 비효율적이라 판단해 원칙을 뒤집고
[`../src/`](../src/)(`ViT_robust` 최상위)로 합쳤다(`ViT_robust/README.md` 참고) — 이제
이 프로젝트도 `patch_size_tradeoff`/`patch_attack_detector`에 의존한다, 정확히는 셋 다 같은 상위 위치에
의존한다. (2) 탐지기(`detector/`)는 그 전부터 이미 원칙의 의도적인 예외였다 — 여러 방어
프로젝트가 공유해야 하는 단일 소스이길 원했기 때문에, `sys.path`에 `../patch_attack_detector`(ROOT
전체)를 추가해 `from detector.topk_mass_v1 import ...`로 가져온다. `defense/`의
`all_switch.py`/`local_switch.py` 자체는 탐지기를 호출하지 않는다(이미 계산된 `flag_idx`를
파라미터로만 받음)는 점은 이전과 동일하다. `system_comparison_test.py`/
`cost_comparison_test.py`/`local_switch_posenc_ablation/posenc_ablation_test.py` 3곳만
`detector/`를 import한다(공유 `src/`는 이 프로젝트의 거의 모든 실험 스크립트가 씀).
리팩터링 전후(detector.py 이동, `detector/` 재배치, 공유 `src/`로 합치기 — 세 시점 전부)
`system_comparison`(n=250, seed=42) 결과가 완전히 동일함을 회귀 테스트로 확인했다(아래
"System comparison" 절 수치 그대로 재현, job 2302715 / 2303199).

`experiments/.../*.py`가 결과를 저장할 때는 자기 파일과 같은 폴더(`HERE`)에 그대로 쓴다
— 예전엔 `/experiments/`를 `/results/`로 바꾼 별도 경로에 썼지만(위 "재구성" 참고), 지금은
그 변환 없이 제자리에 쓴다. 그래서 폴더를 옮기거나 이름을 바꿔도 결과 경로가 항상 자동으로
따라온다. 여러 실험이
같은 공격 로직(예: joint attack)을 쓸 때는 모듈을 폴더마다 **복사**해서 넣었다 — 폴더 간
import를 없애서 폴더 하나만 통째로 옮기거나 지워도 다른 실험이 안 깨지게 하기 위함이다(단,
`system_comparison`/`cost_comparison`은 예외적으로 `defense/`의 공유 모듈을 import한다 —
`defense/`는 애초에 공유되려고 만든 것이라 이 규칙의 대상이 아니다). 몇몇 `viz.py`는 다른
실험의 확정된 숫자를 (npz를 다시 열지 않고) 하드코딩된 상수로 인용한다(예: `local_switch/`의
스크립트들이 `all_switch/`의 §7/§14 수치를 참고용 상수로 가짐).

**결과 파일 이름은 `NN_설명.확장자`** 형식을 유지한다(예: `01_signature_P16.png`) — `NN`은
최초 설계 당시의 실험 번호이고, 이 문서와 코드 전체에서 그 번호로 실험을 지칭한다. `system_
comparison`/`cost_comparison`/`paper_summary`는 번호 체계 밖의 신규 실험이라 번호가 없다.

**2026-09-19/20 재구성**: `full_reclassification/`→`all_switch/`, `local_token_subdivision/`→
`local_switch/`로 폴더명을 바꿔 "local switch가 목표, all switch는 비교 기준선"이라는 서사를
이름에 반영했다. 그다음 `defense/`를 순수 방어 메커니즘 3개(`detector.py`/`all_switch.py`/
`local_switch.py`)만 남긴 라이브러리로 정리하고, 나머지 모든 평가 스크립트를 `experiments/`로
옮겨 `defense/`·`experiments/`·`results/`·`src/`가 최상위에서 나란한 형제가 되도록 재구성했다
(`git mv`로 진행해 히스토리 보존). 이 과정에서 두 가지가 드러났다: (1) 기존 §6(all_switch
최종검증)과 §16(local_switch 복원율)은 seed=42를 공유하긴 했지만 calibration/eval 경계가 서로
달라 **완전히 같은 이미지·같은 탐지 결과로 비교된 게 아니었다** (2) **local_switch는 비용을
측정한 적이 한 번도 없었다**(§10/expected_cost_analysis.py는 all_switch 전용). §6/§16/§10/
expected_cost_analysis.py는 제거했고(`git rm`, 히스토리에서 복구 가능), 그 자리를
`system_comparison`/`cost_comparison`이 대체한다 — 이번엔 두 방어가 **정확히 같은 calibration,
같은 eval 이미지, 같은 탐지 판정**을 쓴다.

## npz / 로그 정리 정책

**현재 살아있는 실험의 `.npz`는 전부 유지한다.** `viz.py`(또는 원본 실험 스크립트 자신)가
그림을 다시 그리는 데 쓰는 원자료라, 하나라도 지우면 그 실험의 그림을 재생성할 방법이 없어진다.

**`nohup_*.txt` 실행 로그는 대부분 삭제했다.** 숫자가 이미 README·npz·그림에 다 들어있어서
로그 자체는 중복이었던 것들은 지웠다. 예외 1개는 **그 실행 로그가 유일한 원자료라 보존**:
- `experiments/diversity_diagnostic/08_diversity_original_seed456_run_2143709.txt`
  — §8의 원래 seed=456 결과(52.3%)의 `.npz`가 나중에 seed=123 재실행 때 같은 파일명으로
  덮어써져서, 이 로그만 그 수치의 유일한 증거로 남음(아래 "샘플링 감사" 참고)

## System comparison — 두 방어를 같은 이미지·같은 탐지 결과로 (`experiments/system_comparison/`)

두 방어를 공정하게 비교하려면 같은 calibration, 같은 eval 이미지, 같은 탐지 판정이 필요하다는
문제의식에서 만든 실험(위 "재구성" 참고). 단일 스크립트가:

1. n=250(seed=42)을 calibration 100 / eval 150으로 분리, **같은 calibration 풀**에서 탐지
   임계값(Youden's J)과 local_switch의 아핀 어댑터를 각각 피팅.
2. eval 150개 전체에 PatchFool 공격을 걸고, 공유 탐지기로 **한 번만** flag 여부를 계산 — 이
   판정을 all_switch와 local_switch가 그대로 공유해서 쓴다.
3. (a) 무조건 복원율(공격 성공 113개에 방어를 적용했을 때 flag 여부와 무관하게 복원되는 비율,
   메커니즘 자체의 능력) (b) 시스템 정확도(flag된 것만 방어 적용, 나머지는 P16 그대로 — 실배포
   조건) (c) clean 오탐 비용(clean 이미지에 무조건 방어를 적용했을 때의 정확도 변화)을 두
   방어에 대해 동일한 방식으로 계산.

**결과** (job 2292052, RTX 4090): threshold=0.5116, FPR=13.3%(20/150, [8.8%,19.7%]), recall=72.7%
(109/150, [65.0%,79.2%]).

- **(a) 복원율** (같은 113개 공격-성공 이미지): all_switch **96.5%**(109/113, [91.3%,98.6%]),
  local_switch **89.4%**(101/113, [82.4%,93.8%]) — 신뢰구간이 겹친다.
- **(b) 시스템 정확도** (eval 150개, 공격 상황): P16 단독 8.7% → all_switch **66.0%**,
  local_switch **65.3%** — 사실상 동률.
- **(c) clean 오탐 비용**: P16 clean 84.0%(126/150) → all_switch 적용 시 88.0%(132/150,
  +6장/+4.0%p, P8이 원래 P16보다 clean 정확도가 높아서[89.2% vs 85.2%] 오탐이 나도 오히려
  소폭 도움이 됐다), local_switch 적용 시 84.7%(127/150, +1장/+0.7%p). 세 조건의 Wilson
  95% CI([77.3%,89.0%]/[81.8%,92.3%]/[78.0%,89.6%])가 전부 겹치고, local_switch는 150장
  중 1장 차이라 사실상 노이즈 — **논문에 "세 번째 강점"으로 올리지 말고 "두 방법 모두 clean
  비용 무시 가능한 수준"으로만 쓸 것.**

**결과**: [`experiments/system_comparison/system_comparison_viz.png`](experiments/system_comparison/system_comparison_viz.png)

**왜 FPR/recall이 §6의 5.0%/64.0%에서 13.3%/72.7%로 바뀌었나**: calibration 이미지 자체는
§6과 완전히 동일(seed=42의 첫 100장)하지만, PatchFool 공격의 초기 랜덤 섭동
(`src/attacks/patch_fool.py`의 `torch.randn_like`)이 배치를 몇 개씩 나눠(chunk) 도는지에
따라 전역 RNG 스트림에서 다른 값을 뽑는다 — §6은 `CHUNK=50`(하드코딩), system_comparison은
`--chunk 20`을 써서 같은 이미지에 다른 공격 초기화가 적용됐다. 직접 검증(2026-09-20, job
2294207): 같은 100장에 chunk=50으로 다시 공격을 생성하면 §6의 원본 탐지 점수와 거의 완벽히
일치(최대 차이 0.0026)하지만, chunk=20으로 생성하면 크게 갈린다(최대 차이 0.43, 적대적 이미지
자체의 최대 픽셀 차이도 39) — **chunk 크기가 원인임을 직접 확정**했다. 즉 "다르게 구성된
calibration 표본" 때문이 아니라 "같은 표본에 다른 공격 초기화가 적용돼 Youden's J가 다른
지점(더 낮은 임계값 0.5116 vs 0.5567)을 골랐다"는 것 — 임계값이 낮아지며 FPR·recall이 같이
오른 것과 정확히 들어맞는다. 두 값 다 유효한 calibration 결과이고, **system_comparison이
all_switch·local_switch에 정확히 같은 임계값을 공유시키는 유일한 실험**이므로 논문에는
13.3%/72.7%를 최종값으로 쓴다(§6은 이미 제거됨, 재현 불가).

## Cost comparison — 실제 파이프라인 비용 (`experiments/cost_comparison/`)

기존 §10은 P8/P16을 각각 고립된 상태로(탐지 hook도 없이) 쟀고, local_switch는 비용을 잰 적이
아예 없었다. 이 실험은 실제 파이프라인을 **기저 비용**(모든 이미지에 항상 발생 — P16 예측+탐지
+위치특정을 한 번의 forward로 계산, patch_attack_detector의 `detector.predict_detect_localize`)과 **추가
비용**(escalate된 이미지에서만 발생 — `apply_all_switch` vs `apply_local_switch`를 실제로 호출)
으로 나눠 batch=1로 측정한다.

**결과** (job 2292084, RTX 4090): 기저 비용 **3.73ms**(44.2 GFLOPs). 추가 비용은 all_switch
**6.91ms**(156.3 GFLOPs) vs local_switch **2.58ms**(36.8 GFLOPs) — **local_switch가 2.7배
저렴**. escalate 시 총 latency: all_switch 10.64ms(기저 대비 2.85배), local_switch 6.31ms
(1.69배).

**E[cost(π)] — 배포 시나리오별 기대 비용** (system_comparison의 FPR=13.3%/recall=72.7%를
재사용해 all_switch/local_switch 둘 다 계산, `E[cost(π)] = base + [(1-π)·FPR + π·recall]·extra`):

| π (공격 이미지 비율) | all_switch | local_switch |
|---|---|---|
| 0% (오탐만 반영) | 4.65ms (1.25x 기저) | 4.08ms (1.09x 기저) |
| 1% | 4.69ms (1.26x) | 4.09ms (1.10x) |
| 10% | 5.07ms (1.36x) | 4.23ms (1.13x) |
| 50% | 6.70ms (1.80x) | 4.84ms (1.30x) |

모든 π에서 local_switch가 더 싸고, π가 커질수록(공격이 흔해질수록) 격차가 더 벌어진다 — π=50%
에서 all_switch는 기저 대비 1.80배까지 늘어나지만 local_switch는 1.30배에 그친다("매번 P16+P8
둘 다 돈다"는 naive 대안은 π와 무관하게 항상 10.64ms=2.85배).

주의: local_switch의 추가 비용은 P8의 patch_embed를 이미지 전체(784개 서브패치)에 대해 계산하고
그중 4개만 쓰는 구현이라(§16/§17/§18과 동일한, 이미 검증된 메커니즘 코드 그대로 사용) 실제
필요한 것보다 더 계산한다 — 여기 나온 2.58ms는 미래에 4개만 계산하도록 최적화하면 더 내려갈 수
있는 **보수적 상한**이다.

**결과**: [`experiments/cost_comparison/cost_comparison_viz.png`](experiments/cost_comparison/cost_comparison_viz.png)

## 종료된 탐색: local_switch 위치 인코딩 개선 시도 (2026-09-20)

**동기**: system_comparison에서 local_switch의 복원율(89.4%)이 all_switch(96.5%)보다 낮게
나온 원인 중 하나로, `local_switch.py`의 4개 P8 서브패치가 "이게 4개 중 몇 번째 서브패치인지"를
구분하는 명시적 위치 신호 없이 브릿지되고 있다는 점이 있었다(대화 기록 참고 — P8 서브패치는
P8 자신의 native pos_embed만 쓰고, 어댑터는 4개 전부에 대해 하나의 공유 아핀 변환만 학습).
APT(arXiv 2510.18091)의 `TokenizedZeroConvPatchAttn`이 여러 sub-patch를 합칠 때 위치 구분
벡터를 쓰는 걸 참고해서, 재학습 없이 빠르게 검증 가능한 옵션부터 시도했다.

**시도 (옵션 A, 재학습 없음)**: closed-form 아핀 브릿지는 그대로 두고, 4개 P8 서브패치
임베딩에 브릿지 적용 전 고정된(학습 안 된) quadrant 구분 벡터(4개의 orthogonal 벡터)를 더함.
system_comparison과 동일한 n=250/seed=42 split, 같은 공격 이미지에서 baseline과 직접 비교.

**결과 (job 2293640/2293693)**: 벡터 크기를 calibration 평균 patch embedding 노름의 10%와
100%(10배 차이) 두 스케일로 시험했는데, **둘 다 완전히 동일한 결과**(101/112=90.18%,
delta=0.0000 — 단 하나의 예측도 안 바뀜)가 나왔다. 스케일을 10배 키워도 전혀 변화가 없다는
건 "스케일이 작아서"가 아니라 **구조적으로 이 방식이 효과를 낼 수 없다**는 뜻이다:
어댑터는 4개 서브패치 전부에 대해 **하나의 공유 아핀 변환**(W, b)만 쓰고, 피팅 타깃도 4개
다 동일(부모 P16 패치 하나)하다 — 고정 벡터를 더해도 최소제곱 피팅이 그걸 그냥 상수 이동으로
흡수해서 W, b를 재조정할 뿐, quadrant별로 다른 보정을 학습하도록 만들지 못한다. "공유된 하나의
선형 변환"이라는 브릿지 설계 자체가 이런 종류의 위치 구분 신호를 살릴 수 없는 구조다.

**결론**: 이 방향은 여기서 닫는다. 더 큰 변경(옵션 B — APT처럼 토큰 수를 안 늘리고 conv로
집약한 걸 zero-init 선형층을 거쳐 원래 토큰에 더하는 방식)은 재학습이 필요해 스코프가 커지는
데다, system_comparison·cost_comparison이 이미 이 개선 없이도 핵심 주장(시스템 정확도 동률,
비용은 1/3)을 성립시키고 있어서 지금 밀어붙일 이유가 없다고 판단해 진행하지 않았다. 코드·양쪽
스케일의 원자료는 `experiments/local_switch_posenc_ablation/`
에 보존.

## Local Switch 강건성 검증 (`experiments/local_switch/`)

복원율/시스템 정확도/비용은 위 system_comparison·cost_comparison이 대표 수치다. 이 섹션은
**adaptive attacker에 대한 강건성**(all switch가 이미 §7/§14로 검증한 것과 동급) 검증만 다룬다.

- **§17 joint attack stress test** (`joint_attack/`): 정렬 메커니즘(아핀 변환)이 §8이 경고한
  "정렬된 표현=joint attack에 더 취약" 함정에 빠지는지 확인하기 위해 §7과 동일한 방법론을
  적용. 나이브 전이 2.5%(1/40) → **joint attack 완전 무력화율 16.7%**(7/42, 95% CI
  [8.3%, 30.6%]) — §7의 18.4%(다른 patch size)에 가깝고, §8의 74.4%(정렬된 표현)와는 한참
  떨어져 있다. **함정을 피했다.**
- **§18 완전판 adaptive evasion** (`adaptive_evasion_full/`): §14와 동일한 방법론(STRAP-ViT류
  제약 최적화)을 적용, target_bound를 clean_max/calibrated_threshold 두 가지로 각각 최적화.
  **calibrated_threshold 기준(공격자가 실배포 임계값을 직접 알고 최적화하는, 더 보수적인
  위협 모델) worst-case = 21.4%**(9/42, 95% CI [11.7%, 35.9%]) — 대표 수치.
  clean_max 기준으로는 11.9%(5/42, [5.2%, 25.0%], 참고치). §14의 15.8%와 신뢰구간이 크게
  겹쳐 **통계적으로 구분되지 않는다**.

**한계 (원인 미규명)**: 탐지기 flag율이 §17/§18(4%)에서 §7/§14(20%)보다 뚜렷이 낮다. 원인은
아직 분석하지 않았고, 열린 한계로 남겨둔다.

## All Switch 강건성 검증 (`experiments/all_switch/`)

의심되면 이미지 전체를 P8로 다시 분류하는, local switch보다 훨씬 단순한 설계. 별도 정렬
메커니즘 없이 완결된 두 번째 모델을 그대로 돌리기만 하면 되기 때문에, local switch를 만들기
전에 먼저 이쪽을 완결적으로 검증해 비교 기준선으로 삼았다.

- **§7 joint attack** (`joint_attack/`): P16+P8 손실을 합쳐 하나의 perturbation으로 동시
  최적화 → 나이브 전이(2.9%) 대비 joint attack은 **18.4%로 6배 위험**, 탐지기도 약해짐
  (joint attack의 20%만 flag).
- **§14 완전판 adaptive evasion** (`adaptive_evasion_full/`): joint attack 손실에 "탐지
  점수를 clean 범위 안으로 유지"하는 미분가능 페널티(STRAP-ViT류 설계)를 추가 → 회피 제약을
  걸어도 결과가 거의 그대로(18.4%) — 두 모델을 동시에 속이는 목표 자체가 이미 탐지 회피를
  "공짜로" 어느 정도 포함하고 있었다는 뜻. **최종 worst-case(무력화+미탐지) = 15.8%**
  (6/38, 95% CI [7.4%, 30.4%]), target_bound를 clean_max/calibrated_threshold 어느 쪽으로
  잡아도 동일하게 나옴.

## 공유 인프라: 탐지 + 위치특정 (`experiments/detection_localization/`)

두 방어가 공통으로 의존하는 raw-attention 기반 탐지기(현재 `../patch_attack_detector/detector/topk_mass_v1.py`,
원래 `defense/detector.py` — 위 "탐지기가 별도 프로젝트로 분리됨" 참고)의 신뢰성을 검증하는
섹션. 여기 있는 §1~§4 실험 스크립트 자신은 재검증 리스크를 피하려고 검증된 인라인 사본을
그대로 유지하며, 이번 이동의 대상이 아니다.

- **§1 탐지 가능성** (`signature/`): clean/LaVAN/PatchFool 각 30장, 층 1~12별 top-4 mass의
  AUROC 측정(rollout vs raw) → **L=12 raw attention이 최고, AUROC 0.879(clean 대비)~0.891
  (LaVAN 대비)**.
- **§2 위치 특정** (`localization/`): top-1(가장 attention 큰 토큰)이 실제 공격 토큰과
  일치하는 비율 → **recall@1 96.7%(29/30)**.
- **§3 회피 시도 3종** (`evasion_robustness/`): (1) 가장 낮은 saliency 위치를 강제 공격
  (2) attention/saliency 정규화한 새 탐지기 시도 (3) 실패 샘플들의 실제 saliency 범위로
  재타겟 → raw 탐지기는 전부 recall 0.900으로 안 뚫림, 대안(정규화) 탐지기만 recall 0.000으로
  자체 실패.
- **§4 attn_layer_idx 일반화** (`layeridx_generalization/`): 공격의 attn_layer_idx를
  1,2,4,6,8,10으로 바꿔가며 반복 → recall@1이 항상 0.900~0.967로 안정적.
- **탐지기는 LaVAN을 못 잡는다 (한계)**: §1의 npz를 재계산해 LaVAN vs Clean AUROC를 레이어별로
  따로 측정하면 L=1: 0.522, L=6: 0.434, **L=12(실제 방어가 쓰는 레이어): 0.363** —
  0.5(chance)보다 낮다. 즉 이 raw-attention 집중도 탐지기는 PatchFool에는 통하지만
  **LaVAN은 원리적으로 이 신호로 안 잡힌다**(§1의 "AUROC 0.891 LaVAN 대비"는 "PatchFool을
  LaVAN-or-clean 배경과 구분하는 능력"이지 "LaVAN 자체를 clean과 구분하는 능력"이 아니다 —
  혼동하기 쉬워서 명시해둠). LaVAN을 잡으려면 별도의 보완 게이트(후보: 패치별 patch_embed
  activation norm outlier)가 필요하며, 아직 구현·검증 안 됨.

## 보조 진단: Diversity diagnostic (`experiments/diversity_diagnostic/`)

방어력이 정확히 어디서 오는지("토큰화 구조가 다름" 자체인지, 그냥 "두 모델이 다름"인지)를
확인하고, local switch의 정렬 메커니즘이 밟을 수 있는 함정을 미리 특정해둔 §8 실험.

- **방법**: 같은 P16, 학습 레시피만 다른 두 번째 모델로 P16-A vs P16-B joint attack (§7과
  동일 이미지, seed=123으로 페어링 — 아래 "샘플링 감사" 참고).
- **결과**: 같은 patch size·다른 학습 = **74.4%**(29/39, [58.9%, 85.4%]) 뚫림 vs 다른 patch
  size(§7) = 18.4% → **방어력의 핵심은 "다른 patch size"이지 "다른 모델"이 아니다.** 동시에
  이 결과는 "표현을 정렬하면 joint attack에 취약해질 수 있다"는 구체적인 경고이기도 하다 —
  local switch가 도입하는 아핀 정렬이 바로 이 함정에 빠지는지를 검증한 것이 위 §17이다.

## 샘플링 감사

지금까지의 실험들이 매번 같은 고정 이미지 집합으로 평가됐는지(→ 비교가 공정한지), 아니면
실행마다 독립적으로 무작위 재샘플링했는지(→ 숫자 차이가 진짜 원인 때문인지 그냥 다른 표본
때문인지 불분명) 감사했다.

**메커니즘**: `src/dataset.py`의 `get_dataloader(seed, num_samples)`는 seed로 고정한
`torch.Generator`로 전체 데이터셋을 한 번 섞은 뒤 앞에서 `num_samples`개를 자른다. 그래서
**seed와 num_samples가 같으면 항상 같은 이미지가 같은 순서로 나온다** (num_samples가 다르면
작은 쪽이 큰 쪽의 prefix가 되지만 calibration/eval 경계는 각 스크립트가 따로 정하므로,
"같은 seed"만으로는 "같은 조건"이 보장되지 않는다 — 정확히 이 문제 때문에 system_comparison이
calibration/eval 분리를 한 스크립트 안에서 한 번만 하도록 설계됐다, 위 참고).

**감사 결과**:

| 비교 | 방식 | 판정 |
|---|---|---|
| `patch_size_tradeoff`의 정식 실험 1~7, area-matched #4/#5 포함 | `experiments/main.py`가 loader를 P×attack 루프 밖에서 1회만 생성, 재사용 | ✅ 고정 이미지, 페어링됨 |
| §1 layer sweep, §4 attn_layer_idx 일반화 | 루프 밖에서 1회 호출 | ✅ 고정 이미지, 페어링됨 |
| **§8 vs §7** — "52.3% vs 18.4%" 반전 결과의 근거 | §8는 seed=456, §7는 seed=123 — **서로 다른 50장으로 비교되고 있었음** | ❌ 발견 → 재실행으로 수정 |
| **§6 vs §16** (제거됨) — "97.1% vs 89.8%" 복원율 비교의 근거 | 둘 다 seed=42지만 calibration/eval 경계가 달라 **부분적으로만 겹치는 이미지로 비교되고 있었음** | ❌ 발견 → system_comparison으로 대체 |

**§8 수정** — §7과 같은 이미지(seed=123)로 재실행(job 2147607):

| 비교 (동일 50장, seed=123) | 둘 다 속음(완전 무력화) | 95% CI |
|---|---|---|
| P16-A vs P16-B (같은 patch size, 다른 학습) | **29/39 = 74.4%** | [58.9%, 85.4%] |
| P16 vs P8 (다른 patch size) | 7/38 = 18.4% | [9.2%, 33.4%] |

**결론: 숫자는 바뀌었지만(52.3%→74.4%) 결론은 안 바뀌었고 오히려 더 뚜렷해졌다.** 두 신뢰구간
겹침이 전혀 없어졌다. **논문/인용에는 페어링된 74.4%를 쓸 것.** 이후 §17/§18도 이 교훈을 따라
항상 동일 seed=123으로 §7/§8과 직접 비교 가능하게 실행했고, system_comparison은 한발 더 나아가
calibration/eval 분리 자체를 단일 스크립트 안에서 한 번만 하도록 만들어 이 종류의 실수가
구조적으로 재발하지 않게 했다.

## 한계 (논문 Limitations에 반영할 것)

- **탐지기 flag율 불일치, 원인 미규명**: local switch(§17/§18)의 탐지기 flag율(4%)이 all
  switch(§7/§14, 20%)보다 뚜렷이 낮다. 원인은 분석하지 않았고, 열린 질문으로 남긴다.
- **LaVAN 비탐지**: raw-attention 탐지기는 LaVAN을 원리적으로 못 잡는다(AUROC 0.36~0.57,
  모든 레이어에서 chance 수준 이하). 별도 보완 게이트가 필요하며 아직 미구현.
- **표본 크기**: adaptive evasion 비교는 n=38~42 규모라 신뢰구간이 넓다(§17/§18 상한이 30%대).
  system_comparison은 n=113(공격 성공 기준)/150(eval 기준)으로 더 크지만, "local switch와
  all switch가 통계적으로 구분되지 않는다"는 주장이지 "local switch가 더 낫다"는 주장이 아니다.
- **local_switch 비용의 보수적 상한**: cost_comparison의 local_switch 추가 비용(2.58ms)은
  P8 patch_embed를 이미지 전체(784개)에 대해 계산하고 4개만 쓰는, 최적화되지 않은 구현
  기준이다 — 실제로는 더 낮아질 여지가 있다.
- **위협 모델 범위**: 검증한 공격은 LaVAN/PatchFool과 그 joint/adaptive 변형에 한정된 empirical
  보장이다. [PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  같은 certified 방어와는 성격이 다르다는 점을 명시할 필요가 있다.

## 로드맵 / 다음 단계 후보

- local switch의 탐지기 flag율 4% vs all switch의 20% 차이 원인 분석 (우선순위 낮음, 의도적으로 보류 중)
- local_switch의 P8 patch_embed 계산을 실제 필요한 4개 서브패치로만 제한 — cost_comparison의
  2.58ms를 더 낮출 수 있는 확실한 최적화 (아직 구현 안 함)
- LaVAN용 보완 탐지 게이트 (후보: patch_embed activation norm outlier)
- 라우터를 L=6으로 당겼을 때의 recall/accuracy trade-off (L=6의 PatchFool vs Clean AUROC가
  0.639로 L=12의 0.879보다 낮아 recall 손실이 예상됨 — 아직 실행 안 함)
- 탐지 recall을 올리는 방법 — 현재 시스템의 실질적 병목
- 기존 baseline([PatchCleanser](https://www.usenix.org/conference/usenixsecurity22/presentation/xiang)
  등)을 같은 P8/16/32 세팅에 직접 돌려 비교 — 코드 공개돼 있고 아키텍처 무관이라 이식 쉬움
- (참고) [ViTGuard](https://arxiv.org/abs/2409.13828)는 attention+CLS token+MAE 재구성을
  결합한 탐지기를 7개 기존 detector·9개 attack과 비교해 검증한 바 있어, 이 프로젝트의
  raw-attention 탐지기는 그보다 단순한 버전이다.

## §9. Protocol C — patch_size_tradeoff로 이동

면적 대신 토큰 개수를 P8/P16/**P32**에서 동일하게 고정하는 실험이라(P32 포함) 이 프로젝트
범위 밖으로 판단, [`patch_size_tradeoff/09_protocol_c/`](../patch_size_tradeoff/09_protocol_c/)로 옮겼다.
결과(P8 RA 22.0% vs P16/P32 0.0%)는 그쪽 README 참고.
