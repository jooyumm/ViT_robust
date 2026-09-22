# experiments/ — 비어있음 (의도적)

2026-09-\*까지 여기 있던 모든 평가/공격/비교 실험(system_comparison, cost_comparison,
paper_summary, local_switch/all_switch의 joint_attack·adaptive_evasion_full,
detection_localization의 §1~§4, diversity_diagnostic §8, attack_type_comparison,
local_switch_posenc_ablation, pos_embed_inspection)을 전부 지웠다.

**이유**: 이 실험들은 전부 `../defense/`(all_switch/local_switch)를
`01_patch_attack_detector`의 탐지기와 엮어서 평가한 것인데, 그 탐지기 쪽이 아직
레이어별 sweep·특성 관찰 단계라 로직이 안정되지 않았다(`01_patch_attack_detector/README.md`
참고). 탐지기가 자리 잡기 전에 나온 방어 평가 숫자는 다시 재현해야 할 게 뻔해서, 지금
단계에서는 `defense/`의 방어 메커니즘 코드 자체(all_switch.py/local_switch.py)를
다듬는 데 집중하고, 평가 실험은 탐지기가 완료된 뒤 다시 짠다.

**지운 것들은 git 이력에 남아있다** — 이전 커밋에서 언제든 복구 가능
(`git log --diff-filter=D -- experiments/` 로 삭제 커밋 확인).

**재개할 때**: `01_patch_attack_detector`의 탐지기가 확정되면, 이전 `system_comparison`/
`cost_comparison`과 같은 방법론(같은 calibration, 같은 eval 이미지, 같은 탐지 판정으로
두 방어 직접 비교)으로 다시 시작하는 게 좋다 — 그 설계 자체는 유효했고, 재검증이 필요한
건 어떤 탐지기를 쓰느냐였다.
