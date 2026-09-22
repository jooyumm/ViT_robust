"""
conditions.py — ablation 비교 조건(CONDITIONS) 정의 모음
각 실험의 fig_*.py(그래프)와 tbl_*.py(표) 스크립트가 똑같은 CONDITIONS를 공유해야 해서
(둘 다 숫자로 시작하는 모듈이라 서로 import 불가) 여기 한곳에 모아둔다.

포맷: [(label, attack, P, tag, hatch, alpha), ...]
      hatch/alpha는 그래프 전용 스타일이고 표에서는 무시된다.
"""

# 실험 7: LaVAN 위치 고정
LAVAN_FIXEDLOC = [
    ('P=8\nRandom loc',  'lavan', 8,  '',        '...', 0.55),
    ('P=8\nFixed loc',   'lavan', 8,  'fixedloc', '...', 0.9),
    ('P=16\nRandom loc', 'lavan', 16, '',        '',    0.7),
    ('P=16\nFixed loc',  'lavan', 16, 'fixedloc', '',    1.0),
]
LAVAN_FIXEDLOC_TITLE = 'LaVAN: Random vs. Fixed (Center) Attack Location'

# 실험 4+5: PatchFool 면적 통제 (scattered vs contiguous)
# 기본(baseline, tag='') P=8/P=16은 01_baseline과 완전히 같은 hatch/alpha(P_HATCH/P_ALPHA)를
# 써서 "이게 그 baseline이다"가 바로 보이게 하고, 비교 대상(ablation, tag 있음)은 더 연한
# alpha + baseline과 다른 hatch를 줘서 "이건 변형이다"가 눈에 띄게 한다.
PATCHFOOL_AREAMATCH = [
    ('P=8\n1 token (64px)',        'patch_fool', 8,  '',                  '...',   0.85),
    ('P=8\n4 tokens, scattered',   'patch_fool', 8,  'areamatch16',       '\\\\\\', 0.4),
    ('P=8\n4 tokens, contiguous',  'patch_fool', 8,  'areamatch16contig', 'xxx',   0.4),
    ('P=16\n1 token (256px)',      'patch_fool', 16, '',                  '',      0.9),
]
PATCHFOOL_AREAMATCH_TITLE = 'PatchFool: Area match (scattered/contiguous)'

# 실험 6: PatchFool 토큰 선택 방식 (Attn vs Rand)
PATCHFOOL_RANDSEL = [
    ('P=8\nAttn-select',  'patch_fool', 8,  '',        '...',   0.85),
    ('P=8\nRand-select',  'patch_fool', 8,  'randsel', '\\\\\\', 0.4),
    ('P=16\nAttn-select', 'patch_fool', 16, '',        '',      0.9),
    ('P=16\nRand-select', 'patch_fool', 16, 'randsel', '\\\\\\', 0.4),
]
PATCHFOOL_RANDSEL_TITLE = 'PatchFool: Attention-Guided vs. Random Token Selection'