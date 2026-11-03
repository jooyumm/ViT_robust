"""vis/colors.py — attention 맵에 쓰는 순차·발산 색상표. 02_compare_attention_success_failure.py에
있던 정의를 그대로 옮겼다 — 새 스크립트(예: 탐지기 결과 시각화)에서 또 만들지 않고 여기서 가져다 쓴다.
"""
from matplotlib.colors import LinearSegmentedColormap

# 양 끝이 검정으로 가면 원본/공격/차이가 다 어두워져서 구분이 안 된다 -- 끝 색을 원색으로 둔다.
# raw attention: 흰색 -> 순수 파랑. 흰색에서 바로 연한 색으로 건너뛴 뒤 원색으로 진해진다 --
# 선형으로 흰색->원색을 이으면 작은 값이 흰색과 구분이 안 된다.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'white_blue', [(0.0, '#ffffff'), (0.05, '#d0d0ff'), (0.2, '#8080ff'), (1.0, '#0000ff')])

# 차이: 감소=연초록->밝은 초록->진한 초록(#008000), 0=흰색, 증가=연빨강->밝은 빨강->진한 빨강(#800000).
DIVERGING = LinearSegmentedColormap.from_list('green_white_red', [
    (0.0, '#008000'), (0.3, '#00ff00'), (0.425, '#66ff66'), (0.475, '#ccffcc'), (0.5, '#ffffff'),
    (0.525, '#ffcccc'), (0.575, '#ff6666'), (0.7, '#ff0000'), (1.0, '#800000')])

# Semantic roles: attack GT outline and population line colors are shared.
GROUP_COLORS = {'clean': '#2465a0', 'patchfool': '#d03b3b', 'lavan': '#0ca30c', 'attentionfool': '#963ac4'}
CLEAN_BOX_COLOR = 'blue'
ATTACK_BOX_COLORS = {'patchfool': 'red', 'lavan': 'green', 'attentionfool': 'purple'}
TRACKED_PATCH_COLOR = '#ffb000'  # selected by attention, NOT ground truth

def attack_color(attack):
    return ATTACK_BOX_COLORS[attack.lower()]
