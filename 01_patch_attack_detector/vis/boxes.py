"""vis/boxes.py — patch 위치를 박스로 표시하는 원시 함수들. 실제 사진(픽셀 좌표)과 attention
맵(imshow 기본 extent, 칸 중심이 정수 좌표) 양쪽에서 같은 박스 로직을 쓴다 — 좌표계 차이는
`offset` 하나로 흡수한다(사진은 offset=0, attention 맵은 offset=-.5).

이전에는 ground_truth/draw_boxes.py와 experiments/characterization/02_compare_...py 두 곳에
비슷한 박스 코드가 나뉘어 있었다. 여기 하나로 모으고 양쪽에서 가져다 쓴다.
"""
from matplotlib.patches import Rectangle

PPL = 14           # 14x14 patch 격자 (ViT-B/16, 224px)
PATCH_SIZE = 16     # 픽셀 좌표에서 patch 한 변(px). attention 맵처럼 patch_size=1인 격자는 호출부에서 넘긴다.

# clean 패널은 파랑, 공격(patchfool/lavan) 패널은 노랑 — clean도 박스를 지우지 않는다(지우면
# adv 패널과 같은 위치인지 비교하기 어려워진다), 색만 달라서 clean인지 공격인지 구분된다.
CLEAN_BOX_COLOR = 'blue'
ATTACK_BOX_COLOR = 'yellow'


def draw_patch_grid(ax, ppl=PPL, patch_size=PATCH_SIZE):
    for i in range(1, ppl):
        ax.axhline(i * patch_size, color='white', lw=0.3, alpha=0.4)
        ax.axvline(i * patch_size, color='white', lw=0.3, alpha=0.4)


def draw_patch_box(ax, patch_idx, color, label=None, ppl=PPL, patch_size=PATCH_SIZE, lw=2.0, offset=0.0):
    """patch_idx 위치에 박스 하나. offset=0(기본값)은 픽셀 좌표(사진), offset=-.5는 imshow 기본
    extent로 그린 격자(칸 중심이 정수 좌표)용 반 칸 보정이다."""
    r, c = patch_idx // ppl, patch_idx % ppl
    ax.add_patch(Rectangle((c * patch_size + offset, r * patch_size + offset), patch_size, patch_size,
                            fill=False, edgecolor=color, lw=lw))
    if label:
        ax.text(c * patch_size + offset + 1, r * patch_size + offset + patch_size - 2, label,
                color=color, fontsize=7, fontweight='bold', va='bottom')


def draw_bbox(ax, gt_patches, color, ppl=PPL, patch_size=PATCH_SIZE, lw=2.0):
    """gt_patches(여러 patch에 걸친 GT)를 patch별로 따로따로가 아니라, 전부를 감싸는
    사각형 하나로 그린다 — LaVAN처럼 겹친 patch가 여러 개면 patch 경계마다 박스를 치는
    게 오히려 격자처럼 보여서 attack이 안 보인다는 피드백 반영."""
    rows = [p // ppl for p, _ in gt_patches]
    cols = [p % ppl for p, _ in gt_patches]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ax.add_patch(Rectangle((c0 * patch_size, r0 * patch_size),
                            (c1 - c0 + 1) * patch_size, (r1 - r0 + 1) * patch_size,
                            fill=False, edgecolor=color, lw=lw))
