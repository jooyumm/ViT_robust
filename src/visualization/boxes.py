"""Shared patch geometry. A ViT-B/16 cell is 16×16 pixels, not a 16×16-cell grid.

224×224 inputs have 14×14 cells. With imshow's default extent, boundaries
are shifted by -0.5 for both photographs and token maps (patch_size=1).
"""
from matplotlib.patches import Rectangle
from .colors import CLEAN_BOX_COLOR, ATTACK_BOX_COLORS, attack_color

PPL = 14
PATCH_SIZE = 16


def draw_patch_grid(ax, ppl=PPL, patch_size=PATCH_SIZE, offset=-.5):
    for i in range(1, ppl):
        ax.axhline(i * patch_size + offset, color='white', lw=.3, alpha=.4)
        ax.axvline(i * patch_size + offset, color='white', lw=.3, alpha=.4)


def draw_patch_box(ax, patch_idx, color, label=None, ppl=PPL, patch_size=PATCH_SIZE,
                   lw=2., offset=-.5):
    if not 0 <= patch_idx < ppl * ppl:
        raise ValueError('Patch index outside grid.')
    row, col = divmod(int(patch_idx), ppl)
    rect = Rectangle((col * patch_size + offset, row * patch_size + offset),
                     patch_size, patch_size, fill=False, edgecolor=color, lw=lw)
    ax.add_patch(rect)
    if label:
        ax.text((col + .05) * patch_size + offset, (row + .9) * patch_size + offset,
                label, color=color, fontsize=7, fontweight='bold', va='bottom')
    return rect


def draw_bbox(ax, gt_patches, color, ppl=PPL, patch_size=PATCH_SIZE, lw=2., offset=-.5):
    if not len(gt_patches):
        return None
    indices = [p for p, _ in gt_patches]
    rows, cols = zip(*(divmod(int(p), ppl) for p in indices))
    rect = Rectangle((min(cols) * patch_size + offset, min(rows) * patch_size + offset),
                     (max(cols)-min(cols)+1) * patch_size,
                     (max(rows)-min(rows)+1) * patch_size,
                     fill=False, edgecolor=color, lw=lw)
    ax.add_patch(rect)
    return rect


def draw_attack_region(ax, coverage, attack, clean=False, ppl=PPL, patch_size=PATCH_SIZE,
                       offset=-.5, min_fraction=0., lw=2.):
    """Enclosing token-aligned GT rectangle; not an exact changed-pixel boundary."""
    patches = [(i, v) for i, v in enumerate(coverage) if v > min_fraction]
    return draw_bbox(ax, patches, CLEAN_BOX_COLOR if clean else attack_color(attack),
                     ppl=ppl, patch_size=patch_size, offset=offset, lw=lw)
