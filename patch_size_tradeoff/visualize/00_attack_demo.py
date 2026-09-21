"""
00_attack_demo.py — 공격 데모: P=8 vs P=16 vs P=32 비교 (이미지 1장, 논문용 가로 레이아웃)

PGD / LaVAN / PatchFool 세 공격이 실제로 어떻게 동작하는지,
패치 크기(PATCH_SIZES)에 따라 perturbation 형태와 공격 성공 여부가
어떻게 달라지는지 이미지 한 장으로 시각화합니다. (발표/논문용)

사용법:
  python visualize/00_attack_demo.py --seed 42
  # 결과: results/figures/00_attack_demo_P8_vs_P16_vs_P32.png (전체 공격 비교 표 1장)

  python visualize/00_attack_demo.py --attacks PGD LaVAN     # 표에 넣을 공격만 골라서 실행

자주 수정하게 되는 값 (모두 파일 상단에 상수로 모아둠):
  PATCH_SIZES         비교할 patch size 목록
  ATTACKS             공격별 실행 함수 + 행 색상
  INNER_WIDTH_RATIOS  Result : Perturbation 칸 너비 비율
  HEADER_FS(함수 내)   표 전체 글자 크기 기준값
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from pgd import pgd_attack        # patch_size_tradeoff-only attack, not shared
from src.attacks.lavan import lavan_attack
from src.attacks.patch_fool import patch_fool_attack
from results_io import FIG_DIR

try:
    from timm.data import ImageNetInfo
    _INFO = ImageNetInfo()
    def class_name(idx):
        return _INFO.index_to_description(idx).split(',')[0]
except Exception:
    def class_name(idx):
        return str(idx)

PATCH_SIZES   = [8, 16, 32]
PSTR          = '_vs_'.join(f'P{p}' for p in PATCH_SIZES)  # 파일명에 쓰는 "P8_vs_P16_vs_P32" 형태
# src/dataset.py와 동일 (augreg 체크포인트가 요구하는 정규화 상수)
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

# ImageNet-1k 표준 클래스 순서에서 0~397번이 동물(어류/조류/파충류/포유류 등)
ANIMAL_LABEL_MAX = 397
# 151~294 = 개/고양이/여우/늑대 등 육상 포유류 — 바다/파란색 배경이 적어
# 초록색 성공 표시가 잘 보이는 사진이 나올 확률이 높다
PREFERRED_LABEL_RANGE = (151, 294)

# 변화 비율이 이 값을 넘으면(=전체 이미지 공격) bbox를 그리지 않음
GLOBAL_ATTACK_THRESHOLD = 0.5

def make_caption(attack_names):
    """attack_names에 맞춰 캡션 문구를 만든다 (전체 표 / 공격별 개별 이미지 공용)."""
    if len(attack_names) > 1:
        names, plural = ', '.join(attack_names[:-1]) + f', and {attack_names[-1]}', 'attacks'
    else:
        names, plural = attack_names[0], 'attack'
    p_str = ' vs '.join(f'P={p}' for p in PATCH_SIZES)
    return f'Fig. 1. Qualitative characterization of the {names} {plural} on ViT-Base ({p_str}).'

ATTACKS = {
    'PGD': dict(
        row_color='#eaf2ff',
        fn=lambda model, img, lbl, device, P: pgd_attack(
            model, img, lbl, device,
            epsilon=8/255, alpha=2/255, steps=40, random_start=True),
    ),
    'LaVAN': dict(
        row_color='#eafbea',
        fn=lambda model, img, lbl, device, P: lavan_attack(
            model, img, lbl, device,
            patch_ratio=0.02, steps=40, alpha=2/255),
    ),
    'PatchFool': dict(
        row_color='#fff2e8',
        fn=lambda model, img, lbl, device, P: patch_fool_attack(
            model, img, lbl, device, patch_size_model=P,
            attack_mode='CE_loss', train_attack_iters=250,
            num_patch=1, patch_select='Attn')[0],
    ),
}


def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (t.cpu() * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def diff_heatmap(orig, adv):
    d = (adv - orig).abs().sum(dim=0).cpu().numpy()
    return d / (d.max() + 1e-8)


def changed_bbox(heat, thresh=0.1, pad=6):
    """실제 변화 영역보다 pad(px)만큼 여유를 둔 bbox — 표시선이 변화 픽셀을 가리지 않도록."""
    ys, xs = np.where(heat > thresh)
    if len(xs) == 0:
        return None
    h, w = heat.shape
    x0 = max(0, xs.min() - pad)
    x1 = min(w - 1, xs.max() + pad)
    y0 = max(0, ys.min() - pad)
    y1 = min(h - 1, ys.max() + pad)
    return x0, x1, y0, y1


def find_shared_correct_sample(models, loader, device):
    """
    모든 P의 모델이 공통으로 맞히는 샘플 탐색.
    가능하면 PREFERRED_LABEL_RANGE(육상 포유류)를 우선하고,
    없으면 동물 클래스, 그것도 없으면 아무 샘플이나 사용한다.
    """
    preferred = animal = fallback = None
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        ok = True
        for model in models.values():
            with torch.no_grad():
                pred = model(images).argmax(dim=1)
            if pred[0] != labels[0]:
                ok = False
                break
        if not ok:
            continue

        lbl = labels[0].item()
        sample = (images[0:1], labels[0:1])
        if fallback is None:
            fallback = sample
        if animal is None and lbl <= ANIMAL_LABEL_MAX:
            animal = sample
        if preferred is None and PREFERRED_LABEL_RANGE[0] <= lbl <= PREFERRED_LABEL_RANGE[1]:
            preferred = sample
            break

    return preferred or animal or fallback or (None, None)


HEADER_BG    = '#d9d9d9'  # P=8/P=16 헤더 + Result/Perturbation 서브헤더 + 공격명 칸 공통 색
INNER_WSPACE = 0.12
# Result / Perturbation 칸 너비 비율. [1, 1]이면 완전히 동일한 폭.
INNER_WIDTH_RATIOS = [1, 1]


def _blank_axes(ax, facecolor):
    ax.set_facecolor(facecolor)
    ax.patch.set_visible(True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure(true_name, orig_np, results, out_path, attack_names=None):
    """
    results[attack_name][P] = dict(adv_np, heat, pred_adv_name, success, frac)
    실제 공격 실행과 분리해서, 그림 레이아웃만 독립적으로 테스트할 수 있게 한다.

    결과 표는 (1 + len(attack_names))행(헤더 + 공격별 행) x (1 + len(PATCH_SIZES))열
    (공격명 + P별 열)의 실제 표 형태로 그린다. 각 셀은 배경색으로 채워지며(헤더는 회색,
    공격별 행은 고유 색), P별 데이터 셀 내부에만 Adversarial/Heatmap 두 패널을 나눠서 넣는다.

    attack_names: 표에 넣을 공격 이름 리스트. None이면 ATTACKS 전체(3개) 사용.
    """
    attack_names = attack_names or list(ATTACKS.keys())
    n_attacks    = len(attack_names)
    n_p          = len(PATCH_SIZES)
    n_rows       = n_attacks + 2  # P별 헤더 + Result/Perturbation 서브헤더 + 공격들
    HEADER_FS    = 27  # 공격명 / P별 헤더 / Result,Perturbation 서브헤더 / 캡션 공통 폰트 크기

    # P 열 하나당 폭 10.08in, 공격명 열 2.52in, 원본 이미지+여백 6.84in (P=8/16 2열일 때 27.0in과 일치)
    fig = plt.figure(figsize=(6.84 + 10.08 * n_p, 6.6 * n_attacks + 2.6))

    # 원본은 참고용이라 작아도 되고, 표가 핵심이므로 표에 더 넓은 영역을 준다.
    # (여백 크기 자체는 그대로 유지)
    MARGIN    = 0.02
    ORIG_LEFT = MARGIN
    ORIG_RIGHT = 0.14
    TABLE_LEFT = ORIG_RIGHT + MARGIN
    TABLE_RIGHT = 1 - MARGIN

    # ── 원본 이미지: 별도 영역(왼쪽), 표보다 작게 ────────────────────────
    gs_orig = gridspec.GridSpec(1, 1, left=ORIG_LEFT, right=ORIG_RIGHT,
                                top=0.95, bottom=0.10, figure=fig)
    ax_orig = fig.add_subplot(gs_orig[0, 0])
    ax_orig.imshow(orig_np)
    ax_orig.axis('off')
    ax_orig.text(0.5, -0.03, f'Original image\n({true_name})',
                transform=ax_orig.transAxes, ha='center', va='top',
                fontsize=20, fontweight='bold')

    # ── 결과 표: (공격명 | P별 열...) ─────────────────────────────────
    gs = gridspec.GridSpec(n_rows, 1 + n_p,
                           left=TABLE_LEFT, right=TABLE_RIGHT, top=0.95, bottom=0.10,
                           width_ratios=[0.7] + [2.8] * n_p,
                           height_ratios=[0.55, 0.4] + [2.6] * n_attacks,
                           wspace=0.0, hspace=0.0, figure=fig)

    # 1행: P=8 + P=16 — 셀 전체를 배경색으로 채운다(상자 아님)
    for pi, P in enumerate(PATCH_SIZES):
        ax_h = fig.add_subplot(gs[0, pi + 1])
        _blank_axes(ax_h, HEADER_BG)
        ax_h.text(0.5, 0.5, f'P = {P}', ha='center', va='center',
                 fontsize=HEADER_FS, fontweight='bold', transform=ax_h.transAxes)

    # 1~2행의 공격명 칸(1열)은 하나로 합친다 — 색은 헤더와 동일하게 통일
    ax_corner = fig.add_subplot(gs[0:2, 0])
    _blank_axes(ax_corner, HEADER_BG)

    # 2행: Result / Perturbation 서브헤더 — 1행과 같은 색으로 통일
    sub_labels = ['Result', 'Perturbation']
    for pi, P in enumerate(PATCH_SIZES):
        cell_spec = gs[1, pi + 1]
        ax_bg2 = fig.add_subplot(cell_spec)
        _blank_axes(ax_bg2, HEADER_BG)
        inner2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell_spec,
                                                  width_ratios=INNER_WIDTH_RATIOS, wspace=INNER_WSPACE)
        for si, label in enumerate(sub_labels):
            ax_s = fig.add_subplot(inner2[0, si])
            _blank_axes(ax_s, HEADER_BG)
            ax_s.text(0.5, 0.5, label, ha='center', va='center',
                     fontsize=HEADER_FS, fontweight='bold', transform=ax_s.transAxes)

    for ri, atk_name in enumerate(attack_names):
        row_color = ATTACKS[atk_name]['row_color']

        ax_label = fig.add_subplot(gs[ri + 2, 0])
        _blank_axes(ax_label, row_color)
        ax_label.text(0.5, 0.5, atk_name, ha='center', va='center',
                      fontsize=HEADER_FS, fontweight='bold', transform=ax_label.transAxes)

        for pi, P in enumerate(PATCH_SIZES):
            r = results[atk_name][P]
            cell_spec = gs[ri + 2, pi + 1]

            # 셀 배경(행 색상)을 먼저 깔고, 그 위에 이미지 2개 + 그 아래 값 2개를 넣는다
            ax_bg = fig.add_subplot(cell_spec)
            _blank_axes(ax_bg, row_color)

            # 0행은 빈 여백 행 — 이미지 위/아래 여백을 동일하게 맞추기 위함
            inner = gridspec.GridSpecFromSubplotSpec(
                3, 2, subplot_spec=cell_spec,
                height_ratios=[0.18, 1, 0.3], width_ratios=INNER_WIDTH_RATIOS,
                hspace=0.1, wspace=INNER_WSPACE)
            ax1     = fig.add_subplot(inner[1, 0])
            ax2     = fig.add_subplot(inner[1, 1])
            ax1_lab = fig.add_subplot(inner[2, 0])
            ax2_lab = fig.add_subplot(inner[2, 1])
            _blank_axes(ax1_lab, row_color)
            _blank_axes(ax2_lab, row_color)

            if r['success']:
                badge_color, badge_text = '#2e7d32', 'SUCCESS'  # 진한(눈에 잘 띄는) 그린
            else:
                badge_color, badge_text = '#b71c1c', 'FAIL'

            ax1.imshow(r['adv_np'])

            if r['frac'] < GLOBAL_ATTACK_THRESHOLD:
                bbox = changed_bbox(r['heat'])
                if bbox is not None:
                    x0, x1, y0, y1 = bbox
                    ax1.add_patch(mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False,
                        edgecolor='#00e5ff', linewidth=3))

            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_color(badge_color)
                spine.set_linewidth(5)
            ax1.set_xticks([]); ax1.set_yticks([])
            # 값만: 색 있는 글씨만 사용 (상자 없음). SUCCESS/FAIL은 크게, pred:는 작게
            ax1_lab.text(0.5, 0.66, badge_text, ha='center', va='center',
                        fontsize=22, fontweight='bold', color=badge_color,
                        transform=ax1_lab.transAxes)
            ax1_lab.text(0.5, 0.28, f"pred: {r['pred_adv_name']}", ha='center', va='center',
                        fontsize=18, fontweight='bold', color=badge_color,
                        transform=ax1_lab.transAxes)

            ax2.imshow(r['heat'], cmap='inferno', vmin=0, vmax=1)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2_lab.text(0.5, 0.5, f"{100*r['frac']:.1f}% px\nchanged",
                        ha='center', va='center', fontsize=22, fontweight='bold', transform=ax2_lab.transAxes)

    # ── 표 격자선 (헤더 포함 전체) ───────────────────────────────────
    full_bbox = gs[:, :].get_position(fig)
    col0_x1 = gs[:, 0].get_position(fig).x1  # 1열(공격명) 오른쪽 경계
    fig.add_artist(mpatches.Rectangle(
        (full_bbox.x0, full_bbox.y0), full_bbox.width, full_bbox.height,
        transform=fig.transFigure, fill=False, edgecolor='black', linewidth=2, zorder=5))
    for ri in range(n_rows - 1):
        y = gs[ri, :].get_position(fig).y0
        # 0행/1행 경계는 1열(공격명)이 두 행에 걸쳐 합쳐진 칸이므로 그 부분은 선을 긋지 않는다
        x_start = col0_x1 if ri == 0 else full_bbox.x0
        fig.add_artist(mlines.Line2D([x_start, full_bbox.x1], [y, y],
                                     transform=fig.transFigure, color='black', linewidth=1, zorder=5))
    for ci in range(n_p):
        x = gs[:, ci].get_position(fig).x1
        fig.add_artist(mlines.Line2D([x, x], [full_bbox.y0, full_bbox.y1],
                                     transform=fig.transFigure, color='black', linewidth=1.5, zorder=5))

    # Result/Perturbation(그리고 그 아래 Adversarial/Heatmap 이미지)을 나누는 선은
    # 옅게 — Result/Perturbation 서브헤더부터 표 맨 아래까지
    row1_bbox = gs[1, :].get_position(fig)
    for pi in range(len(PATCH_SIZES)):
        inner2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, pi + 1],
                                                  width_ratios=INNER_WIDTH_RATIOS, wspace=INNER_WSPACE)
        b0 = inner2[0, 0].get_position(fig)
        b1 = inner2[0, 1].get_position(fig)
        dx = (b0.x1 + b1.x0) / 2
        fig.add_artist(mlines.Line2D([dx, dx], [full_bbox.y0, row1_bbox.y1],
                                     transform=fig.transFigure, color='#c4c4c4', linewidth=1, zorder=4))

    # ── 하단 캡션 (한 줄, P=8/P=16 라벨과 동일한 폰트 크기) ─────────────
    fig.text(0.5, 0.025, make_caption(attack_names), ha='center', va='bottom', fontsize=HEADER_FS)

    plt.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def run_demo(seed=42, attacks=None):
    """attacks: 표에 넣을 공격 이름 리스트. None이면 ATTACKS 전체."""
    attacks = attacks or list(ATTACKS.keys())

    device = get_device()
    out_dir = FIG_DIR

    print("모델 로딩...")
    models = {}
    for P in PATCH_SIZES:
        m = load_vit_model(P, device)
        if m is None:
            print(f"  P={P} 로드 실패 — 종료")
            return
        m.eval()
        models[P] = m

    loader, _ = get_dataloader(batch_size=1, num_samples=500, seed=seed)

    img, lbl = find_shared_correct_sample(models, loader, device)
    if img is None:
        print("모든 모델이 공통으로 맞히는 샘플을 찾지 못함")
        return

    true_name = class_name(lbl.item())
    print(f"\n선택된 이미지: label={lbl.item()} ({true_name})")

    orig_np = denorm(img[0])
    results = {atk_name: {} for atk_name in attacks}

    for atk_name in attacks:
        atk_info = ATTACKS[atk_name]
        for P in PATCH_SIZES:
            model = models[P]

            with torch.no_grad():
                pred_clean = model(img).argmax(dim=1).item()

            print(f"  [{atk_name} | P={P}] 실행 중...")
            adv = atk_info['fn'](model, img, lbl, device, P)

            with torch.no_grad():
                pred_adv = model(adv).argmax(dim=1).item()

            adv_np  = denorm(adv[0])
            heat    = diff_heatmap(img[0], adv[0])
            # SUCCESS = 모델이 공격을 받고도 여전히 맞힘(견고함), FAIL = 틀림(공격에 당함)
            success = pred_adv == pred_clean
            frac    = float((heat > 0.01).sum()) / heat.size

            print(f"    pred: {pred_clean}({class_name(pred_clean)}) -> "
                  f"{pred_adv}({class_name(pred_adv)}) | "
                  f"{'success' if success else 'fail'} | changed px {100*frac:.1f}%")

            results[atk_name][P] = dict(
                adv_np=adv_np, heat=heat, success=success, frac=frac,
                pred_adv_name=class_name(pred_adv),
            )

    out = os.path.join(out_dir, f'00_attack_demo_{PSTR}.png')
    build_figure(true_name, orig_np, results, out, attack_names=attacks)
    print(f"저장 완료: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attacks', nargs='+', default=list(ATTACKS.keys()),
                        choices=list(ATTACKS.keys()), help='표에 넣을 공격 목록 (기본: 전체)')
    args = parser.parse_args()
    run_demo(seed=args.seed, attacks=args.attacks)
