"""run.py의 집계 수치 대신, 개별 이미지에서 '공격 적용 -> GT 없는 블라인드 탐지 -> 방어 교체'
순서가 실제로 일어났는지 직접 확인한다. GT(patchfool__gt_coverage)는 탐지 함수에 전혀
전달하지 않고, 탐지 결과가 나온 뒤 사후 대조에만 쓴다.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '02_detection_defense')]
from src.models import load_vit_model
from src.data.paths import PATCHFOOL
from src.data.protocol import validate_attack_manifest
from region_switch.core.attention import PatchAttentionDetector, patch_to_region
from region_switch.core.switch import converted_patch_kernel
from src.provenance import model_hash, sha256
from src.patches import coverage_from_images
from region_switch.core.inference import predict_defended

GRID, PATCH = 14, 16
MEAN = STD = 0.5


def pixel_diff_patch(clean, attacked):
    """저장된 gt_coverage 필드를 쓰지 않고, 두 이미지의 raw pixel 차이로 공격 위치를 직접 구한다."""
    coverage=coverage_from_images(clean.unsqueeze(0),attacked.unsqueeze(0),PATCH)[0]
    return int(coverage.argmax()),coverage


def patch_xy(patch):
    row, col = divmod(patch, GRID)
    return col * PATCH, row * PATCH


def region_xy(region):
    row, col = divmod(region, 2)
    return col * 112, row * 112


def draw_full_2x2_grid(ax):
    """선택된 사분면과 무관하게, 이미지 전체를 가로지르는 2x2 사분면 경계선을 항상 그린다."""
    ax.plot([112, 112], [0, 224], color='white', lw=1.6, alpha=0.9)
    ax.plot([0, 224], [112, 112], color='white', lw=1.6, alpha=0.9)
    ax.plot([112, 112], [0, 224], color='black', lw=0.6, alpha=0.9, ls=(0, (4, 3)))
    ax.plot([0, 224], [112, 112], color='black', lw=0.6, alpha=0.9, ls=(0, (4, 3)))


def draw_quadrant_grid(ax, region, color, filled):
    """선택된 사분면을 112x112 영역 + 내부 7x7 패치 격자선으로 그린다. filled면 반투명 색 채움도 넣는다."""
    rx, ry = region_xy(region)
    if filled:
        ax.add_patch(mpatches.Rectangle((rx, ry), 112, 112, facecolor=color, alpha=0.22, edgecolor='none'))
    for k in range(8):  # 7x7 패치 경계선 8개
        ax.plot([rx + k * PATCH, rx + k * PATCH], [ry, ry + 112], color=color, lw=0.7, alpha=0.9)
        ax.plot([rx, rx + 112], [ry + k * PATCH, ry + k * PATCH], color=color, lw=0.7, alpha=0.9)
    ax.add_patch(mpatches.Rectangle((rx, ry), 112, 112, fill=False, edgecolor=color, lw=2))


def to_image(tensor):
    return (tensor.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)


def collect_records(model, kernel8, detector, manifest, run):
    """500장 전체(clean+patchfool)에 대해 블라인드 탐지·방어 결과와 위치를 모은다."""
    n = len(manifest['sample_indices'])
    records = []
    offset = 0
    for chunk in manifest['chunks']:
        path=(run/chunk['path']).resolve()
        if run.resolve() not in path.parents or sha256(path)!=chunk['sha256']:raise ValueError('청크 경로/해시 불일치')
        with np.load(path) as z:
            count = chunk['n']
            label = z['label']
            clean_images = torch.from_numpy(z['clean__images'])
            attack_images = torch.from_numpy(z['patchfool__images'])
            base_c, def_c, dec_c = predict_defended(model, kernel8, clean_images, detector)
            base_a, def_a, dec_a = predict_defended(model, kernel8, attack_images, detector)
            base_c, def_c = base_c.argmax(-1).numpy(), def_c.argmax(-1).numpy()
            base_a, def_a = base_a.argmax(-1).numpy(), def_a.argmax(-1).numpy()
            for i in range(count):
                records.append(dict(
                    chunk_path=str(path), local=i, label=int(label[i]),
                    clean_ok=bool(base_c[i] == label[i]), attack_ok=bool(base_a[i] == label[i]),
                    clean_flag=bool(dec_c['flag'][i]), clean_region=int(dec_c['region'][i]),
                    clean_patch=int(dec_c['patch'][i]), clean_defended_ok=bool(def_c[i] == label[i]),
                    attack_flag=bool(dec_a['flag'][i]), attack_region=int(dec_a['region'][i]),
                    attack_patch=int(dec_a['patch'][i]), attack_defended_ok=bool(def_a[i] == label[i]),
                    attack_score=float(dec_a['score'][i]), clean_score=float(dec_c['score'][i])))
        offset += count
        print(f'사례 수집 {offset}/{n}', flush=True)
    return records


def load_pair(chunk_path, local):
    with np.load(chunk_path) as z:
        return (torch.from_numpy(z['clean__images'][local]),
                torch.from_numpy(z['patchfool__images'][local]))


def verify_attack_case(ax, model, kernel8, detector, record, kind):
    """공격 재계산(픽셀) -> 탐지 결과(GT 미사용) -> 방어 전후 예측을 한 이미지에 그린다."""
    clean, attacked = load_pair(record['chunk_path'], record['local'])
    gt_patch, coverage = pixel_diff_patch(clean, attacked)
    assert coverage[gt_patch] > 0.3 and (coverage.sum() - coverage[gt_patch]) < 1.0, \
        '공격이 한 패치에 국소적으로 들어가지 않았습니다.'
    ax.imshow(to_image(attacked))
    draw_full_2x2_grid(ax)
    gx, gy = patch_xy(gt_patch)
    ax.add_patch(mpatches.Rectangle((gx, gy), PATCH, PATCH, fill=False, edgecolor='#f1c40f', lw=2.5, ls='--'))
    px, py = patch_xy(record['attack_patch'])
    ax.add_patch(mpatches.Rectangle((px - 1, py - 1), PATCH + 2, PATCH + 2, fill=False, edgecolor='#e74c3c', lw=2))
    # 최고 점수 사분면은 flag 여부와 무관하게 항상 표시한다. flag=False는 "defended"가 실행 안 됐다는 뜻일 뿐,
    # 사분면 자체는 항상 계산된다 (빨강 채움=임계값 통과/실제 교체됨, 회색 격자만=후보였지만 미발동).
    region_color = '#e74c3c' if record['attack_flag'] else '#7f8c8d'
    draw_quadrant_grid(ax, record['attack_region'], region_color, filled=record['attack_flag'])
    region_match = patch_to_region(np.array([gt_patch]))[0] == record['attack_region']
    verdict = 'RECOVERED' if kind == 'recovered' else 'MISS'
    ax.set_title(
        f"[{verdict}] pixel-diff patch={gt_patch} (yellow) vs detector peak patch={record['attack_patch']}\n"
        f"flag={record['attack_flag']} score={record['attack_score']:.2f} region_match={region_match} | "
        f"P16={'correct' if record['attack_ok'] else 'wrong'}->defended={'correct' if record['attack_defended_ok'] else 'wrong'} "
        f"(true label id={record['label']})", fontsize=8.3)
    ax.set_xticks([]); ax.set_yticks([])


def verify_false_positive_case(ax, record):
    """공격이 없는데 탐지가 발동한 사례. 탐지 위치와 방어 적용 후에도 정답이 유지되는지 보여준다."""
    clean, _ = load_pair(record['chunk_path'], record['local'])
    ax.imshow(to_image(clean))
    draw_full_2x2_grid(ax)
    px, py = patch_xy(record['clean_patch'])
    ax.add_patch(mpatches.Rectangle((px - 1, py - 1), PATCH + 2, PATCH + 2, fill=False, edgecolor='#e74c3c', lw=2))
    draw_quadrant_grid(ax, record['clean_region'], '#e74c3c', filled=True)
    ax.set_title(
        f"[FALSE POSITIVE] clean image, detector fired at QUADRANT {record['clean_region']} (peak patch={record['clean_patch']}) "
        f"(score={record['clean_score']:.2f})\nP16={'correct' if record['clean_ok'] else 'wrong'}->"
        f"defended={'correct' if record['clean_defended_ok'] else 'wrong'} (true label id={record['label']})", fontsize=8.3)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'results/case_studies.png')
    parser.add_argument('--run',type=Path,default=PATCHFOOL,help='저장된 기본 500장 공격 데이터셋')
    args = parser.parse_args()
    out_path = args.output
    if out_path.suffix.lower() != '.png':
        raise ValueError('PNG 경로가 필요합니다.')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if out_path.exists():
        raise FileExistsError(out_path)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    validate_attack_manifest(manifest)
    model = load_vit_model(16, torch.device('cpu'))
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    assert model_hash(model) == manifest['model_sha256']
    kernel8 = converted_patch_kernel(model)
    detector = PatchAttentionDetector()

    records = collect_records(model, kernel8, detector, manifest, args.run)
    successful = [r for r in records if r['clean_ok'] and not r['attack_ok']]
    recovered = [r for r in successful if r['attack_flag'] and r['attack_defended_ok']]
    missed = [r for r in successful if not (r['attack_flag'] and r['attack_defended_ok'])]
    false_positive = [r for r in records if r['clean_flag'] and r['clean_ok']]
    print(f'성공 공격 {len(successful)} 중 복구 {len(recovered)}, 미복구 {len(missed)}; '
          f'clean 오탐(방어 전 정답) {len(false_positive)}', flush=True)

    rng = np.random.default_rng(7)
    def sample(pool, k):
        if not pool:
            return []
        chosen = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in chosen]
    picks = [('recovered', r) for r in sample(recovered, 2)] + \
            [('missed', r) for r in sample(missed, 1)] + \
            [('false_positive', r) for r in sample(false_positive, 1)]
    if not picks:
        raise RuntimeError('표시할 사례가 하나도 없습니다.')

    fig, axes = plt.subplots(len(picks), 1, figsize=(7.2, 4.4 * len(picks)), layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, (kind, record) in zip(axes, picks):
        if kind == 'false_positive':
            verify_false_positive_case(ax, record)
        else:
            verify_attack_case(ax, model, kernel8, detector, record, kind)
    fig.suptitle('Per-image verification: recomputed attack location vs\n'
                 'blind detection vs pre/post-defense prediction\n'
                 'gridded quadrant = top-scoring region (red fill = triggered & swapped, gray grid = below threshold)', fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('완료:', out_path, flush=True)


if __name__ == '__main__':
    main()
