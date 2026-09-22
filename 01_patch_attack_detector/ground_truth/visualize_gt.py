"""
ground_truth/visualize_gt.py — 대표 이미지 5장 x 2개 공격(PatchFool/LaVAN) 전부에 대해
"공격이 실제로 어느 patch를 건드렸는가"(GT, gt_lib.gt_coverage로 픽셀 diff에서 계산)를
그림과 표로 뽑아낸다. attention이나 탐지기 출력은 여기서 전혀 안 본다 — 이 폴더의 역할은
정답 위치를 확정하는 것뿐이고, 그걸 attention/탐지기와 비교하는 건 별도의 나중 단계다.

GT 자체(gt_coverage/repr_images)는 이미 characterization의 00_extract_attention.py가
공격을 생성하면서(GPU 필요) 같이 계산해 npz에 저장해 뒀다 — 여기서는 그 npz를 읽기만
한다(GPU 불필요). 두 단계를 분리한 이유: 공격 이미지를 만드는 과정 자체가 GPU로 모델을
돌려야 하는 무거운 작업이라, 이미 그 작업을 하는 00_extract_attention.py 안에서 같이
계산하는 게 자연스럽다 — 여기서 다시 공격을 생성하지 않는다.

사용법:
  python visualize_gt.py --num_samples 100
  (../experiments/characterization/results/00_attention_data_n100.npz에 gt_coverage_repr/
   repr_images 필드가 있어야 함 — 없으면 00_extract_attention.py를 다시 실행할 것)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = os.path.dirname(HERE)   # 01_patch_attack_detector
CHAR_DIR = os.path.join(DETECT_ROOT, 'experiments', 'characterization')
sys.path.insert(0, HERE)       # for `import gt_lib/gt_plot`
sys.path.insert(0, CHAR_DIR)   # for `import data_io` (attention npz 로더)

from data_io import load_attention_data
from gt_lib import gt_patch_summary
from gt_plot import plot_gt_locations

ATTACKED_GROUPS = ('patchfool', 'lavan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    npz_path = os.path.join(DETECT_ROOT, 'results', 'characterization',
                             f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 characterization/00_extract_attention.py " \
        f"--num_samples {args.num_samples} 실행할 것"
    # load_attention_data는 attention 지표까지 같이 반환하지만, 여기서는 repr_by_group의
    # images/gt_coverage만 쓴다 -- all_metrics/n_layers는 이 스크립트의 관심사가 아니다.
    _, repr_by_group, _ = load_attention_data(npz_path)

    for g in ATTACKED_GROUPS:
        if 'gt_coverage' not in repr_by_group.get(g, {}):
            raise SystemExit(
                f"이 npz는 {g}__gt_coverage_repr 필드가 없는 구버전입니다 — "
                "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    out_dir = os.path.join(HERE, 'results')
    n_repr = repr_by_group['patchfool']['gt_coverage'].shape[0]
    rows = []
    for g in ATTACKED_GROUPS:
        group_dir = os.path.join(out_dir, g)
        os.makedirs(group_dir, exist_ok=True)
        for img_idx in range(n_repr):
            gt_cov = repr_by_group[g]['gt_coverage'][img_idx]
            gt_patches = gt_patch_summary(gt_cov)
            gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
            rows.append((g, img_idx, gt_str))

            p = os.path.join(group_dir, f'img{img_idx}.png')
            plot_gt_locations(repr_by_group, img_idx, g, p)

    md_path = os.path.join(out_dir, f'gt_summary_n{args.num_samples}.md')
    with open(md_path, 'w') as f:
        f.write(f"# Ground truth: 공격이 실제로 건드린 patch (n_repr={n_repr})\n\n")
        f.write("clean/adv 이미지를 직접 diff해서 계산한 값이다(`gt_lib.gt_coverage`) — "
                "attention이나 탐지기 출력으로 추정한 게 아니라 픽셀 그 자체다. "
                "PatchFool은 patch 1개만 공격하므로 항상 겹침 100%인 자리 하나만 나오고, "
                "LaVAN은 16px 그리드에 정렬 안 된 위치라 여러 patch에 걸쳐 부분적으로만 "
                "겹친다.\n\n")
        f.write("| group | image | GT patch(es) (겹침 %) |\n")
        f.write("|---|---|---|\n")
        for g, img_idx, gt_str in rows:
            f.write(f"| {g} | {img_idx} | {gt_str} |\n")
    print(f"Saved: {md_path}")
    print(f"Saved: results/{{patchfool,lavan}}/img*.png ({len(ATTACKED_GROUPS) * n_repr} files)")


if __name__ == '__main__':
    main()
