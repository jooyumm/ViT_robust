"""Render GT examples from the common paired 500-image run."""
import argparse
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from src.data.paired import load_images
from src.data.paths import PATCHFOOL
from src.ground_truth.draw_boxes import plot_gt_locations

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PATCHFOOL)
    parser.add_argument('--sample-index', type=int, help='Original ImageFolder ID; default: first stored sample')
    parser.add_argument('--output', type=Path, default=HERE.parents[1]/'data/gt/previews')
    args = parser.parse_args()
    manifest, data = load_images(args.run)
    ids = data['sample_index'].tolist()
    idx = 0 if args.sample_index is None else ids.index(args.sample_index)
    args.output.mkdir(parents=True, exist_ok=True)
    for attack in manifest.get('attacks',['patchfool']):
        plot_gt_locations(data['clean__images'][idx], data[attack + '__images'][idx],
                          data[attack + '__gt_coverage'][idx], attack, args.output / f'{args.run.name}_{attack}.png',
                          label=f'sample {ids[idx]}', gt_min_frac=1e-12)
        print(args.output / f'{args.run.name}_{attack}.png')

if __name__ == '__main__':
    main()
