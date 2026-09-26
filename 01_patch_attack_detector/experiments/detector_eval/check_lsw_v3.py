"""v3(detector/lsw_v3.py) 확인: (1) v2와 점수가 같은지, (2) "정상 이미지로 한 번 보정 -> 다른 run에서 평가"하는 배포 방식의 오탐률/탐지율.

보정: --calibration_run의 clean 이미지로 fit(fpr=5%). 평가: --run의 clean(오탐률)과 오분류에 성공한 공격(탐지율, 위치 적중).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1] / 'detector'))
spec = importlib.util.spec_from_file_location('compare02', HERE.parents[0] / 'characterization' / '02_compare_attention_success_failure.py')
compare02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare02)
import emergent_window_v2 as v2   # noqa: E402
import lsw_v3 as v3               # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration_run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from src.models import load_vit_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vit_model(16, device)
    model.eval()

    _, calib = compare02.load_paired_run(args.calibration_run)
    manifest, data = compare02.load_paired_run(args.run)

    images = torch.from_numpy(data['clean__images'][:20]).to(device)
    rows = v3.cls_patch_rows(model, images)
    assert torch.equal(v3.surge_score(rows), v2.surge_score(rows)), 'v3 점수가 v2와 다릅니다'
    assert torch.equal(v3.suspect_patches(rows), v2.suspect_patches(rows)), 'v3 위치가 v2와 다릅니다'
    print('v3 점수·위치가 v2와 완전히 같음(앞 20장)')

    detector = v3.LSWDetector(model)
    threshold = detector.fit(torch.from_numpy(calib['clean__images']), fpr=0.05)
    print(f'threshold {threshold:.4f} (calibration: {len(calib["label"])} clean images of {args.calibration_run.name})')
    clean = detector.detect(torch.from_numpy(data['clean__images']))
    print(f'clean false-alarm rate on {args.run.name}: {clean["alarm"].float().mean():.3f} (n={len(clean["alarm"])})')
    for attack in manifest['attacks']:
        success = data[attack + '__outcome'] == 1
        result = detector.detect(torch.from_numpy(data[attack + '__images'][success]))
        coverage = data[attack + '__gt_coverage'][success]
        alarm = result['alarm'].numpy()
        hit = np.array([(coverage[i, result['patches'][i].numpy()] > 0).any() for i in range(len(coverage))])
        print(f'{attack}: detected {alarm.mean():.3f} of {success.sum()} successful attacks; '
              f'suspect window overlaps the attacked patch in {hit[alarm].mean():.3f} of the alarms')


if __name__ == '__main__':
    main()
