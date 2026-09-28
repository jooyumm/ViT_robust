"""detector/rise_timing_v1.py 확인: LSW 단독 대비, LSW OR Rise-timing 결합이 실제로 얼마나 더 잡는지.

두 탐지기 모두 --calibration_run의 clean 이미지로 각자 5% FPR 임계값을 정한 뒤(fit), --run에서 평가한다.
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
import lsw_v3               # noqa: E402
import rise_timing_v1       # noqa: E402


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
    calib_images = torch.from_numpy(calib['clean__images'])

    lsw = lsw_v3.LSWDetector(model)
    lsw.fit(calib_images, fpr=0.05)
    timing = rise_timing_v1.RiseTimingDetector(model)
    timing.fit(calib_images, fpr=0.05)
    print(f'thresholds: LSW {lsw.threshold:.4f}, rise-timing {timing.threshold:.0f} '
          f'(calibration: {len(calib["label"])} clean images of {args.calibration_run.name})')

    clean_images = torch.from_numpy(data['clean__images'])
    lsw_clean = lsw.detect(clean_images)['alarm'].numpy()
    timing_clean = timing.detect(clean_images)['alarm'].numpy()
    print(f'clean false-alarm rate on {args.run.name}: LSW {lsw_clean.mean():.3f}, '
          f'rise-timing {timing_clean.mean():.3f}, OR {(lsw_clean | timing_clean).mean():.3f} (n={len(clean_images)})')

    for attack in manifest['attacks']:
        success = data[attack + '__outcome'] == 1
        images = torch.from_numpy(data[attack + '__images'][success])
        lsw_alarm = lsw.detect(images)['alarm'].numpy()
        timing_alarm = timing.detect(images)['alarm'].numpy()
        either = lsw_alarm | timing_alarm
        print(f'{attack} (n={success.sum()} successful): LSW {lsw_alarm.mean():.3f}, '
              f'rise-timing {timing_alarm.mean():.3f}, OR {either.mean():.3f} '
              f'(rise-timing catches {int((timing_alarm & ~lsw_alarm).sum())}/{int((~lsw_alarm).sum())} of what LSW missed)')


if __name__ == '__main__':
    main()
