"""v3(detector/lsw_v3.py) 확인: "정상 이미지로 한 번 보정 -> 다른 run에서 평가"하는 배포 방식의 오탐률/탐지율.

v3의 점수 정의가 이전 v2(emergent_window_v2.py, 삭제됨)와 동일하다는 것은 README "탐지기 v3" 절에 남긴 검증
결과로 이미 확인됐다. 이 스크립트는 그 비교가 아니라 실제 배포 방식(fit/detect)의 동작만 확인한다.

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
