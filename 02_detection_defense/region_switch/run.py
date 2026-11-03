"""학습·보정 없이 공격 여부를 모르는 입력에 같은 탐지·국소 8×8 분할 정책을 적용한다."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'02_detection_defense')]
from src.models import load_vit_model
from src.data.paths import PATCHFOOL
from src.data.protocol import validate_attack_manifest
from region_switch.core.attention import PatchAttentionDetector
from region_switch.core.switch import converted_patch_kernel
from region_switch.core.inference import predict_defended,verify_transform
from src.provenance import model_hash
from defense_common.evaluation import evaluate_pairs
from defense_common.plotting import plot_comparison



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/comparison.png')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--run',type=Path,default=PATCHFOOL,help='저장된 기본 500장 공격 데이터셋')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.output.suffix.lower()!='.png' or args.batch_size<1:raise ValueError('PNG 경로와 양의 batch-size가 필요합니다.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    # The fixed kernel identity checks need full FP32 convolution/matmul.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    manifest=json.loads((args.run/'manifest.json').read_text());validate_attack_manifest(manifest)
    model=load_vit_model(16,torch.device(args.device))
    if model is None:raise RuntimeError('P16 체크포인트 로드 실패')
    model.eval().requires_grad_(False)
    for block in model.blocks:block.attn.fused_attn=False
    original_hash=model_hash(model)
    assert original_hash==manifest['model_sha256']
    kernel8=converted_patch_kernel(model)
    detector=PatchAttentionDetector()  # fit/calibration 없음. 모든 상수는 attention.py에 고정.
    with np.load(args.run/manifest['chunks'][0]['path']) as first:
        verify_transform(model,kernel8,torch.from_numpy(first['clean__images'][:1]).to(args.device))
    report, _ = evaluate_pairs(
        lambda images: predict_defended(model, kernel8, images.to(args.device), detector), manifest,
        batch_size=args.batch_size, seed=42, run=args.run, progress='무학습 혼합 입력 평가')
    assert model_hash(model) == original_hash
    report.update(unchanged_pretrained_weights=True, training=False, calibration=False,
                  threshold_selection=False, fixed_spatial_z_threshold=detector.threshold,
                  fixed_neighbor_ratio=detector.neighbor_ratio_threshold,
                  embedding='fixed P16 kernel conversion to 8x8; not a separately pretrained ViT-P8',
                  elapsed_seconds=time.monotonic()-started, attack_protocol=manifest['patchfool_protocol'],
                  attack_target=manifest['attack_target'], defense_aware_attack=False,
                  defense_response='detected quadrant -> 8x8 subdivision; no merging or random response')
    report['attack_archive']=str(args.run.resolve())
    plot_comparison(report, args.output, f'Clean / attacked / defended (n={report["n"]})')
    result=args.output.with_suffix('.json')
    pending=result.with_suffix('.pending')
    pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(result)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
