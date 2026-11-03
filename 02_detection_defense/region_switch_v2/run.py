"""region_switch의 탐지는 그대로 쓰고, 방어 응답만 세 축(변환 종류·강도·경계)으로 무작위화한다.

학습·보정 없이 공격 여부를 모르는 입력에 적용한다. flag==True인 사분면마다 매번
P16->P8 세분화와 토큰 병합(49->소수) 중 하나를 무작위로 고르고, 그 변환의 강도와
교체 경계도 함께 무작위화한다 -- 고정된 단일 변환에 맞춰 최적화하는 적응형 공격을
어렵게 만드는 것이 목적이며, 이 자체가 공격을 막는다는 증명은 아니다.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '02_detection_defense')]
from src.models import load_vit_model
from src.data.paths import PATCHFOOL
from src.data.protocol import validate_attack_manifest
from region_switch.core.attention import AttentionCapture, PatchAttentionDetector
from region_switch.core.switch import (converted_patch_kernel, build_region_swap_batch, forward_with_replacement_mass,
                    full_forward_from_tokens, patch_embed_with_pos)
from src.provenance import model_hash
from defense_common.evaluation import evaluate_pairs
from defense_common.plotting import plot_comparison
from region_switch_v2.randomize import randomized_defend, region_bounds, GRID


@torch.inference_mode()
def verify_consistency(model, kernel8, pos8, image):
    """v2의 가변 경계 함수가 기존 region_switch 고정 사분면 결과와 일치하는지, 병합이 permutation만일 때
    baseline과 완전히 같은지 확인한다. 둘 다 학습이 아니라 고정 수식·행렬 연산의 동치성 검증이다."""
    from region_switch_v2.randomize import subdivide_one, merge_one
    fns = (patch_embed_with_pos, full_forward_from_tokens, forward_with_replacement_mass)
    patch = patch_embed_with_pos(model, image)
    cls = model.cls_token + model.pos_embed[:, :1]
    baseline = full_forward_from_tokens(model, torch.cat([cls, patch], dim=1))
    for region in range(4):
        bounds = region_bounds(region, jitter=0, rng=None)  # jitter=0 -> 항상 기존 7x7 경계
        v2_out = subdivide_one(model, kernel8, pos8, patch_embed_with_pos, forward_with_replacement_mass,
                                image, bounds, key_weight=.25)
        v1_out = forward_with_replacement_mass(model, build_region_swap_batch(model, kernel8, image, torch.tensor([region], device=image.device)), key_weight=.25)
        torch.testing.assert_close(v2_out, v1_out, atol=2e-4, rtol=2e-4)
        n_region = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2])
        identity = merge_one(model, patch_embed_with_pos, full_forward_from_tokens, image, bounds, k=n_region)
        torch.testing.assert_close(identity, baseline, atol=2e-4, rtol=2e-4)
    print('검증: 가변 경계 세분화가 기존 고정 사분면과 일치, k=영역 크기 병합은 순서만 바뀐 것이라 baseline과 일치', flush=True)


@torch.inference_mode()
def predict_defended(model, kernel8, pos8, images, detector, rng, jitter, key_weight_range, k_range):
    """입력은 이미지뿐이다. 공격 여부, 정상 짝, label, GT를 받지 않는다. baseline/탐지는 배치로 묶고,
    flag==True인 것만 무작위 응답(변환 종류가 매번 달라 배치 불가)을 한 장씩 적용한다."""
    capture = AttentionCapture(model)
    baseline_logits = capture(images)
    decision = detector.predict(images, capture.arrays())
    logits = baseline_logits.clone()
    choices = [None] * len(images)
    fns = (patch_embed_with_pos, full_forward_from_tokens, forward_with_replacement_mass)
    for i in np.flatnonzero(decision['flag']):
        region = int(decision['region'][i])
        out, choice = randomized_defend(model, kernel8, pos8, fns, images[i:i+1], region, rng,
                                         jitter=jitter, key_weight_range=key_weight_range, k_range=k_range)
        logits[i] = out[0]
        choices[i] = choice
    return baseline_logits, logits, decision, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/comparison.png')
    parser.add_argument('--jitter', type=int, default=1, help='사분면 경계를 늘리고 줄이는 패치 단위 범위')
    parser.add_argument('--batch-size', type=int, default=4, help='baseline/탐지 배치 크기(무작위 응답은 1장씩)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset',type=Path,default=PATCHFOOL)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or not 0 <= args.jitter <= 6:
        raise ValueError('양의 batch-size와 jitter 0..6이 필요합니다.')
    if args.output.suffix.lower() != '.png':
        raise ValueError('PNG 경로가 필요합니다.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    validate_attack_manifest(manifest)
    model = load_vit_model(16, torch.device('cpu'))
    if model is None:
        raise RuntimeError('P16 체크포인트 로드 실패')
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.attn.fused_attn = False
    original_hash = model_hash(model)
    assert original_hash == manifest['model_sha256']
    kernel8 = converted_patch_kernel(model)
    import torch.nn.functional as F
    pos16 = model.pos_embed[:, 1:].reshape(1, GRID, GRID, -1).permute(0, 3, 1, 2)
    pos8 = F.interpolate(pos16, size=(28, 28), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    detector = PatchAttentionDetector()  # fit/calibration 없음. 모든 상수는 region_switch/core/attention.py에 고정.
    rng = np.random.default_rng(args.seed)
    key_weight_range, k_range = (0.15, 0.35), (1, 4)
    with np.load(args.dataset / manifest['chunks'][0]['path']) as first:
        verify_consistency(model, kernel8, pos8, torch.from_numpy(first['clean__images'][:1]))

    report, kinds = evaluate_pairs(
        lambda images: predict_defended(model, kernel8, pos8, images, detector, rng,
                                       args.jitter, key_weight_range, k_range), manifest,
        batch_size=args.batch_size, seed=args.seed, run=args.dataset, progress='무학습 랜덤화 방어 평가')
    assert model_hash(model) == original_hash
    report.update(subdivide_chosen=int((kinds['patchfool']=='subdivide').sum()),
                  merge_chosen=int((kinds['patchfool']=='merge').sum()), jitter_patches=args.jitter,
                  key_weight_range=list(key_weight_range), k_range=list(k_range), seed=args.seed,
                  unchanged_pretrained_weights=True, training=False, calibration=False,
                  threshold_selection=False, fixed_spatial_z_threshold=detector.threshold,
                  fixed_neighbor_ratio=detector.neighbor_ratio_threshold,
                  embedding='fixed P16 kernel conversion to 8x8; not a separately pretrained ViT-P8',
                  elapsed_seconds=time.monotonic()-started)
    plot_comparison(report, args.output, f'Randomized region switch (n={report["n"]})')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
