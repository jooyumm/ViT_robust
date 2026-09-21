"""
experiments/cost_comparison/cost_comparison_test.py — all_switch와 local_switch의 실제
파이프라인 비용(latency/FLOPs)을 같은 조건에서 직접 비교한다 (§10 + expected_cost_analysis.py
대체).

배경
----
기존 §10은 P8/P16을 각각 **고립된 상태**로(탐지 hook도 안 붙이고, escalate 판단도 없이)
벤치마크했다 — "P8이 P16보다 2.7배/4.5배 비싸다"는 사실은 맞지만, 이건 "P8을 통째로 다시
돌리는" all_switch의 실제 파이프라인 비용과 같지 않다(all_switch는 P16 forward도 먼저
해야 함). 그리고 **local_switch는 비용을 측정한 적이 자체가 없었다** — "패치 1개만 추가하니
거의 공짜일 것"이라는 주장이 토큰 수(196->200)로만 뒷받침되고 있었다.

방법
----
파이프라인을 두 단계로 나눠 측정한다:
1) **기저 비용(baseline)**: 모든 이미지에 항상 발생 — P16 예측 + 탐지(+ local_switch에 필요한
   위치특정까지) 를 **한 번의 forward**로 계산(detector.predict_detect_localize, patch_attack_detector
   프로젝트의 공유 탐지기).
   위치특정은 탐지와 같은 attention에서 나오므로 추가 forward가 필요 없다 — all_switch와
   local_switch가 이 기저 비용을 100% 공유한다.
2) **추가 비용(extra, escalate된 이미지에서만 발생)**: all_switch는
   defense.all_switch.apply_all_switch(model8, x)(P8 전체 재실행), local_switch는
   defense.local_switch.apply_local_switch(model16, model8, adapter, x, flag_idx)(패치
   1개만 국소 교체 + P16 12블록 재통과)를 각각 실제로 호출해 측정한다 — 두 "escalate 시
   추가로 도는 함수"를 완전히 동일한 방식(batch=1, warm-up, torch.cuda.synchronize())으로
   재는 것이 핵심.

주의: local_switch.apply_local_switch는 P8의 patch_embed를 이미지 전체(784개 서브패치)에
대해 계산하고 그중 4개만 쓴다 — 실제 필요한 것보다 더 계산하는 구현이라, 여기서 나오는
local_switch 비용은 "미래에 필요한 4개만 계산하도록 최적화하면 더 내려갈 수 있는" 보수적
(pessimistic) 상한이다. 이미 검증된 §16/§17/§18과 완전히 같은 mechanism 코드를 그대로
쓰는 것이므로 재구현하지 않았다.

E[cost(π)] 곡선(공격 비율 π에 따른 기대 비용)도 이 실험에서 함께 계산한다 — 기존
expected_cost_analysis.py의 아이디어를 두 방어 모두에 대해 확장한 것. FPR/recall은
experiments/system_comparison의 결과를 그대로 재사용한다(같은 탐지기라 두 방어에 동일).

사용법:
  python cost_comparison_test.py --n_warmup 20 --n_iters 100
  (같은 폴더에 system_comparison의 npz가 이미 있어야 π-sweep까지 계산됨 — 없으면 latency만)
"""
import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
DETECT_ROOT = os.path.join(SHARED_SRC_ROOT, 'patch_attack_detector')  # for `from detector.topk_mass_v1 import ...`
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, DETECT_ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from detector.topk_mass_v1 import predict_detect_localize
from defense.all_switch import apply_all_switch
from defense.local_switch import fit_adapter, apply_local_switch


def benchmark_fn(fn, device, n_warmup, n_iters):
    """fn: 인자 없는 호출 가능 객체 (이미 입력을 클로저로 들고 있음)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for _ in range(n_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                       if device.type == 'cuda' else float('nan'))

    times = np.array(times)
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            fn()
    flops = fcm.get_total_flops()

    return dict(mean_ms=times.mean(), std_ms=times.std(), median_ms=np.median(times),
                peak_mem_mb=peak_mem_mb, flops=flops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_warmup', type=int, default=20)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--calib_seed', type=int, default=42)
    parser.add_argument('--num_calib', type=int, default=100, help='local_switch 어댑터 피팅용')
    parser.add_argument('--detect_layer', type=int, default=12)
    args = parser.parse_args()

    device = get_device()
    if device.type != 'cuda':
        print("[경고] device.type != 'cuda' — 이 클러스터는 간헐적으로 CUDA 초기화에 실패해 "
              "CPU로 조용히 넘어가는 일이 있었다(README 참고). latency 숫자를 논문에 쓰기 전에 "
              "반드시 GPU에서 재확인할 것.")

    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    calib_loader, _ = get_dataloader(batch_size=args.num_calib, num_samples=args.num_calib,
                                      seed=args.calib_seed)
    calib_images, _ = next(iter(calib_loader))
    calib_images = calib_images.to(device)
    adapter = fit_adapter(model16, model8, calib_images)
    print(f"어댑터 피팅 완료 (calibration {args.num_calib}개, seed={args.calib_seed})")

    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, flag_idx1 = predict_detect_localize(model16, x, detect_layer=args.detect_layer)

    print("\n=== (1) 기저 비용 baseline: P16 예측 + 탐지 + 위치특정 (한 번의 forward, 모든 이미지에 발생) ===")
    r_base = benchmark_fn(lambda: predict_detect_localize(model16, x, detect_layer=args.detect_layer),
                           device, args.n_warmup, args.n_iters)
    print(f"  {r_base['mean_ms']:.3f} +- {r_base['std_ms']:.3f} ms  "
          f"peak_mem={r_base['peak_mem_mb']:.1f}MB  {r_base['flops']/1e9:.3f} GFLOPs")

    print("\n=== (2) all_switch 추가 비용: apply_all_switch (escalate된 이미지에서만 발생) ===")
    r_all = benchmark_fn(lambda: apply_all_switch(model8, x), device, args.n_warmup, args.n_iters)
    print(f"  {r_all['mean_ms']:.3f} +- {r_all['std_ms']:.3f} ms  "
          f"peak_mem={r_all['peak_mem_mb']:.1f}MB  {r_all['flops']/1e9:.3f} GFLOPs")

    print("\n=== (3) local_switch 추가 비용: apply_local_switch (escalate된 이미지에서만 발생) ===")
    r_local = benchmark_fn(
        lambda: apply_local_switch(model16, model8, adapter, x, flag_idx1),
        device, args.n_warmup, args.n_iters)
    print(f"  {r_local['mean_ms']:.3f} +- {r_local['std_ms']:.3f} ms  "
          f"peak_mem={r_local['peak_mem_mb']:.1f}MB  {r_local['flops']/1e9:.3f} GFLOPs")

    print(f"\n{'조건':<45}{'latency(ms)':>16}{'GFLOPs':>12}")
    print("-" * 75)
    print(f"{'baseline (P16+detect+localize)':<45}{r_base['mean_ms']:>10.3f}+-{r_base['std_ms']:<5.3f}"
          f"{r_base['flops']/1e9:>12.3f}")
    print(f"{'+ all_switch extra (P8 재실행)':<45}{r_all['mean_ms']:>10.3f}+-{r_all['std_ms']:<5.3f}"
          f"{r_all['flops']/1e9:>12.3f}")
    print(f"{'+ local_switch extra (국소 교체)':<45}{r_local['mean_ms']:>10.3f}+-{r_local['std_ms']:<5.3f}"
          f"{r_local['flops']/1e9:>12.3f}")
    total_all_ms = r_base['mean_ms'] + r_all['mean_ms']
    total_local_ms = r_base['mean_ms'] + r_local['mean_ms']
    print(f"\n  escalate 시 총 latency: all_switch={total_all_ms:.3f}ms "
          f"({total_all_ms/r_base['mean_ms']:.2f}x baseline), "
          f"local_switch={total_local_ms:.3f}ms ({total_local_ms/r_base['mean_ms']:.2f}x baseline)")

    # ── E[cost(π)]: system_comparison의 FPR/recall을 재사용해 π-sweep ──────
    sys_cand = glob.glob(os.path.join(HERE.replace('/cost_comparison', '/system_comparison'),
                                       'system_comparison_n*.npz'))
    pi_sweep = None
    if sys_cand:
        sys_npz = np.load(max(sys_cand, key=os.path.getmtime))
        fpr, recall = float(sys_npz['fpr']), float(sys_npz['recall'])
        pis = np.linspace(0, 0.5, 200)
        p_escalate = (1 - pis) * fpr + pis * recall
        exp_all_ms = r_base['mean_ms'] + p_escalate * r_all['mean_ms']
        exp_local_ms = r_base['mean_ms'] + p_escalate * r_local['mean_ms']
        naive_ms = r_base['mean_ms'] + r_all['mean_ms']  # 매번 P8까지 돎(all_switch 기준 naive)
        pi_sweep = dict(pis=pis, p_escalate=p_escalate,
                         exp_all_ms=exp_all_ms, exp_local_ms=exp_local_ms, naive_ms=naive_ms,
                         fpr=fpr, recall=recall)
        print(f"\n[π-sweep] system_comparison의 FPR={fpr:.3f}/recall={recall:.3f} 재사용")
        for pi in (0.0, 0.01, 0.1, 0.5):
            pe = (1 - pi) * fpr + pi * recall
            print(f"  π={pi*100:5.1f}%  P(escalate)={pe*100:5.1f}%  "
                  f"all_switch={r_base['mean_ms']+pe*r_all['mean_ms']:.2f}ms  "
                  f"local_switch={r_base['mean_ms']+pe*r_local['mean_ms']:.2f}ms")
    else:
        print("\n[참고] results/system_comparison/에 npz가 아직 없어 π-sweep은 건너뜀 "
              "(system_comparison_test.py를 먼저 돌릴 것)")

    out_dir = HERE
    os.makedirs(out_dir, exist_ok=True)
    save_dict = dict(
        base_mean_ms=r_base['mean_ms'], base_std_ms=r_base['std_ms'], base_flops=r_base['flops'],
        base_peak_mem_mb=r_base['peak_mem_mb'],
        all_extra_mean_ms=r_all['mean_ms'], all_extra_std_ms=r_all['std_ms'], all_extra_flops=r_all['flops'],
        all_extra_peak_mem_mb=r_all['peak_mem_mb'],
        local_extra_mean_ms=r_local['mean_ms'], local_extra_std_ms=r_local['std_ms'],
        local_extra_flops=r_local['flops'], local_extra_peak_mem_mb=r_local['peak_mem_mb'],
        device_type=device.type,
    )
    if pi_sweep is not None:
        save_dict.update({f'pi_{k}': v for k, v in pi_sweep.items()})
    np.savez(os.path.join(out_dir, 'cost_comparison.npz'), **save_dict)
    print(f"\nSaved: {os.path.join(out_dir, 'cost_comparison.npz')}")


if __name__ == '__main__':
    main()
