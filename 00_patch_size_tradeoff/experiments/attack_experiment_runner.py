import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

import torch
from src.attacks.patch_fool import patch_fool_attack
from pgd import pgd_attack        # 00_patch_size_tradeoff-only attack, not shared
from src.attacks.lavan import lavan_attack
from metrics import (compute_confusion_counts, compute_metrics_from_counts,  # 00_patch_size_tradeoff-only
                      aggregate_counts)


def run_attack_on_batch(model, images, labels, device,
                        attack_name: str,
                        patch_size_model: int = 16,
                        attack_kwargs: dict = None):
    """
    배치 하나에 대해 공격 실행 → 지표 반환.

    Returns
    -------
    metrics    : dict  (CA, RA, ASR, A, B, C, D, valid, acc_drop) — 케이스 정의는 metrics.py 참고
    adv_images : Tensor
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    kwargs = attack_kwargs or {}

    with torch.no_grad():
        preds_clean = model(images).argmax(dim=1).cpu().tolist()

    if attack_name == 'patch_fool':
        adv_images, _ = patch_fool_attack(
            model, images, labels, device,
            patch_size_model=patch_size_model, **kwargs)
    elif attack_name == 'pgd':
        adv_images = pgd_attack(model, images, labels, device, **kwargs)
    elif attack_name == 'lavan':
        adv_images = lavan_attack(model, images, labels, device, **kwargs)
    else:
        raise ValueError(f"Unknown attack: '{attack_name}'")

    with torch.no_grad():
        preds_adv = model(adv_images).argmax(dim=1).cpu().tolist()

    labels_list = labels.cpu().tolist()

    # -1 (remapping 실패) 샘플 제외
    filtered = [(pc, pa, l) for pc, pa, l in
                zip(preds_clean, preds_adv, labels_list) if l != -1]
    total = len(labels_list)

    if filtered:
        pc_list, pa_list, l_list = zip(*filtered)
        counts  = compute_confusion_counts(pc_list, pa_list, l_list)
        metrics = compute_metrics_from_counts(**counts)
    else:
        metrics = compute_metrics_from_counts(0, 0, 0, 0)

    metrics['total'] = total
    return metrics, adv_images


def run_experiment(model, loader, device, attack_name, patch_size_model,
                   num_batches, attack_kwargs):
    """loader에서 배치를 num_batches개 뽑아 run_attack_on_batch()를 반복 실행하고 누적 지표를 낸다."""
    model.eval()
    batch_counts = []   # 각 배치의 {'A','B','C','D'} — 케이스 정의는 metrics.py 참고

    batch_iter = iter(loader)
    for b_idx in range(num_batches):
        try:
            images, labels = next(batch_iter)
        except StopIteration:
            break

        metrics, _ = run_attack_on_batch(
            model, images, labels, device,
            attack_name=attack_name,
            patch_size_model=patch_size_model,
            attack_kwargs=attack_kwargs,
        )

        batch_counts.append({k: metrics[k] for k in ('A', 'B', 'C', 'D')})

        print(f"    Batch {b_idx+1:2d}/{num_batches} | "
              f"CA:{metrics['CA']:5.1f}% "
              f"RA:{metrics['RA']:5.1f}% "
              f"ASR:{metrics['ASR']:5.1f}% "
              f"[A:{metrics['A']} B:{metrics['B']} "
              f"C:{metrics['C']} D:{metrics['D']}]")

    total_counts = aggregate_counts(*batch_counts)
    return compute_metrics_from_counts(**total_counts)
