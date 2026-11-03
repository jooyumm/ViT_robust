"""00 내부 공통 실행: CLI 조건을 받아 모델·배치 평가·최종 수치 저장를 한 번 수행한다."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks import pgd_attack, patch_fool_attack, patch_fool_config, lavan_attack
from metrics import compute_confusion_counts, compute_metrics_from_counts
from result_store import save_result


def build_attack_kwargs(args, attack_name):
    """args의 값 중 None이 아닌 것만 kwargs에 넣는다 — None으로 남은(CLI에서 지정 안 한)
    항목은 아예 안 넣어서, 호출되는 공격 함수 자신의 기본값이 적용되게 한다."""
    if attack_name == 'patch_fool':
        raw = {
            'attack_mode':        args.pf_attack_mode,
            'train_attack_iters': args.pf_iters,
            'num_patch':          args.pf_num_patch,
            'patch_select':       args.pf_patch_select,
            'mild_l_inf':         args.pf_mild_l_inf,
            'sparse_pixel_num': args.pf_sparse_pixel_num,
            'clip_valid_range': args.pf_clip_valid_range,
            'attention_source': args.pf_attention_source,
            'initialization': args.pf_initialization,
        }
    elif attack_name == 'pgd':
        raw = {'epsilon': args.pgd_epsilon, 'alpha': args.pgd_alpha, 'steps': args.pgd_steps}
    elif attack_name == 'lavan':
        raw = {'patch_ratio': args.lavan_patch_ratio, 'steps': args.lavan_steps,
               'alpha': args.lavan_alpha, 'fixed_loc': args.lavan_loc}
    else:
        raw = {}
    return {k: v for k, v in raw.items() if v is not None}

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



def run_case(args):
    """단일 모델·공격. 마지막 작은 batch도 평가하고 최종 JSON만 저장한다."""
    if len(args.attacks) != 1 or len(args.patch_sizes) != 1:
        raise ValueError('run_case에는 모델과 공격을 하나씩 전달하세요.')
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError('num_samples와 batch_size는 양수여야 합니다.')
    attack, patch = args.attacks[0], args.patch_sizes[0]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    folder = Path(args.result_dir)
    folder.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f'Attack: {attack} | P={patch} | Tokens: {(224//patch)**2}')
    print(f'Seed: {args.seed} | Samples: {args.num_samples} | Batch size: {args.batch_size}')
    loader, _ = get_dataloader(batch_size=args.batch_size, num_samples=args.num_samples, seed=args.seed)
    model = load_vit_model(patch, device, strict=True)
    kwargs = build_attack_kwargs(args, attack)
    if attack == 'patch_fool':
        print('Effective PatchFool config: '+json.dumps(patch_fool_config(patch_size_model=patch, **kwargs),sort_keys=True))
    counts = dict(A=0,B=0,C=0,D=0)
    for i, (images, labels) in enumerate(loader, 1):
        metrics, _ = run_attack_on_batch(model,images,labels,device,attack,patch,kwargs)
        for key in counts:
            counts[key] += metrics[key]
        print(f'Batch {i}/{len(loader)} | CA:{metrics["CA"]:.1f}% RA:{metrics["RA"]:.1f}% ASR:{metrics["ASR"]:.1f}%')
    overall = compute_metrics_from_counts(**counts)
    print('── Overall ──')
    print(f'CA  (Clean Accuracy)   : {overall["CA"]:.2f}%')
    print(f'RA  (Robust Accuracy)  : {overall["RA"]:.2f}%')
    print(f'ASR (Attack Success)   : {overall["ASR"]:.2f}%')
    print(f'Δ   (CA - RA)          : {overall["acc_drop"]:.2f}%p')
    print(f'Sample breakdown: {counts}, valid={overall["valid"]}')
    path=save_result(folder,dict(attack=attack,patch_size=patch,tag=args.tag,seed=args.seed,
                                num_samples=args.num_samples,batch_size=args.batch_size,
                                attack_config=patch_fool_config(patch_size_model=patch,**kwargs) if attack=='patch_fool' else kwargs,
                                metrics=overall))
    print(f'Result saved: {path}')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall
