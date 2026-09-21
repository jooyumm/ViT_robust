"""
probes/patch_fool_lowsal.py — [탐색적, 롤백 가능] PatchFool의 "위치 선택"만 바꾼 변형들.

원본 patch_fool_attack(patch_select='Attn')은 attention이 가장 높은 토큰을 공격 대상으로
고른다. 이 파일은 그 대신 이미지 자체의 saliency(패치 내부 픽셀 표준편차, 텍스처가 적을수록
낮음 — 모델 attention과 무관, 순수 픽셀 통계)를 기준으로 위치를 강제하는 변형들을 제공한다:
  - select_lowest_saliency_token : saliency 최저 토큰 (1차 회피 시도, 결과: 회피 안 됨)
  - select_range_saliency_token  : saliency가 특정 [lo,hi] 구간(실측한 실패 샘플 범위)인 토큰

CE_loss 공격 루프 자체는 patch_fool_attack의 CE_loss 경로와 동일(공정한 비교를 위해) —
_run_ce_attack_loop()로 공용화해서 선택 전략별로 재사용한다.
_build_mask는 원본에서 그대로 import(수정 없음).

주의: src/attacks/patch_fool.py는 import만 하고 전혀 수정하지 않음. 지우면 그만.
"""
import torch
from src.attacks.patch_fool import _build_mask


def compute_patch_saliency(images, patch_size):
    """패치별 saliency 프록시 (B, num_patches) — 낮을수록 밋밋(저saliency)."""
    B, C, H, W = images.shape
    ppl = H // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, ppl * ppl, patch_size * patch_size)
    std = patches.std(dim=-1).mean(dim=1)  # (B, num_patches), RGB 평균
    return std


def select_lowest_saliency_token(images, patch_size, num_patch, device):
    sal = compute_patch_saliency(images, patch_size).to(device)
    return sal.argsort(dim=1, descending=False)[:, :num_patch]


def select_range_saliency_token(images, patch_size, lo, hi, num_patch, device):
    """saliency가 [lo, hi] 구간 안인 토큰을 우선 선택. 구간 안에 후보가 없는 이미지는
    구간에 가장 가까운(모자라거나 넘치는 정도가 최소인) 토큰을 대신 선택한다."""
    sal = compute_patch_saliency(images, patch_size).to(device)  # (B, num_patches)
    dist = torch.clamp(lo - sal, min=0) + torch.clamp(sal - hi, min=0)  # 구간 안이면 0
    return dist.argsort(dim=1, descending=False)[:, :num_patch]


def _run_ce_attack_loop(model, images, labels, device, max_patch_index, patch_size_model,
                        train_attack_iters=250, attack_lr=0.22, step_size=10, gamma=0.95,
                        mild_l_inf=0.0):
    """patch_fool_attack(attack_mode='CE_loss')와 동일한 공격 최적화 루프.
    위치(max_patch_index)는 밖에서 미리 정해서 넘긴다."""
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    mask = _build_mask(max_patch_index, B, H, patch_size_model, device)

    if mild_l_inf == 0.0:
        delta = torch.randn_like(images) * 0.1
    else:
        delta = (2 * mild_l_inf * torch.rand_like(images) - mild_l_inf)
    original_images = images.clone()
    delta = delta.to(device).requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=attack_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    for it in range(train_attack_iters):
        model.zero_grad()
        opt.zero_grad()
        perturbed = original_images + torch.mul(delta, mask)
        out = model(perturbed)
        ce_loss = criterion(out, labels)
        ce_loss.backward()
        if delta.grad is not None:
            delta.grad = -delta.grad
        opt.step()
        scheduler.step()
        with torch.no_grad():
            if mild_l_inf != 0.0:
                delta.data = delta.data.clamp(-mild_l_inf, mild_l_inf)

    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()
    return adv_images


def patch_fool_attack_lowsal(model, images, labels, device, patch_size_model=16,
                             num_patch=1, train_attack_iters=250, attack_lr=0.22,
                             step_size=10, gamma=0.95, mild_l_inf=0.0):
    """토큰 선택 = saliency 최저. Returns: adv_images, max_patch_index."""
    max_patch_index = select_lowest_saliency_token(images, patch_size_model, num_patch, device)
    adv_images = _run_ce_attack_loop(
        model, images, labels, device, max_patch_index, patch_size_model,
        train_attack_iters, attack_lr, step_size, gamma, mild_l_inf)
    return adv_images, max_patch_index


def patch_fool_attack_rangesal(model, images, labels, device, lo, hi, patch_size_model=16,
                               num_patch=1, train_attack_iters=250, attack_lr=0.22,
                               step_size=10, gamma=0.95, mild_l_inf=0.0):
    """토큰 선택 = saliency가 [lo,hi] 구간(실측한 실패 샘플 범위). Returns: adv_images, max_patch_index."""
    max_patch_index = select_range_saliency_token(images, patch_size_model, lo, hi, num_patch, device)
    adv_images = _run_ce_attack_loop(
        model, images, labels, device, max_patch_index, patch_size_model,
        train_attack_iters, attack_lr, step_size, gamma, mild_l_inf)
    return adv_images, max_patch_index
