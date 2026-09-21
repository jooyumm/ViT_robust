"""
probes/patch_fool_joint.py — [탐색적, 롤백 가능] P16과 P8을 동시에 속이는 "joint attack".

배경
----
지금까지의 PatchFool 공격은 원문 그대로 P16(타깃 분류기)만 아는 화이트박스 공격이라,
우리 탐지기·P8 전환 로직을 전혀 모른 채로 만들어졌다. 만약 공격자가 "P16이 실패하면 P8로
넘어간다"는 방어 구조 자체를 안다면, 처음부터 P16과 P8을 동시에 속이도록 공격을 만들 수
있다 — 이게 이 방어의 진짜 최악의 경우(adaptive attacker)를 흉내내는 stress test다.

방법
----
- 공격 영역(토큰)은 지금까지와 동일하게 P16의 attention(attn_layer_idx=4 기본값)으로 정한
  16x16 픽셀 영역을 그대로 사용한다 — 마스크는 raw pixel 영역이라 두 모델에 공통 적용 가능
  (perturbation은 픽셀 공간에서 일어나고, 각 모델이 그걸 각자의 방식으로 토큰화해서 봄).
- loss = CE(P16(perturbed), label) + CE(P8(perturbed), label) — 두 모델의 gradient를
  합쳐서 하나의 delta를 최적화한다 (CE_loss 모드와 동일한 Adam 루프, 손실만 두 모델 합).
- _build_mask, _collect_attn, _select_patch_attn은 원본에서 그대로 import(수정 없음).

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일도 지우면 원상복구.
"""
import torch
from src.attacks.patch_fool import _build_mask, _collect_attn, _select_patch_attn


def joint_patch_fool_attack(model16, model8, images, labels, device,
                            attn_layer_idx=4, num_patch=1,
                            train_attack_iters=250, attack_lr=0.22,
                            step_size=10, gamma=0.95, mild_l_inf=0.0):
    """
    P16 attention으로 공격 토큰 위치를 정하고, 그 물리적 영역을 P16+P8 CE loss 합으로
    동시에 공격한다.
    Returns: adv_images, true_idx (P16 attention 기준 선택된 토큰, (B,))
    """
    model16.eval()
    model8.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    attn_weights, hooks = _collect_attn(model16)
    with torch.no_grad():
        model16(images)
    for h in hooks:
        h.remove()
    max_patch_index = _select_patch_attn(attn_weights, attn_layer_idx, num_patch, device)

    mask = _build_mask(max_patch_index, B, H, 16, device)  # P16 토큰 기준 16px 영역

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
        model16.zero_grad()
        model8.zero_grad()
        opt.zero_grad()
        perturbed = original_images + torch.mul(delta, mask)
        out16 = model16(perturbed)
        out8 = model8(perturbed)
        loss = criterion(out16, labels) + criterion(out8, labels)
        loss.backward()
        if delta.grad is not None:
            delta.grad = -delta.grad
        opt.step()
        scheduler.step()
        with torch.no_grad():
            if mild_l_inf != 0.0:
                delta.data = delta.data.clamp(-mild_l_inf, mild_l_inf)

    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()
    return adv_images, max_patch_index[:, 0]
