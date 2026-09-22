import torch
import torch.nn.functional as F

# 우리가 쓰는 timm 체크포인트(augreg 계열)의 정규화 상수 — src/dataset.py와 동일 (mean=std=0.5)
_MU  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def pgd_attack(
    model,
    images,
    labels,
    device,
    epsilon=8/255,
    alpha=2/255,
    steps=40,
    random_start=True,
):
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    mu  = _MU.to(device)
    std = _STD.to(device)

    # 정규화 공간에서의 epsilon/alpha
    eps_t   = epsilon / std   # (3,1,1)
    alpha_t = alpha   / std

    lower = (0 - mu) / std
    upper = (1 - mu) / std

    if random_start:
        delta = (2 * eps_t * torch.rand_like(images) - eps_t)
    else:
        delta = torch.zeros_like(images)

    delta = delta.detach()

    for _ in range(steps):
        delta.requires_grad_(True)
        out  = model(images + delta)
        loss = F.cross_entropy(out, labels)
        loss.backward()

        with torch.no_grad():
            delta = delta + alpha_t * delta.grad.sign()
            delta = torch.max(torch.min(delta,  eps_t), -eps_t)   # L_inf box
            delta = torch.clamp(images + delta, lower, upper) - images  # 이미지 범위
        delta = delta.detach()

    return (images + delta).detach()