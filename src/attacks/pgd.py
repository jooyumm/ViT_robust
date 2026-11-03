"""공유 전역 L∞ PGD. epsilon/alpha는 [0,1] 픽셀 단위, 입력은 mean/std=.5 정규화.

기존 00의 최종 iterate·random-start·projection 의미를 유지한다.
모델 모드/가중치/기존 parameter gradient는 보존하고 입력 gradient만 계산한다.
"""
import torch
import torch.nn.functional as F
from ..attack_utils import attack_context

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
    with attack_context(model, device=device):
        if (images.ndim != 4 or not len(images) or images.shape[1] != 3
                or labels.shape != (len(images),) or not torch.isfinite(images).all()):
            raise ValueError('유한한 BCHW RGB 입력과 B labels가 필요합니다.')
        if (epsilon < 0 or alpha <= 0 or not isinstance(steps, int) or steps < 1):
            raise ValueError('epsilon>=0, alpha>0, 양의 정수 steps가 필요합니다.')
        if images.min() < -1.00001 or images.max() > 1.00001:
            raise ValueError('mean/std=.5 정규화의 유효 이미지 범위가 필요합니다.')
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
            grad = torch.autograd.grad(loss, delta)[0]

            with torch.no_grad():
                delta = delta + alpha_t * grad.sign()
                delta = torch.max(torch.min(delta,  eps_t), -eps_t)   # L_inf box
                delta = torch.clamp(images + delta, lower, upper) - images  # 이미지 범위
            delta = delta.detach()

        return (images + delta).detach()
