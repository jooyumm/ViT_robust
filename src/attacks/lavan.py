"""LaVAN: 고정 픽셀 면적(patch_ratio)의 정사각 패치 하나를 PGD로 최적화하는 공격.
반환값은 `adv_images`뿐이며 `return_info`는 지원하지 않는다(GT는 호출자가 원본과의
픽셀 차이로 직접 계산). `fixed_loc=None`이면 배치 내 모든 샘플이 같은 무작위 위치를
공유한다(논문 기본 동작) — 표본별로 다른 위치를 쓰지 않는다는 점에 주의.
"""
import torch
import torch.nn.functional as F
from ..attack_utils import attack_context

# 우리가 쓰는 timm 체크포인트(augreg 계열)의 정규화 상수 — src/dataset.py와 동일 (mean=std=0.5)
_MU  = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)

def lavan_attack(
    model,
    images,
    labels,
    device,
    patch_ratio=0.02,
    steps=40,
    alpha=2/255,
    targeted=False,
    target_labels=None,
    fixed_loc=None,        # None → 랜덤 위치(기존). 'center' → 항상 이미지 중앙 고정
):
    with attack_context(model, device=device):
        images = images.clone().detach().to(device)
        labels = labels.to(device=device, dtype=torch.long)
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
            raise ValueError('images는 비어 있지 않은 B×3×H×W 텐서여야 합니다.')
        B, C, H, W = images.shape
        if labels.shape != (B,) or not torch.isfinite(images).all():
            raise ValueError('labels는 B 크기이고 images는 유한해야 합니다.')
        if not isinstance(steps, int) or steps < 1 or not alpha > 0 or not 0 < patch_ratio <= 1:
            raise ValueError('steps/alpha는 양수, patch_ratio는 (0,1]이어야 합니다.')
        if targeted:
            if target_labels is None or target_labels.shape != labels.shape:
                raise ValueError('targeted=True에는 B 크기 target_labels가 필요합니다.')
            target_labels = target_labels.to(device=device, dtype=torch.long)
        if fixed_loc not in (None, 'center'):
            raise ValueError('fixed_loc는 None 또는 center여야 합니다.')

        mu  = _MU.to(device)
        std = _STD.to(device)

        # 패치 크기 계산 (면적 기준 patch_ratio)
        patch_px = max(1, int((H * W * patch_ratio) ** 0.5))
        if patch_px > min(H, W):
            raise ValueError('정사각 패치가 이미지보다 큽니다. patch_ratio를 줄이세요.')
        max_top, max_left = H - patch_px, W - patch_px

        if fixed_loc == 'center':
            # 위치를 항상 이미지 중앙으로 고정 — 위치 변동성을 배제하고
            # 같은 물리적 영역에서 patch size별 강건성을 비교하기 위함
            s_h = max_top // 2
            s_w = max_left // 2
        else:
            # 패치 위치 랜덤 선택 (배치 내 모든 샘플 동일 위치 — 논문 기본)
            s_h = torch.randint(0, max_top + 1, (1,)).item()
            s_w = torch.randint(0, max_left + 1, (1,)).item()
        e_h = s_h + patch_px
        e_w = s_w + patch_px

        # 정규화 공간 alpha
        alpha_t = alpha / std.to(device)

        # perturbation 초기화 (패치 영역만)
        delta = torch.zeros_like(images)

        for _ in range(steps):
            delta.requires_grad_(True)
            out = model(images + delta)

            if targeted:
                t    = target_labels.to(device)
                loss = -F.cross_entropy(out, t)
            else:
                loss = F.cross_entropy(out, labels)

            gradient = torch.autograd.grad(loss, delta)[0]

            with torch.no_grad():
                grad_sign = gradient.detach().sign()
                new_delta  = delta.detach().clone()
                # 패치 영역만 업데이트
                new_delta[:, :, s_h:e_h, s_w:e_w] += \
                    alpha_t * grad_sign[:, :, s_h:e_h, s_w:e_w]
                # 정규화 공간 [-mu/std, (1-mu)/std] clamp
                lo = (-mu / std).to(device)
                hi = ((1 - mu) / std).to(device)
                new_delta[:, :, s_h:e_h, s_w:e_w] = torch.max(
                    torch.min(
                        new_delta[:, :, s_h:e_h, s_w:e_w],
                        hi - images[:, :, s_h:e_h, s_w:e_w]
                    ),
                    lo - images[:, :, s_h:e_h, s_w:e_w]
                )
                # 패치 외부 영역 0으로 유지
                new_delta[:, :, :s_h, :]  = 0
                new_delta[:, :, e_h:, :]  = 0
                new_delta[:, :, :, :s_w]  = 0
                new_delta[:, :, :, e_w:]  = 0
                delta = new_delta

        return (images + delta).detach()
