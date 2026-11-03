"""탐지된 사분면에 적용할 방어를 변환 종류·강도·경계 세 축으로 무작위화한다.

탐지(PatchAttentionDetector.predict)는 region_switch의 것을 그대로 쓰고, 이 모듈은
flag==True로 판정된 사분면에 "무엇을·얼마나·어디까지" 적용할지만 담당한다.
patch_embed 변환은 경계를 넘어 섞이지 않으므로 패치(16px) 단위로만 지터링한다.
"""
import torch
import torch.nn.functional as F

GRID, PATCH = 14, 16


def region_bounds(region, jitter, rng):
    """사분면(0-3)의 기본 7x7 경계를 패치 단위로 +-jitter만큼 무작위로 넓히거나 좁힌다."""
    row0, col0 = divmod(region, 2)
    r0, r1, c0, c1 = row0 * 7, row0 * 7 + 7, col0 * 7, col0 * 7 + 7
    if jitter:
        r0 = max(0, r0 + int(rng.integers(-jitter, jitter + 1)))
        r1 = min(GRID, r1 + int(rng.integers(-jitter, jitter + 1)))
        c0 = max(0, c0 + int(rng.integers(-jitter, jitter + 1)))
        c1 = min(GRID, c1 + int(rng.integers(-jitter, jitter + 1)))
        if r1 <= r0:
            r1 = r0 + 1
        if c1 <= c0:
            c1 = c0 + 1
    return r0, r1, c0, c1


def subdivide_one(model16, kernel8, pos8, patch_embed_with_pos_fn, forward_with_replacement_mass_fn,
                   image, bounds, key_weight):
    """경계 안의 P16 패치들을 8x8 토큰으로 교체한다(기존 switch.py의 고정 사분면 버전을 가변 경계로 일반화)."""
    r0, r1, c0, c1 = bounds
    patch16 = patch_embed_with_pos_fn(model16, image)
    cls = model16.cls_token + model16.pos_embed[:, :1]
    crop = image[:, :, r0 * PATCH:r1 * PATCH, c0 * PATCH:c1 * PATCH]
    sub = F.conv2d(crop, kernel8, model16.patch_embed.proj.bias, stride=8).flatten(2).transpose(1, 2)
    sub = model16.patch_embed.norm(sub)
    position = pos8[:, r0 * 2:r1 * 2, c0 * 2:c1 * 2].reshape(1, -1, pos8.shape[-1])
    replacement = sub + position
    keep = torch.ones((GRID, GRID), dtype=torch.bool, device=image.device)
    keep[r0:r1, c0:c1] = False
    sequence = torch.cat([cls[0], patch16[0, keep.flatten()], replacement[0]], dim=0).unsqueeze(0)
    return forward_with_replacement_mass_fn(model16, sequence, replacement_count=replacement.shape[1], key_weight=key_weight)


def merge_one(model16, patch_embed_with_pos_fn, full_forward_from_tokens_fn, image, bounds, k):
    """경계 안의 P16 패치들을 k개로 평균 풀링해서 합친다(세분화의 반대 방향, 토큰 수를 줄임)."""
    r0, r1, c0, c1 = bounds
    patch16 = patch_embed_with_pos_fn(model16, image)
    cls = model16.cls_token + model16.pos_embed[:, :1]
    keep = torch.ones((GRID, GRID), dtype=torch.bool, device=image.device)
    keep[r0:r1, c0:c1] = False
    region_idx = (~keep).flatten().nonzero(as_tuple=True)[0]
    region_tokens = patch16[0, region_idx]
    k = max(1, min(k, len(region_idx)))
    groups = torch.chunk(region_tokens, k, dim=0)
    merged = torch.stack([g.mean(0) for g in groups], dim=0)
    sequence = torch.cat([cls[0], patch16[0, keep.flatten()], merged], dim=0).unsqueeze(0)
    return full_forward_from_tokens_fn(model16, sequence)


def randomized_defend(model16, kernel8, pos8, fns, image, region, rng,
                       jitter=1, key_weight_range=(0.15, 0.35), k_range=(1, 4)):
    """flag==True 사분면 하나에 변환 종류·강도·경계를 동시에 무작위로 골라 방어를 적용한다.

    fns는 (patch_embed_with_pos, full_forward_from_tokens, forward_with_replacement_mass)
    튜플 — region_switch/core/switch.py의 함수를 그대로 전달받는다.
    """
    patch_embed_with_pos_fn, full_forward_from_tokens_fn, forward_with_replacement_mass_fn = fns
    bounds = region_bounds(region, jitter, rng)
    kind = 'subdivide' if rng.integers(0, 2) else 'merge'
    if kind == 'subdivide':
        key_weight = float(rng.uniform(*key_weight_range))
        logits = subdivide_one(model16, kernel8, pos8, patch_embed_with_pos_fn,
                                forward_with_replacement_mass_fn, image, bounds, key_weight)
        choice = dict(kind=kind, bounds=bounds, key_weight=round(key_weight, 3))
    else:
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        logits = merge_one(model16, patch_embed_with_pos_fn, full_forward_from_tokens_fn, image, bounds, k)
        choice = dict(kind=kind, bounds=bounds, k=k)
    return logits, choice
