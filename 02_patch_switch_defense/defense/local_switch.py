"""
defense/local_switch.py — "국소 전환" 방어 메커니즘의 단일 정본(canonical) 구현.

배경
----
이 프로젝트의 목표(README 참고). 탐지기가 flag한 P16 패치 1개만 4개의 P8 서브패치로
국소 교체하고, 나머지 195개 P16 토큰과 12개 transformer block은 전혀 건드리지 않는다.
P8 서브패치는 closed-form(최소제곱, gradient descent 없음) 아핀 변환으로 patch_embed
레벨에서 P16 좌표계에 투영한다.

이 파일이 생기기 전까지는 이 메커니즘(AffineAdapter + 교체 + forward)이
recovery_test/joint_attack/adaptive_evasion_full 세 곳에 거의 동일하게 복사돼 있었다
(감사 결과: cosmetic한 docstring 차이만 있고 로직은 동일). 여기서는 그 로직을 한 곳으로
모으되, 이미 검증이 끝난 §16/§17/§18 스크립트 자신은 그대로 둔다(재검증 없이 자기완결
유지) — 이 모듈은 새로 작성하는 실험(system_comparison, cost_comparison 등)이 참조할
단일 소스다.

2026-09-\* 최적화: 실배포 경로(`apply_local_switch`/`local_swap_logits`)는 예전에 P8의
patch_embed를 이미지 전체(784개 서브패치)에 대해 계산하고 그중 4개만 썼다("cost_comparison의
local_switch 비용은 보수적 상한" — README 로드맵에 적혀 있던 미구현 항목). `model8.patch_embed`
가 padding 없는 stride=kernel_size Conv2d(겹침 없음)라는 사실을 이용해, 이제 flag된 P16
패치에 대응하는 16×16 픽셀 crop 하나에만 conv를 돌린다 — 겹치지 않는 conv는 각 출력
위치가 자기 receptive field(여기선 crop)만 보므로, 전체 이미지에 conv를 돌리고 슬라이스한
것과 수학적으로 완전히 동일한 값이 나온다(자동미분도 crop을 그대로 통과하므로
`local_swap_logits`의 gradient 경로도 그대로 유지됨). `_p8_subpatch_embed_only`가 이
최적화를 담당한다.
"""
import torch
import torch.nn as nn


class AffineAdapter(nn.Module):
    """토큰별 아핀 변환 y = xW + b. 최소제곱으로 닫힌 형태 1회 피팅 (gradient descent 아님)."""

    def __init__(self, dim=768):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.W + self.b

    @torch.no_grad()
    def fit(self, x_src, y_tgt):
        C = x_src.shape[1]
        ones = torch.ones(x_src.shape[0], 1, dtype=x_src.dtype, device=x_src.device)
        x_aug = torch.cat([x_src, ones], dim=1)
        sol = torch.linalg.lstsq(x_aug, y_tgt).solution
        self.W.copy_(sol[:C])
        self.b.copy_(sol[C])


def patch_embed_with_pos(model, images):
    """patch_embed(x) + pos_embed[patch part]. CLS 제외 (B, N, D)."""
    x = model.patch_embed(images)
    return x + model.pos_embed[:, 1:, :]


def p16_to_p8_subpatch_indices(idx16, ppl16=14, ppl8=28):
    """P16 flat index -> 대응하는 P8 flat index 4개 (2x2 블록, row-major)."""
    r16, c16 = idx16 // ppl16, idx16 % ppl16
    r8, c8 = r16 * 2, c16 * 2
    return [r8 * ppl8 + c8, r8 * ppl8 + c8 + 1, (r8 + 1) * ppl8 + c8, (r8 + 1) * ppl8 + c8 + 1]


def _p8_subpatch_embed_only(model8, images, flag_idx, ppl16=14, patch16_px=16):
    """flag된 P16 패치 1개에 대응하는 P8 서브패치 4개의 patch_embed(+pos_embed)만 계산한다
    — 784개 전부 계산하지 않음(모듈 docstring의 "2026-09-\* 최적화" 참고).

    images: (B, 3, 224, 224). flag_idx: (B,) 각 이미지에서 flag된 P16 flat index(이미지마다
    다를 수 있음). 반환: (B, 4, D) — p16_to_p8_subpatch_indices와 같은 순서(2x2 블록,
    row-major: top-left, top-right, bottom-left, bottom-right).
    """
    B = images.shape[0]
    r16 = flag_idx // ppl16
    c16 = flag_idx % ppl16

    crops = []
    for i in range(B):
        y0 = int(r16[i]) * patch16_px
        x0 = int(c16[i]) * patch16_px
        crops.append(images[i:i + 1, :, y0:y0 + patch16_px, x0:x0 + patch16_px])
    crops = torch.cat(crops, dim=0)                             # (B, 3, 16, 16)

    # model8.patch_embed.forward()는 입력 크기가 224x224인지 assert하므로 못 씀 — proj(conv)만
    # 직접 호출. stride=kernel_size, padding 없는 conv라 crop만 넣어도 전체 이미지에 conv를
    # 돌린 뒤 같은 위치를 슬라이스한 것과 동일하다(모듈 docstring 참고).
    out = model8.patch_embed.proj(crops)                        # (B, D, 2, 2)
    out = out.flatten(2).transpose(1, 2)                        # (B, 4, D), row-major 2x2
    out = model8.patch_embed.norm(out)

    pos_full = model8.pos_embed[:, 1:, :]                       # (1, 784, D)
    pos = torch.stack(
        [pos_full[0, p16_to_p8_subpatch_indices(int(flag_idx[i]), ppl16=ppl16)]
         for i in range(B)], dim=0)                             # (B, 4, D)
    return out + pos


@torch.no_grad()
def fit_adapter(model16, model8, calib_images, ppl16=14):
    """calib_images(공격 없는 clean 이미지)로 P8<->P16 patch-embed 쌍을 만들어 어댑터를 피팅.
    반환된 어댑터는 calib_images와 겹치지 않는 held-out 이미지에만 적용할 것."""
    p16_full = patch_embed_with_pos(model16, calib_images)   # (N,196,D)
    p8_full = patch_embed_with_pos(model8, calib_images)     # (N,784,D)
    N, n16, D = p16_full.shape
    src_list, tgt_list = [], []
    for idx16 in range(n16):
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16)
        for si in sub_idx:
            src_list.append(p8_full[:, si, :])
            tgt_list.append(p16_full[:, idx16, :])
    x_src = torch.cat(src_list, dim=0)
    y_tgt = torch.cat(tgt_list, dim=0)
    adapter = AffineAdapter(dim=D)
    adapter.fit(x_src.cpu(), y_tgt.cpu())
    return adapter.to(calib_images.device)


def _swap_sequence(model16, model8, adapter, images, flag_idx, ppl16=14):
    """공통 교체 로직: (B, 200, D) 시퀀스(CLS+195 P16 토큰+4 P8 서브패치, 어댑터 보정).
    build_local_swap_batch(no_grad)와 local_swap_logits(미분 가능)이 이 함수를 감싸 쓴다.

    P8 쪽은 784개 서브패치를 전부 계산하지 않고 flag된 4개만 계산한다
    (`_p8_subpatch_embed_only`, 모듈 docstring의 "2026-09-\* 최적화" 참고)."""
    B = images.shape[0]
    p16_full = patch_embed_with_pos(model16, images)          # (B, 196, D)
    p8_sub = _p8_subpatch_embed_only(model8, images, flag_idx, ppl16=ppl16)   # (B, 4, D)
    p8_adapted = adapter(p8_sub.reshape(-1, p8_sub.shape[-1])).reshape(p8_sub.shape)

    cls = model16.cls_token.expand(B, -1, -1) + model16.pos_embed[:, :1, :]

    seqs = []
    for i in range(B):
        idx16 = int(flag_idx[i])
        keep_mask = torch.ones(p16_full.shape[1], dtype=torch.bool)
        keep_mask[idx16] = False
        kept = p16_full[i, keep_mask]                          # (195, D)
        replacement = p8_adapted[i]                             # (4, D) — 이미 이 이미지의 4개뿐
        seq = torch.cat([cls[i], kept, replacement], dim=0)     # (1+195+4=200, D)
        seqs.append(seq)
    return torch.stack(seqs, dim=0)


def full_forward_from_tokens(model16, cls_plus_patches):
    """CLS+패치 임베딩 시퀀스(이미 pos_embed 포함)를 P16의 12개 레이어+head에 그대로 통과."""
    x = model16.pos_drop(cls_plus_patches)
    for blk in model16.blocks:
        x = blk(x)
    x = model16.norm(x)
    x = x[:, 0]
    x = model16.fc_norm(x)
    return model16.head(x)


@torch.no_grad()
def apply_local_switch(model16, model8, adapter, images, flag_idx, ppl16=14):
    """국소 교체를 적용한 뒤 P16 12개 레이어를 통과시킨 logits — 실배포 경로(no_grad)."""
    seq = _swap_sequence(model16, model8, adapter, images, flag_idx, ppl16=ppl16)
    return full_forward_from_tokens(model16, seq)


def local_swap_logits(model16, model8, adapter, images, flag_idx, ppl16=14):
    """apply_local_switch와 동일한 계산을 미분 가능하게(torch.no_grad 없이) 수행 —
    joint/adaptive-evasion attack 스크립트가 이 함수에 대해 gradient를 구한다."""
    seq = _swap_sequence(model16, model8, adapter, images, flag_idx, ppl16=ppl16)
    return full_forward_from_tokens(model16, seq)
