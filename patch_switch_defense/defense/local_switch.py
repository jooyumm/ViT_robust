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
        ones = torch.ones(x_src.shape[0], 1, dtype=x_src.dtype)
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
    build_local_swap_batch(no_grad)와 local_swap_logits(미분 가능)이 이 함수를 감싸 쓴다."""
    B = images.shape[0]
    p16_full = patch_embed_with_pos(model16, images)          # (B, 196, D)
    p8_full = patch_embed_with_pos(model8, images)             # (B, 784, D)
    p8_adapted = adapter(p8_full.reshape(-1, p8_full.shape[-1])).reshape(p8_full.shape)

    cls = model16.cls_token.expand(B, -1, -1) + model16.pos_embed[:, :1, :]

    seqs = []
    for i in range(B):
        idx16 = int(flag_idx[i])
        keep_mask = torch.ones(p16_full.shape[1], dtype=torch.bool)
        keep_mask[idx16] = False
        kept = p16_full[i, keep_mask]                          # (195, D)
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16)
        replacement = p8_adapted[i, sub_idx]                    # (4, D)
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
