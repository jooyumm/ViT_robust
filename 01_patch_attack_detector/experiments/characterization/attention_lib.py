"""
experiments/characterization/attention_lib.py — 헤드 평균 없는 전체 attention을 뽑고, 그
분포의 집중도(top-1 mass/entropy/Gini)를 재는 저수준 함수 모음. CLI 없음, 00_extract_attention.py
가 이 함수들만 써서 데이터를 만든다.

detector/topk_mass_v1.py와의 관계: 그 파일의 _attn_hook/collect_layer_attn은 (1) 헤드
평균, (2) CLS row만 다루는 실배포 탐지 전용 함수라 건드리지 않는다. 여기서는 헤드별
attention과 column(전체 쿼리 집계) 분포까지 보려고 별도로 정의한다.
"""
import torch

N_PATCH = 196   # P16, 14x14
PPL = 14
UNIFORM = 1.0 / N_PATCH


def _full_attn_hook(weights_list):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)          # (B, heads, N, N) — 헤드 평균 없음
            weights_list.append(attn.detach().cpu())
    return hook


def collect_full_layer_attn(model, images):
    """forward 1회로 12개 block 전부의 (B, heads, N, N) post-softmax attention을 그대로 반환."""
    weights = []
    hooks = [blk.attn.register_forward_hook(_full_attn_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def row_distribution(attn):
    """CLS(query 0)가 각 patch(key 1..196)에 주는 attention, 정규화. attn: (..., N, N)."""
    cls_to_patch = attn[..., 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=-1, keepdim=True)


def full_row(attn):
    """CLS(query 0)가 자기 자신 포함 전체 197개 토큰(CLS+196 patch)에 주는 raw attention.
    post-softmax라 이미 합이 1 — row_distribution과 달리 정규화하지 않는다(정규화하면 CLS
    자기 자신에게 주는 attention이 사라져서 03_token_bars.py가 원하는 "토큰 197개 중
    어디가 튀는가"를 CLS 포함해서 볼 수 없게 됨). attn: (..., N, N) -> (..., N)."""
    return attn[..., 0, :]


def full_col(attn, key_idx):
    """모든 쿼리 토큰(CLS+196 patch) 각각이 특정 key 토큰 하나(key_idx: 0=CLS, 1~196=patch)에게
    주는 raw attention. full_row를 뒤집은 버전 — full_row는 "CLS 하나가 전체 197개를 보는
    분포"이고, 이건 "전체 197개 쿼리 각각이 특정 토큰 하나를 보는 값"이다(03_token_bars.py의
    발견: CLS가 patch 17을 튀게 봄 -- 그게 진짜 sink라면 다른 토큰들도 17을 봐야 하는지
    확인하는 데 씀). 정규화 안 함(각 쿼리 행의 softmax 합은 이미 1). attn: (..., N, N) -> (..., N)."""
    return attn[..., :, key_idx]


def col_distribution(attn):
    """전체 쿼리(CLS+patch)가 각 patch(key)에 주는 attention의 합, 정규화. attn: (..., N, N)."""
    col_sum = attn[..., :, 1:].sum(dim=-2)
    return col_sum / col_sum.sum(dim=-1, keepdim=True)


def top1_mass(dist):
    return dist.max(dim=-1).values


def normalized_entropy(dist):
    import numpy as np
    ent = -(dist * dist.clamp(min=1e-12).log()).sum(dim=-1)
    return ent / np.log(dist.shape[-1])


def gini_coefficient(dist):
    sorted_d, _ = torch.sort(dist, dim=-1)
    n = sorted_d.shape[-1]
    idx = torch.arange(1, n + 1, dtype=sorted_d.dtype, device=sorted_d.device)
    numer = 2 * (idx * sorted_d).sum(dim=-1) - (n + 1) * sorted_d.sum(dim=-1)
    denom = n * sorted_d.sum(dim=-1)
    return numer / denom
