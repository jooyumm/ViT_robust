"""
detector/value_outlier_v2.py — value 이상치 경로. emergent_window_v2(LSW)의 attention 경로를 보완하는 후보.

패치(key) j의 value가 CLS의 attention 출력 o와 얼마나 다른지 ||v_j - o||^2 를 head 별로 구해 합친다(RSA식 value 거리,
attention 가중치 a/(1-a)는 개발 단계 실험에서 오히려 성능을 떨어뜨려 쓰지 않는다). 레이어마다 크기가 달라서 정상
reference 이미지로 log 거리의 레이어별 중앙값/MAD를 구해 z로 바꾸고, 이미지 점수는 (레이어, patch) 전체의 z 최댓값이다.

개발 단계(seed 42, 100쌍)에서 확인한 것: 이 신호는 1~4번 블록에 몰려 있다(PatchFool 1·2번 블록 AUROC 0.97~0.98,
LaVAN 3·4번 블록 0.89~0.91, 7~9번 블록은 우연 수준). LSW식 "후반에 새로 생긴 변화"로 바꾸면 신호가 지워진다.
결합: 두 경로 점수를 각자 정상 reference의 중앙값/MAD로 표준화한 뒤 max (combined_score). 임계값은 결합 점수로 따로 보정한다.

주의: 이 신호가 입력 픽셀 범위 이탈이나 고주파 잡음 같은 픽셀 수준 이상을 그대로 반영할 수 있다. 픽셀 통계 기준선과
반드시 함께 비교할 것(experiments/detector_eval).
"""
import numpy as np
import torch


def _hook(store):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            a = ((module.q_norm(q[:, :, :1]) * module.scale) @ module.k_norm(k).transpose(-2, -1)).softmax(-1)[:, :, 0]
            o = (a.unsqueeze(-1) * v).sum(2)
            store.append(((v - o.unsqueeze(2)) ** 2).sum(-1).sum(1)[:, 1:])     # (B,196), head 합
    return hook


@torch.no_grad()
def value_distance(model, images):
    """(B,3,H,W) -> (B,12,196): 블록별·patch별 head 합 ||v_j - o_cls||^2."""
    store = []
    hooks = [blk.attn.register_forward_hook(_hook(store)) for blk in model.blocks]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(store, dim=1)


class Reference:
    """정상 이미지의 log 거리에서 레이어별 중앙값/MAD를 구해 z로 바꾼다."""

    def __init__(self, distance):
        log = np.log(np.asarray(distance) + 1e-12)
        self.median = np.median(log, axis=(0, 2))
        self.mad = np.median(np.abs(log - self.median[None, :, None]), axis=(0, 2)) * 1.4826

    def z(self, distance):
        log = np.log(np.asarray(distance) + 1e-12)
        return (log - self.median[None, :, None]) / self.mad[None, :, None]


def value_score(z):
    """z (N,12,196) -> 이미지 점수 (N,)."""
    return z.reshape(len(z), -1).max(1)


def standardize(scores, reference_scores):
    """점수를 정상 reference 점수의 중앙값/MAD로 표준화."""
    median = np.median(reference_scores)
    mad = np.median(np.abs(reference_scores - median)) * 1.4826
    return (scores - median) / mad


def combined_score(lsw_scores, value_scores, lsw_reference, value_reference):
    """두 경로를 각자 표준화한 뒤 max."""
    return np.maximum(standardize(lsw_scores, lsw_reference), standardize(value_scores, value_reference))
