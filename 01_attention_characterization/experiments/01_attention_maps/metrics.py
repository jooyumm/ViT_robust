"""Attention comparison metrics, shared by the map experiments."""
import numpy as np
MASS_ATOL = 1e-4

def compute_metrics(clean_heads, attacked_heads, coverage):
    """헤드 평균 raw 분포로 같은 위치의 차이를 계산한다. CLS는 재정규화하지 않는다."""
    clean = clean_heads.mean(axis=2)
    attacked = attacked_heads.mean(axis=2)
    delta = attacked - clean
    support = coverage > 0
    attacked_delta = (delta[..., 1:] * support[:, None]).sum(-1)
    other_delta = (delta[..., 1:] * ~support[:, None]).sum(-1)
    cls_delta = delta[..., 0]
    np.testing.assert_allclose(attacked_delta + other_delta + cls_delta, 0, atol=MASS_ATOL, rtol=0)
    return {
        'clean': clean, 'attacked': attacked, 'delta': delta,
        'tv': .5 * np.abs(delta).sum(-1),
        'gt_delta': attacked_delta, 'other_delta': other_delta, 'cls_delta': cls_delta,
    }
