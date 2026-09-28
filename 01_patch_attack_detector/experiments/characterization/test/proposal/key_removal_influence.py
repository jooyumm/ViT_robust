"""연구 후보: key 제외 시 단일 attention head 출력 변화량. 탐지 성능은 미검증이다.

attention: (..., query, key), value: (..., key, channel).
전체 모델에서 입력 패치를 지운 결과가 아니라, 현재 레이어의 Q/K/V를 고정한 국소 개입이다.
"""
import numpy as np


def removal_energy(attention, value):
    """각 query-key 쌍의 ||o - o_without_key||²를 반환한다.

수치 검증용 float64 참조 구현이다. GPU 지연시간을 검증한 구현이 아니다.
attention을 강하게 지배하는 key도 제외 후 나머지 질량을 직접 더해 안정적으로 계산한다.
"""
    attention = np.asarray(attention, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if attention.ndim < 2 or value.ndim != attention.ndim:
        raise ValueError('attention과 value의 차원 수가 일치해야 합니다.')
    if attention.shape[:-2] != value.shape[:-2] or attention.shape[-1] != value.shape[-2]:
        raise ValueError('배치·헤드 차원과 key 개수가 일치해야 합니다.')
    if attention.shape[-1] < 2 or not np.isfinite(attention).all() or not np.isfinite(value).all():
        raise ValueError('최소 두 key와 유한한 입력이 필요합니다.')
    if (attention < 0).any():
        raise ValueError('attention은 음수가 될 수 없습니다.')
    np.testing.assert_allclose(attention.sum(-1), 1, atol=1e-6, rtol=1e-6)
    output = attention @ value
    energies = []
    for key_i in range(attention.shape[-1]):
        keep = np.arange(attention.shape[-1]) != key_i
        weights = attention[..., keep]
        mass = weights.sum(-1, keepdims=True)
        if (mass <= 0).any():
            raise ValueError('제외 후 확률 질량이 0입니다. 원래 logits로 재계산해야 합니다.')
        without = (weights / mass) @ value[..., keep, :]
        energies.append(np.square(output - without).sum(-1))
    return np.stack(energies, axis=-1)


def removal_energy_fast_reference(attention, value):
    """중간 벡터 텐서 없이 거리 항등식을 쓴 후보 구현. 극단 확률에서는 참조 경로로 전환한다."""
    attention = np.asarray(attention, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if np.any(attention > 1 - 1e-6):
        return removal_energy(attention, value)
    output = attention @ value
    distance_squared = (
        np.square(output).sum(-1, keepdims=True)
        + np.square(value).sum(-1)[..., None, :]
        - 2 * (output @ np.swapaxes(value, -1, -2)))
    return np.square(attention / (1 - attention)) * np.maximum(distance_squared, 0)
