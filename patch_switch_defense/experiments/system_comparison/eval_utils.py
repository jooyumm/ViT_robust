"""
probes/eval_utils.py — [탐색적, 롤백 가능] probes/ 전역에서 반복되던 평가 유틸리티 3종을
공용 모듈로 추출.

배경 — 이 파일이 왜 생겼는가
---------------------------
probes/ 스크립트들을 하나씩 새로 짜면서 아래 세 가지 패턴이 여러 파일에 흩어져서
(부분적으로는 복붙으로) 반복됐다. 감사(§ patch_switch_defense/README.md "샘플링 감사") 과정에서
이 중복이 실수를 낼 여지가 있다고 판단해서 한 곳으로 모았다:

  1. calibration/evaluation split 강제
     vitguard_final_validation.py에서 처음 만든 패턴. 탐지기 임계값을 정하는 데 쓴 표본으로
     그 탐지기의 recall/FPR을 다시 재면 순환평가로 낙관 편향이 생긴다 — 실제로
     vitguard_p8_rescue_test.py의 초기 n=30 버전이 이 문제로 recall이 부풀려져 있었다
     (patch_switch_defense/README.md §6). calibration_eval_split()은 "임계값은 calibration 슬라이스에서만,
     성능 지표는 evaluation 슬라이스에서만"이라는 규칙을 코드 레벨에서 강제한다.

  2. Wilson 이항비율 신뢰구간
     vitguard_final_validation.py에서 정의했던 것 그대로(로직 변경 없음, 위치만 이동).

  3. paired sampling 헬퍼
     샘플링 감사에서 발견된 실제 문제: diversity_test.py(seed=456)와
     joint_attack_test.py(seed=123)가 서로 다른 50장으로 "같은 조건" 비교를
     하고 있었다(재실행으로 수정, patch_switch_defense/README.md 참고). load_paired_batch()는 두 실행이
     같은 (seed, num_samples)를 쓰기만 하면 항상 같은 이미지가 같은 순서로 나온다는 사실을
     명시적인 함수로 만들어서, 다음에 비슷한 비교 실험을 짤 때 seed를 따로따로 고르는 실수를
     구조적으로 줄인다.

이 파일은 src/experiments/visualize를 전혀 건드리지 않는다 — Wilson CI/Youden 임계값/
calibration split은 ViTGuard류 탐지기 평가에서만 쓰는 개념이라 그 파이프라인 쪽 코드가
아니라 여기 probes/ 안에 둔다(다른 probes/ 파일들과 마찬가지로 지워도 나머지 프로젝트에
영향 없음). src.dataset.get_dataloader는 import만(수정 없음).
"""
import numpy as np

from src.dataset import get_dataloader


def calibration_eval_split(num_total, frac=0.5):
    """num_total개를 앞/뒤로 나눠 (calibration 슬라이스, evaluation 슬라이스) 반환.

    임계값/통계는 반드시 calibration 슬라이스에서만 도출하고, recall/FPR 등 성능 지표는
    반드시 evaluation 슬라이스에서만 계산할 것 — 같은 표본으로 임계값도 정하고 성능도
    재면 순환평가가 된다(vitguard_p8_rescue_test.py의 n=30 초기 버전이 실제로 이 문제를
    겪었음, patch_switch_defense/README.md §6 참고).
    """
    assert 0.0 < frac < 1.0
    n_cal = int(round(num_total * frac))
    n_eval = num_total - n_cal
    assert n_cal > 0 and n_eval > 0, \
        f"calibration({n_cal})/evaluation({n_eval}) 둘 다 1개 이상이어야 함 (num_total={num_total})"
    return slice(0, n_cal), slice(n_cal, n_cal + n_eval)


def youden_threshold(pos_scores, neg_scores):
    """ROC 상에서 TPR-FPR(Youden's J)이 최대인 지점의 점수를 임계값으로 채택.

    pos_scores/neg_scores에는 calibration 표본만 넣을 것 — 이 함수 자체는 그걸 강제하지
    않으므로 호출부에서 calibration_eval_split()으로 나눈 슬라이스만 넘겨야 한다.
    (vitguard_final_validation.py에서 그대로 옮김, 로직 변경 없음)
    """
    candidates = np.unique(np.concatenate([pos_scores, neg_scores]))
    best_j, best_t = -1.0, candidates[0]
    for t in candidates:
        tpr = (pos_scores > t).mean()
        fpr = (neg_scores > t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def wilson_ci(k, n, z=1.96):
    """이항비율 k/n의 95% Wilson 신뢰구간.

    (정규근사 기반 신뢰구간과 달리 n이 작거나 비율이 0%/100%에 가까울 때도 안정적.
    vitguard_final_validation.py에서 그대로 옮김, 로직 변경 없음)
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_paired_batch(seed, num_samples, batch_size=None):
    """여러 스크립트/실행에 걸쳐 '같은 이미지 집합'으로 비교하고 싶을 때 이 함수로만 로드할 것.

    src.dataset.get_dataloader(seed=seed, num_samples=num_samples)는 seed가 같으면(그리고
    num_samples가 같거나, 다르더라도 더 작은 쪽이 더 큰 쪽의 앞부분과 정확히 일치하는
    prefix 관계라면) 항상 같은 이미지를 같은 순서로 반환한다 — 내부적으로 seed로 고정한
    torch.Generator로 전체 데이터셋을 한 번 섞은 뒤 앞에서 num_samples개를 자르는 방식이기
    때문(batch_size는 순서/내용에 영향 없음). 두 실행을 비교하려면 seed만 맞추면 된다.

    이 규칙이 안 지켜져서 실제로 diversity_test.py(seed=456)와
    joint_attack_test.py(seed=123)가 서로 다른 50장으로 비교됐던 적이 있다
    (patch_switch_defense/README.md "샘플링 감사" 참고, 재실행으로 수정함). 새 비교 실험을 짤 때 두
    조건에 넘기는 seed를 이 함수를 통해 명시적으로 통일해서 같은 실수를 구조적으로 막는다.

    Returns: (images, labels, dataset) — images/labels는 아직 .to(device) 안 된 CPU 텐서.
    """
    loader, dataset = get_dataloader(
        batch_size=batch_size or num_samples, num_samples=num_samples, seed=seed)
    images, labels = next(iter(loader))
    return images, labels, dataset


def both_correct_mask(pred_a, pred_b, labels):
    """두 모델(A, B) 모두 clean에서 원래 맞춘 표본만 골라내는 마스크.

    joint attack류 실험(joint_attack_test.py, diversity_test.py,
    hybrid_joint_attack_test.py)마다 각자 인라인으로 반복하던 한 줄을 모음 (필수 3종은
    아니지만 중복이 많아 덤으로 추출)."""
    return (pred_a == labels) & (pred_b == labels)
