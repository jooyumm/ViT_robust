"""
metrics.py — CA / RA / ASR 계산 공용 모듈

케이스 정의 (공격 전 정답 여부 → 공격 후 정답 여부):
  A: 맞음 → 맞음   (강건한 샘플)
  B: 맞음 → 틀림   (공격 성공 — ASR 분자)
  C: 틀림 → 맞음   (노이즈성, RA에는 포함되나 ASR 분모에서는 제외)
  D: 틀림 → 틀림   (공격과 무관하게 원래 틀린 샘플)

  CA  (Clean Accuracy)      = (A + B) / N                Higher is Better
  RA  (Robust Accuracy)     = (A + C) / N                Higher is Better
  ASR (Attack Success Rate) = B / (A + B) = B / clean_correct   Lower is Better
  Δ   (Accuracy Drop)       = CA - RA
"""


def compute_confusion_counts(preds_clean, preds_adv, labels):
    """
    배치 단위로 A/B/C/D 케이스 카운트 계산.

    Parameters
    ----------
    preds_clean : array-like, clean 이미지에 대한 예측 클래스 인덱스
    preds_adv   : array-like, adversarial 이미지에 대한 예측 클래스 인덱스
    labels      : array-like, 정답 클래스 인덱스

    Returns
    -------
    dict with keys: A, B, C, D
    """
    A = B = C = D = 0
    for pc, pa, l in zip(preds_clean, preds_adv, labels):
        clean_correct = (pc == l)
        adv_correct   = (pa == l)
        if clean_correct and adv_correct:
            A += 1
        elif clean_correct and not adv_correct:
            B += 1
        elif not clean_correct and adv_correct:
            C += 1
        else:
            D += 1
    return {'A': A, 'B': B, 'C': C, 'D': D}


def compute_metrics_from_counts(A, B, C, D):
    """
    A/B/C/D 카운트로부터 CA/RA/ASR/acc_drop 계산.
    배치 단위든 전체 집계든 동일하게 사용 가능.

    Returns
    -------
    dict with keys: CA, RA, ASR, acc_drop, A, B, C, D, valid
    """
    valid = A + B + C + D
    clean_correct_count = A + B   # 공격 전 맞은 샘플
    adv_correct_count   = A + C   # 공격 후 맞은 샘플

    CA  = 100.0 * clean_correct_count / valid if valid > 0 else 0.0
    RA  = 100.0 * adv_correct_count   / valid if valid > 0 else 0.0
    ASR = 100.0 * B / clean_correct_count if clean_correct_count > 0 else 0.0

    return {
        'CA':       CA,
        'RA':       RA,
        'ASR':      ASR,
        'acc_drop': CA - RA,
        'A':        A,
        'B':        B,
        'C':        C,
        'D':        D,
        'valid':    valid,
    }


def aggregate_counts(*count_dicts):
    """
    여러 배치의 A/B/C/D를 합산.
    main.py에서 배치 루프 후 전체 집계할 때 사용.

    Parameters
    ----------
    *count_dicts : dict, 각각 {'A':.., 'B':.., 'C':.., 'D':..} 형태

    Returns
    -------
    dict with summed A, B, C, D
    """
    total = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for d in count_dicts:
        for k in total:
            total[k] += d.get(k, 0)
    return total
