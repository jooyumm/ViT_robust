"""두 region switch 방어가 공유하는 paired 평가와 명시적인 지표 분모.

GT·label은 predict 콜백에 전달하지 않는다. 전체 공격본 평가와 성공한 공격만의
탐지 평가를 둘 다 보고하며, 임계값 탐색/보정은 하지 않는다.
"""
import time
import numpy as np
import torch
from src.cohort import DEFAULT_DATASET
from src.provenance import sha256
from src.patches import patch_to_region


def rate(values, population=None):
    values = np.asarray(values, dtype=bool)
    if population is not None:
        values = values[np.asarray(population, dtype=bool)]
    return float(values.mean()) if len(values) else None


def auroc(positive_scores, negative_scores):
    """높은 점수=공격. 동점은 0.5로 세되 N×N 행렬은 만들지 않는다."""
    pos, neg = np.asarray(positive_scores), np.sort(negative_scores)
    if not len(pos) or not len(neg):
        return None
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError('AUROC 점수는 유한해야 합니다.')
    left = np.searchsorted(neg, pos, side='left')
    right = np.searchsorted(neg, pos, side='right')
    return float(((left + right) * .5).mean() / len(neg))


def summarize(labels, baseline, defended, decisions, gt_patch):
    clean_ok = baseline['clean'] == labels
    attack_ok = baseline['patchfool'] == labels
    defended_clean_ok = defended['clean'] == labels
    defense_ok = defended['patchfool'] == labels
    success = clean_ok & ~attack_ok
    failure = clean_ok & attack_ok
    clean_flag = decisions['clean']['flag'].astype(bool)
    attack_flag = decisions['patchfool']['flag'].astype(bool)
    region_ok = decisions['patchfool']['region'] == patch_to_region(gt_patch)
    clean_score = decisions['clean']['score']
    attack_score = decisions['patchfool']['score']
    return dict(
        n=len(labels), clean_accuracy=rate(clean_ok), attack_accuracy=rate(attack_ok),
        defended_attack_accuracy=rate(defense_ok), clean_policy_accuracy=rate(defended_clean_ok),
        false_positive_rate=rate(clean_flag), attack_detection_rate=rate(attack_flag),
        successful_attack_detection_rate=rate(attack_flag, success),
        failed_attack_detection_rate=rate(attack_flag, failure),
        clean_wrong_attack_detection_rate=rate(attack_flag, ~clean_ok),
        flagged_region_accuracy=rate(region_ok, attack_flag),
        end_to_end_region_accuracy=rate(attack_flag & region_ok),
        detector_auroc=auroc(attack_score, clean_score),
        successful_attack_detector_auroc=auroc(attack_score[success], clean_score),
        attack_success_rate=rate(~attack_ok, clean_ok),
        recovery_rate=rate(defense_ok, success),
        clean_damage_rate=rate(~defended_clean_ok, clean_ok),
        successful_attacks=int(success.sum()), recovered_successful_attacks=int((success & defense_ok).sum()),
        newly_broken_clean_inputs=int((clean_ok & ~defended_clean_ok).sum()),
        populations=dict(all_attacks=len(labels), successful_attacks=int(success.sum()),
                         failed_attacks=int(failure.sum()), clean_already_wrong=int((~clean_ok).sum()),
                         originally_correct_clean=int(clean_ok.sum()), flagged_attacks=int(attack_flag.sum())),
        clean_definition='original model without defense',
        detection_population='attack_detection_rate/detector_auroc: all attack images; successful_attack_*: clean-correct -> attack-wrong',
    )


def evaluate_pairs(predict, manifest, *, batch_size=4, seed=42, run=DEFAULT_DATASET, progress='방어 평가'):
    """predict(images) -> (baseline logits, defended logits, decision[, choices])."""
    if batch_size < 1:
        raise ValueError('batch_size는 양수여야 합니다.')
    started = time.monotonic()
    n = len(manifest['sample_indices'])
    groups = ('clean', 'patchfool')
    baseline = {g: np.empty(n, dtype=np.int64) for g in groups}
    defended = {g: np.empty(n, dtype=np.int64) for g in groups}
    decisions = {g: {k: np.empty(n) for k in ('score', 'flag', 'patch', 'region')} for g in groups}
    kinds = {g: np.full(n, 'none', dtype=object) for g in groups}
    labels, gt = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    offset, max_error = 0, 0.
    for chunk in manifest['chunks']:
        path = (run / chunk['path']).resolve()
        if run.resolve() not in path.parents or sha256(path) != chunk['sha256']:
            raise ValueError('청크 경로/해시 불일치: '+str(path))
        with np.load(path, allow_pickle=False) as z:
            count = chunk['n']
            np.testing.assert_array_equal(z['sample_index'], manifest['sample_indices'][offset:offset+count])
            images = np.concatenate([z[g+'__images'] for g in groups])
            order = rng.permutation(len(images))
            for start in range(0, len(order), batch_size):
                indices = order[start:start+batch_size]
                result = predict(torch.from_numpy(images[indices]))
                base, corrected, decision = result[:3]
                choices = result[3] if len(result) == 4 else [None]*len(indices)
                base, corrected = base.detach().cpu().numpy(), corrected.detach().cpu().numpy()
                if not (np.isfinite(base).all() and np.isfinite(corrected).all()):
                    raise ValueError('추론 logits에 비유한 값이 있습니다.')
                for i, index in enumerate(indices):
                    group, local = groups[index//count], index % count
                    position = offset + local
                    expected = z[group+'__logits'][local]
                    # GPU/CPU, batch size and fused attention change FP32 rounding.
                    # Classification must match exactly; record logit drift separately.
                    if int(base[i].argmax()) != int(z[group+'__prediction'][local]):
                        raise ValueError('baseline 예측이 저장된 예측과 다릅니다.')
                    max_error = max(max_error, float(np.abs(base[i]-expected).max()))
                    baseline[group][position], defended[group][position] = base[i].argmax(), corrected[i].argmax()
                    for key in decisions[group]:
                        decisions[group][key][position] = decision[key][i]
                    if choices[i] is not None:
                        kinds[group][position] = choices[i]['kind']
            # label과 GT를 꺼내는 시점은 블라인드 추론 이후다.
            labels[offset:offset+count] = z['label']
            coverage = z['patchfool__gt_coverage']
            if not np.all((coverage > 0).sum(-1) == 1):
                raise ValueError('현재 위치 평가는 P16 한 패치 공격 데이터 전용입니다.')
            gt[offset:offset+count] = coverage.argmax(-1)
        offset += count
        print(f'{progress} {offset}/{n}, elapsed={time.monotonic()-started:.1f}s', flush=True)
    if offset != n:
        raise ValueError('manifest와 실제 표본 수가 다릅니다.')
    for group in groups:
        inactive = ~decisions[group]['flag'].astype(bool)
        np.testing.assert_array_equal(baseline[group][inactive], defended[group][inactive])
    report = summarize(labels, baseline, defended, decisions, gt)
    report['baseline_max_logit_error'] = max_error
    report['baseline_prediction_mismatches'] = 0  # Any mismatch raises above.
    report['source_manifest_sha256'] = sha256(run/'manifest.json')
    return report, kinds
