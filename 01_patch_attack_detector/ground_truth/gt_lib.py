"""
ground_truth/gt_lib.py — 공격이 실제로 어느 patch를 건드렸는지 "눈으로 보고 추측"하지
않고, clean/adv 이미지 텐서를 직접 diff해서 계산하는 ground truth(GT) 함수. 이 폴더의
유일한 역할은 이 GT를 뽑아내는 것 — attention이나 탐지기 출력과 비교하는 로직은 여기
없다(그건 이 GT를 가져다 쓰는 별도의, 나중 단계).

PatchFool은 선택된 patch 하나(16x16 grid cell)에만 마스크를 씌워 perturbation을 넣으므로
diff가 정확히 그 셀 하나를 가득 채운다. LaVAN은 면적 비율(patch_ratio)로 정한 정사각형을
16px 그리드에 정렬 없이 랜덤 위치에 놓으므로, 여러 grid cell에 걸쳐 부분적으로만 겹친다 —
그래서 "정확히 어느 patch"가 아니라 "각 patch가 얼마나 겹쳤는지(0~1 비율)"로 표현한다.
patch 인덱스 규약(patch 0~195, row*ppl+col)은 attention_lib.py와 동일하게 맞춰뒀다 — 나중에
탐지기 출력과 대조할 때 같은 축을 바로 쓸 수 있도록.
"""
import torch

PATCH_SIZE = 16
PPL = 14


def gt_coverage(clean_img, adv_img, patch_size=PATCH_SIZE, ppl=PPL, threshold=1e-6):
    """clean_img/adv_img: (3, H, W) 텐서 한 장. 196개 patch 각각에서 실제로 픽셀 값이
    바뀐 비율(0~1)을 (196,) ndarray로 반환 — attention_lib의 row_avg/col_avg와 같은
    patch 인덱스 규약(row*ppl+col). PatchFool이면 정확히 한 자리만 1.0, 나머지는 0.0.
    LaVAN이면 여러 자리에 0~1 사이 분수 값이 걸쳐 있을 수 있다."""
    diff_mask = (adv_img - clean_img).abs().sum(dim=0) > threshold   # (H, W) bool
    cov = torch.zeros(ppl * ppl)
    for pr in range(ppl):
        for pc in range(ppl):
            cell = diff_mask[pr * patch_size:(pr + 1) * patch_size,
                              pc * patch_size:(pc + 1) * patch_size]
            cov[pr * ppl + pc] = cell.float().mean()
    return cov.numpy()


def gt_patch_summary(coverage, min_frac=0.01):
    """coverage: (196,) ndarray (gt_coverage 출력). min_frac 이상 겹친 patch만 골라
    [(patch_idx, coverage_frac), ...]를 **patch 번호 오름차순**으로 반환(겹침 비율 순이
    아님 — 어느 자리들인지 공간적으로 훑어보기 쉽도록). 공격이 없으면(coverage 전부 0)
    빈 리스트."""
    idxs = [i for i in range(len(coverage)) if coverage[i] >= min_frac]
    idxs.sort()
    return [(i, float(coverage[i])) for i in idxs]
