"""패치별 incoming attention을 유지하는 탐지/영역 선택용 특징.

영역 합산 전에 생긴 단일 패치의 집중도를 보존한다. CLS attention과
자기 자신을 제외한 patch-query incoming을 헤드 평균/최대로 요약한다.
"""
import numpy as np
import torch
from src.attention import extract_attention

MAP_NAMES = ('incoming_mean', 'incoming_head_max', 'incoming_head_std', 'cls_mean', 'cls_head_max')

class AttentionCapture:
    def __init__(self, model):
        self.maps = []
        self.model = model

    def __call__(self, images):
        logits, attention = extract_attention(self.model, images)
        self.maps.clear()
        for a in attention:
            self.capture(a)
        return logits

    def capture(self, attention):
        a = attention.detach()
        if a.shape[1:] != (12, 197, 197):
            raise ValueError('P16 ViT-B attention이 필요합니다.')
        pp = a[:, :, 1:, 1:]
        incoming = (pp.sum(-2) - pp.diagonal(dim1=-2, dim2=-1)) / 195
        incoming = incoming / incoming.sum(-1, keepdim=True).clamp_min(1e-12)
        cls = a[:, :, 0, 1:]
        cls = cls / cls.sum(-1, keepdim=True).clamp_min(1e-12)
        maps = torch.stack((incoming.mean(1), incoming.amax(1), incoming.std(1, correction=0), cls.mean(1), cls.amax(1)), dim=1)
        self.maps.append(maps.cpu().numpy())

    def arrays(self):
        return np.stack(self.maps, axis=1)  # image, layer, map kind, patch


from src.patches import patch_to_region


REGION_OF_PATCH = patch_to_region(np.arange(196))
REGION_PATCHES = [np.flatnonzero(REGION_OF_PATCH == r) for r in range(4)]


def _robust_z(values):
    median = np.median(values, axis=-1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=-1, keepdims=True)
    return (values - median) / np.maximum(1.4826 * mad, 1e-4)


class PatchAttentionDetector:
    """사분면 4개를 직접 비교하는 고정 규칙. 학습/보정/GT 입력이 없다.

    (attention 단독으로도 시도해봤지만, 뒷단 레이어의 attention 집중은 공격과
    무관하게 ViT의 정상 동작이라 오탐률이 99%까지 치솟아 폐기했다 — "공격받아서
    많이 본다"와 "원래 중요한 피사체라 많이 본다"를 attention만으로는 구분하지
    못했다. 그래서 픽셀 TV로 되돌렸다.)

    패치 단위 TV 이상치(이웃 대비 비율, 전역/지역 z)와 incoming/CLS attention을
    먼저 196개 패치에서 계산하지만, 결정은 패치가 아니라 사분면 단위로 내린다:
    각 사분면 내부에서 가장 튀는 패치의 값을 그 사분면의 대표값(최댓값)으로 삼고,
    그 대표값 4개(사분면별 1개)만 서로 비교해 임계값을 넘는 사분면이 있으면 발동한다.
    patch 필드는 선택된 사분면 내부에서 어느 패치가 대표였는지 보여주는 진단용이며
    결정(flag/region)에는 쓰이지 않는다. 고정 상수는 이 실험의 정답으로 탐색하지 않는다.
    """
    threshold = 6.0
    neighbor_ratio_threshold = 2.0
    minimum_support_layers = 2

    def predict(self, images, maps):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError('정규화된 224×224 입력이 필요합니다.')
        if maps.shape != (len(images), 12, 5, 196) or not np.isfinite(maps).all():
            raise ValueError('유한한 (N,12,5,196) attention 맵이 필요합니다.')
        cells = images.detach().unfold(2, 16, 16).unfold(3, 16, 16)
        horizontal = (cells[..., 1:] - cells[..., :-1]).abs().mean((-1, -2))
        vertical = (cells[..., 1:, :] - cells[..., :-1, :]).abs().mean((-1, -2))
        tv = ((horizontal + vertical) * .5).mean(1).cpu().numpy().reshape(-1, 196)
        global_z = _robust_z(tv)
        local_z = np.empty_like(tv)
        ratio = np.empty_like(tv)
        global_mad = np.median(np.abs(tv-np.median(tv, axis=1, keepdims=True)), axis=1)
        for patch in range(196):
            row, col = divmod(patch, 14)
            neighbors = [r * 14 + c for r in range(max(0, row-1), min(14, row+2))
                         for c in range(max(0, col-1), min(14, col+2)) if (r, c) != (row, col)]
            values = tv[:, neighbors]
            med = np.median(values, axis=1)
            mad = np.median(np.abs(values-med[:, None]), axis=1)
            local_z[:, patch] = (tv[:, patch]-med) / np.maximum(1.4826 * (mad + .25 * global_mad), 1e-4)
            ratio[:, patch] = tv[:, patch] / np.maximum(med, 1e-4)
        z = np.maximum(global_z, local_z)
        n = len(tv)
        rows = np.arange(n)
        # 사분면별 대표(최댓값) 패치: 결정 이전에 196개 패치를 4개 사분면 값으로 먼저 집계한다.
        region_z = np.empty((n, 4)); region_ratio = np.empty((n, 4)); region_peak_patch = np.empty((n, 4), dtype=np.int64)
        for r, idx in enumerate(REGION_PATCHES):
            local_argmax = z[:, idx].argmax(1)
            region_peak_patch[:, r] = idx[local_argmax]
            region_z[:, r] = z[rows, region_peak_patch[:, r]]
            region_ratio[:, r] = ratio[rows, region_peak_patch[:, r]]
        region_support = np.zeros((n, 4), dtype=np.int64)
        for layer in (9, 10, 11):
            layer_support = np.zeros((n, 4), dtype=bool)
            for kind in (1, 4):  # patch incoming head-max, CLS head-max
                region_attention = np.stack([maps[:, layer, kind, idx].max(1) for idx in REGION_PATCHES], axis=1)
                layer_support[rows, region_attention.argmax(1)] = True  # 4개 사분면 중 그 층·종류에서 가장 집중받은 1개
            region_support += layer_support
        eligible = (region_ratio > self.neighbor_ratio_threshold) & (region_support >= self.minimum_support_layers)
        region_score = np.where(eligible, np.maximum(region_z, 0), 0.)
        region = region_score.argmax(-1)  # 196개 패치가 아니라 4개 사분면 중에서 결정
        score = region_score[rows, region]
        patch = region_peak_patch[rows, region]  # 진단/시각화용: 선택된 사분면 내부의 대표 패치
        return dict(score=score, flag=score > self.threshold, patch=patch, region=region,
                    spatial_z=z[rows, patch], neighbor_ratio=ratio[rows, patch],
                    support_layers=region_support[rows, region])
