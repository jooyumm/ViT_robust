"""
detector/lsw_v3.py — v3: Late-Surge Window(LSW)를 배포용 하나의 탐지기로 정리한 것.

v2(emergent_window_v2.py)와 점수 정의가 **똑같다**(새로 튜닝하지 않았다, check_lsw_v3.py가 v2와 같은 점수임을
확인한다). 달라진 것은 (1) 정상 이미지로 임계값을 정하는 fit(), (2) 경보·점수·의심 위치를 함께 돌려주는 detect(),
(3) 이 파일 하나로 자기완결이라는 점이다. v1(topk_mass_v1.py)은 그대로 둔다(02_patch_switch_defense가 쓰는 배포본).
value 이상치 경로(value_outlier_v2.py)와 결합은 새 시드에서 이득이 재현되지 않아 v3에 넣지 않았다.

점수: 모든 3x3 patch 창에 대해 max( mean_{블록 10,11,12} 창합 - max_{블록 7,8,9} 창합 ). CLS query attention 행만
쓰고(head 평균, patch 위에서 합 1로 재정규화) forward 한 번, 학습 없음, 정상 이미지로 임계값만 정한다.
왜 이렇게 하는가: 자연 attention sink/물체는 7~9 블록에 이미 형성돼 있어 점수가 안 오르고, 공격 패치는 후반에 새로
몰린다. 3x3 창은 2% 면적 패치(한 변 ~32px)가 차지할 수 있는 최대 크기다.

검증된 범위(README "탐지기 v2" 절 참고): ViT-B/16(224px), PatchFool·LaVAN, 정상 보정 후 새 시드에서 평가.
알려진 한계: 탐지기를 아는 적응형 공격에는 뚫린다(탐지율 0%). 픽셀 수준 이상이 큰 공격(예: 이 저장소의 PatchFool)은
모델 없이 국소 변화량만으로도 더 잘 잡힌다. ViT-B/16 외 모델, 다른 패치 면적에서는 창·블록 구간을 다시 확인해야 한다.

사용:
    detector = LSWDetector(model)
    detector.fit(clean_images, fpr=0.05)        # 정상 이미지 수백 장 이상 권장
    result = detector.detect(images)            # {'alarm': (B,) bool, 'score': (B,), 'patches': (B,9) 의심 patch index}
"""
import torch
import torch.nn.functional as F

N_BLOCKS = 12
GRID = 14
LATE = (10, 11, 12)
REF = (7, 8, 9)
WINDOW = 3


def _cls_row_hook(rows):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((module.q_norm(q[:, :, :1]) * module.scale) @ module.k_norm(k).transpose(-2, -1)).softmax(-1)
        rows.append(attn[:, :, 0].mean(1))
    return hook


@torch.no_grad()
def cls_patch_rows(model, images):
    """(B,3,224,224) -> (B,12,196): 블록별 CLS->patch attention(head 평균), patch 위에서 합 1."""
    if len(model.blocks) != N_BLOCKS:
        raise ValueError(f'블록 {N_BLOCKS}개인 모델만 지원합니다(현재 {len(model.blocks)}).')
    rows = []
    hooks = [blk.attn.register_forward_hook(_cls_row_hook(rows)) for blk in model.blocks]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()
    out = torch.stack(rows, dim=1)[:, :, 1:]
    if out.shape[-1] != GRID * GRID:
        raise ValueError(f'patch {GRID * GRID}개(14x14)인 모델만 지원합니다(현재 {out.shape[-1]}).')
    return out / out.sum(-1, keepdim=True)


def surge_map(rows):
    """rows (B,12,196) -> (B, 144): 3x3 창별 surge. 창 index = 좌상단 patch의 row*12+col."""
    B = rows.shape[0]
    win = F.avg_pool2d(rows.reshape(B * N_BLOCKS, 1, GRID, GRID), WINDOW, stride=1) * WINDOW ** 2
    win = win.reshape(B, N_BLOCKS, -1)
    late = win[:, [b - 1 for b in LATE]].mean(1)
    reference = win[:, [b - 1 for b in REF]].max(1).values
    return late - reference


def surge_score(rows):
    return surge_map(rows).max(1).values


def suspect_patches(rows):
    """argmax 창이 덮는 patch index (B, 9)."""
    k = GRID - WINDOW + 1
    best = surge_map(rows).argmax(1)
    r, c = best // k, best % k
    offsets = torch.tensor([dr * GRID + dc for dr in range(WINDOW) for dc in range(WINDOW)], device=rows.device)
    return (r * GRID + c).unsqueeze(1) + offsets.unsqueeze(0)


class LSWDetector:
    def __init__(self, model, batch_size=50):
        self.model = model.eval()
        self.batch_size = batch_size
        self.threshold = None

    @torch.no_grad()
    def _rows(self, images):
        device = next(self.model.parameters()).device
        for start in range(0, len(images), self.batch_size):
            yield cls_patch_rows(self.model, images[start:start + self.batch_size].to(device))

    @torch.no_grad()
    def score(self, images):
        """(N,3,H,W) -> (N,) 이상치 점수(CPU)."""
        return torch.cat([surge_score(rows).cpu() for rows in self._rows(images)])

    def fit(self, clean_images, fpr=0.05):
        """정상 이미지 점수의 (1-fpr) 분위수를 임계값으로 정한다. GT나 공격 성공 여부는 쓰지 않는다."""
        scores = self.score(clean_images)
        self.threshold = float(torch.quantile(scores, 1 - fpr))
        return self.threshold

    @torch.no_grad()
    def detect(self, images):
        if self.threshold is None:
            raise RuntimeError('fit()으로 임계값을 먼저 정해야 합니다.')
        scores, patches = [], []
        for rows in self._rows(images):
            scores.append(surge_score(rows).cpu())
            patches.append(suspect_patches(rows).cpu())
        scores, patches = torch.cat(scores), torch.cat(patches)
        return {'alarm': scores > self.threshold, 'score': scores, 'patches': patches}
