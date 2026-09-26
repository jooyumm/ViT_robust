"""
detector/emergent_window_v2.py — Late-Surge Window (LSW) 탐지·위치특정. v1(topk_mass_v1.py)과 나란히 두는 v2.

아이디어
--------
v1은 마지막 블록(12) CLS attention의 top-4 mass 하나만 본다. 원본 이미지에도 자연 attention sink와
물체 attention이 이미 있어서 기준선이 높고(원본 top-4 mass 중앙값 ~40%), LaVAN처럼 attention이 여러
patch에 퍼지는 공격은 오히려 점수가 낮아진다(v1 AUROC 0.34). LSW는 두 가지를 바꾼다.

1. **늦게 생긴 몰림만 본다(emergence)**: 자연 sink/물체 attention은 블록 7~9에 이미 형성돼 있고 이후 유지된다.
   공격 패치는 블록 7~9에서는 눈에 안 띄다가(순위 ~4, attention 2~3%) 블록 10~12에서 급증한다. 그래서
   (후반 블록 평균 attention) - (참조 블록 7~9에서의 최댓값)을 쓴다. 참조 때 이미 높았던 곳은 점수가 안 오른다.
2. **공격이 차지할 수 있는 크기의 창(window)으로 합친다**: 패치 공격은 면적이 제한돼 있다(2% 면적이면 한 변
   ~32px = 최대 3x3 patch). patch 하나가 아니라 3x3 창의 attention 합을 써서, 여러 patch에 걸친 LaVAN도 한 덩어리로
   잡는다. 창을 3x3으로 제한하므로 물체 전체처럼 넓게 퍼진 attention은 창 하나에 다 담기지 않는다.

점수 = 모든 3x3 창에 대한 max( mean_{b in 10,11,12} 창합_b - max_{b in 7,8,9} 창합_b ). 임계값을 넘으면 경보,
argmax 창이 의심 위치. 한 번의 forward, 학습 없음, 정상 이미지로 임계값만 보정한다(v1과 같은 방식).
CLS query 행만 계산하므로 전체 N x N attention 행렬도 필요 없다.

한계(정직하게): 창 크기·블록 구간은 ViT-B/16, 224px, 2% 면적 패치 기준으로 정했다. 탐지기를 아는 적응형 공격
(후반 블록 attention을 일부러 억제하는 공격)은 평가하지 않았다.
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
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            # CLS query만 필요하다. CLS key까지 포함해 softmax한 뒤 head 평균.
            attn = ((q[:, :, :1] * module.scale) @ k.transpose(-2, -1)).softmax(-1)
            rows.append(attn[:, :, 0].mean(1).detach())
    return hook


@torch.no_grad()
def cls_patch_rows(model, images):
    """(B,3,H,W) -> (B,12,196): 블록별 CLS->patch attention(head 평균), patch 위에서 합 1로 재정규화(v1과 동일)."""
    rows = []
    hooks = [blk.attn.register_forward_hook(_cls_row_hook(rows)) for blk in model.blocks]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()
    out = torch.stack(rows, dim=1)[:, :, 1:]
    return out / out.sum(-1, keepdim=True)


def surge_map(rows, late=LATE, ref=REF, window=WINDOW):
    """rows (B,12,196) -> (B, k*k) 창별 surge(k=14-window+1). 창 index = 좌상단 patch의 row*k+col."""
    B = rows.shape[0]
    win = F.avg_pool2d(rows.reshape(B * N_BLOCKS, 1, GRID, GRID), window, stride=1) * window ** 2
    win = win.reshape(B, N_BLOCKS, -1)
    late_mass = win[:, [b - 1 for b in late]].mean(1)
    ref_mass = win[:, [b - 1 for b in ref]].max(1).values
    return late_mass - ref_mass


def surge_score(rows, **kw):
    """이미지 단위 이상치 점수 (B,)."""
    return surge_map(rows, **kw).max(1).values


def suspect_patches(rows, window=WINDOW, **kw):
    """argmax 창이 덮는 patch index (B, window*window). 위치특정용."""
    k = GRID - window + 1
    best = surge_map(rows, window=window, **kw).argmax(1)
    r, c = best // k, best % k
    offsets = torch.tensor([dr * GRID + dc for dr in range(window) for dc in range(window)], device=rows.device)
    return (r * GRID + c).unsqueeze(1) + offsets.unsqueeze(0)


@torch.no_grad()
def detection_score(model, images):
    """images (B,3,H,W) -> 이상치 점수 (B,). v1의 detection_score와 같은 입출력."""
    return surge_score(cls_patch_rows(model, images))
