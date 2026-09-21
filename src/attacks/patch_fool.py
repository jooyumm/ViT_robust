import torch
import torch.nn.functional as F
import numpy as np


def _clamp(x, lower, upper):
    return torch.max(torch.min(x, upper), lower)


def _pcgrad(atten_grad_flat, ce_grad_flat, cos_sim, original_shape):
    """PCGrad: 논문 utils.py 그대로."""
    pcgrad = atten_grad_flat.clone()
    conflict = cos_sim < 0
    if conflict.any():
        g_a = pcgrad[conflict]
        g_c = ce_grad_flat[conflict]
        dot  = (g_a * g_c).sum(dim=-1, keepdim=True)
        norm = g_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pcgrad[conflict] = g_a - (dot / norm) * g_c
    return pcgrad.view(original_shape)


def _make_attn_hook(weights_list):
    """
    timm Attention 모듈의 forward hook.
    input[0] = norm1(x) shape (B, seq, dim)
    """
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]                          # (B, N, C)
            B, N, C = x.shape
            qkv = module.qkv(x)                   # (B, N, 3*C)
            qkv = qkv.reshape(B, N, 3, module.num_heads,
                               C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)               # (B, heads, N, head_dim)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)            # (B, heads, N, N)
            weights_list.append(attn.detach().cpu())
    return hook


def _collect_attn(model):
    """hook 등록 → forward → hook 해제 → attn_weights 반환."""
    weights = []
    hooks   = []
    for blk in model.blocks:
        hooks.append(blk.attn.register_forward_hook(_make_attn_hook(weights)))
    return weights, hooks


def _select_patch_attn(attn_weights, layer_idx, num_patch, device):
    """
    Attention map에서 상위 num_patch 인덱스 반환.
    CLS → 각 패치로 가는 attention row 사용 (column 0, row 1:)
    """
    # attn_weights[layer_idx]: (B, heads, N, N), N = 1+patches
    a = attn_weights[layer_idx].to(device)  # (B, heads, N, N)
    # CLS 토큰(row 0)이 각 패치(col 1:)에 주는 attention
    a = a.mean(dim=1)[:, 0, 1:]            # (B, num_patches)
    return a.argsort(descending=True)[:, :num_patch]  # (B, num_patch)


def _select_contiguous_block(attn_weights, layer_idx, patch_size_model, H, device,
                             ref_patch_size=16):
    """
    개별 상위 attention 패치가 아니라, ref_patch_size(기본 16px) 하나와 정확히 같은
    물리적 영역을 덮는 block x block 개의 인접 패치를 고른다 (block = ref_patch_size // patch_size_model).
    각 후보 블록의 attention 합이 가장 큰 블록을 선택 — "면적은 같지만 뭉쳐있는" 버전.
    Returns: (B, block**2) 패치 인덱스 텐서 — _select_patch_attn과 같은 포맷
    """
    a = attn_weights[layer_idx].to(device).mean(dim=1)[:, 0, 1:]   # (B, num_patches)
    ppl   = H // patch_size_model                # patches per line
    block = ref_patch_size // patch_size_model    # 한 변에 들어가는 패치 수
    assert ppl % block == 0, \
        f"ppl({ppl})이 block({block})으로 안 나눠떨어짐 — ref_patch_size 확인 필요"

    B = a.shape[0]
    grid = a.view(B, ppl, ppl)
    n_blocks = ppl // block
    # block x block 단위로 attention 합산 → 블록별 점수
    block_scores = grid.unfold(1, block, block).unfold(2, block, block)  # (B, n_blocks, n_blocks, block, block)
    block_scores = block_scores.sum(dim=(-1, -2))                        # (B, n_blocks, n_blocks)
    best_block   = block_scores.reshape(B, -1).argmax(dim=1)  # (B,)
    block_row    = best_block // n_blocks
    block_col    = best_block % n_blocks

    offsets = torch.tensor(
        [(dr, dc) for dr in range(block) for dc in range(block)], device=device)  # (block**2, 2)
    rows = block_row.unsqueeze(1) * block + offsets[:, 0].unsqueeze(0)  # (B, block**2)
    cols = block_col.unsqueeze(1) * block + offsets[:, 1].unsqueeze(0)  # (B, block**2)
    return rows * ppl + cols


def _build_mask(max_patch_index, B, H, patch_size, device):
    """선택된 패치 위치에 1인 spatial mask (B, 1, H, W)."""
    ppl  = H // patch_size   # patches per line
    mask = torch.zeros(B, 1, H, H, device=device)
    for b in range(B):
        for idx in max_patch_index[b]:
            row = (idx // ppl) * patch_size
            col = (idx  % ppl) * patch_size
            mask[b, :, row:row+patch_size, col:col+patch_size] = 1.0
    return mask


def patch_fool_attack(
    model,
    images,
    labels,
    device,
    patch_size_model=16,
    num_patch=1,
    attack_mode='Attention',      # 'Attention' | 'CE_loss'
    patch_select='Attn',          # 'Attn' | 'Rand' | 'Contiguous'
    attn_layer_idx=4,
    train_attack_iters=250,
    attack_lr=0.22,
    step_size=10,
    gamma=0.95,
    atten_loss_weight=0.002,
    mild_l_inf=0.0,
):
    """
    Returns: adv_images (B,3,H,W), attn_weights (list, 시각화용)
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    # ── 1. 공격 전 forward → attention 수집 ──────────────────────────────
    attn_weights, hooks = _collect_attn(model)
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()

    # ── 2. 패치 선택 ──────────────────────────────────────────────────────
    num_patches = (H // patch_size_model) ** 2
    if patch_select == 'Attn' and len(attn_weights) > attn_layer_idx:
        max_patch_index = _select_patch_attn(
            attn_weights, attn_layer_idx, num_patch, device)
    elif patch_select == 'Contiguous' and len(attn_weights) > attn_layer_idx:
        # num_patch 대신 ref_patch_size(16px)와 동일 면적을 덮는 인접 블록을 선택
        max_patch_index = _select_contiguous_block(
            attn_weights, attn_layer_idx, patch_size_model, H, device)
    else:
        max_patch_index = torch.from_numpy(
            np.random.randint(0, num_patches, (B, num_patch))).to(device)

    # ── 3. 공간 마스크 ────────────────────────────────────────────────────
    mask = _build_mask(max_patch_index, B, H, patch_size_model, device)

    # ── 4. delta 초기화 ───────────────────────────────────────────────────
    # 논문: 정규화 공간에서 랜덤 init
    # mild_l_inf=0 → unbounded (vanilla Patch-Fool)
    if mild_l_inf == 0.0:
        delta = torch.randn_like(images) * 0.1
    else:
        delta = (2 * mild_l_inf * torch.rand_like(images) - mild_l_inf)

    original_images = images.clone()
    delta = delta.to(device).requires_grad_(True)

    opt       = torch.optim.Adam([delta], lr=attack_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    # attention loss용 index matrix (논문 코드 그대로)
    seq_len = num_patches + 1                              # patches + CLS
    mpim    = max_patch_index[:, 0].long()                 # (B,)
    mpim    = mpim.unsqueeze(0).expand(seq_len, -1).T      # (B, seq)
    mpim    = mpim.reshape(-1)                             # (B*seq,)

    # ── 5. 공격 반복 ──────────────────────────────────────────────────────
    for it in range(train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        perturbed = original_images + torch.mul(delta, mask)

        if attack_mode == 'Attention':
            # attention 수집하면서 forward
            attn_iter, hooks_iter = _collect_attn(model)
            out = model(perturbed)
            for h in hooks_iter:
                h.remove()
        else:
            out = model(perturbed)

        ce_loss = criterion(out, labels)

        if attack_mode == 'Attention' and len(attn_iter) > 0:
            grad = torch.autograd.grad(ce_loss, delta, retain_graph=True)[0]
            ce_grad_flat = grad.view(B, -1).detach().clone()

            for li in range(len(attn_iter) // 2):
                if li == 0:
                    continue
                a = attn_iter[li].to(device).mean(dim=1)  # (B, N, N)
                a = a.reshape(-1, a.size(-1))              # (B*N, N)
                a = -torch.log(a.clamp(min=1e-8))
                # CLS 토큰 +1 offset
                a_loss = F.nll_loss(a, (mpim + 1).to(device))
                a_grad = torch.autograd.grad(a_loss, delta, retain_graph=True)[0]
                a_flat = a_grad.view(B, -1)
                cos    = F.cosine_similarity(a_flat, ce_grad_flat, dim=1)
                a_grad = _pcgrad(a_flat, ce_grad_flat, cos, grad.shape)
                grad   = grad + a_grad * atten_loss_weight

            opt.zero_grad()
            delta.grad = -grad
        else:
            ce_loss.backward()
            if delta.grad is not None:
                delta.grad = -delta.grad

        opt.step()
        scheduler.step()

        with torch.no_grad():
            if mild_l_inf != 0.0:
                delta.data = delta.data.clamp(-mild_l_inf, mild_l_inf)

    # ── 6. 최종 adversarial image ─────────────────────────────────────────
    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()

    return adv_images, attn_weights