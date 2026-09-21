"""
probes/patch_fool_joint_evasive.py — [탐색적, 롤백 가능] 탐지 회피 항이 추가된 joint attack.
"완전판 adaptive attacker": 방어 구조(P16+P8 동시 존재)뿐 아니라 탐지기(L=12 raw attention
top-4 mass) 존재 자체도 알고, 그 탐지 지표를 clean 범위 안에 붙잡아두면서 P16+P8을 동시에
속이는 perturbation을 찾는다. STRAP-ViT류 adaptive attack 설계(탐지 지표를 clean 최댓값
아래로 강제하는 제약 최적화)를 참고해서 구현.

배경
----
patch_fool_joint.py의 joint_patch_fool_attack은 P16+P8을 동시에 속이지만 탐지기 존재를
전혀 모른다 — 그 결과가 patch_switch_defense/README.md §7이다(무력화율 18.4%, 그 중 탐지기가 20%는 그래도
flag함). 이 파일은 그 공격에 탐지 회피 제약을 추가해서 "탐지기까지 아는 진짜 최악의
공격자"를 흉내낸다.

방법 (제약 최적화)
------------------
- 목적: CE16(perturbed) + CE8(perturbed)를 키우면서(오분류), 동시에 L=12 raw attention의
  top-4 mass 점수를 target_bound(기본: calibration에서 관찰된 clean 최댓값) 아래로 유지
- 매 iteration마다:
    combined = CE16 + CE8 - lambda * relu(score - target_bound)^2
    combined.backward(); delta.grad = -delta.grad; opt.step()
  (기존 코드베이스의 "backward 후 grad 부호 반전" 관례를 그대로 유지 — CE 항은 ascent로
  오분류를 키우고, 동시에 -lambda*penalty 항도 ascent되므로 결과적으로 penalty는 descent됨
  = score가 target_bound 밑으로 내려가도록 최적화된다. 하나의 backward 호출, 하나의 부호
  반전으로 두 목표가 동시에 최적화되는 구조.)
- lambda는 고정하지 않고, `lambda_step_every` iteration마다 여전히 제약을 어기고 있으면
  (mean penalty > eps) lambda_growth배로 키움(최대 lambda_max) — 단순화된 augmented
  Lagrangian(듀얼 변수 업데이트 없이 페널티 계수만 키우는 방식).
- score는 model16(perturbed)의 마지막 block(L=12) attention에 forward hook을 걸어
  **grad를 끊지 않고**(patch_fool.py의 관측용 hook들과 달리 no_grad/detach 없음) 계산 —
  이래야 delta까지 역전파가 이어짐.
- 공격 영역 선택(_select_patch_attn 등)은 기존과 동일하게 attn_layer_idx(기본 4)의
  model16 attention으로 정함 — 탐지 회피 항은 "어디를 공격할지"가 아니라 "어떻게 공격
  perturbation을 만들지"에만 영향을 준다.

주의: src/attacks/patch_fool.py는 import만(수정 없음), patch_fool_joint.py의
joint_patch_fool_attack도 그대로 둔 채 옆에 새 함수만 추가. 이 파일 지우면 원상복구.
"""
import torch
import torch.nn.functional as F
from src.attacks.patch_fool import _build_mask, _collect_attn, _select_patch_attn


def _make_diff_attn_hook(store, key='attn'):
    """관측용(no_grad) 버전과 달리 grad를 안 끊는다 -- delta까지 역전파가 이어져야 하므로."""
    def hook(module, input, output):
        x = input[0]                          # (B, N, C), 그래프에 연결된 텐서
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(
            B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)           # (B, heads, N, N)
        store[key] = attn.mean(dim=1)         # (B, N, N) -- 헤드 평균, grad 유지
    return hook


def detection_score(attn):
    """attn: (B, N, N) -- CLS->patch top-4 mass, raw_at_layer/top4_mass와 동일 정의(미분 가능)."""
    cls_to_patch = attn[:, 0, 1:]
    v = cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)
    return v.topk(4, dim=1).values.sum(dim=1)   # (B,)


def joint_patch_fool_attack_evasive(
    model16, model8, images, labels, device,
    attn_layer_idx=4, num_patch=1,
    train_attack_iters=250, attack_lr=0.22,
    step_size=10, gamma=0.95, mild_l_inf=0.0,
    detect_layer=12, target_bound=0.5567,
    lambda0=5.0, lambda_growth=1.5, lambda_max=500.0,
    lambda_step_every=25, penalty_eps=1e-3,
):
    """
    P16 attention으로 공격 토큰 위치를 정하고, 그 물리적 영역을 P16+P8 CE 합으로 동시에
    공격하되, model16의 L=detect_layer raw attention top-4 mass가 target_bound를 넘으면
    페널티를 부과해서 탐지를 회피하도록 제약을 건다.

    Returns:
      adv_images, true_idx (B,), score_history (list of (B,) numpy, iteration마다 기록),
      lambda_history (list of float)
    """
    model16.eval()
    model8.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    attn_weights, hooks = _collect_attn(model16)
    with torch.no_grad():
        model16(images)
    for h in hooks:
        h.remove()
    max_patch_index = _select_patch_attn(attn_weights, attn_layer_idx, num_patch, device)

    mask = _build_mask(max_patch_index, B, H, 16, device)

    if mild_l_inf == 0.0:
        delta = torch.randn_like(images) * 0.1
    else:
        delta = (2 * mild_l_inf * torch.rand_like(images) - mild_l_inf)
    original_images = images.clone()
    delta = delta.to(device).requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=attack_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    detect_block = model16.blocks[detect_layer - 1]
    store = {}
    diff_hook = detect_block.attn.register_forward_hook(_make_diff_attn_hook(store))

    lam = lambda0
    score_history = []
    lambda_history = []

    try:
        for it in range(train_attack_iters):
            model16.zero_grad()
            model8.zero_grad()
            opt.zero_grad()

            perturbed = original_images + torch.mul(delta, mask)
            out16 = model16(perturbed)          # store['attn']이 여기서 채워짐 (grad 유지)
            out8 = model8(perturbed)
            ce = criterion(out16, labels) + criterion(out8, labels)

            score = detection_score(store['attn'])          # (B,)
            penalty = F.relu(score - target_bound).pow(2).mean()

            combined = ce - lam * penalty
            combined.backward()
            if delta.grad is not None:
                delta.grad = -delta.grad
            opt.step()
            scheduler.step()
            with torch.no_grad():
                if mild_l_inf != 0.0:
                    delta.data = delta.data.clamp(-mild_l_inf, mild_l_inf)

            score_history.append(score.detach().cpu().numpy())
            lambda_history.append(lam)

            if (it + 1) % lambda_step_every == 0:
                mean_violation = F.relu(score.detach() - target_bound).mean().item()
                if mean_violation > penalty_eps:
                    lam = min(lam * lambda_growth, lambda_max)
    finally:
        diff_hook.remove()

    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()
    return adv_images, max_patch_index[:, 0], score_history, lambda_history
