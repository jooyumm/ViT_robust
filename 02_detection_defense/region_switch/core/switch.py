"""P16 가중치의 고정 선형 변환으로 선택 사분면만 8×8 패치로 세분화한다."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def converted_patch_kernel(model16):
    """8×8 입력을 bilinear로 16×16 확대한 후 원래 projection을 적용하는 것과 같은 커널.

    보간 행렬과 사전학습 W16의 곱만 사용한다. 이미지, label, GT, 피팅은 사용하지 않는다.
    반환 커널은 P16의 기존 임베딩 좌표계에 있으므로 별도 어댑터가 필요 없다.
    """
    weight = model16.patch_embed.proj.weight
    basis = torch.eye(64, dtype=weight.dtype, device=weight.device).reshape(64, 1, 8, 8)
    basis = F.interpolate(basis, size=(16, 16), mode='bilinear', align_corners=False).reshape(64, 256)
    return torch.einsum('ocq,pq->ocp', weight.flatten(2), basis).reshape(weight.shape[0], 3, 8, 8)


def patch_embed_with_pos(model, images):
    return model.patch_embed(images) + model.pos_embed[:, 1:]


def full_forward_from_tokens(model16, cls_plus_patches):
    x = model16.pos_drop(cls_plus_patches)
    for block in model16.blocks:
        x = block(x)
    x = model16.norm(x)[:, 0]
    return model16.head(model16.fc_norm(x))


def build_region_swap_batch(model16, kernel8, images, region_idx):
    """별도 P8 모델/어댑터 없이 P16 49개를 변환된 P8 196개로 교체한다."""
    if images.shape[1:] != (3,224,224) or region_idx.shape != (len(images),):
        raise ValueError('224×224 이미지와 이미지별 사분면 인덱스가 필요합니다.')
    if not bool(((region_idx >= 0) & (region_idx < 4)).all()):
        raise ValueError('사분면 인덱스는 0..3이어야 합니다.')
    patch16 = patch_embed_with_pos(model16, images)
    cls = model16.cls_token.expand(len(images), -1, -1) + model16.pos_embed[:, :1]
    pos16 = model16.pos_embed[:, 1:].reshape(1,14,14,-1).permute(0,3,1,2)
    pos8 = F.interpolate(pos16, size=(28,28), mode='bilinear', align_corners=False).permute(0,2,3,1)
    result = []
    for i, region in enumerate(region_idx.tolist()):
        row, col = divmod(region,2)
        crop = images[i:i+1,:,row*112:(row+1)*112,col*112:(col+1)*112]
        sub = F.conv2d(crop,kernel8,model16.patch_embed.proj.bias,stride=8).flatten(2).transpose(1,2)
        sub = model16.patch_embed.norm(sub)
        position = pos8[:,row*14:(row+1)*14,col*14:(col+1)*14].reshape(1,196,-1)
        replacement = sub + position
        keep = torch.ones((14,14),dtype=torch.bool,device=images.device)
        keep[row*7:(row+1)*7,col*7:(col+1)*7] = False
        result.append(torch.cat([cls[i],patch16[i,keep.flatten()],replacement[0]],dim=0))
    return torch.stack(result)


def forward_with_replacement_mass(model16, sequence, replacement_count=196, key_weight=0.25):
    """P16 하나를 P8 네 개로 세분화할 때 key당 면적 비중을 보정한다.

    replacement_count개의 마지막 토큰에 key_weight를 곱하고 attention 행을
    재정규화한다. 0.25는 P8/P16 면적 비율이다. 같은 부모 토큰을 네 번 복제한
    경우 원래 시퀀스와 같은 출력을 내는 질량 보존 조건을 만족한다.
    """
    if not 0 < key_weight <= 1 or not 0 < replacement_count < sequence.shape[1]:
        raise ValueError('replacement_count/key_weight가 올바르지 않습니다.')
    handles, previous_fused = [], []
    def reweight(module, inputs):
        a = inputs[0]
        weights = a.new_ones(a.shape[-1])
        weights[-replacement_count:] = key_weight
        weighted = a * weights
        return (weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-12),)
    try:
        for block in model16.blocks:
            previous_fused.append(block.attn.fused_attn)
            block.attn.fused_attn = False
            handles.append(block.attn.attn_drop.register_forward_pre_hook(reweight))
        return full_forward_from_tokens(model16, sequence)
    finally:
        for handle in handles:
            handle.remove()
        for block, fused in zip(model16.blocks, previous_fused):
            block.attn.fused_attn = fused

