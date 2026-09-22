"""PatchFool/LaVAN 공격을 청크 단위로 생성. 00/05가 같은 파라미터를 써야 해서(재현성)
공유 파일로 뺐다.
"""
import torch

from src.attacks.patch_fool import patch_fool_attack
from src.attacks.lavan import lavan_attack


def chunked_patch_fool(model16, images, labels, device, attn_layer_idx, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        # attack_mode='Attention': CE + attention loss 같이 최적화(논문 원래 방식)
        adv_chunk, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='Attention', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=attn_layer_idx)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)


def chunked_lavan(model16, images, labels, device, patch_ratio, steps, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        adv_chunk = lavan_attack(
            model16, images[s:e], labels[s:e], device, patch_ratio=patch_ratio, steps=steps)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)
