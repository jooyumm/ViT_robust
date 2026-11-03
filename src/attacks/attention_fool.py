"""Attention-Fool의 L_kq / L_kq* 목적함수 구현. 공식 학습 설정 전체의 재현은 아니다.
논문: https://openaccess.thecvf.com/content/CVPR2022/papers/Lovisotto_Give_Me_Your_Attention_Dot-Product_Attention_Considered_Harmful_for_Adversarial_CVPR_2022_paper.pdf

논문 L_kq/L_kq*에 따라 Q/K를 head별 평균 L2 norm으로 정규화하고, 대상 key로 가는 pre-softmax
내적을 head·layer logsumexp로 집계한다. 여러 image token에 걸친 마스크는 대상 key의 평균으로
확장했다 — 논문의 단일 target key와 다르므로 이 설정을 보고해야 한다. `attention_terms`(공유
hook, attack_utils.py)를 그대로 쓰므로 단일 CLS, 일반 timm ViT만 지원하며 Swin/DETR/register
token/동적 patch 격자는 지원하지 않는다.
"""
import torch
from ..attack_utils import projected_attack, classification_loss, token_support
from ..attack_utils import attention_terms


def attention_fool_attack(model, images, labels, device, *, layers=None, query='all',
                          attention_weight=1., normalize_qk=True, loss='ce',
                          targeted=False, target_labels=None, **kwargs):
    """head별 Q/K 평균 L2 정규화, 대상 key 내적, head/layer logsumexp. 다중 key는 평균 확장.
    `query='all'`은 CLS를 포함한 모든 query, `'cls'`는 CLS query다. `layers=None`은 전체
    블록, 직접 지정하면 0-based 인덱스다. 기본은 cosine PGD 스케줄(`attack_utils.projected_attack`의
    `cosine=True`)이다."""
    if query not in ('all','cls') or attention_weight < 0:
        raise ValueError('query=all/cls, attention_weight>=0이어야 합니다.')
    def objective(x, y, targets, mask):
        target = token_support(model, mask)
        def reducer(q, k, scale):
            if k.shape[-2] != target.shape[-1]+1:
                raise ValueError('이미지 token과 픽셀 마스크 격자가 다릅니다.')
            if normalize_qk:
                q = q / q.norm(dim=-1).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12)
                k = k / k.norm(dim=-1).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-12)
            query_vector = q.mean(-2) if query == 'all' else q[:,:,0]
            key_vector = (k[:,:,1:] * target[:,None,:,None]).sum(-2) / target.sum(-1)[:,None,None]
            return (query_vector*key_vector).sum(-1)*scale
        with attention_terms(model, reducer, layers) as terms:
            logits = model(x)
        attraction = torch.stack(list(terms.values()), -1).flatten(1).logsumexp(-1)
        score = classification_loss(logits,y,loss,targeted,targets) + attention_weight*attraction
        return score, logits, None
    kwargs.setdefault('cosine', True)
    return projected_attack(model, images, labels, device, objective,
                            targeted=targeted, target_labels=target_labels, **kwargs)
