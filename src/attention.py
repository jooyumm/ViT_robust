"""timm 공식 추출기로 logits와 softmax attention을 한 forward에서 얻는다."""
from timm.utils import AttentionExtract


def extract_attention(model, images):
    """eval 모델 전용. 반환: logits, 레이어 순서의 (B,H,Q,K) attention 튜플.

    forward_pre는 attention dropout 직전의 softmax 확률을 읽는다.
    timm의 hook 해제 API가 없어 FeatureHooks의 handle 목록만 정리한다.
    """
    if model.training:
        raise ValueError('Attention 추출에는 model.eval()이 필요합니다.')
    names = [f'blocks.{i}.attn.attn_drop' for i in range(len(model.blocks))]
    fused = [block.attn.fused_attn for block in model.blocks]
    extractor = None
    logits = []
    handle = model.register_forward_hook(lambda module, inputs, output: logits.append(output))
    try:
        for block in model.blocks:
            block.attn.fused_attn = False
        extractor = AttentionExtract(model, names=names, method='hook', hook_type='forward_pre')
        attention = extractor(images)
        if list(attention) != names:
            raise RuntimeError('전체 레이어의 attention이 추출되지 않았습니다.')
        return logits[0], tuple(attention[name] for name in names)
    finally:
        handle.remove()
        if extractor is not None:
            for hook in extractor.hooks._handles:
                hook.remove()
            extractor.hooks._feature_outputs.clear()
        for block, original in zip(model.blocks, fused):
            block.attn.fused_attn = original
