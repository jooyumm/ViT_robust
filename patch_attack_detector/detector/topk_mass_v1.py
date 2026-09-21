"""
detector/topk_mass_v1.py — raw attention 기반 Top-K Mass 탐지·위치특정의 **v1(초기) 구현**.

이름에 "v1"을 붙인 이유
----------------------
이 파일 하나가 patch_attack_detector의 전부이던 시절엔 그냥 `detector.py`였다. 이 프로젝트의 존재
이유 자체가 "탐지기를 여러 버전으로 독립적으로 발전시키는 것"(README 참고 — 레이어별
sweep, LaVAN 보완용 Dual-Gate 등)이므로, 두 번째 버전이 생기기 전에 미리 `detector/`
폴더 밑에 버전이 드러나는 이름으로 옮겨뒀다. 이후 개선된 탐지기는
`detector/dual_gate_v2.py`처럼 이 파일과 나란히 추가될 것이고, 이 파일 자체는 바뀌지
않는다(patch_switch_defense가 지금 실제로 배포에 쓰는 버전이 바로 이것이므로).

원래 위치와 배경 (patch_switch_defense/defense/detector.py였을 때)
--------------------------------------------------------------
두 방어 메커니즘(all_switch, local_switch)이 공통으로 쓰는 탐지·위치특정 메커니즘의 단일
정본(canonical) 구현. 이 파일이 생기기 전까지는 이 로직(_attn_hook -> collect_layer_attn
-> raw_at_layer -> top4_mass/argmax)이 defense/ 아래 13곳에 독립적으로 복사돼 있었다
(감사 결과: 전부 detect_layer=12, topk(4)로 완전히 동일, 분기 없음). 원본은
experiments/detection_localization/localization/vitguard_localization.py였다.
여기서는 그 로직을 한 곳으로 모으되, 각 실험 스크립트(§1~§18 등)는 이미 검증이 끝난
자기 복사본을 그대로 유지한다(프로젝트의 폴더별 자기완결 관례 — 재검증 없이 그대로 둠).
2026-09-21에 이 파일 자체가 patch_switch_defense/defense/에서 이 프로젝트로 옮겨졌고(git
이력 보존, patch_attack_detector README "이력" 참고), 다시 여기서 `detector/topk_mass_v1.py`로
재배치됐다(이번엔 git mv, 같은 저장소 안이라 이력이 자동으로 이어짐).

메커니즘 (실배포 기준)
----
raw attention: 12번째 transformer block의 attention을 head 평균 낸 뒤, CLS->patch 행을
정규화한 벡터. top-4 원소의 합("top4_mass")이 이미지 단위 이상치 점수이고, 그 argmax
1개가 "의심 패치" 위치다(recall@1 96.7%, §2). 점수를 calibration에서 정한 임계값과
비교해 escalate 여부를 결정한다(FPR/recall은 그 임계값의 함수 — 임계값 자체는 여기서
만들지 않고, 매 실험이 자기 calibration 표본으로 정한다).
"""
import torch


def _attn_hook(weights_list):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            weights_list.append(attn.mean(dim=1).detach())
    return hook


def collect_layer_attn(model, images):
    """model의 12개 block 전부에서 head-평균 attention을 뽑아 층별 리스트로 반환."""
    weights = []
    hooks = [blk.attn.register_forward_hook(_attn_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def raw_at_layer(layer_weights, L):
    """collect_layer_attn 결과에서 L번째 층의 CLS->patch 정규화 벡터 (B, N_patch)."""
    attn = layer_weights[L - 1]
    cls_to_patch = attn[:, 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)


def top4_mass(v):
    """이상치 점수: 벡터의 top-4 원소 합 (B,)."""
    return v.topk(4, dim=1).values.sum(dim=1)


@torch.no_grad()
def detection_score(model, images, detect_layer=12):
    """images (B,3,H,W) -> 이상치 점수 (B,). 실배포 탐지기와 100% 동일한 경로(no_grad)."""
    layer_weights = collect_layer_attn(model, images)
    v = raw_at_layer(layer_weights, detect_layer)
    return top4_mass(v)


@torch.no_grad()
def localize_top1(model, images, detect_layer=12):
    """images (B,3,H,W) -> 의심 패치 flat index (B,). L=detect_layer raw attention argmax."""
    layer_weights = collect_layer_attn(model, images)
    v = raw_at_layer(layer_weights, detect_layer)
    return v.argmax(dim=1)


def should_escalate(score, threshold):
    """점수 > 임계값이면 방어(all_switch/local_switch)를 발동할지 여부 (B,) bool 텐서."""
    return score > threshold


@torch.no_grad()
def predict_detect_localize(model, images, detect_layer=12):
    """실배포에서 실제로 필요한 단일 forward: P16 예측(logits) + 탐지 점수 + 위치특정을
    **한 번의 forward pass**로 함께 계산한다 (detection_score/localize_top1을 따로 부르면
    같은 forward를 두 번 하게 됨 — cost_comparison처럼 실제 파이프라인 비용을 잴 때는 이
    함수를 쓸 것). 다른 실험 스크립트들이 진작부터 관례로 써 온 "예측 1회 + collect_layer_attn
    1회"(2 forward) 방식은 연구/측정 편의를 위한 것이었지, 실배포 비용의 정답은 아니다."""
    weights = []
    hooks = [blk.attn.register_forward_hook(_attn_hook(weights)) for blk in model.blocks]
    logits = model(images)
    for h in hooks:
        h.remove()
    v = raw_at_layer(weights, detect_layer)
    score = top4_mass(v)
    flag_idx = v.argmax(dim=1)
    return logits, score, flag_idx
