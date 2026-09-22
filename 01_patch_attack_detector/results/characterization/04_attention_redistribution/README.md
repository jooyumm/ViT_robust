# 04 — Attention redistribution

Input: `/home/jooyumm/ViT_robust/01_patch_attack_detector/results/characterization/00_attention_data_n100.npz`. No new attacks/model inference.

- Argmax location change: n=100 paired images. Argmax is among patch tokens; CLS excluded.
- Spatial maps and regional analysis: 5 stored examples only, not the full sample.
- Raw head-averaged attention retains CLS; attacked + other + CLS deltas sum to zero.
- All queries means the raw attention averaged over CLS and patch queries.
- Positive delta is gained attention, negative is lost attention; this is not causal flow.
- Total variation = 0.5 * sum(abs(attack - clean)), including CLS.
- GT support includes any token with coverage > 0; partial tokens are counted as whole tokens.
- Overview bands show example min–max, not confidence intervals. CSV values are fractions.
- Stored example IDs map to full-array sample offsets [40, 41, 42, 43, 44] by unique metric matching across all groups. These are not ImageNet dataset IDs. Source has no labels or reliable run metadata; do not join with independently regenerated 03 results or infer success/failure.
- Figures are saved as PNG only. Original inputs are not modified.
