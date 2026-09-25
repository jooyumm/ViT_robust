# Attention before/after attack: success vs failure, by layer

Generated from run `/home/jooyumm/ViT_robust/01_patch_attack_detector/results/characterization/attention_comparison/run_s42` — its `manifest.json` and `data/` hold the reproducible source (seeds, hashes, raw images/attention); re-running this script only overwrites the files below, not that run.

- One folder per attack (`patchfool/`, `lavan/`), no outcome subfolders — the outcome is in the filename instead: `ex_attack_success_*` (attack succeeded, image misclassified) vs. `ex_attack_fail_*` (attack failed, image still correct). These names describe the ATTACK's outcome, not a pipeline failure.
- Maps show CLS-query attention to each patch key, head-averaged, CLS key mass kept (not renormalized away). GT patches (any pixel actually changed) are outlined with ground_truth/draw_boxes.py's colors, reused directly: blue on Clean panels, yellow on attacked/difference panels.
- Raw attention: white -> pure blue. Attack-minus-clean: pure green (attention lost) -> white (no change) -> pure red (attention concentrated more). Ends are pure colors, not dark/black, so panels stay distinguishable at the extremes. Both use a 99.5th-percentile cap (not the true max) shared across every block of a given attack, so a single extreme sample or patch can't blank out the rest, while the real block-to-block story (near-zero early on, then increasingly concentrated) still shows. Values above the cap are drawn at the top color, marked by the colorbar's arrow, not silently redrawn.
- GT Δ: attention change summed over GT patches (raw, not renormalized). TV: 0.5 × sum(|attack - clean|) over all 197 tokens including CLS.
- The layer table uses the *median* over all clean-correct samples; flagged blocks (⚠) are exploratory candidates by descriptive statistics, not a significance test or causal claim.
- No CSVs here by design — the table above is the numeric result worth keeping. Per-sample/per-head raw numbers are recomputable from `/home/jooyumm/ViT_robust/01_patch_attack_detector/results/characterization/attention_comparison/run_s42/data/` with `compute_metrics()` in this folder's script if ever needed.

## patchfool

- Attack succeeded (misclassified): n=82. Attack failed (still correct): n=5. (Clean-incorrect samples are excluded from this comparison.)
- Flagged blocks (largest median TV / largest TV jump / largest success-vs-fail GT gap): 11, 12.
- `patchfool/layer_comparison.png`: population median + IQR by block.
- `patchfool/ex_attack_success_<id>.png`, `patchfool/ex_attack_fail_<id>.png`: one representative example each, clean vs. attack vs. difference at every block (1-12) — the flagged blocks above are where the population-level chart says to look closest, not the only blocks shown.

| Block | TV (median) | GT Δ success (pp) | GT Δ fail (pp) | flagged |
|---|---|---|---|---|
| 1 | 0.21 | -0.19 | -0.29 |  |
| 2 | 0.61 | +0.48 | +0.20 |  |
| 3 | 0.56 | +0.45 | +0.36 |  |
| 4 | 0.92 | +0.60 | +0.33 |  |
| 5 | 1.14 | +0.28 | +0.19 |  |
| 6 | 2.92 | -0.77 | -1.15 |  |
| 7 | 6.96 | -4.44 | -4.78 |  |
| 8 | 14.52 | -10.69 | -12.20 |  |
| 9 | 18.63 | -11.31 | -16.11 |  |
| 10 | 16.72 | +1.23 | -9.53 |  |
| 11 | 29.69 | +23.67 | -6.93 | ⚠ |
| 12 | 40.26 | +35.79 | +25.36 | ⚠ |

## lavan

- Attack succeeded (misclassified): n=55. Attack failed (still correct): n=32. (Clean-incorrect samples are excluded from this comparison.)
- Flagged blocks (largest median TV / largest TV jump / largest success-vs-fail GT gap): 11, 12.
- `lavan/layer_comparison.png`: population median + IQR by block.
- `lavan/ex_attack_success_<id>.png`, `lavan/ex_attack_fail_<id>.png`: one representative example each, clean vs. attack vs. difference at every block (1-12) — the flagged blocks above are where the population-level chart says to look closest, not the only blocks shown.

| Block | TV (median) | GT Δ success (pp) | GT Δ fail (pp) | flagged |
|---|---|---|---|---|
| 1 | 0.07 | -0.07 | -0.06 |  |
| 2 | 0.17 | -0.13 | -0.14 |  |
| 3 | 0.39 | +0.29 | +0.19 |  |
| 4 | 1.02 | +1.03 | +0.53 |  |
| 5 | 1.14 | +0.96 | +0.66 |  |
| 6 | 2.21 | +1.47 | +1.01 |  |
| 7 | 2.78 | +1.22 | +0.69 |  |
| 8 | 4.49 | +1.65 | +0.97 |  |
| 9 | 6.78 | +4.26 | +2.92 |  |
| 10 | 12.19 | +9.09 | +7.32 |  |
| 11 | 25.63 | +17.27 | +17.47 | ⚠ |
| 12 | 31.55 | +14.62 | +21.33 | ⚠ |
