# Attention before/after attack: success vs failure, by layer

Generated from run `/home/jooyumm/ViT_robust/data/attention/p16_patchfool_official` — its `manifest.json` and `data/` hold the reproducible source (seeds, hashes, raw images/attention); re-running this script only overwrites the files below, not that run.

- One folder per attack (`patchfool/`, `lavan/`), no outcome subfolders — the outcome is in the filename instead: `attack_success*` (attack succeeded, image misclassified) vs. `attack_fail*` (attack failed, image still correct). These names describe the ATTACK's outcome, not a pipeline failure.
- Maps show CLS-query attention to each patch key, head-averaged, CLS key mass kept (not renormalized away). GT patches (any pixel actually changed) are outlined with src/ground_truth/draw_boxes.py's colors, reused directly: blue on Clean panels, attack-specific red/green on attacked/difference panels.
- Raw attention: white -> pure blue. Attack-minus-clean: pure green (attention lost) -> white (no change) -> pure red (attention concentrated more). Ends are pure colors so panels stay distinguishable at the extremes; the color ramps jump quickly off white so small values stay visible, while the colorbars are linear with evenly spaced ticks. Both use a 99.5th-percentile cap (not the true max) shared across every block of a given attack, so a single extreme sample or patch can't blank out the rest, while the real block-to-block story (near-zero early on, then increasingly concentrated) still shows. Values above the cap are drawn at the top color, marked by the colorbar's arrow, not silently redrawn.
- GT Δ: attention change summed over GT patches (raw, not renormalized). TV: 0.5 × sum(|attack - clean|) over all 197 tokens including CLS.
- The layer table uses the *median* over all clean-correct samples; flagged blocks (⚠) are exploratory candidates by descriptive statistics, not a significance test or causal claim.
- No CSVs here by design — the table above is the numeric result worth keeping. Per-sample/per-head raw numbers are recomputable from `/home/jooyumm/ViT_robust/data/attention/p16_patchfool_official/data/` with `compute_metrics()` in this folder's script if ever needed.

## patchfool

- Attack succeeded (misclassified): n=259. Attack failed (still correct): n=160. (Clean-incorrect samples are excluded from this comparison.)
- Flagged blocks (largest median TV / largest TV jump / largest success-vs-fail GT gap): 8, 11, 12.
- `patchfool/layer_comparison.png`: population median + IQR by block.
- `patchfool/attack_success.png`, `patchfool/attack_fail.png`: one representative example each (sample id is in the fig title), clean vs. attack vs. difference at every block (1-12) — the flagged blocks above are where the population-level chart says to look closest, not the only blocks shown.

| Block | TV (median) | GT Δ success (pp) | GT Δ fail (pp) | flagged |
|---|---|---|---|---|
| 1 | 0.20 | -0.19 | -0.21 |  |
| 2 | 0.20 | +0.09 | +0.09 |  |
| 3 | 0.26 | +0.19 | +0.21 |  |
| 4 | 0.49 | +0.26 | +0.25 |  |
| 5 | 0.81 | +0.07 | +0.15 |  |
| 6 | 2.71 | -1.41 | -1.54 |  |
| 7 | 6.83 | -5.02 | -5.37 |  |
| 8 | 14.69 | -11.80 | -12.69 | ⚠ |
| 9 | 20.13 | -15.78 | -17.23 |  |
| 10 | 15.30 | -7.39 | -9.56 |  |
| 11 | 17.03 | +8.90 | +4.01 | ⚠ |
| 12 | 20.82 | +5.67 | +6.06 | ⚠ |
