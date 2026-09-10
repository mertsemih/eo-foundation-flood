# Lab notebook

One entry per working day. What was tried, what happened, what is next. Numbers go here first, then into the results CSV.

Template:

```
## YYYY-MM-DD (week N)
- Did:
- Result:
- Broke / surprised me:
- Next:
```

---

## 2026-09-08 (week 1)
- Did: repository skeleton — dataset loader, transforms, metrics, U-Net baseline, Prithvi wrapper (untested), train/evaluate CLIs, configs for the five models, download and stats scripts, unit tests, CI, 12-week plan.
- Result: tests pass on synthetic rasters; nothing trained yet.
- Broke / surprised me: —
- Next: download Sen1Floods11, EDA notebook, compute band statistics.

## 2026-09-10 (week 1)
- Did: download script rewritten to plain HTTPS (bucket is public, no gcloud needed); downloaded all 446 chips x 3 folders (1.75 GB, 4.7 min). Band stats computed and written into every config (`scripts/apply_stats.py`). EDA notebook generated from `scripts/make_eda_notebook.py` and executed. CUDA torch installed; local RTX 3070 Ti (8 GB) used for training. Started `unet_scratch` seed 0.
- Result (EDA): split sizes 252 / 89 / 90 / 15 match the paper. Ten regions in train/valid/test, Bolivia only in its own split. Median water fraction per chip is only ~2.5 % (mean 9–14 %), so water IoU is the right headline metric. No-data share: mean 13 % in train/valid/test but **27 % in Bolivia** (median 27 %, max 66 %). One train chip has zero valid pixels; three valid chips are fully no-data. Band means (x10 000 reflectance): B2 1396, B3 1364, B4 1218, B8A 3077, B11 2031, B12 1179.
- Broke / surprised me: Bolivia is harder than "a new region" suggests: it is also the cloudiest split. Any OOD claim must mention this.
- Next: finish unet_scratch, look at Bolivia predictions, run unet_imagenet, install terratorch and try loading Prithvi.

## 2026-09-10, later (week 1 → 2)
- Did: `unet_scratch` seed 0, full labels, 50 epochs on the local RTX 3070 Ti. Added the RAM cache (epoch 17 s → 4.5 s) and `scripts/run_matrix.py`; launched the U-Net label-fraction sweep (both U-Nets × {1.0, 0.25, 0.10, 0.05}, seed 0). Started installing terratorch + peft for Prithvi.
- Result (unet_scratch, f=1.0, s=0, best epoch 45, 3.9 GPU-min): **test water IoU 0.823** (mIoU 0.898, F1 0.903, P 0.915, R 0.892); **Bolivia water IoU 0.750** (mIoU 0.850, F1 0.857). Val water IoU is noisy epoch to epoch (0.58–0.81) because valid is 89 chips with random 224 crops — evaluating on full 512 chips is what we do, the noise is just in the selection signal. A 7-point in-distribution → Bolivia drop is a healthy-size OOD gap for the paper's claim 4.
- Visual check of Bolivia (`docs/figures/bolivia_unet_scratch.png`): (1) the split is dominated by cumulus clouds and their shadows; (2) worst chip (290290, IoU 0.53) is flooded *vegetation*: water under canopy looks dark green in RGB and the model under-segments it; (3) some cloud shadows are predicted as water; (4) swath edges (no-data triangles) are handled fine.
- Broke / surprised me: editing `sen1floods11.py` while a run was in progress killed it (Windows DataLoader workers re-import the module every epoch). Rule added to PLAN.md: never edit `src/` while a run is going; queue runs through `run_matrix.py`.
- Next: sweep results → figure 1; load Prithvi via terratorch and fix the `# CHECK` items.

## 2026-09-10, Prithvi probe (week 4 item, pulled forward)
- Did: installed terratorch 1.2.13 + peft 0.20.0 with torch pinned (only new packages, no changes to the loaded ones). `scripts/probe_prithvi.py` downloaded Prithvi-EO-2.0-300M and inspected the API.
- Result: backbone is `PrithviViT` (303.9 M, embed 1024, 24 blocks). 4-D input works; each block returns (B, N+1, 1024) tokens with a CLS token; 224 -> 197, 512 -> 1025, so full-chip evaluation is possible without sliding windows. `PrithviSegmenter` builds and runs in all three modes on CPU: trainable 1.44 M (frozen) / 2.23 M (LoRA r=8, qkv) / 305.3 M (full). All `# CHECK` items resolved without code changes.
- Broke / surprised me: HF hub warns about symlinks on Windows (cache stores duplicates); harmless. `terratorch` has no `__version__`, use `importlib.metadata`.
- Next: once the U-Net sweep frees the GPU, run `prithvi_frozen` at 100 % labels and watch memory at batch 8 x 224 with AMP; then `prithvi_lora`.

## 2026-09-10, U-Net sweep (week 3 item, done early)
- Did: both U-Nets x {1.0, 0.25, 0.10, 0.05} labels, seed 0, fixed 1,550-step budget. 8 runs, ~4 min each. `collect_results.py` -> `results/results.csv`, `plot_results.py` -> `docs/figures/fig1..3`.
- Result (test water IoU / Bolivia water IoU):

  | labels | U-Net scratch | U-Net ImageNet |
  |---|---|---|
  | 100 % (252 chips) | 0.823 / 0.750 | 0.812 / 0.789 |
  | 25 % (63) | 0.777 / 0.758 | 0.803 / 0.722 |
  | 10 % (25) | 0.803 / 0.770 | 0.822 / 0.787 |
  | 5 % (13) | 0.776 / 0.790 | 0.804 / 0.784 |

- Broke / surprised me: **the low-label regime is not hard for a U-Net on this dataset.** With the same number of updates, 13 chips give test IoU 0.78-0.80 vs 0.81-0.82 with 252 chips. Water is spectrally distinctive in B8A/B11/B12, so a few chips already cover the concept. Consequences: (1) claim 2 ("gap widens below 10 %") may not hold at 5 %; we should extend the sweep to 2 % (5 chips) and 1 % (2-3 chips) where the U-Net presumably breaks; (2) one-seed differences of 0.02-0.04 are within what looks like run-to-run noise (val IoU wobbles ±0.1 between epochs), so seeds 1-2 are needed before reading anything into scratch vs ImageNet; (3) Bolivia is not systematically worse for the 5 % runs than for the 100 % ones, which again points at noise + the cloud confound rather than a clean OOD effect.
- Next: `prithvi_frozen` f=1.0 (running), then `prithvi_lora`; queue U-Net seeds 1-2 and fractions 0.02 / 0.01 overnight.

## 2026-09-10, Prithvi frozen (week 4 item)
- Did: `prithvi_frozen` f=1.0 seed 0 on the 3070 Ti: 2.7 GB at batch 8 x 224, ~12 s/epoch, 8.7 GPU-min. Then launched the detached queue `scripts/queue_2026-09-10.cmd` (LoRA -> full -> Prithvi label sweep -> U-Net seeds 1-2).
- Result: **test water IoU 0.699** (mIoU 0.825, F1 0.823), **Bolivia 0.527** (precision 0.98, recall 0.53). Best epoch 25 of 50; val IoU noisy (0.60-0.72).
- Broke / surprised me: a frozen MAE encoder with a 1.4 M-param conv head is far below the U-Net (0.823 / 0.750). High precision + low recall on Bolivia = the head is conservative on unfamiliar-looking water. This is consistent with the MAE literature (features need fine-tuning; linear probes are weak), and it makes the LoRA vs full comparison the interesting one. Possible head-side improvements if LoRA also underperforms: use all 24 layers or a UNet-style decoder, larger crop (Prithvi was pretrained at 224 but the head sees a 14x14 grid), longer schedule.
- Next: LoRA and full results from the queue.

## 2026-09-10, Prithvi LoRA
- Result (`prithvi_lora` f=1.0 s0, r=8 on qkv, 2.23 M trainable, 4.3 GB, 13.3 GPU-min): **test water IoU 0.778, Bolivia 0.725**. +8 / +20 points over frozen; still 4.5 / 2.5 points below U-Net scratch at full labels.
- Reading: adapting the encoder matters much more than the head; the ranking at 100 % labels so far is U-Net (0.823) > LoRA (0.778) > frozen (0.699). Whether Prithvi overtakes at 5 % labels is the open question the queue answers next.

## 2026-09-10, Prithvi full fine-tune
- Result (`prithvi_full` f=1.0 s0, 305 M trainable, batch 4, 7.2 GB, 1,575 steps, 9.0 GPU-min): **test water IoU 0.779, Bolivia 0.637**.
- Reading: identical to LoRA in-distribution (0.778 vs 0.779) and 9 points worse on Bolivia. Claim 3 (LoRA recovers full-FT accuracy with <1 % of the parameters) holds at 100 % labels; full FT seems to overfit the training regions. Caveat: full FT ran at batch 4 with backbone lr x0.05, LoRA at batch 8 with x0.5; a small lr sweep for full FT would make the comparison airtight.
- Full-label ranking (seed 0): U-Net scratch 0.823 > U-Net ImageNet 0.812 > Prithvi LoRA 0.778 = Prithvi full 0.779 > Prithvi frozen 0.699. On Bolivia: ImageNet 0.789 > scratch 0.750 > LoRA 0.725 > full 0.637 > frozen 0.527.
