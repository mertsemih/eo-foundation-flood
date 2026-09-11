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

## 2026-09-10, Prithvi label sweep complete (17 local runs, seed 0)
- Result, test water IoU / Bolivia water IoU:

  | labels | U-Net scratch | U-Net ImageNet | Prithvi frozen | Prithvi LoRA | Prithvi full |
  |---|---|---|---|---|---|
  | 100 % | 0.823 / 0.750 | 0.812 / 0.789 | 0.699 / 0.527 | 0.778 / 0.725 | 0.779 / 0.637 |
  | 25 % | 0.777 / 0.758 | 0.803 / 0.722 | 0.699 / 0.755 | 0.751 / 0.744 | - |
  | 10 % | 0.803 / 0.770 | 0.822 / 0.787 | 0.686 / 0.674 | 0.734 / 0.751 | - |
  | 5 % | 0.776 / 0.790 | 0.804 / 0.784 | 0.695 / 0.625 | 0.719 / 0.741 | - |

- Reading (single seed, so ±0.02-0.03 is noise):
  1. **Claim 1 holds strongly, in the negative direction:** at full labels the foundation model does not merely fail to beat the U-Net, it trails it by 4-5 points in every regime.
  2. **Claim 2 does not hold down to 5 %:** the U-Net degrades by ~2-5 points from 100 % to 5 %, LoRA by ~6 points. The gap does not close, it widens slightly against Prithvi. On this dataset 13 chips still contain enough water pixels (~1 M) for a CNN.
  3. **Claim 3 holds:** LoRA (2.2 M) equals full FT (305 M) in-distribution and beats it by 9 points on Bolivia.
  4. **Claim 4 is unresolved:** Bolivia numbers fluctuate ±0.05 across label fractions for every model, larger than any between-model difference except frozen/full. Seeds 1-2 (running on Colab) are needed; the cloud confound (27 % no-data) probably dominates.
- Cost: U-Net 3.7-4.5 GPU-min, frozen 9-10, LoRA 13-14, full 9 (batch 4) on the 3070 Ti.
- Honest framing for the paper: "a small, well-tuned CNN remains the better choice for Sentinel-2 flood mapping on Sen1Floods11, even with 13 labeled chips; the foundation model's advantage, if any, must lie below 5 % labels or in cross-sensor / cross-region transfer that this benchmark does not test." That is a publishable negative result if the seeds and a 1-2 % sweep confirm it, and it is more useful to practitioners than a marginal win.
- Caveats to close before writing: (a) Prithvi head is a plain FCN on a 14x14 grid; a UNet-style decoder over multiple scales might recover part of the gap (ablation); (b) LoRA rank / targets untested; (c) Prithvi crop 224 vs U-Net crop 224 is matched, but Prithvi's positional grid was pretrained at 224 so this is its comfort zone, not a handicap; (d) dataset-vs-Prithvi normalization stats.
- Next: seeds from Colab; add fractions 0.02 / 0.01 (needs the batch-size guard); decoder ablation for Prithvi.

## 2026-09-10, 2 % / 1 % labels (seed 0, running)
- Did: added the batch-size guard, launched fractions 0.02 (5 chips) and 0.01 (3 chips) for U-Nets, Prithvi frozen and LoRA (seed 0). Same 1,550-step budget, eval every 31 epochs.
- Result so far (U-Net scratch): **2 % -> test 0.788 / Bolivia 0.788; 1 % -> test 0.812 / Bolivia 0.775.** Three chips reach the full-label number minus one point.
- Why: the seed-0 1 % subset is Mekong_16233 (24 % water, 62 k px), Somalia_886726 (32 %, 83 k px) and Spain_7786924. Its 149 k water pixels are 80 % of what the 13-chip 5 % subset holds (185 k) because that subset drew several zero-water chips. **At <= 5 % labels the subset draw, not the label count, is the dominant variable.**
- Consequence for the protocol: (1) report tiny-fraction results over >= 3 subset seeds (mean +/- std), (2) consider a *water-pixel budget* axis (e.g. 10 k / 50 k / 150 k labeled water pixels) as the honest x-axis instead of chip count, or stratify the draw by water fraction; (3) the paper's claim 2 should be reframed around water-pixel budget.
- Next: remaining 6 tiny runs; then seeds 1-2 for 0.02 / 0.01 (U-Net scratch + LoRA first).

## 2026-09-10, end of day: 25 local runs (seed 0), GPU released
- Tiny-fraction results (test / Bolivia water IoU):

  | labels | U-Net scratch | U-Net ImageNet | Prithvi frozen | Prithvi LoRA |
  |---|---|---|---|---|
  | 2 % (5 chips) | 0.788 / 0.788 | 0.792 / 0.789 | 0.684 / 0.742 | 0.706 / 0.706 |
  | 1 % (3 chips) | 0.812 / 0.775 | 0.810 / 0.767 | 0.676 / 0.670 | 0.722 / 0.715 |

- Reading: with the same 3 chips, U-Net 0.81 vs LoRA 0.72 vs frozen 0.68. The ordering U-Net > LoRA > frozen is identical at every fraction from 100 % down to 1 %; LoRA's curve keeps sliding (0.778 -> 0.706) while the U-Net stays flat within noise. The seed-0 1 % subset is water-rich (see above), so the absolute numbers at 1-2 % need subset seeds, but the *ordering* between models on the same subset is a fair comparison.
- Local queue stopped after this run at the user's request (laptop needed for other work). Colab is running U-Net seeds 1-2 in parallel; results to be merged from Drive.
- Remaining for the paper: Prithvi seeds 1-2 (Colab), tiny-fraction subset seeds, Prithvi decoder ablation, LoRA rank ablation, then write.

## 2026-09-10, Colab U-Net seeds merged (41 runs total)
- Did: 16 U-Net runs (seeds 1-2, fractions 1.0/0.25/0.10/0.05) from Colab T4 (~5 GPU-min each) merged into `runs/`. Table below is test / Bolivia water IoU, mean ± std over seeds where n>1.

| labels | unet_scratch | unet_imagenet | prithvi_frozen | prithvi_lora | prithvi_full |
|---|---|---|---|---|---|
| 100 % | 0.823 ± 0.008 / 0.766 ± 0.021 (n=3) | 0.829 ± 0.015 / 0.781 ± 0.007 (n=3) | 0.699 / 0.527 (n=1) | 0.778 / 0.725 (n=1) | 0.778 / 0.637 (n=1) |
| 25 % | 0.802 ± 0.031 / 0.678 ± 0.112 (n=3) | 0.820 ± 0.016 / 0.744 ± 0.035 (n=3) | 0.699 / 0.755 (n=1) | 0.750 / 0.744 (n=1) | – |
| 10 % | 0.803 ± 0.003 / 0.769 ± 0.006 (n=3) | 0.827 ± 0.005 / 0.773 ± 0.043 (n=3) | 0.686 / 0.674 (n=1) | 0.734 / 0.751 (n=1) | – |
| 5 % | 0.807 ± 0.031 / 0.770 ± 0.032 (n=3) | 0.814 ± 0.010 / 0.778 ± 0.008 (n=3) | 0.695 / 0.625 (n=1) | 0.719 / 0.741 (n=1) | – |
| 2 % | 0.788 / 0.788 (n=1) | 0.792 / 0.789 (n=1) | 0.683 / 0.742 (n=1) | 0.706 / 0.706 (n=1) | – |
| 1 % | 0.812 / 0.775 (n=1) | 0.810 / 0.767 (n=1) | 0.676 / 0.670 (n=1) | 0.722 / 0.715 (n=1) | – |

- Reading: with 3 seeds, in-distribution std is 0.003-0.03 for the U-Nets; **every U-Net point is >= 0.79 and every Prithvi point is <= 0.78**, so the ordering is outside seed noise. ImageNet init is consistently +1-2 points over scratch (0.827 vs 0.823 at 100 %, 0.827 vs 0.803 at 10 %). Bolivia std is 0.01-0.11 (scratch at 25 % has one outlier seed at 0.55), which confirms the OOD split is too small/cloudy to rank models finely; report it with error bars and do not over-interpret.
- Next (tomorrow): Prithvi frozen + LoRA seeds 1-2 on Colab; tiny-fraction subset seeds; decoder and LoRA-rank ablations.

## 2026-09-11 (week 2, day 3)
- Did: disk was at 30 GB free; `runs/` held 32 GB because every Prithvi run saved the full 1.2 GB state dict twice. Deleted all `last.pt` and the non-essential Prithvi `best.pt` (27 GB freed; kept the three f=1.0 seed-0 Prithvi checkpoints). Training now saves **only trainable tensors + buffers** (`trainable_state_dict`), and `load_checkpoint` rebuilds frozen weights from the config and verifies nothing trainable is missing; old full checkpoints still load (LoRA f=1.0 Bolivia 0.7248 reproduced).
- Did: `MultiScaleUNetDecoder` (four encoder depths 6/12/18/24 -> x4/x2/x1/x0.5 pyramid -> UNet merge). Prithvi LoRA trainable params: FCN head 2.23 M, UNet decoder 4.65 M (head 3.86 M). Config `prithvi_lora_unetdec`. LoRA ablation configs: `prithvi_lora_r4`, `prithvi_lora_r16`, `prithvi_lora_wide` (qkv+proj+fc1+fc2).
- Queues: Colab runs Prithvi frozen + LoRA seeds 1-2 (16 runs). Local: q6 = 2 % / 1 % subset seeds 1-2 for both U-Nets and LoRA (12 runs), then q7 = decoder ablation at 100 % and 5 %, then q8 = LoRA rank / target ablation at 100 %.
