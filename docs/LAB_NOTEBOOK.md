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
