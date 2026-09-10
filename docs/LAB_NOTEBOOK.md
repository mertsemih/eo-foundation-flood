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
