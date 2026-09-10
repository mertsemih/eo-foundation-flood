# eo-foundation-flood

Parameter-efficient fine-tuning of Earth-observation foundation models (Prithvi-EO 2.0, later Clay) for flood segmentation on Sentinel-2, benchmarked against from-scratch and ImageNet-pretrained U-Nets on **Sen1Floods11**.

## Research question

> How much does an EO foundation model buy over a conventional U-Net in the **low-label regime**, and which fine-tuning strategy (frozen encoder / LoRA / full fine-tune) gives the best accuracy per trainable parameter, including on the **out-of-distribution Bolivia split**?

The experiment matrix is fixed up front (see [docs/PLAN.md](docs/PLAN.md)) so results are comparable and the write-up is straightforward:

| axis | values |
|---|---|
| model | `unet_scratch`, `unet_imagenet`, `prithvi_frozen`, `prithvi_lora`, `prithvi_full` |
| training labels | 100 %, 25 %, 10 %, 5 % of hand-labeled chips |
| seeds | 3 |
| evaluation | Sen1Floods11 test split (in-distribution) + Bolivia split (held-out region) |
| metrics | water IoU, mIoU, water F1, trainable params, GPU-hours |

## Status

- [x] Repository skeleton, config-driven training loop, metrics, tests
- [x] Dataset downloaded and band statistics computed (`scripts/compute_stats.py`)
- [x] U-Net baselines + label sweep (seed 0): scratch 0.823 / ImageNet 0.812 test water IoU at 100 % labels; still 0.78–0.80 at 5 % (`docs/figures/fig1_iou_vs_labels.png`)
- [x] Prithvi frozen / LoRA / full at 100 % labels (seed 0): 0.699 / 0.778 / 0.779 test water IoU; LoRA beats full FT on Bolivia (0.725 vs 0.637)
- [~] Label sweep 100 % → 1 % for all models (seed 0) and 3 seeds for the U-Nets: U-Nets ≥ 0.79 test water IoU at every fraction, Prithvi ≤ 0.78; Prithvi seeds and ablations pending
- [ ] Write-up / preprint (weeks 10–12)

The Prithvi wrapper (`src/eoflood/models/prithvi.py`) was verified against terratorch 1.2.13 on 10 Sep (weights load, 224 and 512 inputs, frozen / LoRA / full modes: 1.4 M / 2.2 M / 305 M trainable parameters). Training runs on a local RTX 3070 Ti (8 GB); the U-Net baseline takes ~4 GPU-minutes with the in-RAM chip cache.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + tests
pip install -e ".[foundation]"     # adds terratorch + peft for Prithvi experiments
```

Python 3.10+ and a CUDA GPU are assumed for training. CPU is enough for the tests and for dataset inspection.

## Data

Sen1Floods11 (Bonafilia et al., 2020) — 446 hand-labeled 512×512 chips over 11 flood events, with Sentinel-1 (VV, VH), Sentinel-2 (13 bands) and a binary water label (`-1` = no data, `0` = no water, `1` = water). Official train/valid/test/Bolivia splits are shipped with the dataset.

```bash
python scripts/download_sen1floods11.py --root data/sen1floods11     # public bucket over HTTPS, ~1.8 GB, no account needed
python scripts/compute_stats.py --root data/sen1floods11             # per-band mean/std -> configs/data/stats.yaml
```

Details and band mapping in [docs/DATASET.md](docs/DATASET.md).

## Quickstart

```bash
# from-scratch U-Net on all labels
python -m eoflood.train --config configs/unet_scratch.yaml

# Prithvi-EO 2.0 + LoRA on 10 % of the labels, seed 1
python -m eoflood.train --config configs/prithvi_lora.yaml data.train_fraction=0.10 seed=1

# evaluate a checkpoint on the Bolivia split
python -m eoflood.evaluate --run runs/prithvi_lora_f0.10_s1 --split bolivia
```

Every run writes `config.yaml`, `metrics.csv`, `best.pt` and `test_metrics.json` under `runs/<name>/`.

## Layout

```
configs/            experiment configs (one file = one model recipe; override on the CLI)
src/eoflood/
  data/             Sen1Floods11 dataset + augmentations
  models/           U-Net baseline, Prithvi wrapper (frozen / LoRA / full), registry
  metrics.py        confusion-matrix based IoU / F1 with ignore index
  train.py          training loop (AMP, cosine LR, best-checkpoint on val water IoU)
  evaluate.py       evaluation on any split, optional prediction dumps
scripts/            download, statistics
tests/              unit tests on synthetic rasters (no data download needed)
docs/               PLAN.md (12-week plan), DATASET.md, LAB_NOTEBOOK.md
```

## Citation

If this work is useful, cite the Sen1Floods11 and Prithvi-EO 2.0 papers; a preprint of this study will be linked here when available.

- Bonafilia et al., *Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, CVPRW 2020.
- Szwarcman et al., *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications*, 2024.

## License

MIT — see [LICENSE](LICENSE).
