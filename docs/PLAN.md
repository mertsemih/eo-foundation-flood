# Plan — 12 weeks at 1–2 days/week

Start: week of 8 September 2026. Target: preprint on arXiv by early December 2026, submission to SIU 2027 / RAST 2027 / an IGARSS 2027 student session after that.

## Research question and claims

**RQ.** How much does an EO foundation model (Prithvi-EO 2.0) improve flood segmentation over a conventional U-Net when labels are scarce, and which fine-tuning strategy is the most parameter-efficient, including under geographic shift (Bolivia)?

Claims we want the experiments to be able to confirm or refute:

1. In the full-label regime, foundation-model fine-tuning is at best marginally better than a good U-Net.
2. Under 5–10 % labels the gap widens substantially in favour of the foundation model.
3. LoRA recovers most of the full fine-tune accuracy with <5 % of its trainable parameters.
4. Pretraining helps generalization to the held-out Bolivia region more than it helps in-distribution accuracy.

Each claim maps to a figure: (1)/(2) accuracy vs label fraction curves; (3) accuracy vs trainable-params scatter; (4) test vs Bolivia bars.

## Experiment matrix

| model | pretraining | trainable | notes |
|---|---|---|---|
| `unet_scratch` | none | all | primary baseline |
| `unet_imagenet` | ImageNet | all | "generic pretraining" control |
| `prithvi_frozen` | Prithvi-EO 2.0 300M | head only | linear-probe-style |
| `prithvi_lora` | Prithvi-EO 2.0 300M | LoRA r=8 + head | main method |
| `prithvi_full` | Prithvi-EO 2.0 300M | all | upper bound |

× label fraction {1.0, 0.25, 0.10, 0.05} × seeds {0, 1, 2} = **60 runs**. At ~20–40 GPU-minutes each on a T4 (Prithvi runs longer), that is roughly 30–40 GPU-hours total: feasible on Colab Pro / Kaggle across the weeks. Run seeds last; get one seed of everything first.

Report on every run: val water IoU (selection), test water IoU / mIoU / F1, Bolivia water IoU / mIoU / F1, trainable params, GPU minutes. All of this is already written to `runs/<name>/test_metrics.json` by `eoflood.train`.

## Weekly schedule

### Week 1 (8–14 Sep) — orientation and data
- [ ] Read: Sen1Floods11 paper (CVPRW 2020); Prithvi-EO 2.0 paper (skim architecture, read fine-tuning section); LoRA paper (Hu et al. 2021), sections 1–4.
- [ ] Install Google Cloud CLI, run `scripts/download_sen1floods11.py`, confirm chip counts against `docs/DATASET.md`.
- [ ] `pip install -e ".[dev]"`, run `pytest`; all green on CPU.
- [ ] Notebook `notebooks/01_eda.ipynb`: plot 6 chips (RGB, false colour SWIR, S1 VV, label). Histogram of water fraction per chip. Note anything odd in `docs/LAB_NOTEBOOK.md`.
- [ ] `scripts/compute_stats.py` → paste mean/std into all five configs.

### Week 2 (15–21 Sep) — first baseline
- [ ] Run `unet_scratch` on a GPU (Colab/Kaggle). Make it converge; tune lr / epochs until val water IoU is stable across two seeds. Sanity target from the literature: ≈0.7–0.8 water IoU on the test split for S2-only U-Nets.
- [ ] Run `eoflood.evaluate --split bolivia --dump`, look at 10 predictions by eye. Write down failure modes (clouds, shadows, narrow rivers).
- [ ] Fix anything the real data broke in the loader. Commit.

### Week 3 (22–28 Sep) — second baseline + label sweep infra
- [ ] Run `unet_imagenet`. Compare to scratch at 100 % labels.
- [ ] Sweep `data.train_fraction` ∈ {0.25, 0.10, 0.05} for both U-Nets, seed 0. Write a small `scripts/collect_results.py` that reads every `test_metrics.json` into one CSV.
- [ ] First plot: water IoU vs label fraction (two lines). This is figure 1 of the paper, even if it changes later.

### Week 4 (29 Sep–5 Oct) — Prithvi comes alive
- [ ] `pip install -e ".[foundation]"`. Load `prithvi_eo_v2_300` via terratorch in a notebook, run one forward pass on a batch, inspect the token shapes. Fix the `# CHECK` items in `models/prithvi.py`.
- [ ] Get `prithvi_frozen` training end to end. It will be the fastest Prithvi variant; use it to debug.
- [ ] NASA Space Apps Challenge is usually the first weekend of October: register with the Ankara local event, use this repo as your project demo if a flood/water challenge is offered.

### Weeks 5–6 — LoRA and full fine-tune
- [ ] `prithvi_lora`: confirm trainable-parameter count is a few million, not 300M. Tune `backbone_lr_mult`, `lora_r` ∈ {4, 8, 16}.
- [ ] `prithvi_full` with the largest batch that fits; gradient checkpointing if needed.
- [ ] All five models at 100 % labels, seed 0. Table 1 draft.

### Weeks 7–8 — the sweep
- [ ] Label-fraction sweep for the three Prithvi variants, seed 0.
- [ ] Bolivia numbers for every run (already automatic). Figure 2: test vs Bolivia.
- [ ] Start `paper/` (LaTeX, IEEE conference template). Write the method and dataset sections now while the details are fresh.

### Week 9 — seeds and ablations
- [ ] Seeds 1 and 2 for the main table. Report mean ± std.
- [ ] One ablation you can afford: LoRA targets (`qkv` vs `qkv+proj+fc1+fc2`) or Prithvi's own normalization stats vs dataset stats.

### Weeks 10–11 — write
- [ ] Results, discussion, limitations (single dataset, S2-only, 300M model only, no Clay yet).
- [ ] Figures regenerated from `collect_results.py` output, not by hand.
- [ ] Ask your BraTS advisor to read it; offer co-authorship if they contribute substantively.

### Week 12 — release
- [ ] arXiv preprint (cs.CV, cross-list eess.IV). README gets the link and a results table.
- [ ] Tag `v1.0`, upload best checkpoints to Hugging Face or a GitHub release.
- [ ] Update CV: "Paper (preprint): ...; N runs, X GPU-hours, key number."

## Stretch (if ahead of schedule)
- Clay v1 as a second foundation model (same six bands work).
- S1+S2 fusion for the U-Net and for Prithvi via an extra input adapter.
- Weak supervision: pretrain on `S1OtsuLabelHand`, fine-tune on hand labels.

## Working rules
- One config = one run = one folder under `runs/`. Never edit a run's `config.yaml` after the fact.
- Every experiment day ends with a dated entry in `docs/LAB_NOTEBOOK.md`, even if it only says what broke.
- Commit small and often; push at the end of every working day.
