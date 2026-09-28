"""Qualitative predictions and per-chip error analysis for the seed-0, full-label checkpoints.

    python scripts/qualitative.py [--out docs/figures] [--results results]

1. Predicts every test and Bolivia chip with each model and caches the masks in
   ``runs/<run>/preds_<split>.npz`` (skipped when the cache exists).
2. Writes ``results/per_chip_iou.csv`` (water IoU per chip and model) and
   ``results/error_analysis.csv``: per model and split, water precision / recall, the share of
   errors within ``--boundary`` px of a label water edge, and recall on small vs. large water
   bodies (connected components below / above ``--small`` px).
3. Draws ``fig4_predictions.png``. Chips are chosen by a fixed rule, not by eye: among chips
   with at least 5 % water in the valid pixels, sort by the per-chip IoU difference
   U-Net (ImageNet) minus Prithvi LoRA and take the chips at the 50th and 75th percentile,
   for the test and the Bolivia split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eoflood.data import build_datasets  # noqa: E402
from eoflood.metrics import IGNORE_INDEX  # noqa: E402
from eoflood.models import build_model  # noqa: E402
from eoflood.train import load_checkpoint  # noqa: E402
from eoflood.utils import get_device, load_config  # noqa: E402

MODELS = ["unet_scratch", "unet_imagenet", "prithvi_frozen", "prithvi_lora", "prithvi_full", "prithvi_lora_unetdec"]
FIG_MODELS = ["unet_imagenet", "prithvi_frozen", "prithvi_lora", "prithvi_full"]
LABELS = {
    "unet_scratch": "U-Net (scratch)",
    "unet_imagenet": "U-Net (ImageNet)",
    "prithvi_frozen": "Prithvi frozen",
    "prithvi_lora": "Prithvi LoRA",
    "prithvi_full": "Prithvi full FT",
    "prithvi_lora_unetdec": "Prithvi LoRA + UNet dec.",
}
SPLITS = ["test", "bolivia"]
# error-map colours: true negative, true positive, false positive, false negative, no-data
COLORS = np.array([[240, 240, 240], [33, 102, 172], [230, 97, 0], [197, 27, 125], [90, 90, 90]], dtype=np.uint8)


def predict(run: Path, split: str) -> tuple[np.ndarray, list[str]]:
    cache = run / f"preds_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pred"], list(z["names"])
    cfg = load_config(run / "config.yaml")
    cfg["data"]["cache"] = False
    device = get_device()
    ds = build_datasets(cfg)[split]
    model = build_model(cfg, in_channels=ds.in_channels).to(device)
    load_checkpoint(model, run / "best.pt", device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=cfg["train"].get("eval_batch_size", 4), shuffle=False):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy().astype(np.uint8))
    pred = np.concatenate(preds)
    names = [label.replace("_LabelHand.tif", "") for _, label in ds.items]
    np.savez_compressed(cache, pred=pred, names=np.array(names))
    del model
    torch.cuda.empty_cache()
    print(f"  predicted {run.name} / {split}: {len(names)} chips", flush=True)
    return pred, names


def water_iou(p: np.ndarray, y: np.ndarray) -> float:
    v = y != IGNORE_INDEX
    tp = np.sum((p == 1) & (y == 1) & v)
    fp = np.sum((p == 1) & (y == 0) & v)
    fn = np.sum((p == 0) & (y == 1) & v)
    return float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan")


def error_stats(preds: np.ndarray, labels: np.ndarray, boundary: int, small: int) -> dict:
    tp = fp = fn = err = err_near = 0
    small_hit = small_tot = large_hit = large_tot = 0
    for p, y in zip(preds, labels):
        v = y != IGNORE_INDEX
        w = y == 1
        tp += np.sum(p.astype(bool) & w & v)
        fp += np.sum((p == 1) & (y == 0) & v)
        fn += np.sum((p == 0) & w & v)
        e = (p != y) & v
        # band of +-boundary px around every label water edge
        near = ndimage.binary_dilation(w, iterations=boundary) & ~ndimage.binary_erosion(w, iterations=boundary)
        err += e.sum()
        err_near += (e & near).sum()
        comp, n = ndimage.label(w)
        if n:
            sizes = np.bincount(comp.ravel())[1:]
            hits = np.bincount(comp.ravel(), weights=(p == 1).ravel())[1:]
            s = sizes < small
            small_hit += hits[s].sum()
            small_tot += sizes[s].sum()
            large_hit += hits[~s].sum()
            large_tot += sizes[~s].sum()
    return {
        "water_iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "error_px": int(err),
        "error_share_near_edge": err_near / err,
        "recall_small_bodies": small_hit / small_tot if small_tot else float("nan"),
        "recall_large_bodies": large_hit / large_tot if large_tot else float("nan"),
    }


def rgb(x: np.ndarray) -> np.ndarray:
    """True-colour composite (B4, B3, B2) with a 2-98 % stretch over the chip's valid pixels."""
    img = x[[2, 1, 0]].transpose(1, 2, 0)
    lo, hi = np.percentile(img[img.sum(-1) > 0], [2, 98]) if (img.sum(-1) > 0).any() else (0, 1)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def error_map(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    code = np.zeros(y.shape, dtype=np.uint8)
    code[(p == 1) & (y == 1)] = 1
    code[(p == 1) & (y == 0)] = 2
    code[(p == 0) & (y == 1)] = 3
    code[y == IGNORE_INDEX] = 4
    return COLORS[code]


def label_map(y: np.ndarray) -> np.ndarray:
    code = np.where(y == 1, 1, 0).astype(np.uint8)
    code[y == IGNORE_INDEX] = 4
    return COLORS[code]


def pick(chips: pd.DataFrame) -> list[str]:
    c = chips[chips.water_frac >= 0.05].copy()
    c["diff"] = c["unet_imagenet"] - c["prithvi_lora"]
    c = c.sort_values("diff").reset_index(drop=True)
    return [c.loc[int(round(q * (len(c) - 1))), "chip"] for q in (0.5, 0.75)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--boundary", type=int, default=3)
    ap.add_argument("--small", type=int, default=1000, help="water-body size threshold in pixels (10 m)")
    args = ap.parse_args()
    runs, res, out = Path(args.runs), Path(args.results), Path(args.out)
    res.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(runs / "unet_imagenet_f1.00_s0" / "config.yaml")
    cfg["data"]["cache"] = False
    ds_all = build_datasets(cfg)
    data = {}
    for split in SPLITS:
        ds = ds_all[split]
        raw = [ds.load_raw(i) for i in range(len(ds))]
        data[split] = (np.stack([x for x, _ in raw]), np.stack([y for _, y in raw]))

    preds, rows, stats = {}, [], []
    for split in SPLITS:
        xs, ys = data[split]
        for m in MODELS:
            p, names = predict(runs / f"{m}_f1.00_s0", split)
            preds[(m, split)] = p
            stats.append({"model": m, "split": split, **error_stats(p, ys, args.boundary, args.small)})
        for i, name in enumerate(names):
            y = ys[i]
            v = y != IGNORE_INDEX
            row = {"split": split, "chip": name, "water_frac": float((y[v] == 1).mean()) if v.any() else 0.0}
            row.update({m: water_iou(preds[(m, split)][i], y) for m in MODELS})
            rows.append(row)
    chips = pd.DataFrame(rows)
    chips.to_csv(res / "per_chip_iou.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(res / "error_analysis.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(st.to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sel = [(s, c) for s in SPLITS for c in pick(chips[chips.split == s])]
    ncol = 2 + len(FIG_MODELS)
    fig, axes = plt.subplots(len(sel), ncol, figsize=(1.55 * ncol, 1.85 * len(sel)))
    for r, (split, chip) in enumerate(sel):
        names = list(chips[chips.split == split].chip)
        i = names.index(chip)
        xs, ys = data[split]
        panels = [(rgb(xs[i]), f"{chip.replace('_', ' ')}"), (label_map(ys[i]), "label")]
        for m in FIG_MODELS:
            p = preds[(m, split)][i]
            panels.append((error_map(p, ys[i]), f"{LABELS[m]}\nIoU {water_iou(p, ys[i]):.2f}"))
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=7)
        axes[r, 0].set_ylabel("test" if split == "test" else "Bolivia", fontsize=8)
    legend = [Patch(color=COLORS[k] / 255, label=t) for k, t in enumerate(["no water", "water (correct)", "false water", "missed water", "no data / cloud"])]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.05, wspace=0.04, hspace=0.3)
    fig.savefig(out / "fig4_predictions.png", dpi=200)
    print(f"figure written to {out / 'fig4_predictions.png'} ({sel})")


if __name__ == "__main__":
    main()
