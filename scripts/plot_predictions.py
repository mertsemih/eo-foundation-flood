"""Side-by-side RGB / SWIR composite / label / prediction panels for a finished run.

    python -m eoflood.evaluate --run runs/unet_scratch_f1.00_s0 --split bolivia --dump
    python scripts/plot_predictions.py --run runs/unet_scratch_f1.00_s0 --split bolivia --out docs/figures/bolivia_unet_scratch.png

Per-chip water IoU is printed in the panel title; red in the label panel is no-data / cloud.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eoflood.data import S2_BANDS, Sen1Floods11  # noqa: E402
from eoflood.metrics import IGNORE_INDEX  # noqa: E402


def stretch(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    lo_v, hi_v = np.percentile(a, [lo, hi])
    return np.clip((a - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="bolivia")
    ap.add_argument("--root", default="data/sen1floods11")
    ap.add_argument("--indices", nargs="*", type=int, default=None, help="chip indices; default = 5 evenly spaced")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    run = Path(args.run)
    pred_dir = run / f"preds_{args.split}"
    if not pred_dir.exists():
        raise SystemExit(f"{pred_dir} missing: run eoflood.evaluate --dump first")
    ds = Sen1Floods11(args.root, args.split, bands=S2_BANDS)
    idx = args.indices or list(np.linspace(0, len(ds) - 1, 5, dtype=int))

    fig, axes = plt.subplots(len(idx), 4, figsize=(12, 3 * len(idx)))
    axes = np.atleast_2d(axes)
    for row, i in zip(axes, idx, strict=True):
        x, y = ds.load_raw(i)
        b = {n: x[k] for k, n in enumerate(S2_BANDS)}
        rgb = np.dstack([stretch(b["B4"]), stretch(b["B3"]), stretch(b["B2"])])
        swir = np.dstack([stretch(b["B12"]), stretch(b["B8A"]), stretch(b["B4"])])
        lab = np.zeros((*y.shape, 3))
        lab[y == 1] = 1
        lab[y == IGNORE_INDEX] = (1, 0, 0)
        pred = np.array(Image.open(pred_dir / ds.items[i][1].replace("_LabelHand.tif", ".png"))) > 0
        valid = y != IGNORE_INDEX
        inter = ((pred == 1) & (y == 1) & valid).sum()
        union = (((pred == 1) | (y == 1)) & valid).sum()
        name = ds.items[i][0].split("_S1")[0]
        titles = ["RGB", "SWIR composite", "label (red = no-data)", f"pred  IoU={inter / max(union, 1):.2f}"]
        for ax, img, t in zip(row, [rgb, swir, lab, pred], titles, strict=True):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(f"{name}  {t}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    out = Path(args.out or run / f"panels_{args.split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
