"""Generate notebooks/01_eda.ipynb programmatically (keeps the notebook reproducible and diff-friendly).

    python scripts/make_eda_notebook.py            # write the notebook
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Sen1Floods11 — exploratory look

Goals of this notebook (week 1 of `docs/PLAN.md`):

1. Confirm chip counts per split and the label value set.
2. Look at a handful of chips: RGB, false-colour SWIR composite, Sentinel-1 VV, hand label.
3. Water fraction per chip, per split, per region (class imbalance, Bolivia vs rest).
4. Per-band statistics sanity check against `configs/data/stats.yaml`.
""",
    ),
    (
        "code",
        """import sys, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml

ROOT = Path("..") / "data" / "sen1floods11"
sys.path.insert(0, str(Path("..") / "src"))
from eoflood.data import PRITHVI_BANDS, S2_BANDS, Sen1Floods11, read_split
from eoflood.metrics import IGNORE_INDEX

splits = {s: read_split(ROOT, s) for s in ("train", "valid", "test", "bolivia")}
{s: len(v) for s, v in splits.items()}""",
    ),
    (
        "markdown",
        "## Regions per split\nEach chip name starts with the flood event / country. Bolivia should appear **only** in the `bolivia` split.",
    ),
    (
        "code",
        """for s, items in splits.items():
    c = collections.Counter(n.split("_")[0] for n, _ in items)
    print(f"{s:8s} {len(items):3d}  ", dict(sorted(c.items())))""",
    ),
    (
        "markdown",
        "## A few chips\nRGB = B4/B3/B2, SWIR composite = B12/B8A/B4 (water is dark, vegetation green), S1 VV in dB, label (black = no water, white = water, red = no data).",
    ),
    (
        "code",
        """ds_all = Sen1Floods11(ROOT, "train", bands=S2_BANDS, use_s1=True)   # raw, unnormalized

def stretch(a, lo=2, hi=98):
    lo_v, hi_v = np.percentile(a, [lo, hi])
    return np.clip((a - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

def show_chip(i, axes):
    x, y = ds_all.load_raw(i)
    b = {name: x[k] for k, name in enumerate(S2_BANDS)}
    vv = x[len(S2_BANDS)]
    rgb = np.dstack([stretch(b["B4"]), stretch(b["B3"]), stretch(b["B2"])])
    swir = np.dstack([stretch(b["B12"]), stretch(b["B8A"]), stretch(b["B4"])])
    lab = np.zeros((*y.shape, 3)); lab[y == 1] = 1; lab[y == IGNORE_INDEX] = (1, 0, 0)
    for ax, img, t in zip(axes, [rgb, swir, stretch(vv), lab], ["RGB", "SWIR composite", "S1 VV", "label"]):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray"); ax.set_title(f"{ds_all.items[i][0].split('_S1')[0]}  {t}", fontsize=8); ax.axis("off")

rng = np.random.RandomState(0)
idx = rng.choice(len(ds_all), 6, replace=False)
fig, axes = plt.subplots(len(idx), 4, figsize=(14, 3.4 * len(idx)))
for row, i in zip(axes, idx):
    show_chip(i, row)
plt.tight_layout()""",
    ),
    (
        "markdown",
        "## Water fraction per chip\nFraction of *valid* pixels labelled water. Also how much of each chip is no-data (`-1`).",
    ),
    (
        "code",
        """rows = []
for s in splits:
    ds = Sen1Floods11(ROOT, s, bands=["B2"])
    for i in range(len(ds)):
        _, y = ds.load_raw(i)
        valid = y != IGNORE_INDEX
        rows.append((s, ds.items[i][0].split("_")[0], (y[valid] == 1).mean() if valid.any() else np.nan, 1 - valid.mean()))
import pandas as pd
df = pd.DataFrame(rows, columns=["split", "region", "water_frac", "nodata_frac"])
print(df.groupby("split")[["water_frac", "nodata_frac"]].describe().round(3).T)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
df.boxplot(column="water_frac", by="split", ax=ax[0]); ax[0].set_title("water fraction per chip")
df.boxplot(column="water_frac", by="region", ax=ax[1], rot=60); ax[1].set_title("by region")
plt.suptitle(""); plt.tight_layout()""",
    ),
    (
        "markdown",
        "## Band statistics\nCompare with `configs/data/stats.yaml` (written by `scripts/compute_stats.py`). Reflectance ×10 000: means of a few hundred to a few thousand are expected.",
    ),
    (
        "code",
        """stats_path = Path("..") / "configs" / "data" / "stats.yaml"
if stats_path.exists():
    st = yaml.safe_load(stats_path.read_text())
    print(pd.DataFrame({"band": st["bands"], "mean": st["mean"], "std": st["std"]}).to_string(index=False))
else:
    print("run scripts/compute_stats.py first")""",
    ),
    (
        "markdown",
        """## Notes

- Fill in after running: anything surprising about no-data, cloud-heavy chips, regions with almost no water, Bolivia vs the rest.
- Copy the key numbers (chip counts, median water fraction, nodata share) into `docs/LAB_NOTEBOOK.md`.
""",
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    out = Path("notebooks") / "01_eda.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
