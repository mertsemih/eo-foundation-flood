"""Per-band mean / std over the training split, written to configs/data/stats.yaml.

Paste the numbers into ``data.mean`` / ``data.std`` of each config (or pass them as overrides).

    python scripts/compute_stats.py --root data/sen1floods11 [--use-s1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from eoflood.data import PRITHVI_BANDS, S2_BANDS, Sen1Floods11


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sen1floods11")
    ap.add_argument("--bands", nargs="*", default=None, help="subset of S2 bands; default = Prithvi six; 'all' = 13")
    ap.add_argument("--use-s1", action="store_true")
    ap.add_argument("--out", default="configs/data/stats.yaml")
    args = ap.parse_args()

    bands = S2_BANDS if args.bands == ["all"] else (args.bands or PRITHVI_BANDS)
    ds = Sen1Floods11(args.root, "train", bands=bands, use_s1=args.use_s1)

    # Welford-style accumulation over all valid pixels (label != -1 is not required for stats).
    n = 0
    s = np.zeros(ds.in_channels, dtype=np.float64)
    s2 = np.zeros(ds.in_channels, dtype=np.float64)
    for i in range(len(ds)):
        x, _ = ds.load_raw(i)
        flat = x.reshape(x.shape[0], -1).astype(np.float64)
        n += flat.shape[1]
        s += flat.sum(1)
        s2 += (flat**2).sum(1)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))

    names = list(bands) + (["VV", "VH"] if args.use_s1 else [])
    out = {
        "bands": names,
        "n_chips": len(ds),
        "mean": [round(float(v), 4) for v in mean],
        "std": [round(float(v), 4) for v in std],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
