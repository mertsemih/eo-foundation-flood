"""Evaluate a finished run on any split; optionally dump predictions as PNGs.

    python -m eoflood.evaluate --run runs/unet_scratch_f1.00_s0 --split bolivia [--dump]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_datasets
from .metrics import SegMetrics
from .models import build_model
from .train import load_checkpoint
from .utils import get_device, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.yaml and best.pt")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "bolivia"])
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--dump", action="store_true", help="write prediction PNGs to <run>/preds_<split>/")
    args = ap.parse_args(argv)

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    device = get_device()
    ds = build_datasets(cfg)[args.split]
    model = build_model(cfg, in_channels=ds.in_channels).to(device)
    load_checkpoint(model, run / args.ckpt, device)
    model.eval()

    loader = DataLoader(ds, batch_size=cfg["train"].get("eval_batch_size", 4), shuffle=False)
    metrics = SegMetrics()
    dump_dir = run / f"preds_{args.split}"
    if args.dump:
        from PIL import Image  # optional; only needed for dumps

        dump_dir.mkdir(exist_ok=True)

    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            metrics.update(pred, y)
            if args.dump:
                for p in pred.numpy():
                    name = ds.items[idx][1].replace("_LabelHand.tif", ".png")
                    Image.fromarray((p * 255).astype(np.uint8)).save(dump_dir / name)
                    idx += 1

    result = metrics.compute()
    print(json.dumps({args.split: result}, indent=2))
    with open(run / f"eval_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
