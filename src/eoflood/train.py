"""Config-driven training loop.

    python -m eoflood.train --config configs/unet_scratch.yaml [key.sub=value ...]

Writes ``runs/<name>/{config.yaml, metrics.csv, best.pt, last.pt, test_metrics.json}``.
Best checkpoint is selected on validation water IoU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Compose, RandomCrop, RandomFlipRotate, build_datasets
from .metrics import IGNORE_INDEX, SegMetrics
from .models import build_model
from .utils import CSVLogger, count_params, get_device, load_config, save_config, seed_everything


def run_name(cfg: dict) -> str:
    if cfg.get("name"):
        return cfg["name"]
    frac = cfg["data"].get("train_fraction", 1.0)
    return f"{cfg['model']['tag']}_f{frac:.2f}_s{cfg.get('seed', 0)}"


def make_loaders(cfg: dict):
    t = cfg["train"]
    aug = Compose([RandomCrop(t.get("crop_size", 224)), RandomFlipRotate()])
    ds = build_datasets(cfg, train_transform=aug)
    # With data.cache=true the chips live in RAM and num_workers=0 is the right choice on
    # Windows (spawned workers would each receive a pickled copy of the cache). Persistent
    # workers avoid the per-epoch respawn cost when workers are used.
    nw = t.get("num_workers", 0)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    # Tiny label fractions (1-2 % = 3-5 chips) can be smaller than the batch; shrink the batch
    # instead of silently producing zero steps per epoch with drop_last.
    bs = min(int(t["batch_size"]), len(ds["train"]))
    if bs != t["batch_size"]:
        print(f"batch_size {t['batch_size']} > {len(ds['train'])} training chips; using batch_size={bs}")
    loaders = {"train": DataLoader(ds["train"], batch_size=bs, shuffle=True, drop_last=True, **kw)}
    for split in ("valid", "test", "bolivia"):
        loaders[split] = DataLoader(ds[split], batch_size=t.get("eval_batch_size", 4), shuffle=False, **kw)
    return ds, loaders


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    metrics = SegMetrics()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        metrics.update(logits.argmax(1), y)
    return metrics.compute()


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    if hasattr(model, "param_groups"):
        groups = model.param_groups(t["lr"], t.get("backbone_lr_mult", 0.1))
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": t["lr"]}]
    return torch.optim.AdamW(groups, weight_decay=t.get("weight_decay", 0.01))


def resolve_epochs(t: dict, steps_per_epoch: int) -> int:
    """Number of epochs to run.

    If ``train.total_steps`` is set, every run gets the same optimizer-step budget regardless
    of how many training chips it has (label-fraction runs would otherwise get proportionally
    fewer updates). Otherwise ``train.epochs`` is used as-is.
    """
    total = t.get("total_steps")
    if total:
        return max(1, math.ceil(total / max(1, steps_per_epoch)))
    return int(t["epochs"])


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 0))
    device = get_device()
    t = cfg["train"]

    name = run_name(cfg)
    out = Path(cfg.get("runs_dir", "runs")) / name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    ds, loaders = make_loaders(cfg)
    model = build_model(cfg, in_channels=ds["train"].in_channels).to(device)
    trainable, total = count_params(model)
    print(f"[{name}] train chips={len(ds['train'])} params trainable={trainable:,} / total={total:,}")

    class_weights = t.get("class_weights")
    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None,
    )
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(loaders["train"])
    epochs = resolve_epochs(t, steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    # Validate ~evals_per_run times per run so long low-label runs (hundreds of short epochs)
    # do not spend most of their time evaluating; the last epoch is always evaluated.
    eval_every = max(1, epochs // int(t.get("evals_per_run", 50)))
    print(f"[{name}] steps/epoch={steps_per_epoch} epochs={epochs} total steps={total_steps} eval every {eval_every} epochs")
    scheduler = cosine_with_warmup(optimizer, t.get("warmup_steps", steps_per_epoch), total_steps)
    amp = t.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    logger = CSVLogger(out / "metrics.csv")
    best_iou, best_epoch = -1.0, -1
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * x.shape[0]
            n += x.shape[0]

        if (epoch + 1) % eval_every and epoch != epochs - 1:
            continue
        val = evaluate(model, loaders["valid"], device, amp)
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, n), "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"val_{k}": v for k, v in val.items()})
        row["elapsed_min"] = (time.time() - t0) / 60
        logger.log(row)
        print(f"epoch {epoch:3d} loss {row['train_loss']:.4f} val water IoU {val['water_iou']:.4f} mIoU {val['miou']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out / "last.pt")
        if val["water_iou"] > best_iou:
            best_iou, best_epoch = val["water_iou"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg}, out / "best.pt")

    # Final evaluation with the best checkpoint on the held-out splits.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    final = {
        "best_epoch": best_epoch,
        "epochs": epochs,
        "total_steps": total_steps,
        "best_val_water_iou": best_iou,
        "trainable_params": trainable,
        "total_params": total,
        "train_chips": len(ds["train"]),
        "gpu_minutes": (time.time() - t0) / 60,
        "test": evaluate(model, loaders["test"], device, amp),
        "bolivia": evaluate(model, loaders["bolivia"], device, amp),
    }
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
