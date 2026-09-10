"""Gather every ``runs/*/test_metrics.json`` into one CSV and print a summary table.

    python scripts/collect_results.py [--runs runs] [--out results/results.csv]

Summary groups by (model, train_fraction) and reports mean ± std over seeds for
test / Bolivia water IoU, so the table in the paper is regenerated, never hand-edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for tm in sorted(runs_dir.glob("*/test_metrics.json")):
        run = tm.parent
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        m = json.loads(tm.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run.name,
                "model": cfg["model"]["tag"],
                "finetune": cfg["model"].get("finetune", "-"),
                "train_fraction": cfg["data"].get("train_fraction", 1.0),
                "seed": cfg.get("seed", 0),
                "train_chips": m["train_chips"],
                "trainable_params": m["trainable_params"],
                "total_params": m["total_params"],
                "gpu_minutes": round(m["gpu_minutes"], 1),
                "best_epoch": m["best_epoch"],
                "val_water_iou": m["best_val_water_iou"],
                "test_water_iou": m["test"]["water_iou"],
                "test_miou": m["test"]["miou"],
                "test_water_f1": m["test"]["water_f1"],
                "bolivia_water_iou": m["bolivia"]["water_iou"],
                "bolivia_miou": m["bolivia"]["miou"],
                "bolivia_water_f1": m["bolivia"]["water_f1"],
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "train_fraction"])
    out = g.agg(
        seeds=("seed", "count"),
        trainable_params=("trainable_params", "first"),
        test_iou_mean=("test_water_iou", "mean"),
        test_iou_std=("test_water_iou", "std"),
        bolivia_iou_mean=("bolivia_water_iou", "mean"),
        bolivia_iou_std=("bolivia_water_iou", "std"),
        gpu_min=("gpu_minutes", "mean"),
    )
    return out.sort_values(["train_fraction", "model"], ascending=[False, True])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    df = load_runs(Path(args.runs))
    if df.empty:
        print("no finished runs found")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df)
    summary.to_csv(Path(args.out).with_name("summary.csv"))
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string())
    print(f"\n{len(df)} runs -> {args.out}")


if __name__ == "__main__":
    main()
