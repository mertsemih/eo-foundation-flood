"""Run (a slice of) the experiment matrix sequentially, skipping finished runs.

    python scripts/run_matrix.py --models unet_scratch unet_imagenet --fractions 1.0 0.25 0.10 0.05 --seeds 0
    python scripts/run_matrix.py --models prithvi_lora --fractions 0.10 --seeds 0 1 2 --extra train.epochs=30

A run is "finished" when ``runs/<tag>_f<frac>_s<seed>/test_metrics.json`` exists, so the
script can be re-launched after an interruption and continues where it stopped.
Each run's stdout/stderr goes to ``runs/<name>.log``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_name(tag: str, frac: float, seed: int) -> str:
    return f"{tag}_f{frac:.2f}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="config basenames under configs/")
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--extra", nargs="*", default=[], help="extra key=value overrides for every run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = Path(args.runs)
    runs.mkdir(exist_ok=True)
    todo = []
    for tag in args.models:
        for frac in args.fractions:
            for seed in args.seeds:
                name = run_name(tag, frac, seed)
                if (runs / name / "test_metrics.json").exists():
                    print(f"skip {name} (done)")
                    continue
                todo.append((tag, frac, seed, name))
    print(f"{len(todo)} runs to do")

    for k, (tag, frac, seed, name) in enumerate(todo, 1):
        cmd = [
            sys.executable, "-m", "eoflood.train", "--config", f"configs/{tag}.yaml",
            f"data.train_fraction={frac}", f"seed={seed}", f"runs_dir={args.runs}", *args.extra,
        ]
        print(f"[{k}/{len(todo)}] {name}: {' '.join(cmd[2:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(runs / f"{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"    {status} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
