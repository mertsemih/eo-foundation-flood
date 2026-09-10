"""Copy mean/std from configs/data/stats.yaml into every config whose ``data.bands`` matches.

    python scripts/apply_stats.py [--stats configs/data/stats.yaml] [--configs configs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="configs/data/stats.yaml")
    ap.add_argument("--configs", default="configs")
    args = ap.parse_args()

    st = yaml.safe_load(Path(args.stats).read_text(encoding="utf-8"))
    bands, mean, std = st["bands"], st["mean"], st["std"]

    for path in sorted(Path(args.configs).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        d = cfg.get("data") if isinstance(cfg, dict) else None
        if not d:
            continue
        expected = list(d.get("bands", [])) + (["VV", "VH"] if d.get("use_s1") else [])
        if expected != bands:
            print(f"skip {path.name}: bands {expected} != stats bands {bands}")
            continue
        d["mean"], d["std"] = mean, std
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
