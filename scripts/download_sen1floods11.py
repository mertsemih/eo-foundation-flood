"""Download the hand-labeled part of Sen1Floods11 from the public GCS bucket.

Requires the Google Cloud CLI (``gcloud``) on PATH; no account needed for this bucket.
Downloads S1Hand, S2Hand, LabelHand (~4 GB) plus the split CSVs.

    python scripts/download_sen1floods11.py --root data/sen1floods11
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "gs://sen1floods11/v1.1"
HAND_DIRS = ["S1Hand", "S2Hand", "LabelHand"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sen1floods11")
    ap.add_argument("--extra", nargs="*", default=[], help="extra HandLabeled dirs, e.g. JRCWaterHand S1OtsuLabelHand")
    args = ap.parse_args()

    tool = shutil.which("gcloud") or shutil.which("gsutil")
    if tool is None:
        sys.exit("gcloud/gsutil not found. Install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install")
    use_gcloud = tool.endswith("gcloud") or tool.endswith("gcloud.cmd")

    root = Path(args.root)
    hand_dst = root / "v1.1" / "data" / "flood_events" / "HandLabeled"
    split_dst = root / "v1.1" / "splits" / "flood_handlabeled"
    hand_dst.mkdir(parents=True, exist_ok=True)
    split_dst.mkdir(parents=True, exist_ok=True)

    def cp(src: str, dst: Path) -> None:
        if use_gcloud:
            run(["gcloud", "storage", "cp", "-r", src, str(dst)])
        else:
            run(["gsutil", "-m", "cp", "-r", src, str(dst)])

    for d in HAND_DIRS + list(args.extra):
        cp(f"{BUCKET}/data/flood_events/HandLabeled/{d}", hand_dst)
    cp(f"{BUCKET}/splits/flood_handlabeled/*", split_dst)

    n = len(list((hand_dst / "LabelHand").glob("*.tif")))
    print(f"done: {n} labeled chips under {hand_dst}")


if __name__ == "__main__":
    main()
