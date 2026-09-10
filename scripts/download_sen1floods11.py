"""Download the hand-labeled part of Sen1Floods11 (S1Hand, S2Hand, LabelHand + split CSVs).

The bucket ``gs://sen1floods11`` is public, so no Google account or gcloud install is needed:
objects are listed through the JSON API and fetched over plain HTTPS with a small thread
pool. Existing files with the right size are skipped, so the script is safe to re-run.
Total size is roughly 1.8 GB (446 chips per folder).

    python scripts/download_sen1floods11.py --root data/sen1floods11 [--workers 8]
    python scripts/download_sen1floods11.py --extra JRCWaterHand S1OtsuLabelHand   # optional folders
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "sen1floods11"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
HAND_PREFIX = "v1.1/data/flood_events/HandLabeled/"
SPLIT_PREFIX = "v1.1/splits/flood_handlabeled/"
HAND_DIRS = ["S1Hand", "S2Hand", "LabelHand"]


def list_objects(prefix: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    token = None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            d = json.load(r)
        out += [(i["name"], int(i["size"])) for i in d.get("items", []) if not i["name"].endswith("/")]
        token = d.get("nextPageToken")
        if not token:
            return out


def fetch(name: str, size: int, root: Path, retries: int = 4) -> tuple[str, int, bool]:
    """Download one object to ``root/name``; returns (name, bytes, skipped)."""
    dst = root / name
    if dst.exists() and dst.stat().st_size == size:
        return name, size, True
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = MEDIA + urllib.parse.quote(name)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise OSError(f"size mismatch for {name}: {tmp.stat().st_size} != {size}")
            tmp.replace(dst)
            return name, size, False
        except Exception as e:  # noqa: BLE001 - retry on any network / IO error
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  retry {attempt + 1} for {name}: {e}", file=sys.stderr)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sen1floods11")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[], help="extra HandLabeled dirs, e.g. JRCWaterHand S1OtsuLabelHand")
    args = ap.parse_args()
    root = Path(args.root)

    objects: list[tuple[str, int]] = list_objects(SPLIT_PREFIX)
    for d in HAND_DIRS + list(args.extra):
        objects += list_objects(f"{HAND_PREFIX}{d}/")
    total = sum(s for _, s in objects)
    print(f"{len(objects)} objects, {total / 1e9:.2f} GB -> {root}")

    done_bytes, skipped, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, n, s, root) for n, s in objects]
        for k, fut in enumerate(as_completed(futures), 1):
            _, size, was_skipped = fut.result()
            done_bytes += size
            skipped += was_skipped
            if k % 50 == 0 or k == len(objects):
                rate = done_bytes / 1e6 / max(1e-6, time.time() - t0)
                print(f"  {k}/{len(objects)}  {done_bytes / 1e9:.2f} GB  {rate:.1f} MB/s")

    for d in HAND_DIRS:
        n = len(list((root / HAND_PREFIX / d).glob("*.tif")))
        print(f"{d}: {n} chips")
    print(f"done in {(time.time() - t0) / 60:.1f} min ({skipped} files already present)")


if __name__ == "__main__":
    main()
