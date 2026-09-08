"""Build a tiny synthetic Sen1Floods11 tree so the dataset code is tested without a download."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

CHIP = 64
N_TRAIN, N_VALID = 6, 2


def _write_tif(path: Path, arr: np.ndarray, dtype: str) -> None:
    arr = arr.astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1e-4, 1e-4),
    ) as f:
        f.write(arr)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sen1floods11")
    hand = root / "v1.1" / "data" / "flood_events" / "HandLabeled"
    for d in ("S1Hand", "S2Hand", "LabelHand"):
        (hand / d).mkdir(parents=True)
    splits = root / "v1.1" / "splits" / "flood_handlabeled"
    splits.mkdir(parents=True)

    rng = np.random.RandomState(0)
    names = {"train": [], "valid": [], "test": [], "bolivia": []}
    counts = {"train": N_TRAIN, "valid": N_VALID, "test": N_VALID, "bolivia": N_VALID}
    k = 0
    for split, n in counts.items():
        for _ in range(n):
            base = f"Region_{k:06d}"
            k += 1
            s2 = rng.randint(0, 10000, size=(13, CHIP, CHIP))
            s1 = rng.uniform(-30, 0, size=(2, CHIP, CHIP))
            label = rng.randint(0, 2, size=(1, CHIP, CHIP))
            label[0, :4, :4] = -1  # a no-data corner
            _write_tif(hand / "S2Hand" / f"{base}_S2Hand.tif", s2, "uint16")
            _write_tif(hand / "S1Hand" / f"{base}_S1Hand.tif", s1, "float32")
            _write_tif(hand / "LabelHand" / f"{base}_LabelHand.tif", label, "int16")
            names[split].append((f"{base}_S1Hand.tif", f"{base}_LabelHand.tif"))
        with open(splits / f"flood_{split}_data.csv", "w", newline="") as f:
            csv.writer(f).writerows(names[split])
    return root
