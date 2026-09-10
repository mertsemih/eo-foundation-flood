"""Sen1Floods11 hand-labeled chips as a PyTorch dataset.

Expected layout (what ``scripts/download_sen1floods11.py`` produces)::

    <root>/v1.1/data/flood_events/HandLabeled/S1Hand/*.tif      (2 bands: VV, VH, dB)
    <root>/v1.1/data/flood_events/HandLabeled/S2Hand/*.tif      (13 bands, L1C TOA reflectance x 10000)
    <root>/v1.1/data/flood_events/HandLabeled/LabelHand/*.tif   (1 band: -1 nodata, 0 no water, 1 water)
    <root>/v1.1/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv

Each split CSV row is ``<S1 chip>.tif,<Label chip>.tif``; the S2 chip name follows by
replacing ``_S1Hand`` with ``_S2Hand``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ..metrics import IGNORE_INDEX

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
# Six HLS-equivalent bands Prithvi-EO was pretrained on, in the order the model expects.
PRITHVI_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def band_indices(bands: list[str]) -> list[int]:
    return [S2_BANDS.index(b) for b in bands]


def read_split(root: Path, split: str) -> list[tuple[str, str]]:
    path = root / "v1.1" / "splits" / "flood_handlabeled" / SPLIT_FILES[split]
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    return [(r[0].strip(), r[1].strip()) for r in rows]


class Sen1Floods11(Dataset):
    """Returns ``(x, y)`` with ``x`` float32 (C, H, W) and ``y`` int64 (H, W)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        bands: list[str] = PRITHVI_BANDS,
        use_s1: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        transform=None,
        train_fraction: float = 1.0,
        seed: int = 0,
        cache: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.bands = list(bands)
        self.band_idx = band_indices(self.bands)
        self.use_s1 = use_s1
        self.transform = transform

        hand = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled"
        self.s1_dir, self.s2_dir, self.label_dir = hand / "S1Hand", hand / "S2Hand", hand / "LabelHand"

        items = read_split(self.root, split)
        if not 0 < train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        if train_fraction < 1.0:
            # Deterministic subsample for the low-label experiments; the same seed
            # always yields the same subset so runs are comparable across models.
            rng = np.random.RandomState(seed)
            keep = max(1, int(round(len(items) * train_fraction)))
            order = rng.permutation(len(items))[:keep]
            items = [items[i] for i in sorted(order)]
        self.items = items

        n_ch = len(self.bands) + (len(S1_BANDS) if use_s1 else 0)
        self.mean = np.asarray(mean if mean is not None else [0.0] * n_ch, dtype=np.float32)
        self.std = np.asarray(std if std is not None else [1.0] * n_ch, dtype=np.float32)
        if len(self.mean) != n_ch or len(self.std) != n_ch:
            raise ValueError(f"mean/std must have {n_ch} entries, got {len(self.mean)}/{len(self.std)}")

        # Optional in-RAM cache of the raw (band-selected) chips. Reading a 13-band GeoTIFF per
        # sample is the training bottleneck on a local GPU; 446 chips x 6 bands is ~2.8 GB float32.
        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache:
            self._cache = [self._load_raw_from_disk(i) for i in range(len(self.items))]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def in_channels(self) -> int:
        return len(self.mean)

    def _read(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as f:
            return f.read()

    def load_raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Unnormalized (C, H, W) float32 image and (H, W) int64 label. Used by compute_stats."""
        if self._cache is not None:
            x, y = self._cache[i]
            return x.copy(), y.copy()
        return self._load_raw_from_disk(i)

    def _load_raw_from_disk(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s1_name, label_name = self.items[i]
        s2_name = s1_name.replace("_S1Hand", "_S2Hand")
        s2 = self._read(self.s2_dir / s2_name).astype(np.float32)
        x = s2[self.band_idx]
        if self.use_s1:
            s1 = self._read(self.s1_dir / s1_name).astype(np.float32)
            x = np.concatenate([x, s1], axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self._read(self.label_dir / label_name)[0].astype(np.int64)
        y[(y != 0) & (y != 1)] = IGNORE_INDEX
        return x, y

    def __getitem__(self, i: int):
        x, y = self.load_raw(i)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def build_datasets(cfg: dict, train_transform=None) -> dict[str, Sen1Floods11]:
    """Build train/valid/test/bolivia datasets from the ``data`` section of a config."""
    d = cfg["data"]
    common = dict(
        root=d["root"],
        bands=d.get("bands", PRITHVI_BANDS),
        use_s1=d.get("use_s1", False),
        mean=d.get("mean"),
        std=d.get("std"),
        cache=d.get("cache", False),
    )
    out = {
        "train": Sen1Floods11(
            split="train",
            transform=train_transform,
            train_fraction=d.get("train_fraction", 1.0),
            seed=cfg.get("seed", 0),
            **common,
        )
    }
    for split in ("valid", "test", "bolivia"):
        out[split] = Sen1Floods11(split=split, **common)
    return out
