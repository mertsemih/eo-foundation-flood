import numpy as np
import pytest
import torch

from eoflood.data import (
    PRITHVI_BANDS,
    Compose,
    RandomCrop,
    RandomFlipRotate,
    Sen1Floods11,
    build_datasets,
)
from eoflood.metrics import IGNORE_INDEX

from .conftest import CHIP, N_TRAIN


def test_shapes_and_dtypes(fake_root):
    ds = Sen1Floods11(fake_root, "train")
    assert len(ds) == N_TRAIN
    x, y = ds[0]
    assert x.shape == (len(PRITHVI_BANDS), CHIP, CHIP) and x.dtype == torch.float32
    assert y.shape == (CHIP, CHIP) and y.dtype == torch.int64
    assert set(torch.unique(y).tolist()) <= {IGNORE_INDEX, 0, 1}
    assert (y[:4, :4] == IGNORE_INDEX).all()


def test_s1_concat_and_normalization(fake_root):
    mean = [1.0] * 8
    std = [2.0] * 8
    ds = Sen1Floods11(fake_root, "train", use_s1=True, mean=mean, std=std)
    assert ds.in_channels == 8
    x_raw, _ = ds.load_raw(0)
    x, _ = ds[0]
    np.testing.assert_allclose(x.numpy(), (x_raw - 1.0) / 2.0, rtol=1e-5)


def test_wrong_stat_length_raises(fake_root):
    with pytest.raises(ValueError):
        Sen1Floods11(fake_root, "train", mean=[0.0] * 3, std=[1.0] * 3)


def test_train_fraction_is_deterministic(fake_root):
    a = Sen1Floods11(fake_root, "train", train_fraction=0.5, seed=1)
    b = Sen1Floods11(fake_root, "train", train_fraction=0.5, seed=1)
    c = Sen1Floods11(fake_root, "train", train_fraction=0.5, seed=2)
    assert len(a) == N_TRAIN // 2
    assert a.items == b.items
    assert a.items != c.items or N_TRAIN <= 2


def test_transforms_keep_alignment(fake_root):
    aug = Compose([RandomCrop(32), RandomFlipRotate()])
    ds = Sen1Floods11(fake_root, "train", transform=aug)
    x, y = ds[0]
    assert x.shape[1:] == (32, 32) and y.shape == (32, 32)


def test_build_datasets(fake_root):
    cfg = {"seed": 0, "data": {"root": str(fake_root), "train_fraction": 1.0}}
    ds = build_datasets(cfg)
    assert set(ds) == {"train", "valid", "test", "bolivia"}
    assert ds["train"].in_channels == len(PRITHVI_BANDS)


def test_cache_matches_disk(fake_root):
    a = Sen1Floods11(fake_root, "train")
    b = Sen1Floods11(fake_root, "train", cache=True)
    xa, ya = a[1]
    xb, yb = b[1]
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    # the cache must hand out copies so callers cannot corrupt it
    x, _ = b.load_raw(1)
    x[:] = 0
    assert not torch.equal(torch.zeros_like(xb), b[1][0])
