"""Joint image/label augmentations on numpy arrays. Image is (C, H, W), label is (H, W)."""

from __future__ import annotations

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray, y: np.ndarray):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class RandomCrop:
    """Random spatial crop; if the chip is smaller than ``size`` it is returned unchanged."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: np.ndarray, y: np.ndarray):
        _, h, w = x.shape
        s = self.size
        if h <= s and w <= s:
            return x, y
        top = np.random.randint(0, h - s + 1)
        left = np.random.randint(0, w - s + 1)
        return x[:, top : top + s, left : left + s], y[top : top + s, left : left + s]


class RandomFlipRotate:
    """Dihedral-group augmentation: random horizontal / vertical flips and 90-degree rotations."""

    def __call__(self, x: np.ndarray, y: np.ndarray):
        if np.random.rand() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1, :], y[::-1, :]
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(1, 2))
            y = np.rot90(y, k, axes=(0, 1))
        return np.ascontiguousarray(x), np.ascontiguousarray(y)
