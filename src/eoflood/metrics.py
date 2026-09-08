"""Confusion-matrix based segmentation metrics with an ignore index.

Accumulates over batches so the reported numbers are dataset-level
(micro-averaged), which is the convention used in the Sen1Floods11 paper.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -1


class SegMetrics:
    def __init__(self, num_classes: int = 2, ignore_index: int = IGNORE_INDEX, water_class: int = 1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.water_class = water_class
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        self.cm.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """``pred`` and ``target`` are integer tensors of identical shape (N, H, W)."""
        pred = pred.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx = target * self.num_classes + pred
        self.cm += torch.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, float]:
        cm = self.cm.double()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        # clamp(min=1) keeps 0/0 -> 0 without biasing exact values like 1.0
        iou = tp / (tp + fp + fn).clamp(min=1)
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        w = self.water_class
        out = {
            "acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
            "miou": iou.mean().item(),
            "water_iou": iou[w].item(),
            "water_f1": f1[w].item(),
            "water_precision": precision[w].item(),
            "water_recall": recall[w].item(),
        }
        for c in range(self.num_classes):
            out[f"iou_c{c}"] = iou[c].item()
        return out
