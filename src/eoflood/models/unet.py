"""U-Net baselines via segmentation_models_pytorch.

``encoder_weights=None``       -> from-scratch baseline (``unet_scratch``)
``encoder_weights="imagenet"`` -> generic-pretraining baseline (``unet_imagenet``); smp adapts the
                                  first conv to ``in_channels`` != 3 by re-weighting the RGB filters.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    in_channels: int,
    num_classes: int = 2,
    encoder: str = "resnet34",
    encoder_weights: str | None = None,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
