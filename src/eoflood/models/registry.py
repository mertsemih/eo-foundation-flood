"""Model factory. Prithvi is imported lazily so the core package works without terratorch."""

from __future__ import annotations

import torch.nn as nn


def build_model(cfg: dict, in_channels: int) -> nn.Module:
    m = cfg["model"]
    name = m["name"]
    num_classes = m.get("num_classes", 2)

    if name == "unet":
        from .unet import build_unet

        return build_unet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder=m.get("encoder", "resnet34"),
            encoder_weights=m.get("encoder_weights"),  # None (scratch) or "imagenet"
        )

    if name == "prithvi":
        from .prithvi import PrithviSegmenter

        return PrithviSegmenter(
            backbone_name=m.get("backbone", "prithvi_eo_v2_300"),
            num_classes=num_classes,
            in_channels=in_channels,
            finetune=m.get("finetune", "lora"),
            lora_r=m.get("lora_r", 8),
            lora_alpha=m.get("lora_alpha", 16),
            lora_dropout=m.get("lora_dropout", 0.05),
            lora_targets=m.get("lora_targets", ["qkv"]),
            feature_layers=m.get("feature_layers"),
            img_size=m.get("img_size", 224),
            pretrained=m.get("pretrained", True),
            decoder=m.get("decoder", "fcn"),
        )

    raise ValueError(f"unknown model name {name!r}")
