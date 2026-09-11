"""Decoders that turn ViT token maps into dense logits.

``FCNHead`` (in prithvi.py) concatenates a few late layers at one 1/16 scale and upsamples.
``MultiScaleUNetDecoder`` builds a feature pyramid from four encoder depths (shallow -> fine
resolution, deep -> coarse) and merges them UNet-style, the standard remedy when a plain ViT
head under-segments thin structures such as rivers and flood edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class MultiScaleUNetDecoder(nn.Module):
    """Four ViT feature maps (all at 1/patch scale) -> pyramid at x4, x2, x1, x1/2 -> UNet merge.

    ``in_channels`` is the ViT embedding size; every input map is (B, in_channels, h, w).
    Output is at full input resolution (h * patch, w * patch).
    """

    SCALES = (4.0, 2.0, 1.0, 0.5)  # shallow -> deep

    def __init__(self, in_channels: int, num_classes: int, width: int = 256, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_channels, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)) for _ in self.SCALES]
        )
        # bottom-up merge: deepest (x1/2) -> x1 -> x2 -> x4
        self.up3 = conv_block(width * 2, width)      # x1/2 up to x1, concat with x1 map
        self.up2 = conv_block(width * 2, width // 2)  # -> x2
        self.up1 = conv_block(width // 2 + width, width // 4)  # -> x4
        # x4 is patch/4 of full res (56 px for 224 input); two more stages reach full res
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(width // 4, width // 8),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_block(width // 8, width // 8),
            nn.Conv2d(width // 8, num_classes, 1),
        )

    @staticmethod
    def _resize(x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return x
        if scale < 1.0:
            return F.max_pool2d(x, kernel_size=int(round(1 / scale)))
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, maps: list[torch.Tensor]) -> torch.Tensor:
        if len(maps) != len(self.SCALES):
            raise ValueError(f"expected {len(self.SCALES)} feature maps (shallow->deep), got {len(maps)}")
        p4, p2, p1, p05 = [self._resize(proj(m), s) for proj, m, s in zip(self.proj, maps, self.SCALES, strict=True)]
        x = self.up3(torch.cat([self._up_to(p05, p1), p1], 1))
        x = self.up2(torch.cat([self._up_to(x, p2), p2], 1))
        x = self.up1(torch.cat([self._up_to(x, p4), p4], 1))
        return self.final(x)
