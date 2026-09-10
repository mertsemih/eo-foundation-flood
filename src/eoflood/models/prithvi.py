"""Prithvi-EO 2.0 backbone + light decoder, with frozen / LoRA / full fine-tuning.

Backbone weights come through ``terratorch`` (IBM's fine-tuning toolkit for Prithvi), LoRA
adapters through ``peft``. Both are optional dependencies (``pip install -e ".[foundation]"``).

Verified on 2026-09-10 with terratorch 1.2.13 / peft 0.20.0 (``scripts/probe_prithvi.py``):
``prithvi_eo_v2_300`` is ``terratorch.models.backbones.prithvi_mae.PrithviViT`` (303.9 M params,
embed_dim 1024, 24 blocks, patch 16). It accepts 4-D ``(B, C, H, W)`` input directly and returns
a list of 24 token tensors ``(B, N + 1, C)`` with a leading CLS token; 224 -> 197 tokens,
512 -> 1025 tokens, so full 512 x 512 chips work at evaluation time. Linear layers per block are
``qkv``, ``proj``, ``fc1``, ``fc2``. Trainable params: frozen 1.44 M (head only), LoRA r=8 on qkv
2.23 M, full 305.3 M.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# terratorch band identifiers for the six Prithvi input bands, in model order.
PRITHVI_TT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

FINETUNE_MODES = ("frozen", "lora", "full")


class FCNHead(nn.Module):
    """Concatenate selected ViT feature maps and upsample x16 with conv blocks."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256, n_up: int = 4):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        ch = hidden
        for _ in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers.append(nn.Conv2d(ch, num_classes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tokens_to_map(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, N[+1], C) -> (B, C, h, w); drops a leading CLS token if present."""
    if tokens.shape[1] == h * w + 1:
        tokens = tokens[:, 1:]
    elif tokens.shape[1] != h * w:
        raise ValueError(f"token count {tokens.shape[1]} does not match grid {h}x{w}")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)


class PrithviSegmenter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "prithvi_eo_v2_300",
        num_classes: int = 2,
        in_channels: int = 6,
        finetune: str = "lora",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_targets: list[str] | None = None,
        feature_layers: list[int] | None = None,
        img_size: int = 224,
        pretrained: bool = True,
    ):
        super().__init__()
        if finetune not in FINETUNE_MODES:
            raise ValueError(f"finetune must be one of {FINETUNE_MODES}, got {finetune!r}")
        if in_channels != len(PRITHVI_TT_BANDS):
            raise ValueError(
                f"Prithvi expects {len(PRITHVI_TT_BANDS)} bands {PRITHVI_TT_BANDS}; "
                f"config gives {in_channels} channels (set data.bands to PRITHVI_BANDS, use_s1=false)"
            )

        from terratorch.registry import BACKBONE_REGISTRY  # lazy: optional dependency

        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            bands=PRITHVI_TT_BANDS,
            num_frames=1,
            img_size=img_size,
        )
        self.patch_size = getattr(self.backbone, "patch_size", 16)
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[-1]
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("could not read embed_dim from backbone; set it manually")

        self.finetune = finetune
        self._apply_finetune_mode(lora_r, lora_alpha, lora_dropout, lora_targets or ["qkv"])

        # Default: last four layers of a 24-layer ViT-L; overridable for smaller backbones.
        self.feature_layers = feature_layers or [-4, -3, -2, -1]
        self.head = FCNHead(embed_dim * len(self.feature_layers), num_classes)

    # ------------------------------------------------------------------ fine-tuning modes
    def _apply_finetune_mode(self, r, alpha, dropout, targets):
        if self.finetune == "full":
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.finetune == "lora":
            from peft import LoraConfig, inject_adapter_in_model

            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
            # timm-style block names: "qkv", "proj" (attention), "fc1", "fc2" (MLP)
            self.backbone = inject_adapter_in_model(cfg, self.backbone)

    # ------------------------------------------------------------------ forward
    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Prithvi is spatio-temporal (B, C, T, H, W); terratorch's PrithviViT accepts 4-D input
        # and inserts the T=1 axis itself (verified with terratorch 1.2.13).
        feats = self.backbone(x)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        return list(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        gh, gw = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
        feats = self._encode(x)
        maps = [tokens_to_map(feats[i], gh, gw) for i in self.feature_layers]
        logits = self.head(torch.cat(maps, dim=1))
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------ optimizer groups
    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        """Head at ``lr``; trainable backbone params (LoRA or full) at a reduced rate."""
        head = [p for p in self.head.parameters() if p.requires_grad]
        body = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head, "lr": lr}]
        if body:
            groups.append({"params": body, "lr": lr * backbone_lr_mult})
        return groups
