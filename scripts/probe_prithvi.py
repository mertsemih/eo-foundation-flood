"""Load a Prithvi backbone through terratorch and print what the wrapper needs to know.

    python scripts/probe_prithvi.py [--backbone prithvi_eo_v2_300] [--no-pretrained]

Prints: backbone class, patch size, embed dim, number of blocks, the Linear module names a
LoRA config can target, and the output shapes for a 4-D (B, C, H, W) and 5-D (B, C, T, H, W)
input. Use it to settle the ``# CHECK`` items in ``src/eoflood/models/prithvi.py``.
"""

from __future__ import annotations

import argparse
import collections

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="prithvi_eo_v2_300")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    from terratorch.registry import BACKBONE_REGISTRY

    bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    kwargs = dict(pretrained=not args.no_pretrained, bands=bands, num_frames=1)
    print(f"building {args.backbone} with {kwargs}")
    bb = BACKBONE_REGISTRY.build(args.backbone, **kwargs)
    print("class:", type(bb).__module__, type(bb).__name__)
    for attr in ("patch_size", "embed_dim", "img_size", "num_frames", "out_channels", "feature_info"):
        if hasattr(bb, attr):
            print(f"  {attr} = {getattr(bb, attr)}")
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"  params = {n_params / 1e6:.1f} M")

    # Linear layer names (last path component) -> counts; these are LoRA target candidates.
    names = collections.Counter(n.rsplit(".", 1)[-1] for n, m in bb.named_modules() if isinstance(m, torch.nn.Linear))
    print("  Linear module names:", dict(names))
    blocks = [n for n, _ in bb.named_modules() if n.endswith("blocks") or n == "blocks"]
    print("  block containers:", blocks[:5])

    bb.eval()
    for shape in [(2, 6, args.size, args.size), (2, 6, 1, args.size, args.size)]:
        x = torch.randn(*shape)
        try:
            with torch.no_grad():
                out = bb(x)
            if isinstance(out, torch.Tensor):
                out = [out]
            print(f"  input {tuple(shape)} -> {len(out)} outputs; shapes: {[tuple(o.shape) for o in out[:2]]} ... {tuple(out[-1].shape)}")
        except Exception as e:  # noqa: BLE001 - probe script, report and continue
            print(f"  input {tuple(shape)} -> ERROR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
