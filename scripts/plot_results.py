"""Paper figures from results/results.csv (written by scripts/collect_results.py).

    python scripts/collect_results.py
    python scripts/plot_results.py [--results results/results.csv] [--out docs/figures]

Figure 1  water IoU vs training-label fraction, one line per model (test + Bolivia panels).
Figure 2  test vs Bolivia water IoU per model at full labels (OOD gap).
Figure 3  water IoU vs trainable parameters (accuracy per parameter).
Mean over seeds with min-max error bars when more than one seed exists.
With --lang tr the figures get Turkish labels and decimal commas and are written as fig*_tr.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["unet_scratch", "unet_imagenet", "prithvi_frozen", "prithvi_lora", "prithvi_lora_unetdec", "prithvi_lora_r4", "prithvi_lora_r16", "prithvi_lora_wide", "prithvi_full"]
LABELS = {
    "unet_scratch": "U-Net (scratch)",
    "unet_imagenet": "U-Net (ImageNet)",
    "prithvi_frozen": "Prithvi frozen",
    "prithvi_lora": "Prithvi LoRA",
    "prithvi_lora_unetdec": "Prithvi LoRA + UNet dec.",
    "prithvi_lora_r4": "Prithvi LoRA r=4",
    "prithvi_lora_r16": "Prithvi LoRA r=16",
    "prithvi_lora_wide": "Prithvi LoRA qkv+proj+fc",
    "prithvi_full": "Prithvi full FT",
}


LABELS_TR = {
    "unet_scratch": "U-Net (sıfırdan)",
    "unet_imagenet": "U-Net (ImageNet)",
    "prithvi_frozen": "Prithvi donuk",
    "prithvi_lora": "Prithvi LoRA",
    "prithvi_lora_unetdec": "Prithvi LoRA + UNet kod çözücü",
    "prithvi_lora_r4": "Prithvi LoRA r=4",
    "prithvi_lora_r16": "Prithvi LoRA r=16",
    "prithvi_lora_wide": "Prithvi LoRA qkv+proj+fc",
    "prithvi_full": "Prithvi tam ince ayar",
}
TEXT = {
    "en": {"test": "Test (in-distribution)", "bol": "Bolivia (held-out region)", "xfrac": "training labels (% of 252 chips)",
           "iou": "water IoU", "iou100": "water IoU (100 % labels)", "test_iou100": "test water IoU (100 % labels)",
           "params": "trainable parameters", "bar_test": "test", "suffix": ""},
    "tr": {"test": "Test (dağılım içi)", "bol": "Bolivya (dışarıda tutulan bölge)", "xfrac": "eğitim etiketleri (252 karonun %'si)",
           "iou": "su IoU", "iou100": "su IoU (%100 etiket)", "test_iou100": "test su IoU (%100 etiket)",
           "params": "eğitilen parametre sayısı", "bar_test": "test", "suffix": "_tr"},
}
T = TEXT["en"]


def label(m: str) -> str:
    return (LABELS_TR if T is TEXT["tr"] else LABELS).get(m, m)


def comma_axis(ax) -> None:
    """Decimal commas on the y axis for the Turkish figures."""
    if T is TEXT["tr"]:
        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))


LABEL_OFFSETS = {
    "unet_imagenet": (-6, 4, "right"),
    "unet_scratch": (-6, -10, "right"),
    "prithvi_lora_r16": (6, -10, "left"),
    "prithvi_full": (-6, 6, "right"),
}


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["model", "train_fraction"])[metric]
    return g.agg(["mean", "min", "max", "count"]).reset_index()


def fig_label_fraction(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(axes, ["test_water_iou", "bolivia_water_iou"], [T["test"], T["bol"]], strict=True):
        a = agg(df, metric)
        for m in ORDER:
            s = a[a.model == m].sort_values("train_fraction")
            if s.empty:
                continue
            yerr = [s["mean"] - s["min"], s["max"] - s["mean"]]
            ax.errorbar(s.train_fraction * 100, s["mean"], yerr=yerr, marker="o", capsize=3, label=label(m))
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 100])
        ax.set_xticklabels(["1", "2", "5", "10", "25", "100"])
        ax.set_xlabel(T["xfrac"])
        comma_axis(ax)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(T["iou"])
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig1_iou_vs_labels{T['suffix']}.png", dpi=150)
    plt.close(fig)


def fig_ood_gap(df: pd.DataFrame, out: Path) -> None:
    full = df[df.train_fraction == 1.0]
    if full.empty:
        return
    a = full.groupby("model")[["test_water_iou", "bolivia_water_iou"]].mean()
    a = a.loc[[m for m in ORDER if m in a.index]]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(a))
    ax.bar([i - 0.2 for i in x], a.test_water_iou, width=0.4, label=T["bar_test"])
    ax.bar([i + 0.2 for i in x], a.bolivia_water_iou, width=0.4, label="Bolivya" if T is TEXT["tr"] else "Bolivia")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in a.index], rotation=35, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_ylabel(T["iou100"])
    comma_axis(ax)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"fig2_ood_gap{T['suffix']}.png", dpi=150)
    plt.close(fig)


def fig_params(df: pd.DataFrame, out: Path) -> None:
    full = df[df.train_fraction == 1.0]
    if full.empty:
        return
    a = full.groupby("model")[["trainable_params", "test_water_iou", "bolivia_water_iou"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, r in a.iterrows():
        ax.scatter(r.trainable_params, r.test_water_iou, s=60)
        # hand-placed offsets where labels of neighbouring points would collide
        dx, dy, ha = LABEL_OFFSETS.get(m, (5, 5, "left"))
        ax.annotate(label(m), (r.trainable_params, r.test_water_iou), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(T["params"])
    ax.set_ylabel(T["test_iou100"])
    comma_axis(ax)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"fig3_iou_vs_params{T['suffix']}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--lang", choices=["en", "tr"], default="en")
    args = ap.parse_args()
    global T
    T = TEXT[args.lang]
    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_label_fraction(df, out)
    fig_ood_gap(df, out)
    fig_params(df, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
