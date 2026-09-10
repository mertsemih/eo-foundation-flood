"""Paper figures from results/results.csv (written by scripts/collect_results.py).

    python scripts/collect_results.py
    python scripts/plot_results.py [--results results/results.csv] [--out docs/figures]

Figure 1  water IoU vs training-label fraction, one line per model (test + Bolivia panels).
Figure 2  test vs Bolivia water IoU per model at full labels (OOD gap).
Figure 3  water IoU vs trainable parameters (accuracy per parameter).
Mean over seeds with min-max error bars when more than one seed exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["unet_scratch", "unet_imagenet", "prithvi_frozen", "prithvi_lora", "prithvi_full"]
LABELS = {
    "unet_scratch": "U-Net (scratch)",
    "unet_imagenet": "U-Net (ImageNet)",
    "prithvi_frozen": "Prithvi frozen",
    "prithvi_lora": "Prithvi LoRA",
    "prithvi_full": "Prithvi full FT",
}


def agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["model", "train_fraction"])[metric]
    return g.agg(["mean", "min", "max", "count"]).reset_index()


def fig_label_fraction(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(axes, ["test_water_iou", "bolivia_water_iou"], ["Test (in-distribution)", "Bolivia (held-out region)"], strict=True):
        a = agg(df, metric)
        for m in ORDER:
            s = a[a.model == m].sort_values("train_fraction")
            if s.empty:
                continue
            yerr = [s["mean"] - s["min"], s["max"] - s["mean"]]
            ax.errorbar(s.train_fraction * 100, s["mean"], yerr=yerr, marker="o", capsize=3, label=LABELS.get(m, m))
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 100])
        ax.set_xticklabels(["5", "10", "25", "100"])
        ax.set_xlabel("training labels (% of 252 chips)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("water IoU")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_iou_vs_labels.png", dpi=150)
    plt.close(fig)


def fig_ood_gap(df: pd.DataFrame, out: Path) -> None:
    full = df[df.train_fraction == 1.0]
    if full.empty:
        return
    a = full.groupby("model")[["test_water_iou", "bolivia_water_iou"]].mean()
    a = a.loc[[m for m in ORDER if m in a.index]]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(a))
    ax.bar([i - 0.2 for i in x], a.test_water_iou, width=0.4, label="test")
    ax.bar([i + 0.2 for i in x], a.bolivia_water_iou, width=0.4, label="Bolivia")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS.get(m, m) for m in a.index], rotation=20, fontsize=8)
    ax.set_ylabel("water IoU (100 % labels)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_ood_gap.png", dpi=150)
    plt.close(fig)


def fig_params(df: pd.DataFrame, out: Path) -> None:
    full = df[df.train_fraction == 1.0]
    if full.empty:
        return
    a = full.groupby("model")[["trainable_params", "test_water_iou", "bolivia_water_iou"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, r in a.iterrows():
        ax.scatter(r.trainable_params, r.test_water_iou, s=60)
        ax.annotate(LABELS.get(m, m), (r.trainable_params, r.test_water_iou), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("test water IoU (100 % labels)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_iou_vs_params.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_label_fraction(df, out)
    fig_ood_gap(df, out)
    fig_params(df, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
