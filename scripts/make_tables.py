"""LaTeX tables for the paper, generated from results/summary.csv (never hand-edited).

    python scripts/collect_results.py && python scripts/make_tables.py

Writes paper/tables/table_full.tex (all models at 100 % labels), paper/tables/table_sweep.tex
(test / Bolivia water IoU vs label fraction) and paper/tables/table_ablation.tex (LoRA rank,
targets, decoder). Cells are mean ± std over seeds; single-seed cells carry a dagger.
GPU minutes in the full-label table come from the seed-0 run of each model, because every
seed-0 run ran on the local RTX 3070 Ti while some later seeds ran on a ~3x slower Colab T4.
With --lang tr the same tables are written with Turkish labels and decimal commas
(default output paper/tables_tr) for paper/main_tr.tex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MAIN = [
    ("unet_scratch", "U-Net (scratch)", "--", "24.4\\,M"),
    ("unet_imagenet", "U-Net (ImageNet)", "ImageNet", "24.4\\,M"),
    ("prithvi_frozen", "Prithvi frozen", "Prithvi-EO 2.0", "1.4\\,M"),
    ("prithvi_lora", "Prithvi LoRA (r=8)", "Prithvi-EO 2.0", "2.2\\,M"),
    ("prithvi_full", "Prithvi full FT", "Prithvi-EO 2.0", "305\\,M"),
]
ABLATION = [
    ("prithvi_lora_r4", "LoRA r=4, qkv"),
    ("prithvi_lora", "LoRA r=8, qkv (main)"),
    ("prithvi_lora_r16", "LoRA r=16, qkv"),
    ("prithvi_lora_wide", "LoRA r=8, qkv+proj+fc1+fc2"),
    ("prithvi_lora_unetdec", "LoRA r=8, qkv + multi-scale UNet decoder"),
]
FRACS = [1.0, 0.25, 0.10, 0.05, 0.02, 0.01]

TR = {
    "U-Net (scratch)": "U-Net (sıfırdan)",
    "Prithvi frozen": "Prithvi donuk",
    "Prithvi full FT": "Prithvi tam ince ayar",
    "LoRA r=8, qkv (main)": "LoRA r=8, qkv (ana)",
    "LoRA r=8, qkv + multi-scale UNet decoder": "LoRA r=8, qkv + çok ölçekli UNet kod çözücü",
    "Model": "Model",
    "Pretraining": "Ön eğitim",
    "Trainable": "Eğitilen",
    "Test IoU$_w$": "Test IoU$_s$",
    "Bolivia IoU$_w$": "Bolivya IoU$_s$",
    "GPU min": "GPU dk",
    "Labels": "Etiket",
    "Variant": "Varyant",
}
LANG = "en"


def t(x: str) -> str:
    return TR.get(x, x) if LANG == "tr" else x


def num(x: str) -> str:
    """Decimal comma for the Turkish tables (numbers are in text mode, so a plain comma is safe)."""
    return x.replace(".", ",") if LANG == "tr" else x


def cell(df: pd.DataFrame, model: str, frac: float, key: str, bold: bool = False) -> str:
    r = df[(df.model == model) & (df.train_fraction == frac)]
    if r.empty:
        return "--"
    r = r.iloc[0]
    m, s, n = r[f"{key}_iou_mean"], r[f"{key}_iou_std"], int(r.seeds)
    txt = num(f"{m:.3f}") + (f" $\\pm$ {num(f'{s:.3f}')}" if n > 1 else "$^\\dagger$")
    return f"\\textbf{{{txt}}}" if bold else txt


def best_model(df: pd.DataFrame, frac: float, key: str, models: list[str]) -> str | None:
    rows = df[(df.train_fraction == frac) & (df.model.isin(models))]
    if rows.empty:
        return None
    return rows.sort_values(f"{key}_iou_mean", ascending=False).iloc[0].model


def table_full(df: pd.DataFrame, runs: pd.DataFrame) -> str:
    models = [m for m, *_ in MAIN]
    bt, bb = best_model(df, 1.0, "test", models), best_model(df, 1.0, "bolivia", models)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & ".join(t(h) for h in ["Model", "Pretraining", "Trainable", "Test IoU$_w$", "Bolivia IoU$_w$", "GPU min"]) + " \\\\",
        "\\midrule",
    ]
    for m, label, pre, params in MAIN:
        r = runs[(runs.model == m) & (runs.train_fraction == 1.0) & (runs.seed == 0)]
        gpu = num(f"{r.iloc[0].gpu_minutes:.1f}") if not r.empty else "--"
        lines.append(f"{t(label)} & {pre} & {num(params)} & {cell(df, m, 1.0, 'test', m == bt)} & {cell(df, m, 1.0, 'bolivia', m == bb)} & {gpu} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_sweep(df: pd.DataFrame, key: str) -> str:
    models = [m for m, *_ in MAIN if m != "prithvi_full"]
    header = " & ".join(t(label) for m, label, *_ in MAIN if m != "prithvi_full")
    lines = ["\\begin{tabular}{l" + "r" * len(models) + "}", "\\toprule", f"{t('Labels')} & {header} \\\\", "\\midrule"]
    for f in FRACS:
        b = best_model(df, f, key, models)
        chips = round(252 * f) or 3
        cells = " & ".join(cell(df, m, f, key, m == b) for m in models)
        pct = f"\\%{int(f * 100)}" if LANG == "tr" else f"{int(f * 100)}\\,\\%"
        lines.append(f"{pct} ({chips}) & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_ablation(df: pd.DataFrame) -> str:
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", " & ".join(t(h) for h in ["Variant", "Trainable", "Test IoU$_w$", "Bolivia IoU$_w$"]) + " \\\\", "\\midrule"]
    for m, label in ABLATION:
        r = df[(df.model == m) & (df.train_fraction == 1.0)]
        params = num(f"{r.iloc[0].trainable_params / 1e6:.2f}\\,M") if not r.empty else "--"
        lines.append(f"{t(label)} & {params} & {cell(df, m, 1.0, 'test')} & {cell(df, m, 1.0, 'bolivia')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default=None, help="default: paper/tables (en) or paper/tables_tr (tr)")
    ap.add_argument("--lang", choices=["en", "tr"], default="en")
    args = ap.parse_args()
    global LANG
    LANG = args.lang
    sys.stdout.reconfigure(encoding="utf-8")  # Turkish labels on a cp1252 Windows console
    args.out = args.out or ("paper/tables_tr" if LANG == "tr" else "paper/tables")
    df = pd.read_csv(args.summary)
    runs = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_full.tex").write_text(table_full(df, runs) + "\n", encoding="utf-8")
    (out / "table_sweep_test.tex").write_text(table_sweep(df, "test") + "\n", encoding="utf-8")
    (out / "table_sweep_bolivia.tex").write_text(table_sweep(df, "bolivia") + "\n", encoding="utf-8")
    (out / "table_ablation.tex").write_text(table_ablation(df) + "\n", encoding="utf-8")
    n_runs = int(df.seeds.sum())
    (out / "stats.tex").write_text(f"\\newcommand{{\\nruns}}{{{n_runs}}}\n", encoding="utf-8")
    print(f"tables written to {out} ({n_runs} runs)")
    print(table_full(df, runs))


if __name__ == "__main__":
    main()
