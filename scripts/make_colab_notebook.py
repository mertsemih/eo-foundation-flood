"""Generate notebooks/colab_run_matrix.ipynb: run a slice of the experiment matrix on Google Colab.

    python scripts/make_colab_notebook.py

Open the notebook in Colab (File > Open notebook > GitHub > mertsemih/eo-foundation-flood), pick a
GPU runtime, edit the parameters cell, Runtime > Run all. Results (config.yaml, metrics.csv,
test_metrics.json per run, plus checkpoints) persist in Google Drive, so a disconnected session
can simply be re-run: finished runs are skipped.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# eo-foundation-flood — run experiments on Colab

1. **Runtime > Change runtime type > GPU** (T4 is enough for U-Nets and Prithvi LoRA; use A100/L4 for `prithvi_full`).
2. Edit the **Parameters** cell.
3. **Runtime > Run all**. Approve the Google Drive mount when asked.

Results land in `MyDrive/eo-foundation-flood/runs/<run name>/`. If the session disconnects, just run all again: finished runs are skipped.
""",
    ),
    (
        "code",
        """#@title Parameters { display-mode: "form" }
MODELS = "unet_scratch unet_imagenet"       #@param {type:"string"}
FRACTIONS = "1.0 0.25 0.10 0.05"            #@param {type:"string"}
SEEDS = "1 2"                               #@param {type:"string"}
EXTRA = ""                                  #@param {type:"string"}
INSTALL_FOUNDATION = False                  #@param {type:"boolean"}
REPO = "https://github.com/mertsemih/eo-foundation-flood.git"
BRANCH = "main"
DRIVE_DIR = "/content/drive/MyDrive/eo-foundation-flood" """,
    ),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    (
        "code",
        """from google.colab import drive
drive.mount("/content/drive")
import os
os.makedirs(f"{DRIVE_DIR}/runs", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/data", exist_ok=True)
print("drive ok:", DRIVE_DIR)""",
    ),
    (
        "code",
        """import os, shutil, subprocess
WORK = "/content/eo-foundation-flood"
if not os.path.exists(os.path.join(WORK, "pyproject.toml")):
    shutil.rmtree(WORK, ignore_errors=True)   # leftovers from a failed clone
    r = subprocess.run(["git", "clone", "--branch", BRANCH, REPO, WORK], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("git clone failed (is the repo public?):
" + r.stderr)
os.chdir(WORK)
subprocess.run(["git", "pull", "-q"], check=False)
assert os.path.exists("configs/unet_scratch.yaml"), "clone failed: " + os.getcwd()
!pip install -q -e ".[dev]" 2>&1 | tail -1
if INSTALL_FOUNDATION:
    !pip install -q terratorch peft 2>&1 | tail -1
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cwd", os.getcwd())""",
    ),
    (
        "code",
        """import os; os.chdir("/content/eo-foundation-flood")
# Data: keep a copy in Drive so later sessions skip the download (1.75 GB, ~2 min first time).
!python scripts/download_sen1floods11.py --root "$DRIVE_DIR/data/sen1floods11" --workers 16
!rm -rf data && mkdir -p data && ln -s "$DRIVE_DIR/data/sen1floods11" data/sen1floods11
!ls data/sen1floods11/v1.1/splits/flood_handlabeled/""",
    ),
    (
        "code",
        """import os; os.chdir("/content/eo-foundation-flood")
# Band statistics: reuse the committed ones (configs already contain them); recompute only if missing.
import yaml
cfg = yaml.safe_load(open("configs/unet_scratch.yaml"))
if not cfg["data"].get("mean"):
    !python scripts/compute_stats.py --root data/sen1floods11 && python scripts/apply_stats.py
else:
    print("configs already carry band statistics")""",
    ),
    (
        "code",
        """import os; os.chdir("/content/eo-foundation-flood")
# runs/ lives in Drive so results survive disconnects and finished runs are skipped on re-run.
!rm -rf runs && ln -s "$DRIVE_DIR/runs" runs
!python scripts/run_matrix.py --models $MODELS --fractions $FRACTIONS --seeds $SEEDS $EXTRA""",
    ),
    (
        "code",
        """import os; os.chdir("/content/eo-foundation-flood")
# Summary + a small zip (no checkpoints) to bring back to the local repo:  unzip into <repo>/runs/
!python scripts/collect_results.py --runs runs --out "$DRIVE_DIR/results/results.csv"
!cd runs && zip -q -r "$DRIVE_DIR/results/runs_small.zip" */config.yaml */metrics.csv */test_metrics.json
print("zip:", f"{DRIVE_DIR}/results/runs_small.zip")""",
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    out = Path("notebooks") / "colab_run_matrix.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
