"""Level 0: run the H3 gradient-precision harness on a tiny random model, Modal CPU only. Not for merge.

modal run modal_runs/h3_grad_l0_modal.py --flags "--layers 50"
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO_LOCAL = HERE.parent
REPO = "/root/UniRL"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
OVERRIDES_TXT = "kernels>=0.14.1,<0.15\\ntokenizers>=0.22,<0.23\\ndiffusers==0.40.0\\n"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .add_local_file(str(REPO_LOCAL / "pyproject.toml"), "/build/pyproject.toml", copy=True)
    .run_commands(
        f"printf '{OVERRIDES_TXT}' > /build/overrides.txt",
        "cd /build && uv pip install --system "
        "-r pyproject.toml --extra train --extra infer "
        "'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' 'av>=14.2,<19' "
        f"--extra-index-url {TORCH_INDEX} --index-strategy unsafe-best-match "
        "--override /build/overrides.txt",
    )
    .env({"PYTHONPATH": f"{REPO}:/root/harness", "REPORT_TO_WANDB": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        str(REPO_LOCAL),
        REPO,
        ignore=[".git", "modal_runs", "dev_notes", "**/__pycache__", "*.html", "outputs", "wandb", "assets"],
    )
    .add_local_file(str(HERE / "h3_grad_precision.py"), "/root/harness/h3_grad_precision.py")
)
app = modal.App("unirl-h3-grad-l0", image=image)


@app.function(cpu=8, memory=16384, timeout=1500)
def run(flags: str = ""):
    subprocess.run(["python", "/root/harness/h3_grad_precision.py", *flags.split()], check=True)


@app.local_entrypoint()
def main(flags: str = ""):
    run.remote(flags)
