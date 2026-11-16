"""Render the figures on a CPU container and write the PNGs next to the results.

modal run modal_runs/h3_rootcause/figures_modal.py
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("matplotlib==3.10.0")
    .add_local_dir(str(HERE / "results"), "/work/results")
    .add_local_file(str(HERE / "make_figures.py"), "/work/make_figures.py")
)
app = modal.App("unirl-h3-rootcause-figures", image=image)


@app.function(cpu=2, memory=2048, timeout=300)
def render() -> dict:
    import subprocess

    subprocess.run(["python", "/work/make_figures.py", "--results", "/work/results", "--out", "/work/figures"], check=True)
    return {p.name: p.read_bytes() for p in Path("/work/figures").glob("*.png")}


@app.local_entrypoint()
def main():
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    for name, data in render.remote().items():
        (out / name).write_bytes(data)
        print("wrote", name, len(data))
