"""Draw the figures from the JSON reports in ``results/`` (matplotlib only).

python make_figures.py --results results --out figures
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = {"light": "#86b6ef", "mid": "#2a78d6", "dark": "#104281"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
    }
)


def load(results: str, name: str) -> dict:
    with open(os.path.join(results, name)) as f:
        return json.load(f)


def deep_band(ax, lo: float = 36.5, hi: float = 48.5, label: bool = True) -> None:
    """Shade blocks 37-48, the range the deep-only suggestion uses."""
    ax.axvspan(lo, hi, color="#f0efec", zorder=0, lw=0)
    if label:
        ax.text((lo + hi) / 2, ax.get_ylim()[0], "blocks 37–48", ha="center", va="bottom", fontsize=9, color=INK2)


def fig1(results: str, out: str) -> None:
    s4, s2 = load(results, "stage4_scaled_fp16_layout.json"), load(results, "stage2_layouts.json")
    x = list(range(50))
    series = [
        ("plain bf16", s4["bf16_baseline"]["block_cos_mean"], BLUE, "-", "o"),
        ("fp32 stream / norm / AdaLN, bf16 matmuls", s4["variants"]["peer_layout_with_bf16_instead_of_fp16__2^16"]["block_cos_mean"], ORANGE, "--", "s"),
        ("same placement, scaled fp16 matmuls, loss scale 2^16", s4["variants"]["peer_fp16_layout__loss_scale_2^16"]["block_cos_mean"], AQUA, "-.", "^"),
        ("blocks 0–36 fp32, 37–49 bf16 with fp32 attention", s2["variants"]["D2_shallow_fp32__deep_bf16_attention_fp32"]["block_cos_mean"], YELLOW, ":", "D"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=150)
    ax.set_ylim(-0.12, 1.08)
    deep_band(ax)
    ax.axhline(1.0, color=INK2, lw=1.0, ls=(0, (1, 2)))
    ax.text(0.2, 1.015, "all fp32 (reference)", fontsize=9, color=INK2, va="bottom")
    for name, y, c, ls, m in series:
        ax.plot(x, y, color=c, ls=ls, marker=m, label=name, markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("cosine with the fp32 LoRA gradient")
    ax.set_title("LoRA gradient vs fp32, per block — same trajectories, same output cotangent")
    ax.legend(loc="upper center", fontsize=9.5, bbox_to_anchor=(0.5, -0.14), ncol=2, columnspacing=2.0)
    fig.text(0.01, 0.01, "Mean over 12 (sample, step) pairs, LoRA at initialisation, single H200.\nSource: results/stage2_layouts.json, results/stage4_scaled_fp16_layout.json",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig1_per_block_cosine_by_layout.png"))
    plt.close(fig)


def fig2(results: str, out: str) -> None:
    s1 = load(results, "stage1_mechanism.json")
    drift = [r["drift_bf16_vs_fp32"]["all"] for r in s1["stream_stats"] if r["layer"] < 50]
    local = [r["lora_bf16_vs_fp32_on_rounded_io"]["rel"] for r in s1["local_fp32_state"] if r["layer"] < 50]
    total = s1["bf16_baseline"]["block_rel_mean"]
    x = list(range(50))
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.6, 7.0), dpi=150, sharex=True)
    a.set_yscale("log")
    a.plot(x, drift, color=BLUE, marker="o", markeredgecolor=SURFACE, markeredgewidth=0.6)
    a.set_ylim(1e-3, 0.5)
    deep_band(a, label=False)
    a.set_ylabel("relative difference of the\nresidual stream entering the block")
    a.set_title("bf16 forward drifts with depth; the gradient error is far larger than any single block's own error")
    a.annotate("0.5 % in the early blocks", (8, drift[8]), (9, 0.0018), fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    a.annotate("20 % by block 40", (40, drift[40]), (27.5, 0.3), fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    b.set_yscale("log")
    b.plot(x, total, color=BLUE, marker="o", label="full bf16 run", markeredgecolor=SURFACE, markeredgewidth=0.6)
    b.plot(x, local, color=ORANGE, ls="--", marker="s", label="this block alone in bf16, fp32 input and incoming cotangent", markeredgecolor=SURFACE, markeredgewidth=0.6)
    b.set_ylim(5e-3, 4)
    deep_band(b)
    b.set_xlabel("transformer block")
    b.set_ylabel("relative error of the block's\nLoRA gradient vs fp32")
    b.legend(loc="center left", fontsize=9.5, bbox_to_anchor=(0.02, 0.50))
    fig.text(0.01, 0.01, "Top and dashed line: one (sample, step) pair. Solid bottom line: mean over 12 pairs. Single B200. Source: results/stage1_mechanism.json",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig2_forward_drift_and_gradient_error.png"))
    plt.close(fig)


def fig3(results: str, out: str) -> None:
    d = load(results, "batch32_stage0c_block_ranges.json")
    rows = [("all 50 blocks", "all_50"), ("blocks 0–36", "blocks_0-36"), ("blocks 30–49", "blocks_30+"), ("blocks 37–49", "blocks_37+"), ("blocks 37–48", "blocks_37-48")]
    vals = [d[k]["batch_cos"] for _, k in rows]
    lo = [d[k]["batch_cos"] - d[k]["batch_cos_draws"]["p10"] for _, k in rows]
    hi = [d[k]["batch_cos_draws"]["p90"] - d[k]["batch_cos"] for _, k in rows]
    lo, hi = [max(v, 0.0) for v in lo], [max(v, 0.0) for v in hi]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    xs = range(len(rows))
    ax.bar(xs, vals, width=0.58, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + hi[i] + 0.03, f"{v:.2f}", ha="center", fontsize=10.5, color=INK)
    ax.axhline(1.0, color=INK2, lw=1.0, ls=(0, (1, 2)))
    ax.set_xticks(list(xs), [r for r, _ in rows])
    ax.set_ylim(-0.08, 1.12)
    ax.set_ylabel("batch cosine, bf16 vs fp32")
    ax.set_title("Which LoRA blocks have an accurate bf16 gradient")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "32 samples x 9 steps, group-centred random advantages; whiskers: 10–90 % over 500 advantage draws.\nSource: results/batch32_stage0c_block_ranges.json",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "fig3_cosine_by_block_range.png"))
    plt.close(fig)


def fig4(results: str, out: str) -> None:
    s1 = load(results, "stage1_mechanism.json")
    x = list(range(50))
    noise = [("fp32 + 1e-4 relative noise on the stream", "eps0.0001_seed0", BLUE_RAMP["light"], "o"),
             ("fp32 + 1e-3", "eps0.001_seed0", BLUE_RAMP["mid"], "s"),
             ("fp32 + 4e-3", "eps0.004_seed0", BLUE_RAMP["dark"], "^")]
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=150)
    ax.set_yscale("log")
    ax.set_ylim(0.015, 4)
    deep_band(ax)
    for name, key, c, m in noise:
        ax.plot(x, s1["noise"][key]["block_rel_mean"], color=c, marker=m, label=name, markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax.plot(x, s1["bf16_baseline"]["block_rel_mean"], color=ORANGE, ls="--", marker="D", label="bf16 (no added noise)", markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("relative error of the block's\nLoRA gradient vs fp32")
    ax.set_title("A small value-only perturbation in pure fp32 damages the same blocks as bf16")
    ax.legend(loc="lower left", fontsize=9.5, bbox_to_anchor=(0.02, 0.06))
    fig.text(0.01, 0.01, "Noise added after every block, not differentiated; 3 pairs. bf16: mean over 12 pairs. Single B200. Source: results/stage1_mechanism.json",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out, "fig4_noise_baseline.png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f(args.results, args.out)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
