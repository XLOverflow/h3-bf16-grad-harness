"""32-sample fixed batch on real MiniMax-H3 weights: bf16 vs fp32 LoRA gradients, sharded over GPUs. Not for merge.

modal run modal_runs/h3_grad_l1_modal.py::download      # checkpoint -> volume (once)
modal run modal_runs/h3_grad_l1_modal.py::embed         # text embeddings -> volume (once)
modal run modal_runs/h3_grad_l1_modal.py::parsmoke      # one-GPU rehearsal of shard + merge
modal run modal_runs/h3_grad_l1_modal.py::par           # four GPU shards, then a CPU merge
modal run modal_runs/h3_grad_l1_modal.py::stage0c --run-id <run>   # CPU analyses on the stored gradients
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO_LOCAL = HERE.parent
REPO = "/root/UniRL"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
OVERRIDES_TXT = "kernels>=0.14.1,<0.15\\ntokenizers>=0.22,<0.23\\ndiffusers==0.40.0\\n"
MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REV = "0919fb684102ef115126612f3fbb84d1027e7317"
MODEL_DIR = "/cache/models/minimax-h3"
OUT_DIR = "/cache/h3_grad"
PATTERNS = ["transformer/*", "text_encoder/*", "tokenizer/*", "processor/*", "model_index.json"]

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
    .env(
        {
            "PYTHONPATH": f"{REPO}:/root/harness",
            "HF_HOME": "/cache/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "REPORT_TO_WANDB": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(REPO_LOCAL),
        REPO,
        ignore=[".git", "modal_runs", "dev_notes", "**/__pycache__", "*.html", "outputs", "wandb", "assets"],
    )
    .add_local_file(str(HERE / "h3_grad_precision.py"), "/root/harness/h3_grad_precision.py")
)
app = modal.App("unirl-h3-grad-l1", image=image)
cache = modal.Volume.from_name("unirl-hf-cache", create_if_missing=True)


@app.function(cpu=16, memory=32768, timeout=5400, volumes={"/cache": cache})
def download():
    """Public repo, no token: transformer (66 GB) + text encoder (67 GB) + tokenizer/processor."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=MODEL_DIR, allow_patterns=PATTERNS, max_workers=16)
    cache.commit()
    print("downloaded", MODEL_DIR)


PAR_DIR = f"{OUT_DIR}/par"
N_PROMPTS, GROUP, STEPS, N_SHARDS = 2, 16, 10, 4
DESIGN = {
    "n_prompts": N_PROMPTS,
    "group": GROUP,
    "steps": STEPS,
    "adv_seed": 11,
    "lora_seed": 0,
    "drop_terminal": True,
    "fp32_main": "fixed_residual",
    "store": "fp32",
    "version": 4,
}
TENSOR_KEYS = (
    "per_sample_bf16",
    "per_sample_fp32",
    "per_step_bf16",
    "per_step_w_bf16",
    "per_step_fp32",
    "per_step_w_fp32",
    "per_step_fp32own",
    "per_step_w_fp32own",
)


@app.function(gpu=["H200", "B200"], cpu=16, memory=131072, timeout=1800, volumes={"/cache": cache})
def embed(n_prompts: int = N_PROMPTS):
    """Encode the first ``n_prompts`` lines of ``datasets/pickscore/train.txt`` with the H3 text encoder -> ``embeds_<n>.pt``."""
    import os

    import torch

    from unirl.models.minimax_h3.text_embed import encode_minimax_h3_prompt, load_minimax_h3_conditioner

    dev = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)
    prompts = [p.strip() for p in open(f"{REPO}/datasets/pickscore/train.txt") if p.strip()][:n_prompts]
    enc, processor, tokenizer = load_minimax_h3_conditioner(MODEL_DIR, torch.bfloat16)
    enc.to(dev)
    embeds = [
        encode_minimax_h3_prompt(text_encoder=enc, tokenizer=tokenizer, processor=processor, prompt=p, device=dev).cpu()
        for p in prompts
    ]
    path = f"{OUT_DIR}/embeds_{n_prompts}.pt"
    shapes = [tuple(e.shape) for e in embeds]
    if os.path.exists(path):  # never overwrite: every stored gradient was computed from the existing file
        old = torch.load(path)
        assert len(old) == len(embeds) and all(a.shape == b.shape for a, b in zip(old, embeds)), "prompt set changed"
        rel = [float((a.float() - b.float()).norm() / a.float().norm()) for a, b in zip(old, embeds)]
        mx = [float((a.float() - b.float()).abs().max()) for a, b in zip(old, embeds)]
        same = all(torch.equal(a, b) for a, b in zip(old, embeds))
        print(
            f"EMBED kept existing {path} on {torch.cuda.get_device_name(0)}; identical: {same}; relative difference {rel}; max abs {mx}; shapes {shapes}"
        )
    else:
        torch.save(embeds, path)
        cache.commit()
        print(f"EMBED wrote {path}; shapes {shapes}")


def _source_hash() -> str:
    import hashlib

    h = hashlib.sha256()
    for path in ("/root/harness/h3_grad_precision.py", __file__):
        h.update(open(path, "rb").read())
    return h.hexdigest()


def _shard(idx: int, run_id: str, ids: list, n_shards: int):
    """Rollouts + bf16 and fp32 gradients for the global sample indices ``ids``; tensors go to ``par/<run_id>/shard<idx>/``."""
    import hashlib
    import json
    import os
    import time

    import h3_grad_precision as H
    import torch

    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[shard {idx} {time.perf_counter() - t_start:6.0f}s] {msg}", flush=True)

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    out_dir = f"{PAR_DIR}/{run_id}/shard{idx}"
    assert not os.path.exists(out_dir), f"{out_dir} already exists: run_id collision"
    os.makedirs(out_dir)
    dev = torch.device("cuda")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be off: the fp32 arm is the reference"
    emb_path = f"{OUT_DIR}/embeds_{N_PROMPTS}.pt"
    assert os.path.exists(emb_path), f"{emb_path} missing"
    embeds = torch.load(emb_path)
    embeds_hash = hashlib.sha256(b"".join(e.float().numpy().tobytes() for e in embeds)).hexdigest()

    tr = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)

    def check_layout(when: str) -> None:
        for name, mod in tr.named_children():
            want = torch.float32 if any(k in name for k in H.KEEP_FP32) else torch.bfloat16
            got = {
                t.dtype
                for n, t in list(mod.named_parameters()) + list(mod.named_buffers())
                if t.dtype.is_floating_point and "lora_" not in n
            }
            assert got <= {want}, f"{when}: {name} is {got}, expected {want}"

    # The rollout runs on this layout untouched, so it must already be the bf16 arm's layout.
    check_layout("after load")
    torch.manual_seed(DESIGN["lora_seed"])  # PEFT draws lora_A from the global RNG; every shard must hold the same A
    inject_lora(tr, rank=32, alpha=64, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    check_layout("after LoRA injection")
    lora_dtypes = sorted({str(p.dtype) for _, p in H.lora_params(tr)})
    lora_sha = H.lora_hash(tr)
    tr.to(dev).eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    check_layout("on device")
    state_before = H.state_fingerprint(tr)
    log(f"loaded; lora dtypes={lora_dtypes}")

    params = DiffusionSamplingParams(
        num_inference_steps=STEPS, guidance_scale=1.0, height=256, width=384, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=STEPS, shift=12.0)
    assert float(sigmas[-1]) == 0.0 and sigmas.numel() == STEPS + 1, f"unexpected sigma schedule {sigmas.tolist()}"
    samples = []
    for i in ids:
        p, s = divmod(i, GROUP)
        samples.append(
            (embeds[p], H.rollout(stage, embeds[p], params, sigmas, geometry, seed=1000 * p + s, key=f"p{p}s{s}"))
        )
    log(f"rollouts done: ids={ids}")

    def put(key: str, value) -> None:
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all()), (
                f"{key}: {value.dtype} or non-finite"
            )
        torch.save(value, f"{out_dir}/{key}.pt")

    adv = H.random_advantages(N_PROMPTS, GROUP, 1, seed=DESIGN["adv_seed"])[0]
    torch.cuda.reset_peak_memory_stats()
    info = H.shard_payload(stage, params, samples, ids, adv, put, acts=(idx == 0), log=log)
    H.cast_arm(tr, "bf16")
    moved = H.fingerprints_match(state_before, H.state_fingerprint(tr))
    assert not moved, f"model state changed across the arms: {moved}"
    info.update(
        {
            "idx": idx,
            "n_shards": n_shards,
            "run_id": run_id,
            "design": DESIGN,
            "lora_sha": lora_sha,
            "lora_dtypes": lora_dtypes,
            "embeds_sha": embeds_hash,
            "source_sha": _source_hash(),
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "total_seconds": time.perf_counter() - t_start,
        }
    )
    with open(f"{out_dir}/info.json", "w") as f:  # written last: its presence marks a complete shard
        json.dump(info, f, default=float)
    cache.commit()
    log(f"saved; peak {info['peak_gb']:.1f} GB")
    return {k: info[k] for k in ("idx", "seconds_bf16", "seconds_fp32", "peak_gb", "total_seconds", "lora_dtypes")}


def _merge(run_id: str, n_shards: int, n_prompts: int, group: int):
    import json
    import resource

    import h3_grad_precision as H
    import torch

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    run_dir = f"{PAR_DIR}/{run_id}"
    cache.reload()  # see the shards committed by other containers even if this container was reused
    torch.set_num_threads(8)
    infos = [json.load(open(f"{run_dir}/shard{idx}/info.json")) for idx in range(n_shards)]
    src = _source_hash()
    for i in infos:
        assert i["run_id"] == run_id and i["design"] == DESIGN and i["n_shards"] == n_shards, (
            f"shard {i['idx']}: other run/design"
        )
        assert i["source_sha"] == src, f"shard {i['idx']} was produced by different harness source"
        for k in ("lora_sha", "embeds_sha", "torch", "gpu"):
            assert i[k] == infos[0][k], f"shards disagree on {k}: {i[k]} vs {infos[0][k]}"

    def get(idx: int, key: str):
        return torch.load(f"{run_dir}/shard{idx}/{key}.pt")

    adv = H.random_advantages(N_PROMPTS, GROUP, 1, seed=DESIGN["adv_seed"])[0]
    adv_draws = H.random_advantages(N_PROMPTS, GROUP, 500, seed=12)
    if (n_prompts, group) != (N_PROMPTS, GROUP):  # smoke: the first n_prompts*group samples of the full design
        sel = [p * GROUP + s for p in range(n_prompts) for s in range(group)]
        adv, adv_draws = adv[sel], adv_draws[:, sel]
    report = {"tag": "full_par4", "run_id": run_id, "design": DESIGN, "n_shards": n_shards, "shards": infos}
    report.update(H.merge_payloads(get, n_shards, n_prompts, group, adv, adv_draws))
    report["merge_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    with open(f"{run_dir}/report.json", "w") as f:
        json.dump(report, f, default=float)
    cache.commit()
    print("REPORT " + json.dumps(report, default=float))
    assert report["checks_passed"], "self-checks failed: do not use this report"


@app.function(gpu="B200", cpu=8, memory=98304, timeout=3600, volumes={"/cache": cache}, single_use_containers=True)
def shard(idx: int, run_id: str):
    """One quarter of the fixed batch."""
    total = N_PROMPTS * GROUP
    return _shard(idx, run_id, [i for i in range(total) if i * N_SHARDS // total == idx], N_SHARDS)


@app.function(cpu=8, memory=131072, timeout=2400, volumes={"/cache": cache})
def merge(run_id: str):
    """Merge one run's shards and write ``par/<run_id>/report.json``; shard files are left for manual removal."""
    _merge(run_id, N_SHARDS, N_PROMPTS, GROUP)


@app.function(gpu="B200", cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache}, single_use_containers=True)
def smoke_shards(run_id: str):
    """GPU rehearsal of the exact shard code on samples 0 and 1 as two one-sample shards, in one container."""
    import gc

    import torch

    out = []
    for idx, ids in enumerate([[0], [1]]):
        out.append(_shard(idx, run_id, ids, 2))
        gc.collect()
        torch.cuda.empty_cache()
    return out


@app.function(cpu=8, memory=98304, timeout=1200, volumes={"/cache": cache})
def smoke_merge(run_id: str):
    _merge(run_id, 2, 1, 2)


def _run_id() -> str:
    import time
    import uuid

    return time.strftime("run%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


@app.local_entrypoint()
def parsmoke():
    """One B200: two one-sample shards through the real code path, then the real merge."""
    run_id = _run_id()
    print("RUN_ID", run_id)
    print("SHARDS", smoke_shards.remote(run_id))
    smoke_merge.remote(run_id)


@app.local_entrypoint()
def par():
    """Four B200 shards in parallel, then one CPU merge."""
    run_id = _run_id()
    print("RUN_ID", run_id)
    print("SHARDS", list(shard.starmap([(i, run_id) for i in range(N_SHARDS)])))
    merge.remote(run_id)


@app.function(cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache})
def stage0(run_id: str):
    """Free analysis of the stored 32-sample gradients: Gram matrices, advantage-sum cancellation, same-prompt differences."""
    import itertools
    import json

    import h3_grad_precision as H
    import torch

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    run_dir = f"{PAR_DIR}/{run_id}"
    torch.set_num_threads(8)
    total = N_PROMPTS * GROUP

    def stacked(key: str) -> torch.Tensor:
        out = None
        for k in range(N_SHARDS):
            ids, t = torch.load(f"{run_dir}/shard{k}/ids.pt"), torch.load(f"{run_dir}/shard{k}/{key}.pt")
            if out is None:
                out = torch.zeros(total, t.shape[1], dtype=torch.float32)
            out[ids] = t
        return out

    b, r = stacked("per_sample_bf16"), stacked("per_sample_fp32")
    gbb, gbr, grr = H.gram(b, b), H.gram(b, r), H.gram(r, r)
    del b, r
    adv = H.random_advantages(N_PROMPTS, GROUP, 1, seed=DESIGN["adv_seed"])[0]
    draws = H.random_advantages(N_PROMPTS, GROUP, 500, seed=12)

    def kappa(g: torch.Tensor, a: torch.Tensor) -> float:
        return float((a.abs() * g.diagonal().sqrt()).sum() / (a @ g @ a).sqrt())

    rep = {
        "run_id": run_id,
        "kappa_fp32": kappa(grr, adv),
        "kappa_bf16": kappa(gbb, adv),
        "kappa_fp32_over_draws_median": float(torch.tensor([kappa(grr, d) for d in draws]).median()),
        "kappa_uniform_weights_fp32": kappa(grr, torch.ones(total, dtype=torch.float64)),
    }
    # Pairwise cosine of the fp32 per-sample gradients themselves: how alike are samples?
    nr = grr.diagonal().sqrt()
    cr = grr / (nr[:, None] * nr[None, :])
    same = [(i, j) for i, j in itertools.combinations(range(total), 2) if i // GROUP == j // GROUP]
    cross = [(i, j) for i, j in itertools.combinations(range(total), 2) if i // GROUP != j // GROUP]
    rep["fp32_pair_cos_same_prompt_mean"] = float(sum(cr[i, j] for i, j in same) / len(same))
    rep["fp32_pair_cos_cross_prompt_mean"] = float(sum(cr[i, j] for i, j in cross) / len(cross))
    # Same-prompt differences g_i - g_j: bf16 against fp32.
    cos_d, rel_d = [], []
    for i, j in same:
        nb = gbb[i, i] + gbb[j, j] - 2 * gbb[i, j]
        nrr = grr[i, i] + grr[j, j] - 2 * grr[i, j]
        dot = gbr[i, i] + gbr[j, j] - gbr[i, j] - gbr[j, i]
        cos_d.append(float(dot / (nb * nrr).sqrt()))
        rel_d.append(float(((nb + nrr - 2 * dot).clamp_min(0) / nrr).sqrt()))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    rep["same_prompt_diff_cos"] = {"p10": q(cos_d, 0.1), "median": q(cos_d, 0.5), "p90": q(cos_d, 0.9), "n": len(cos_d)}
    rep["same_prompt_diff_rel_err"] = {"p10": q(rel_d, 0.1), "median": q(rel_d, 0.5), "p90": q(rel_d, 0.9)}
    rel_s = [float(((gbb[i, i] + grr[i, i] - 2 * gbr[i, i]).clamp_min(0) / grr[i, i]).sqrt()) for i in range(total)]
    rep["per_sample_rel_err"] = {"p10": q(rel_s, 0.1), "median": q(rel_s, 0.5), "p90": q(rel_s, 0.9)}
    rep["per_sample_norm_ratio_median"] = q([float((gbb[i, i] / grr[i, i]).sqrt()) for i in range(total)], 0.5)
    torch.save({"gbb": gbb, "gbr": gbr, "grr": grr}, f"{run_dir}/gram.pt")
    with open(f"{run_dir}/stage0.json", "w") as f:
        json.dump(rep, f)
    cache.commit()
    print("STAGE0 " + json.dumps(rep))


@app.function(cpu=2, memory=2048, timeout=300, volumes={"/cache": cache})
def stage0b(run_id: str):
    """Does bf16 preserve the EXPECTED gradient? Off-diagonal Gram means cancel per-sample noise (uses gram.pt from stage0)."""
    import json

    import torch

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    g = torch.load(f"{PAR_DIR}/{run_id}/gram.pt")
    gbb, gbr, grr = g["gbb"], g["gbr"], g["grr"]
    n = gbb.shape[0]
    grp = torch.arange(n) // GROUP
    off = ~torch.eye(n, dtype=torch.bool)
    masks = {
        "all_pairs": off,
        "same_prompt": off & (grp[:, None] == grp[None, :]),
        "cross_prompt": grp[:, None] != grp[None, :],
    }
    rep = {
        "diag_mean": {
            "bb": float(gbb.diagonal().mean()),
            "br": float(gbr.diagonal().mean()),
            "rr": float(grr.diagonal().mean()),
        }
    }
    for k, m in masks.items():
        bb, br, rr = float(gbb[m].mean()), float(gbr[m].mean()), float(grr[m].mean())
        rep[k] = {
            "mean_bb": bb,
            "mean_br": br,
            "mean_rr": rr,
            "cos_expected_grad_bf16_vs_fp32": br / (bb * rr) ** 0.5 if bb > 0 and rr > 0 else None,
            "bf16_projection_on_fp32_expected": br / rr if rr > 0 else None,
            "expected_power_fraction_fp32": rr / float(grr.diagonal().mean()),
            "expected_power_fraction_bf16": bb / float(gbb.diagonal().mean()),
        }
    # Sampling error of those estimates: leave-one-sample-out jackknife on the all-pairs cosine.
    vals = []
    for i in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[i] = False
        m = off & keep[:, None] & keep[None, :]
        vals.append(float(gbr[m].mean() / (gbb[m].mean() * grr[m].mean()).clamp_min(1e-300).sqrt()))
    v = torch.tensor(vals, dtype=torch.float64)
    rep["jackknife_all_pairs_cos"] = {
        "mean": float(v.mean()),
        "se": float(((n - 1) / n * ((v - v.mean()) ** 2).sum()).sqrt()),
        "min": float(v.min()),
        "max": float(v.max()),
    }
    print("STAGE0B " + json.dumps(rep))


@app.function(cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache})
def stage0c(run_id: str):
    """Deep-only LoRA on the stored 32-sample gradients: every metric restricted to the LoRA of blocks >= a cut."""
    import itertools
    import json

    import h3_grad_precision as H
    import torch

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    run_dir = f"{PAR_DIR}/{run_id}"
    torch.set_num_threads(8)
    total = N_PROMPTS * GROUP
    slices = torch.load(f"{run_dir}/shard0/slices.pt")

    def stacked(key: str) -> torch.Tensor:
        out = None
        for k in range(N_SHARDS):
            ids, t = torch.load(f"{run_dir}/shard{k}/ids.pt"), torch.load(f"{run_dir}/shard{k}/{key}.pt")
            if out is None:
                out = torch.zeros(total, t.shape[1], dtype=torch.float32)
            out[ids] = t
        return out

    b, r = stacked("per_sample_bf16"), stacked("per_sample_fp32")
    adv = H.random_advantages(N_PROMPTS, GROUP, 1, seed=DESIGN["adv_seed"])[0]
    draws = H.random_advantages(N_PROMPTS, GROUP, 500, seed=12)
    same = [(i, j) for i, j in itertools.combinations(range(total), 2) if i // GROUP == j // GROUP]
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    sets = {
        "all_50": list(range(50)),
        "blocks_30+": list(range(30, 50)),
        "blocks_37+": list(range(37, 50)),
        "blocks_37-48": list(range(37, 49)),
        "blocks_40+": list(range(40, 50)),
        "blocks_40-48": list(range(40, 49)),
        "blocks_0-36": list(range(0, 37)),
    }
    rep = {}
    for name, blocks in sets.items():
        spans = [s for blk in blocks for s in slices[blk]]
        bb = torch.cat([b[:, a:c] for a, c in spans], dim=1)
        rr = torch.cat([r[:, a:c] for a, c in spans], dim=1)
        gbb, gbr, grr = H.gram(bb, bb), H.gram(bb, rr), H.gram(rr, rr)
        del bb, rr
        num = torch.einsum("di,ij,dj->d", draws, gbr, draws)
        den = (torch.einsum("di,ij,dj->d", draws, gbb, draws) * torch.einsum("di,ij,dj->d", draws, grr, draws)).sqrt()
        c = (num / den).sort().values
        ps = [float(gbr[i, i] / (gbb[i, i] * grr[i, i]).sqrt()) for i in range(total)]
        rel = [float(((gbb[i, i] + grr[i, i] - 2 * gbr[i, i]).clamp_min(0) / grr[i, i]).sqrt()) for i in range(total)]
        dcos = []
        for i, j in same:
            nb, nr = gbb[i, i] + gbb[j, j] - 2 * gbb[i, j], grr[i, i] + grr[j, j] - 2 * grr[i, j]
            dcos.append(float((gbr[i, i] + gbr[j, j] - gbr[i, j] - gbr[j, i]) / (nb * nr).sqrt()))
        a = adv
        rep[name] = {
            "n_params": int(sum(c2 - a2 for a2, c2 in spans)),
            "batch_cos": float((a @ gbr @ a) / ((a @ gbb @ a) * (a @ grr @ a)).sqrt()),
            "batch_cos_draws": {
                "p10": float(c[len(c) // 10]),
                "median": float(c[len(c) // 2]),
                "p90": float(c[-(len(c) // 10) - 1]),
            },
            "batch_rel_err": float(
                ((a @ gbb @ a) + (a @ grr @ a) - 2 * (a @ gbr @ a)).clamp_min(0).sqrt() / (a @ grr @ a).sqrt()
            ),
            "per_sample_cos": {"min": min(ps), "median": q(ps, 0.5), "max": max(ps)},
            "per_sample_rel_err_median": q(rel, 0.5),
            "same_prompt_diff_cos_median": q(dcos, 0.5),
            "fp32_grad_power_share": float(grr.diagonal().sum()),
        }
    tot = rep["all_50"]["fp32_grad_power_share"]
    for v in rep.values():
        v["fp32_grad_power_share"] /= tot
    with open(f"{run_dir}/stage0c.json", "w") as f:
        json.dump(rep, f)
    cache.commit()
    print("STAGE0C " + json.dumps(rep))


@app.function(cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache})
def stage0d(run_id: str):
    """First Adam update on the stored 32-sample gradients: with eps = 1e-12 it is lr * sign(g), so compare signs per block set."""
    import json

    import h3_grad_precision as H
    import torch

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    run_dir = f"{PAR_DIR}/{run_id}"
    torch.set_num_threads(8)
    total = N_PROMPTS * GROUP
    slices = torch.load(f"{run_dir}/shard0/slices.pt")

    def stacked(key: str) -> torch.Tensor:
        out = None
        for k in range(N_SHARDS):
            ids, t = torch.load(f"{run_dir}/shard{k}/ids.pt"), torch.load(f"{run_dir}/shard{k}/{key}.pt")
            if out is None:
                out = torch.zeros(total, t.shape[1], dtype=torch.float32)
            out[ids] = t
        return out

    b, r = stacked("per_sample_bf16"), stacked("per_sample_fp32")
    draws = torch.cat(
        [
            H.random_advantages(N_PROMPTS, GROUP, 1, seed=DESIGN["adv_seed"]),
            H.random_advantages(N_PROMPTS, GROUP, 4, seed=12),
        ]
    )
    sets = {
        "all_50": list(range(50)),
        "blocks_0-36": list(range(37)),
        "blocks_37-48": list(range(37, 49)),
        "block_49": [49],
    }
    spans = {name: [s for blk in blocks for s in slices[blk]] for name, blocks in sets.items()}
    rep = {name: {"raw_cos": [], "adam_first_step_cos": [], "sign_agreement": []} for name in sets}
    for a in draws:
        gb, gr = H.weighted(b, a).float(), H.weighted(r, a).float()
        for name, sp in spans.items():
            x, y = H.take(gb, sp), H.take(gr, sp)
            keep = y != 0  # lora_A entries have zero gradient at init in both arms
            x, y = x[keep], y[keep]
            sx, sy = torch.sign(x), torch.sign(y)
            rep[name]["raw_cos"].append(H.cos(x, y))
            rep[name]["adam_first_step_cos"].append(H.cos(sx, sy))
            rep[name]["sign_agreement"].append(float((sx == sy).float().mean()))
            rep[name]["n_live_params"] = int(keep.sum())
            rep[name]["raw_norm_share_fp32"] = float(y.double().pow(2).sum())
    tot_n, tot_e = rep["all_50"]["n_live_params"], rep["all_50"]["raw_norm_share_fp32"]
    for v in rep.values():
        v["adam_update_share"] = v["n_live_params"] / tot_n  # sign update: every live parameter moves by the same lr
        v["raw_norm_share_fp32"] /= tot_e
        for k in ("raw_cos", "adam_first_step_cos", "sign_agreement"):
            v[k + "_mean"] = sum(v[k]) / len(v[k])
    # Per-sample (batch of one) signs for reference: how much does averaging 32 samples help the sign?
    one = {name: [] for name in sets}
    for i in range(0, total, 4):
        for name, sp in spans.items():
            x, y = H.take(b[i], sp), H.take(r[i], sp)
            keep = y != 0
            one[name].append(H.cos(torch.sign(x[keep]), torch.sign(y[keep])))
    for name in sets:
        rep[name]["adam_first_step_cos_single_sample_mean"] = sum(one[name]) / len(one[name])
    with open(f"{run_dir}/stage0d.json", "w") as f:
        json.dump(rep, f)
    cache.commit()
    print("STAGE0D " + json.dumps(rep))
