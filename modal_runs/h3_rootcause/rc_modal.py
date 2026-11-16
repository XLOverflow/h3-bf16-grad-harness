"""Stage 1 of the H3 bf16 root-cause diagnosis on one B200. Not for merge.

modal run modal_runs/h3_rootcause/rc_modal.py
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO_LOCAL = HERE.parent.parent
REPO = "/root/UniRL"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
OVERRIDES_TXT = "kernels>=0.14.1,<0.15\\ntokenizers>=0.22,<0.23\\ndiffusers==0.40.0\\n"
MODEL_DIR = "/cache/models/minimax-h3"
OUT_DIR = "/cache/h3_grad"
STEPS = 10
SAMPLES = (0, 15, 2)  # prompt 0: median / worst / best per-sample cosine in the 32-sample run
LORA_SHA_32_SAMPLE_RUN = "e64a560aff9017f276bcb46d01f6cff1bb19c3ac0d95e8452496ceb7c390d5d7"
PAIR_STEPS = (7, 2, 5, 8)  # the first one is the deep-diagnosis step (worst weighted per-step cosine)

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
            "PYTHONPATH": f"{REPO}:/root/harness:/root/rootcause",
            "HF_HOME": "/cache/hf",
            "REPORT_TO_WANDB": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(REPO_LOCAL),
        REPO,
        ignore=[".git", "modal_runs", "dev_notes", "**/__pycache__", "*.html", "outputs", "wandb", "assets"],
    )
    .add_local_file(str(HERE.parent / "h3_grad_precision.py"), "/root/harness/h3_grad_precision.py")
    .add_local_dir(str(HERE), "/root/rootcause", ignore=["**/__pycache__"])
)
app = modal.App("unirl-h3-rootcause", image=image)
cache = modal.Volume.from_name("unirl-hf-cache")


@app.function(gpu="B200", cpu=8, memory=98304, timeout=3600, volumes={"/cache": cache}, single_use_containers=True)
def stage1(run_id: str):
    import json
    import os
    import time

    import h3_grad_precision as H
    import rc
    import torch

    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:6.0f}s] {msg}", flush=True)

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    out_dir = f"{OUT_DIR}/rootcause/{run_id}"
    os.makedirs(out_dir)
    dev = torch.device("cuda")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be off"
    embeds = torch.load(f"{OUT_DIR}/embeds_2.pt")[0]

    tr = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    for name, mod in tr.named_children():
        want = torch.float32 if any(k in name for k in H.KEEP_FP32) else torch.bfloat16
        got = {t.dtype for t in list(mod.parameters()) + list(mod.buffers()) if t.dtype.is_floating_point}
        assert got <= {want}, f"after load: {name} is {got}, expected {want}"
    torch.manual_seed(0)  # same LoRA A as the 32-sample run (its sha is recorded there)
    inject_lora(tr, rank=32, alpha=64, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    lora_sha = H.lora_hash(tr)
    assert {str(p.dtype) for _, p in H.lora_params(tr)} == {"torch.bfloat16"}, "LoRA must be injected in bf16"
    assert lora_sha == LORA_SHA_32_SAMPLE_RUN, f"LoRA differs from the 32-sample run: {lora_sha}"
    tr.to(dev).eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    log(f"loaded; lora sha {lora_sha[:8]}")

    params = DiffusionSamplingParams(
        num_inference_steps=STEPS, guidance_scale=1.0, height=256, width=384, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=STEPS, shift=12.0)
    segs = {s: H.rollout(stage, embeds, params, sigmas, geometry, seed=s, key=f"p0s{s}") for s in SAMPLES}
    pairs = [{"sample": s, "k": k, "seg": segs[s], "embeds": embeds} for k in PAIR_STEPS for s in SAMPLES]
    log(f"rollouts done; {len(pairs)} pairs")

    def save(report: dict) -> None:
        with open(f"{out_dir}/stage1.json", "w") as f:
            json.dump(
                {"run_id": run_id, "lora_sha": lora_sha, "gpu": torch.cuda.get_device_name(0), **report},
                f,
                default=float,
            )
        cache.commit()

    report = rc.stage1(
        rc.Env(stage, params),
        pairs,
        save,
        deadline=t_start + 3000.0,
        tail_start=44,
        range_start=30,
        probe_layers=[5, 20, 36],
        log=log,
    )
    log(f"done; sections={report['sections_done']} skipped={report['skipped']} final={report['final_checks']}")
    return {
        "sections": report["sections_done"],
        "skipped": report["skipped"],
        "final": report["final_checks"],
        "controls": report["controls"],
    }


STORED_RUN = "run20261115-201438-d14be6c3"  # the 32-sample run whose fp32 gradients serve as held-out references
HELD_IDS = (16, 17, 18, 19)  # prompt 1; none of them is used for screening


@app.function(
    gpu=["H200", "B200"], cpu=8, memory=98304, timeout=3600, volumes={"/cache": cache}, single_use_containers=True
)
def stage2(run_id: str):
    import json
    import os
    import time

    import h3_grad_precision as H
    import rc
    import torch

    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:6.0f}s] {msg}", flush=True)

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    out_dir = f"{OUT_DIR}/rootcause/{run_id}"
    os.makedirs(out_dir)
    dev = torch.device("cuda")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be off"
    embeds = torch.load(f"{OUT_DIR}/embeds_2.pt")

    tr = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    for name, mod in tr.named_children():
        want = torch.float32 if any(k in name for k in H.KEEP_FP32) else torch.bfloat16
        got = {t.dtype for t in list(mod.parameters()) + list(mod.buffers()) if t.dtype.is_floating_point}
        assert got <= {want}, f"after load: {name} is {got}, expected {want}"
    torch.manual_seed(0)
    inject_lora(tr, rank=32, alpha=64, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    lora_sha = H.lora_hash(tr)
    assert {str(p.dtype) for _, p in H.lora_params(tr)} == {"torch.bfloat16"}, "LoRA must be injected in bf16"
    assert lora_sha == LORA_SHA_32_SAMPLE_RUN, f"LoRA differs from the 32-sample run: {lora_sha}"
    bidx = rc.b_part_index(tr)

    # Stored references for the held-out samples exist only for B200 trajectories: on another GPU type the bf16 rollouts
    # are not the same trajectories, so the held-out comparison is skipped there instead of producing a mismatched number.
    gpu_name = torch.cuda.get_device_name(0)
    use_held = "B200" in gpu_name
    if use_held:
        stored = f"{OUT_DIR}/par/{STORED_RUN}"
        info = json.load(open(f"{stored}/shard2/info.json"))
        assert (
            info["lora_sha"] == lora_sha
            and info["design"]["drop_terminal"]
            and info["design"]["fp32_main"] == "fixed_residual"
        ), info["design"]
        ids = list(torch.load(f"{stored}/shard2/ids.pt"))
        rows = [ids.index(i) for i in HELD_IDS]
        ref_fp32 = torch.load(f"{stored}/shard2/per_sample_fp32.pt")[rows][:, bidx].clone()
        ref_bf16 = torch.load(f"{stored}/shard2/per_sample_bf16.pt")[rows][:, bidx].clone()

    tr.to(dev).eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    log(f"loaded on {gpu_name}; lora sha {lora_sha[:8]}; held-out {'on' if use_held else 'skipped (not a B200)'}")

    params = DiffusionSamplingParams(
        num_inference_steps=STEPS, guidance_scale=1.0, height=256, width=384, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=STEPS, shift=12.0)

    def roll(prompt: int, s: int, key: str):
        return H.rollout(stage, embeds[prompt], params, sigmas, geometry, seed=1000 * prompt + s, key=key)

    segs = {s: roll(0, s, f"p0s{s}") for s in SAMPLES}
    pairs = [{"sample": s, "k": k, "seg": segs[s], "embeds": embeds[0]} for k in PAIR_STEPS for s in SAMPLES]
    held = None
    if use_held:
        held = {
            "samples": [{"id": i, "seg": roll(1, i - 16, f"p1s{i - 16}"), "embeds": embeds[1]} for i in HELD_IDS],
            "ref_fp32": ref_fp32,
            "ref_bf16": ref_bf16,
        }
    log(f"rollouts done; {len(pairs)} screening pairs, held-out samples: {len(HELD_IDS) if use_held else 0}")

    def save(report: dict) -> None:
        with open(f"{out_dir}/stage2.json", "w") as f:
            json.dump(
                {"run_id": run_id, "lora_sha": lora_sha, "gpu": torch.cuda.get_device_name(0), **report},
                f,
                default=float,
            )
        cache.commit()

    report = rc.stage2(
        rc.Env(stage, params),
        pairs,
        save,
        deadline=t_start + 2400.0,
        cut=37,
        sharp=[45, 49],
        held=held,
        rollout_fn=lambda: roll(0, 0, "p0s0"),
        log=log,
    )
    log(f"done; sections={report['sections_done']} skipped={report['skipped']} final={report['final_checks']}")
    return {
        "sections": report["sections_done"],
        "skipped": report["skipped"],
        "final": report["final_checks"],
        "controls": report["controls"],
    }


@app.function(
    gpu=["H200", "B200"], cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache}, single_use_containers=True
)
def stage3(run_id: str):
    import json
    import os
    import time

    import h3_grad_precision as H
    import rc
    import torch

    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:6.0f}s] {msg}", flush=True)

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    out_dir = f"{OUT_DIR}/rootcause/{run_id}"
    os.makedirs(out_dir)
    dev = torch.device("cuda")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be off"
    embeds = torch.load(f"{OUT_DIR}/embeds_2.pt")
    tr = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr, rank=32, alpha=64, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    lora_sha = H.lora_hash(tr)
    assert lora_sha == LORA_SHA_32_SAMPLE_RUN, f"LoRA differs from the 32-sample run: {lora_sha}"
    tr.to(dev).eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    gpu_name = torch.cuda.get_device_name(0)
    log(f"loaded on {gpu_name}")
    params = DiffusionSamplingParams(
        num_inference_steps=STEPS, guidance_scale=1.0, height=256, width=384, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=STEPS, shift=12.0)
    segs = {s: H.rollout(stage, embeds[0], params, sigmas, geometry, seed=s, key=f"p0s{s}") for s in SAMPLES}
    pairs = [{"sample": s, "k": k, "seg": segs[s], "embeds": embeds[0]} for k in PAIR_STEPS for s in SAMPLES]

    def save(report: dict) -> None:
        with open(f"{out_dir}/stage3.json", "w") as f:
            json.dump({"run_id": run_id, "gpu": gpu_name, **report}, f, default=float)
        cache.commit()

    report = rc.stage3_fp16(rc.Env(stage, params), pairs, save, deadline=t_start + 1200.0, log=log)
    log(f"done; sections={report['sections_done']} skipped={report['skipped']} final={report['final_checks']}")
    return {"sections": report["sections_done"], "skipped": report["skipped"], "final": report["final_checks"]}


@app.function(
    gpu=["H200", "B200"], cpu=8, memory=98304, timeout=1800, volumes={"/cache": cache}, single_use_containers=True
)
def stage4(run_id: str):
    import json
    import os
    import time

    import h3_grad_precision as H
    import rc
    import torch

    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    t_start = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:6.0f}s] {msg}", flush=True)

    assert run_id and "/" not in run_id, f"bad run_id {run_id!r}"
    out_dir = f"{OUT_DIR}/rootcause/{run_id}"
    os.makedirs(out_dir)
    dev = torch.device("cuda")
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be off"
    embeds = torch.load(f"{OUT_DIR}/embeds_2.pt")
    tr = MiniMaxH3Transformer3DModel.from_pretrained(MODEL_DIR, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr, rank=32, alpha=64, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    lora_sha = H.lora_hash(tr)
    assert lora_sha == LORA_SHA_32_SAMPLE_RUN, f"LoRA differs from the 32-sample run: {lora_sha}"
    tr.to(dev).eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    gpu_name = torch.cuda.get_device_name(0)
    log(f"loaded on {gpu_name}")
    params = DiffusionSamplingParams(
        num_inference_steps=STEPS, guidance_scale=1.0, height=256, width=384, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=STEPS, shift=12.0)
    segs = {s: H.rollout(stage, embeds[0], params, sigmas, geometry, seed=s, key=f"p0s{s}") for s in SAMPLES}
    pairs = [{"sample": s, "k": k, "seg": segs[s], "embeds": embeds[0]} for k in PAIR_STEPS for s in SAMPLES]

    def save(report: dict) -> None:
        with open(f"{out_dir}/stage4.json", "w") as f:
            json.dump({"run_id": run_id, "gpu": gpu_name, **report}, f, default=float)
        cache.commit()

    report = rc.stage4_scaled_fp16(rc.Env(stage, params), pairs, save, deadline=t_start + 1200.0, cut=37, log=log)
    log(f"done; sections={report['sections_done']} skipped={report['skipped']} final={report['final_checks']}")
    return {"sections": report["sections_done"], "skipped": report["skipped"], "final": report["final_checks"]}


@app.local_entrypoint()
def s4():
    import time
    import uuid

    run_id = time.strftime("rc4-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    print("RUN_ID", run_id)
    print("RESULT", stage4.remote(run_id))


@app.local_entrypoint()
def s3():
    import time
    import uuid

    run_id = time.strftime("rc3-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    print("RUN_ID", run_id)
    print("RESULT", stage3.remote(run_id))


@app.local_entrypoint()
def s2():
    import time
    import uuid

    run_id = time.strftime("rc2-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    print("RUN_ID", run_id)
    print("RESULT", stage2.remote(run_id))


@app.local_entrypoint()
def main():
    import time
    import uuid

    run_id = time.strftime("rc%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    print("RUN_ID", run_id)
    print("RESULT", stage1.remote(run_id))
