"""Shared helpers for the MiniMax-H3 bf16 vs fp32 LoRA-gradient measurements (verification harness, not for merge)."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from typing import Dict, List, Sequence, Tuple

import torch

from unirl.models.minimax_h3.bundle import MiniMaxH3Bundle
from unirl.models.minimax_h3.conditions import MiniMaxH3Conditions
from unirl.models.minimax_h3.config import MINIMAX_H3_AUDIO_LATENT_CHANNELS
from unirl.models.minimax_h3.diffusion import MiniMaxH3DiffusionStage
from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
from unirl.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
from unirl.sde.kernels import CPSSDEStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.train.lora import inject_lora
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
LORA_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
KEEP_FP32 = tuple(MiniMaxH3Transformer3DModel._keep_in_fp32_modules)


def cast_arm(transformer: torch.nn.Module, arm: str) -> None:
    """Cast per child; keep-fp32 children (true fp32 weights, rope ``inv_freq``) never pass through a narrower dtype."""
    dtype = DTYPES[arm]
    for name, mod in transformer.named_children():
        pinned = any(k in name for k in KEEP_FP32)
        mod.to(torch.float32 if pinned and arm != "fp64" else dtype)


def make_stage(transformer: torch.nn.Module, device: torch.device) -> MiniMaxH3DiffusionStage:
    bundle = MiniMaxH3Bundle(
        transformer=transformer,
        vae=None,
        audio_vae=None,
        text_encoder=None,
        processor=None,
        tokenizer=None,
        dtype=torch.bfloat16,
        device=device,
        pretrained_path="",
        text_encoder_onload_for_embed=False,
        text_embed_store=None,
    )
    return MiniMaxH3DiffusionStage(bundle, CPSSDEStrategy(), audio_shift=3.0, batch_replay_steps=False)


def lora_params(transformer: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]


def block_of(name: str) -> int:
    m = re.search(r"transformer_blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def flat_grad(params: Sequence[Tuple[str, torch.nn.Parameter]], store_dtype: torch.dtype) -> torch.Tensor:
    """Concatenate LoRA grads in name order; CPU, ``store_dtype``. Zeroes the grads."""
    parts = []
    for _, p in params:
        parts.append(p.grad.detach().to("cpu", store_dtype).reshape(-1))
        p.grad = None
    return torch.cat(parts)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if float(den) > 0 else float("nan")


class ActStats:
    """Per-block residual-stream max|x| and rms, recorded on the first forward only."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.rows: List[Tuple[float, float, bool]] = []
        self.done = False
        self.handles = [b.register_forward_hook(self._hook) for b in transformer.transformer_blocks]
        self.n = len(self.handles)

    def _hook(self, _mod, _inp, out) -> None:
        if self.done:
            return
        x = out.detach().float()
        self.rows.append((float(x.abs().max()), float(x.pow(2).mean().sqrt()), bool(torch.isfinite(x).all())))
        if len(self.rows) == self.n:
            self.done = True

    def close(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def rollout(stage, embeds, params, sigmas, geometry, *, seed: int, key: str):
    """One bf16-policy rollout -> CPU ``LatentSegment`` (all steps SDE)."""
    dev = stage.bundle.device
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(1, geometry.num_video_rows, geometry.video_token_dim, generator=g)
    a0 = torch.randn(1, geometry.num_audio_rows, MINIMAX_H3_AUDIO_LATENT_CHANNELS, generator=g)
    cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=embeds.to(dev)))
    seg = stage.generate(
        cond,
        params=params,
        sigmas=sigmas.to(dev),
        geometry=geometry,
        initial_latents=x0,
        initial_audio_latents=a0,
        denoise_seed_keys=[key],
        denoise_base_seed=seed,
    )
    return seg.to("cpu") if hasattr(seg, "to") else seg


def arm_gradients(
    stage,
    params,
    samples,
    arm: str,
    *,
    record_acts: bool = True,
    log=None,
    drop_terminal: bool = False,
    adv=None,
    record_cot: bool = False,
    fixed_cot=None,
) -> Dict[str, object]:
    """Per-(sample, step) LoRA gradients in one arm; with ``fixed_cot`` the main outputs use another arm's d(-logp)/d(pred)."""
    import time

    tr = stage.bundle.transformer
    cast_arm(tr, arm)
    lp = lora_params(tr)
    dev = stage.bundle.device
    acts = ActStats(tr) if record_acts else None
    n_params = sum(p.numel() for _, p in lp)
    n_steps = int(samples[0][1].sde_indices.numel())
    per_sample = torch.zeros(len(samples), n_params, dtype=torch.float32)
    per_step = torch.zeros(n_steps, n_params, dtype=torch.float32)
    per_step_w = torch.zeros(n_steps, n_params, dtype=torch.float32) if adv is not None else None
    # Own-residual gradients, only kept as step sums when the main outputs come from ``fixed_cot``.
    own = fixed_cot is not None
    per_step_own = torch.zeros(n_steps, n_params, dtype=torch.float32) if own else None
    per_step_w_own = torch.zeros(n_steps, n_params, dtype=torch.float32) if own and adv is not None else None
    logps: List[List[float]] = []
    resid: List[Dict[str, float]] = []
    cot: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    preds: List[torch.Tensor] = []
    orig_predict = stage.step.predict_noise

    def predict(*a, **k):
        v, au = orig_predict(*a, **k)
        if v.requires_grad:
            v.retain_grad()
            au.retain_grad()
        preds[:] = [v, au]
        return v, au

    stage.step.predict_noise = predict
    try:
        for i, (embeds, seg) in enumerate(samples):
            t0 = time.perf_counter()
            cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=embeds.to(dev)))
            row = []
            ks = seg.sde_indices.tolist()
            # The terminal step has sigma_next == 0: no sampling noise, so its residual is pure rollout-vs-replay mismatch.
            used = [k for k in ks if not (drop_terminal and float(seg.sigmas[k + 1]) == 0.0)]
            assert len(ks) == n_steps and len(used) >= 1 and (not drop_terminal or len(used) == n_steps - 1)
            for j, k in enumerate(ks):
                res = stage.replay(cond, segment=seg, params=params, step_indices=[k])
                logp = res.log_probs
                row.append(float(logp.detach().sum()))
                if res.prev_sample_means is not None:
                    # Video residual x_prev - mu against what fp16 trajectory storage alone would leave behind.
                    with torch.no_grad():
                        mu = res.prev_sample_means[0, 0].detach().float().cpu()
                        xp = seg.latents[0, seg.indices.tolist().index(k + 1)].float().cpu()
                        resid.append(
                            {
                                "sample": i,
                                "step": k,
                                "resid_rms": float((xp - mu).pow(2).mean().sqrt()),
                                "fp16_quant_rms_of_mu": float((mu.half().float() - mu).pow(2).mean().sqrt()),
                            }
                        )
                v, au = preds
                if own:
                    cv, ca = fixed_cot[(i, j)]
                    torch.autograd.backward(
                        [v, au], [cv.to(v.device, v.dtype), ca.to(au.device, au.dtype)], retain_graph=True
                    )
                    g = flat_grad(lp, torch.float32)
                    (-logp.sum()).backward()
                    g_own = flat_grad(lp, torch.float32)
                    per_step_own[j].add_(g_own)
                    if per_step_w_own is not None:
                        per_step_w_own[j].add_(g_own, alpha=float(adv[i]))
                    del g_own
                else:
                    (-logp.sum()).backward()
                    g = flat_grad(lp, torch.float32)
                    if record_cot:
                        cot[(i, j)] = (v.grad.detach().float().cpu(), au.grad.detach().float().cpu())
                if k in used:
                    per_sample[i].add_(g, alpha=1.0 / len(used))
                per_step[j].add_(g)
                if per_step_w is not None:
                    per_step_w[j].add_(g, alpha=float(adv[i]))
                preds.clear()
            logps.append(row)
            if log is not None:
                log(f"[{arm}] sample {i + 1}/{len(samples)} {time.perf_counter() - t0:.1f}s")
    finally:
        stage.step.predict_noise = orig_predict
        if acts is not None:
            acts.close()
    return {
        "per_sample": per_sample,
        "per_step": per_step,
        "per_step_w": per_step_w,
        "per_step_own": per_step_own,
        "per_step_w_own": per_step_w_own,
        "cot": cot,
        "logp": logps,
        "resid": resid,
        "acts": acts.rows if acts else [],
    }


def gram(a: torch.Tensor, b: torch.Tensor, chunk: int = 4_000_000) -> torch.Tensor:
    """``a @ b.T`` for [N, P] fp32 rows, accumulated in float64."""
    out = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float64)
    for s in range(0, a.shape[1], chunk):
        out += a[:, s : s + chunk].double() @ b[:, s : s + chunk].double().T
    return out


def weighted(rows: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``mean_i w_i * rows_i`` in float64, without materialising a float64 copy of ``rows``."""
    out = torch.zeros(rows.shape[1], dtype=torch.float64)
    for r, wi in zip(rows, w.tolist()):
        out.add_(r.double(), alpha=wi / rows.shape[0])
    return out


def block_slices(lp: Sequence[Tuple[str, torch.nn.Parameter]]) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    off = 0
    for name, p in lp:
        out.setdefault(block_of(name), []).append((off, off + p.numel()))
        off += p.numel()
    return out


def take(v: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> torch.Tensor:
    return torch.cat([v[a:b] for a, b in spans])


def random_advantages(n_prompts: int, group: int, draws: int, seed: int) -> torch.Tensor:
    """[draws, N] group-centred, globally-normalised advantages (the recipe's ``adv_use_global_std``)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(draws, n_prompts, group, generator=g, dtype=torch.float64)
    a = a - a.mean(2, keepdim=True)
    return (a / (a.std(dim=(1, 2), keepdim=True) + 1e-8)).reshape(draws, -1)


def compare(test: Dict, ref: Dict, adv: torch.Tensor, adv_draws: torch.Tensor, slices, group_ids) -> Dict[str, object]:
    """Acceptance metrics of ``test`` against ``ref`` on the same fixed batch; ``adv`` [N] fixed, ``adv_draws`` [D, N]."""
    t, r = test["per_sample"], ref["per_sample"]
    n = t.shape[0]
    gtt, gtr, grr = gram(t, t), gram(t, r), gram(r, r)
    gt, gr = weighted(t, adv), weighted(r, adv)
    res: Dict[str, object] = {
        "batch_cos": cos(gt, gr),
        "batch_norm_ratio": float(gt.norm() / gr.norm()),
        "per_block_cos": {b: cos(take(gt, s), take(gr, s)) for b, s in sorted(slices.items())},
        "per_block_norm_ratio": {b: float(take(gt, s).norm() / take(gr, s).norm()) for b, s in sorted(slices.items())},
        "per_sample_cos": [float(gtr[i, i] / (gtt[i, i] * grr[i, i]).sqrt()) for i in range(n)],
        "per_step_cos": [cos(a, b) for a, b in zip(test["per_step"], ref["per_step"])],
        "per_step_norm_ref": [float(b.double().norm()) for b in ref["per_step"]],
    }
    del gt, gr
    # Batch cosine over many advantage draws: a^T Gtr a / sqrt(a^T Gtt a * a^T Grr a).
    num = torch.einsum("di,ij,dj->d", adv_draws, gtr, adv_draws)
    den = (
        torch.einsum("di,ij,dj->d", adv_draws, gtt, adv_draws) * torch.einsum("di,ij,dj->d", adv_draws, grr, adv_draws)
    ).sqrt()
    c = (num / den).sort().values
    res["batch_cos_over_adv_draws"] = {
        "min": float(c[0]),
        "p10": float(c[len(c) // 10]),
        "median": float(c[len(c) // 2]),
        "p90": float(c[-(len(c) // 10) - 1]),
        "max": float(c[-1]),
        "draws": len(c),
    }
    # Error structure: e_i = g_i^test - g_i^ref. Shared across samples (bias) or independent (noise)?
    gee = gtt - gtr - gtr.T + grr
    en = gee.diagonal().clamp_min(0).sqrt()
    ecos = gee / (en[:, None] * en[None, :])
    same = [float(ecos[i, j]) for i, j in itertools.combinations(range(n), 2) if group_ids[i] == group_ids[j]]
    cross = [float(ecos[i, j]) for i, j in itertools.combinations(range(n), 2) if group_ids[i] != group_ids[j]]
    res["err_pair_cos_same_prompt"] = sum(same) / len(same) if same else None
    res["err_pair_cos_cross_prompt"] = sum(cross) / len(cross) if cross else None
    res["err_rel_norm_per_sample"] = [float(en[i] / grr[i, i].sqrt()) for i in range(n)]
    res["mean_err_norm_over_mean_single_err_norm"] = float(gee.sum().clamp_min(0).sqrt() / n / en.mean())
    res["iid_reference_equal_magnitudes"] = 1.0 / math.sqrt(n)
    res["independent_reference_actual_magnitudes"] = float(en.pow(2).sum().sqrt() / en.sum())
    return res


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """``|a - b| / |b|`` in float64: unlike a cosine this also catches scale errors."""
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def state_fingerprint(transformer: torch.nn.Module) -> Dict[str, List[float]]:
    """Per top-level child [sum, abs-sum, square-sum] over non-LoRA float params and buffers, in float64."""
    out: Dict[str, List[float]] = {}
    with torch.no_grad():
        named = list(transformer.named_parameters()) + list(transformer.named_buffers())
        for name, t in named:
            if "lora_" in name or not t.dtype.is_floating_point:
                continue
            acc = out.setdefault(name.split(".")[0], [0.0, 0.0, 0.0])
            d = t.detach().double()
            acc[0] += float(d.sum())
            acc[1] += float(d.abs().sum())
            acc[2] += float(d.pow(2).sum())
    return out


def fingerprints_match(a: Dict[str, List[float]], b: Dict[str, List[float]], tol: float = 1e-9) -> List[str]:
    """Names of children whose fingerprint moved by more than ``tol`` relative (empty list = identical state)."""
    bad = [k for k in sorted(set(a) | set(b)) if k not in a or k not in b]
    for k in set(a) & set(b):
        if any(abs(x - y) > tol * max(abs(x), abs(y), 1e-300) for x, y in zip(a[k], b[k])):
            bad.append(k)
    return bad


def lora_hash(transformer: torch.nn.Module) -> str:
    """sha256 over the names and raw bytes of every trainable (LoRA) tensor; call on CPU right after injection."""
    import hashlib

    h = hashlib.sha256()
    for name, p in lora_params(transformer):
        h.update(name.encode())
        h.update(str(p.dtype).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


STEP_KEYS = (
    "per_step_bf16",
    "per_step_w_bf16",
    "per_step_fp32",
    "per_step_w_fp32",
    "per_step_fp32own",
    "per_step_w_fp32own",
)


def shard_payload(stage, params, samples, ids, adv_all, put, *, acts: bool = False, log=None) -> Dict[str, object]:
    """bf16 arm, then fp32 arm driven by the bf16 arm's d(-logp)/d(pred); every tensor goes to ``put(key, value)`` in fp32."""
    import time

    adv = adv_all[ids]
    put("ids", list(ids))
    put("slices", block_slices(lora_params(stage.bundle.transformer)))
    info: Dict[str, object] = {"ids": list(ids)}
    cot = None
    for arm in ("bf16", "fp32"):
        t0 = time.perf_counter()
        out = arm_gradients(
            stage,
            params,
            samples,
            arm,
            record_acts=acts,
            log=log,
            drop_terminal=True,
            adv=adv,
            record_cot=(arm == "bf16"),
            fixed_cot=(cot if arm == "fp32" else None),
        )
        info[f"seconds_{arm}"] = time.perf_counter() - t0
        info[f"logp_{arm}"], info[f"resid_{arm}"] = out["logp"], out["resid"]
        if acts:
            info[f"acts_{arm}"] = summarize_acts(out["acts"])
        put(f"per_sample_{arm}", out["per_sample"])  # [n, P] mean over non-terminal steps
        put(f"per_step_{arm}", out["per_step"])  # [S, P] sum over this shard's samples
        put(f"per_step_w_{arm}", out["per_step_w"])  # [S, P] advantage-weighted sum
        if arm == "bf16":
            cot = out["cot"]
            info["lora_sha_in_bf16_arm"] = lora_hash(stage.bundle.transformer)
        else:
            put("per_step_fp32own", out["per_step_own"])  # [S, P] fp32 with its own residual
            put("per_step_w_fp32own", out["per_step_w_own"])
        del out
    # Both arms must have seen the same adapter values: cast back and compare bytes.
    cast_arm(stage.bundle.transformer, "bf16")
    info["lora_sha_after_fp32_arm"] = lora_hash(stage.bundle.transformer)
    assert info["lora_sha_in_bf16_arm"] == info["lora_sha_after_fp32_arm"], "LoRA values changed between the arms"
    return info


def merge_payloads(get, n_shards: int, n_prompts: int, group: int, adv, adv_draws) -> Dict[str, object]:
    """Merge ``get(shard, key)`` tensors in global sample order and compare bf16 against fp32; raises if self-checks fail."""
    total = n_prompts * group
    ids = [list(get(k, "ids")) for k in range(n_shards)]
    assert sorted(i for part in ids for i in part) == list(range(total)), f"sample ids incomplete or duplicated: {ids}"
    slices = get(0, "slices")
    assert all(get(k, "slices") == slices for k in range(n_shards)), "shards disagree on the LoRA parameter layout"
    group_ids = [i // group for i in range(total)]

    def stacked(key: str) -> torch.Tensor:
        out = None
        for k in range(n_shards):
            t = get(k, key)
            assert t.dtype == torch.float32 and t.shape[0] == len(ids[k]), (
                f"{key} shard {k}: {t.dtype} {tuple(t.shape)}"
            )
            if out is None:
                out = torch.zeros(total, t.shape[1], dtype=torch.float32)
            out[ids[k]] = t
            del t
        return out

    def summed(key: str) -> torch.Tensor:
        out = None
        for k in range(n_shards):
            t = get(k, key)
            assert t.dtype == torch.float32, f"{key} shard {k}: {t.dtype}"
            out = t.clone() if out is None else out.add_(t)
            del t
        return out

    steps = {k: summed(k) for k in ("per_step_bf16", "per_step_fp32")}
    b = {"per_sample": stacked("per_sample_bf16"), "per_step": steps["per_step_bf16"]}
    r = {"per_sample": stacked("per_sample_fp32"), "per_step": steps["per_step_fp32"]}
    n_steps = b["per_step"].shape[0]
    t_idx = n_steps - 1  # index of the terminal step
    rep: Dict[str, object] = {
        "labels": {
            "fixed_residual": "fp32 arm reuses the bf16 arm's d(-logp)/d(model output): only the model Jacobian (forward+backward compute precision) differs",
            "own_residual": "fp32 arm uses its own log-prob residual on the bf16-sampled trajectory (adds the bf16-vs-fp32 mean mismatch)",
            "fixed_residual_terminal_excluded.per_step_cos/per_step_norm_ref": "UNWEIGHTED sum over samples, all steps incl. terminal; training-relevant per-step numbers are per_step_w_*",
            "fixed_residual_terminal_excluded.(batch|per_block|per_sample|err)_*": "mean over the non-terminal steps, fixed_residual fp32 reference",
            "advantages": "synthetic: standard normal draws, centred per prompt group, divided by their overall std",
            "logp": "CPS 'log-prob' is the unscaled negative mean squared residual, not a normalised Gaussian log-likelihood",
        },
        "fixed_residual_terminal_excluded": compare(b, r, adv, adv_draws, slices, group_ids),
    }
    del b, r
    for k in STEP_KEYS:
        if k not in steps:
            steps[k] = summed(k)

    def step_cos(x: str, y: str):
        return [cos(a, c) for a, c in zip(steps[x], steps[y])]

    def tot(x: str, upto: int) -> torch.Tensor:
        return steps[x][:upto].double().sum(0)

    def total_cos(x: str, y: str, upto: int) -> float:
        return cos(tot(x, upto), tot(y, upto))

    rep["per_step_w_cos_bf16_vs_fp32fixed"] = step_cos("per_step_w_bf16", "per_step_w_fp32")
    rep["per_step_w_cos_bf16_vs_fp32own"] = step_cos("per_step_w_bf16", "per_step_w_fp32own")
    rep["per_step_w_cos_fp32fixed_vs_fp32own"] = step_cos("per_step_w_fp32", "per_step_w_fp32own")
    for key in STEP_KEYS:
        rep[f"step_sum_norm_{key}"] = [float(x.double().norm()) for x in steps[key]]
    rep["batch_cos_w_terminal_excluded"] = {
        "bf16_vs_fp32fixed": total_cos("per_step_w_bf16", "per_step_w_fp32", t_idx),
        "bf16_vs_fp32own": total_cos("per_step_w_bf16", "per_step_w_fp32own", t_idx),
        "fp32fixed_vs_fp32own": total_cos("per_step_w_fp32", "per_step_w_fp32own", t_idx),
    }
    rep["batch_cos_w_terminal_included"] = {
        "bf16_vs_fp32fixed": total_cos("per_step_w_bf16", "per_step_w_fp32", n_steps),
        "bf16_vs_fp32own": total_cos("per_step_w_bf16", "per_step_w_fp32own", n_steps),
    }
    rep["unweighted_cos_terminal_excluded"] = {
        "bf16_vs_fp32fixed": total_cos("per_step_bf16", "per_step_fp32", t_idx),
        "bf16_vs_fp32own": total_cos("per_step_bf16", "per_step_fp32own", t_idx),
    }
    # mean_i A_i * mean_{non-terminal j} g_ij, i.e. the batch gradient as defined for compare().
    rep["batch_grad_norm_w_terminal_excluded"] = {
        k: float(tot(f"per_step_w_{k}", t_idx).norm() / (total * t_idx)) for k in ("bf16", "fp32", "fp32own")
    }
    # Two independent routes to the advantage-weighted, terminal-excluded batch gradient must agree in direction and scale.
    main = rep["fixed_residual_terminal_excluded"]
    ratio2 = rep["batch_grad_norm_w_terminal_excluded"]["bf16"] / rep["batch_grad_norm_w_terminal_excluded"]["fp32"]
    rep["two_routes_cos_abs_diff"] = abs(main["batch_cos"] - rep["batch_cos_w_terminal_excluded"]["bf16_vs_fp32fixed"])
    rep["two_routes_norm_ratio_rel_diff"] = abs(main["batch_norm_ratio"] - ratio2) / abs(ratio2)
    bad = [k for k, v in _flat(rep).items() if isinstance(v, float) and not math.isfinite(v)]
    rep["checks_passed"] = (
        rep["two_routes_cos_abs_diff"] < 1e-4 and rep["two_routes_norm_ratio_rel_diff"] < 1e-4 and not bad
    )
    rep["non_finite_entries"] = bad[:20]
    return rep


def _flat(x, prefix: str = "") -> Dict[str, object]:
    if isinstance(x, dict):
        return {k2: v2 for k, v in x.items() for k2, v2 in _flat(v, f"{prefix}{k}.").items()}
    if isinstance(x, (list, tuple)):
        return {k2: v2 for i, v in enumerate(x) for k2, v2 in _flat(v, f"{prefix}{i}.").items()}
    return {prefix.rstrip("."): x}


def partest(args) -> None:
    """Tiny-model proof that N independently built shards + merge reproduce the single-process result."""
    dev = torch.device("cpu")
    params = DiffusionSamplingParams(
        num_inference_steps=args.steps, guidance_scale=1.0, height=64, width=64, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=args.steps, shift=12.0)
    g = torch.Generator().manual_seed(args.seed + 1)
    embeds = [
        torch.randn(1, args.text_tokens, args.text_dim, generator=g).to(torch.bfloat16) for _ in range(args.prompts)
    ]
    total = args.prompts * args.group
    adv = random_advantages(args.prompts, args.group, 1, seed=11)[0]
    adv_draws = random_advantages(args.prompts, args.group, 200, seed=12)

    def build(ids, lora_seed):
        """A fresh 'container' in the production order: checkpoint layout -> LoRA -> rollout, with no cast_arm before rollout."""
        tr = build_tiny(
            args.layers, args.hidden, args.heads, args.text_dim, args.seed
        )  # fp32, pinned weights NOT bf16-representable
        for name, mod in tr.named_children():
            if not any(k in name for k in KEEP_FP32):
                mod.to(torch.bfloat16)
        tr.requires_grad_(False)
        torch.manual_seed(lora_seed)
        inject_lora(
            tr, rank=args.rank, alpha=2 * args.rank, target_modules=LORA_TARGETS, module_prefix="transformer_blocks"
        )
        lh = lora_hash(tr)
        tr.eval()
        tr.enable_gradient_checkpointing()
        stage = make_stage(tr, dev)
        fp_before = state_fingerprint(tr)
        samples = []
        for i in ids:
            p, s = divmod(i, args.group)
            samples.append(
                (embeds[p], rollout(stage, embeds[p], params, sigmas, geometry, seed=1000 * p + s, key=f"p{p}s{s}"))
            )
        return stage, samples, lh, fp_before

    def one_shard(ids, lora_seed):
        stage, samples, lh, fp_before = build(ids, lora_seed)
        store: Dict[str, object] = {}
        info = shard_payload(stage, params, samples, ids, adv, store.__setitem__)
        cast_arm(stage.bundle.transformer, "bf16")
        moved = fingerprints_match(fp_before, state_fingerprint(stage.bundle.transformer))
        assert not moved, f"model state changed across bf16->fp32->bf16: {moved}"
        return store, info, lh

    def merged(stores):
        rep = merge_payloads(lambda k, key: stores[k][key], len(stores), args.prompts, args.group, adv, adv_draws)
        return rep, {k: v for k, v in _flat(rep).items() if isinstance(v, float)}

    all_ids = list(range(total))
    split = lambda n: [[i for i in all_ids if i * n // total == k] for k in range(n)]
    serial_p, serial_i, lh0 = one_shard(all_ids, 0)
    serial_rep, serial = merged([serial_p])
    assert serial_rep["checks_passed"], "serial run fails its own self-checks"
    parts = [one_shard(ids, 0) for ids in split(4)]
    assert all(p[2] == lh0 for p in parts), "seeded LoRA init differs between shards"
    par_rep, par = merged([p[0] for p in parts])
    assert par_rep["checks_passed"] and serial.keys() == par.keys()
    # A number passes if it agrees to 1e-7 absolute OR 1e-5 relative; score < 1 means pass.
    worst = max(
        (min(abs(serial[k] - par[k]) / 1e-7, abs(serial[k] - par[k]) / (1e-5 * max(abs(serial[k]), 1e-300))), k)
        for k in serial
    )
    # Vector-level: the merged per-sample rows themselves, not just summary numbers.
    par_rows = torch.zeros_like(serial_p["per_sample_fp32"])
    for store, _, _ in parts:
        par_rows[store["ids"]] = store["per_sample_fp32"]
    rows_rel = rel_diff(par_rows, serial_p["per_sample_fp32"])
    # Negative control: shards with different LoRA A must give a different merged batch gradient.
    _, bad = merged([one_shard(ids, 100 + k)[0] for k, ids in enumerate(split(4))])
    nk = "batch_grad_norm_w_terminal_excluded.fp32"
    neg_rel = abs(bad[nk] - serial[nk]) / serial[nk]

    # Independent reference passes with no shard/merge code and no cotangent plumbing.
    stage, samples, _, _ = build(all_ids, 0)
    ref_b = arm_gradients(
        stage, params, samples, "bf16", record_acts=False, drop_terminal=True, adv=adv, record_cot=True
    )
    # (a) feeding the bf16 arm its own cotangent must reproduce the bf16 gradient, in value.
    self_b = arm_gradients(
        stage, params, samples, "bf16", record_acts=False, drop_terminal=True, adv=adv, fixed_cot=ref_b["cot"]
    )
    self_rel = max(
        rel_diff(self_b["per_sample"], ref_b["per_sample"]), rel_diff(self_b["per_step_own"], ref_b["per_step"])
    )
    # (b) the fp32 own-residual sums produced inside the cotangent pass must equal a plain fp32 pass, in value.
    plain_32 = arm_gradients(stage, params, samples, "fp32", record_acts=False, drop_terminal=True, adv=adv)
    own_rel = max(
        rel_diff(serial_p["per_step_w_fp32own"], plain_32["per_step_w"]),
        rel_diff(serial_p["per_step_fp32own"], plain_32["per_step"]),
    )
    # (c) the shard path's bf16 rows must equal the direct pass, in value.
    direct_rel = rel_diff(serial_p["per_sample_bf16"], ref_b["per_sample"])
    # (d) the plain fp32 batch gradient without the terminal step, built by hand from the plain pass.
    hand = cos(ref_b["per_step_w"][:-1].double().sum(0), plain_32["per_step_w"][:-1].double().sum(0))
    hand_diff = abs(hand - serial["batch_cos_w_terminal_excluded.bf16_vs_fp32own"])
    term = [d for d in serial_i["resid_bf16"] if d["step"] == args.steps - 1]
    term32 = [d for d in serial_i["resid_fp32"] if d["step"] == args.steps - 1]
    out = {
        "n_compared": len(serial),
        "serial_vs_4_shards_worst_score_lt1_passes": worst[0],
        "at": worst[1],
        "merged_rows_rel_diff": rows_rel,
        "two_routes_cos_abs_diff": serial["two_routes_cos_abs_diff"],
        "two_routes_norm_ratio_rel_diff": serial["two_routes_norm_ratio_rel_diff"],
        "negative_control_rel_norm_diff": neg_rel,
        "bf16_self_cotangent_rel_diff": self_rel,
        "fp32_own_inside_cot_pass_vs_plain_rel_diff": own_rel,
        "shard_path_vs_direct_bf16_rel_diff": direct_rel,
        "hand_built_own_batch_cos_abs_diff": hand_diff,
        "batch_cos_w_terminal_excluded": serial_rep["batch_cos_w_terminal_excluded"],
        "batch_cos_w_terminal_included": serial_rep["batch_cos_w_terminal_included"],
        "bf16_terminal_resid_over_quant_max": max(d["resid_rms"] / d["fp16_quant_rms_of_mu"] for d in term),
        "fp32_terminal_resid_over_quant_min": min(d["resid_rms"] / d["fp16_quant_rms_of_mu"] for d in term32),
    }
    print("PARTEST " + json.dumps(out, default=float))
    assert worst[0] < 1.0 and rows_rel < 1e-6, f"shards + merge do not reproduce serial: {worst} rows {rows_rel}"
    assert neg_rel > 1e-3, "negative control did not detect differing LoRA A"
    assert self_rel < 1e-5, "cotangent plumbing does not reproduce the plain gradient"
    assert own_rel < 1e-5, "own-residual gradient inside the cotangent pass differs from a plain pass"
    assert direct_rel < 1e-5, "shard path differs from a direct pass"
    assert hand_diff < 1e-6, "merge's own-residual batch cosine differs from a hand-built one"
    assert out["bf16_terminal_resid_over_quant_max"] < 1.5, (
        "bf16 replay does not reproduce the bf16 rollout at the terminal step"
    )
    print("PARTEST OK")


def summarize_acts(rows) -> Dict[str, object]:
    return {
        "max_by_block": [float(f"{r[0]:.3g}") for r in rows],
        "rms_by_block": [float(f"{r[1]:.3g}") for r in rows],
        "all_finite": all(r[2] for r in rows),
    }


def build_tiny(layers: int, hidden: int, heads: int, text_dim: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    tr = MiniMaxH3Transformer3DModel(
        num_attention_heads=heads,
        attention_head_dim=hidden // heads,
        hidden_size=hidden,
        num_layers=layers,
        num_refiner_layers=2,
        ffn_dim=hidden * 2,
        text_dim=text_dim,
        freq_dim=64,
        time_embed_hidden_dim=hidden,
        time_embed_dim=hidden // 2,
        rope_freq_dim=hidden // heads // 8,
    )
    return tr


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU self-test of the shard + merge path on a tiny random H3")
    ap.add_argument("--partest", action="store_true", help="kept for compatibility; the self-test is the only mode")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--text-dim", type=int, default=96)
    ap.add_argument("--text-tokens", type=int, default=12)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(8)
    partest(args)


if __name__ == "__main__":
    main()
