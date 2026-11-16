"""Mechanism harness for the MiniMax-H3 bf16 gradient gap: per-operation precision, factor split, per-block local VJPs. Not for merge."""

from __future__ import annotations

import argparse
import functools
import json
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h3_grad_precision as H
import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn

from unirl.models.minimax_h3.conditions import MiniMaxH3Conditions
from unirl.models.minimax_h3.vendor.transformer_minimax_h3 import _apply_rotary_emb
from unirl.types.conditions.text import TextEmbedCondition

BF, FP, HF = torch.bfloat16, torch.float32, torch.float16
METRIC_SETS: Dict[str, List[int]] = {}  # name -> block indices; when set, metrics() also reports each restriction
BLOCK_OPS = ("stream", "norm", "modtab", "modar", "qkv", "qknorm", "core", "out", "ff1", "ff2")
HEAD_OPS = ("norm", "tab", "ar")
STEM = ("context_embedder", "token_refiner")


def policy(default: torch.dtype, **over: torch.dtype) -> Dict[str, torch.dtype]:
    """One block's dtype per operation; every op casts its INPUT to its own dtype, so rounding happens on entry to a bf16 op."""
    assert set(over) <= set(BLOCK_OPS), over
    return {**{k: default for k in BLOCK_OPS}, **over}


class Replace(torch.autograd.Function):
    """Forward returns ``value`` exactly; backward passes the gradient straight to ``x``."""

    @staticmethod
    def forward(ctx, x, value):
        return value.clone()

    @staticmethod
    def backward(ctx, g):
        return g, None


class Ctl:
    """Mutable switches read by the patched block / head forwards."""

    def __init__(self, tr: torch.nn.Module) -> None:
        self.tr = tr
        self.n = len(tr.transformer_blocks)
        self.pol: List[Dict[str, torch.dtype]] = [policy(BF) for _ in range(self.n)]
        self.head: Dict[str, torch.dtype] = {k: BF for k in HEAD_OPS}
        self.rec: Optional[Dict[str, object]] = None  # capture h / c / block args when set
        self.rec_layers: Optional[Sequence[int]] = None  # None = all
        self.subst: Optional[Dict[int, torch.Tensor]] = None  # block index -> exact replacement of its output value
        self.noise: Optional[Tuple[float, int]] = None  # (eps, seed): value-only stream perturbation after each block
        self.z_store: Optional[Dict[str, torch.Tensor]] = None  # record lora z here when set
        self.z_for: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # tag -> name -> z used for d z^T
        self.fac: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # tag -> name -> accumulated d z^T
        self.io: Dict[str, object] = {}
        self.cut: Optional[int] = None  # detach the stream on entry to this block: nothing below it is differentiated
        self.probe: Optional[Dict[str, Dict[str, object]]] = (
            None  # "fwd"/"bwd": (block, op) -> [finite, absmax] when set
        )

    def see(self, i: int, op: str, t: torch.Tensor) -> torch.Tensor:
        """Record finiteness and max |value| of an intermediate, and of its gradient when it is differentiated."""
        if self.probe is None:
            return t
        d = t.detach().float()
        self.probe["fwd"][f"{i}.{op}"] = [
            bool(torch.isfinite(d).all()),
            float(d.abs().nan_to_num(posinf=float("inf")).max()),
            str(t.dtype),
        ]
        self.probe["order"].append(f"{i}.{op}")
        if t.requires_grad:

            def hook(g: torch.Tensor, key: str = f"{i}.{op}") -> None:
                gf = g.detach().float()
                self.probe["bwd"][key] = [
                    bool(torch.isfinite(gf).all()),
                    float(gf.abs().nan_to_num(posinf=float("inf")).max()),
                    str(g.dtype),
                    float((gf == 0).float().mean()),
                ]
                self.probe["bwd_order"].append(key)

            t.register_hook(hook)
        return t

    def set_all(self, dt: torch.dtype) -> None:
        self.pol = [policy(dt) for _ in range(self.n)]
        self.head = {k: dt for k in HEAD_OPS}
        self.set_stem(dt)

    def set_stem(self, dt: torch.dtype) -> None:
        """Stem weights are bf16-representable, so this module cast is lossless both ways; pinned fp32 modules are never cast."""
        for name in STEM:
            getattr(self.tr, name).to(dt)

    def want(self, i: int) -> bool:
        return self.rec is not None and (self.rec_layers is None or i in self.rec_layers)


def flin(
    ctl: Ctl, name: str, mod: torch.nn.Module, x: torch.Tensor, dt: torch.dtype, scale: float = 1.0
) -> torch.Tensor:
    """``Linear`` (optionally a PEFT LoRA wrapper) at ``dt`` with the stored weight values; mirrors peft's vanilla forward.

    ``scale`` != 1 runs the matmul on ``x / scale`` and returns ``fp32(output) * scale`` (bias-free layers only)."""
    base = getattr(mod, "base_layer", mod)
    if scale != 1.0:
        assert base.bias is None, f"{name}: input scaling needs a bias-free linear"
        x = x.float() / scale
    x = x.to(dt)
    y = F.linear(x, base.weight.to(dt), None if base.bias is None else base.bias.to(dt))
    if hasattr(mod, "lora_A"):
        (ad,) = mod.active_adapters
        z = F.linear(x, mod.lora_A[ad].weight.to(dt))  # [B, T, r]
        bo = F.linear(z, mod.lora_B[ad].weight.to(dt))  # [B, T, out]
        if ctl.z_store is not None:
            ctl.z_store[name] = z.detach().float()
        if ctl.fac is not None and bo.requires_grad:

            def hook(d: torch.Tensor, name: str = name) -> None:
                d2 = d.detach().float().flatten(0, -2)  # [B*T, out]
                for tag, zs in ctl.z_for.items():
                    g = d2.T @ zs[name].flatten(0, -2)  # [out, r] = B.grad with this pairing of factors
                    ctl.fac[tag][name] = g if name not in ctl.fac[tag] else ctl.fac[tag][name] + g

            bo.register_hook(hook)
        y = y + bo * mod.scaling[ad]
    if scale != 1.0:
        y = y.float() * scale
    return y


def fblock(ctl: Ctl, i: int, blk: torch.nn.Module, h, temb, idx, rot, mask=None) -> torch.Tensor:
    """Functional copy of ``MiniMaxH3TransformerBlock.forward`` with a dtype per operation (``ctl.pol[i]``)."""
    P = ctl.pol[i]
    pre = f"transformer_blocks.{i}."
    h = h.to(P["stream"])
    if ctl.cut is not None and i == ctl.cut:
        h = h.detach()
    if ctl.want(i):
        ctl.rec["h"][i] = h.detach().cpu()
        ctl.rec["args"] = (temb.detach(), idx, tuple(r.detach() for r in rot), mask)
    lin = blk.adaln_proj.linear
    tab = F.linear(F.silu(temb).to(P["modtab"]), lin.weight.to(P["modtab"]), lin.bias.to(P["modtab"]))
    sh1, sc1, g1, sh2, sc2, g2 = tab.view(-1, 6 * blk.adaln_proj.hidden_size).chunk(6, dim=-1)

    def norm(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.to(P["norm"]), m.normalized_shape, m.weight.to(P["norm"]), m.eps)

    def modulate(n: torch.Tensor, sc: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
        return n.to(P["modar"]) * (1.0 + sc[idx].to(P["modar"])) + sh[idx].to(P["modar"])

    attn = blk.attn
    assert not getattr(attn, "fused_projections", False)
    see = functools.partial(ctl.see, i)
    see("modtab", tab)
    x = see("modulated1", modulate(see("norm1", norm(blk.norm1, h)), sc1, sh1))
    q = see("to_q", flin(ctl, pre + "attn.to_q", attn.to_q, x, P["qkv"])).unflatten(-1, (attn.heads, -1))
    k = see("to_k", flin(ctl, pre + "attn.to_k", attn.to_k, x, P["qkv"])).unflatten(-1, (attn.heads, -1))
    v = see("to_v", flin(ctl, pre + "attn.to_v", attn.to_v, x, P["qkv"])).unflatten(-1, (attn.heads, -1))
    qn, c = P["qknorm"], P["core"]
    q = F.rms_norm(q.to(qn), attn.norm_q.normalized_shape, attn.norm_q.weight.to(qn), attn.norm_q.eps).to(c)
    k = F.rms_norm(k.to(qn), attn.norm_k.normalized_shape, attn.norm_k.weight.to(qn), attn.norm_k.eps).to(c)
    see("q_normed", q)
    see("k_normed", k)
    if rot is not None:
        q = _apply_rotary_emb(q, *rot)
        k = _apply_rotary_emb(k, *rot)
    o = dispatch_attention_fn(
        q,
        k,
        v.to(c),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        backend=attn.processor._attention_backend,
        parallel_config=attn.processor._parallel_config,
    )
    see("attn_core", o)
    o = see(
        "to_out",
        flin(ctl, pre + "attn.to_out.0", attn.to_out[0], o.flatten(2, 3).type_as(q), P["out"], P.get("out_scale", 1.0)),
    )
    h = see("stream_after_attn", h + g1[idx].to(P["stream"]) * o.to(P["stream"]))

    x = see("modulated2", modulate(see("norm2", norm(blk.norm2, h)), sc2, sh2))
    act = blk.ff.net[0]
    a, b = see("ff1", flin(ctl, pre + "ff.net.0.proj", act.proj, x, P["ff1"])).chunk(2, dim=-1)
    sw = P.get("swiglu", P["ff1"])  # the gated product follows the first FF linear unless a layout says otherwise
    f = see(
        "ff2",
        flin(
            ctl,
            pre + "ff.net.2",
            blk.ff.net[2],
            see("swiglu", a.to(sw) * act.activation(b.to(sw))),
            P["ff2"],
            P.get("ff2_scale", 1.0),
        ),
    )
    h = see("stream_after_ff", h + g2[idx].to(P["stream"]) * f.to(P["stream"]))

    if ctl.subst is not None and i in ctl.subst:
        h = Replace.apply(h, ctl.subst[i].to(h.device, h.dtype))
    if ctl.noise is not None:
        eps, seed = ctl.noise
        gen = torch.Generator(device=h.device).manual_seed(1_000_003 * seed + i)
        xi = torch.randn(h.shape, generator=gen, device=h.device, dtype=torch.float32)
        h = h + (eps * xi * h.detach().float().abs()).to(h.dtype)
    if ctl.want(i) and h.requires_grad:
        h.register_hook(lambda g, i=i: ctl.rec["c"].__setitem__(i, g.detach().cpu()))
    return h


def fhead(ctl: Ctl, m: torch.nn.Module, h, temb, tidx) -> torch.Tensor:
    """Functional copy of ``MiniMaxH3AdaLayerNormOut.forward`` with a dtype per operation (``ctl.head``)."""
    P = ctl.head
    if ctl.rec is not None:
        ctl.rec["h"][ctl.n] = h.detach().cpu()
        ctl.rec["head_args"] = (temb.detach(), tidx)
    sh, sc = F.linear(F.silu(temb).to(P["tab"]), m.linear.weight.to(P["tab"]), m.linear.bias.to(P["tab"])).chunk(
        2, dim=-1
    )
    n = F.rms_norm(h.to(P["norm"]), m.norm.normalized_shape, m.norm.weight.to(P["norm"]), m.norm.eps)
    return n.to(P["ar"]) * (1.0 + sc[tidx].to(P["ar"])) + sh[tidx].to(P["ar"])


def patch(tr: torch.nn.Module) -> Ctl:
    """LoRA masters to fp32 (exact upcast), then route every block and ``norm_out`` through the functional forwards."""
    for mod in tr.modules():
        if hasattr(mod, "lora_A"):
            (ad,) = mod.active_adapters
            assert ad not in getattr(mod, "lora_variant", {}), "DoRA / variants are not supported by the factor split"
            assert float(mod.lora_B[ad].weight.abs().max()) == 0.0, "factor split needs lora_B == 0"
            drop = mod.lora_dropout[ad]
            assert isinstance(drop, torch.nn.Identity) or getattr(drop, "p", 0.0) == 0.0, "LoRA dropout must be 0"
            mod.lora_A[ad].to(FP)
            mod.lora_B[ad].to(FP)
    ctl = Ctl(tr)
    for i, blk in enumerate(tr.transformer_blocks):
        blk.forward = functools.partial(fblock, ctl, i, blk)
    tr.norm_out.forward = functools.partial(fhead, ctl, tr.norm_out)
    tr.register_forward_pre_hook(lambda m, a, kw: ctl.io.update(kw), with_kwargs=True)
    return ctl


# ----------------------------------------------------------------------------------------------- gradients


def b_params(tr: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    """``lora_B`` weights in name order, keyed by the wrapped module's name (same keys as ``flin``)."""
    out = []
    for n, p in tr.named_parameters():
        if ".lora_B." in n:
            out.append((n.split(".lora_B.")[0], p))
    return out


def flat_b(tr: torch.nn.Module) -> torch.Tensor:
    """Concatenated fp32 ``lora_B`` grads on the parameter device; clears every LoRA grad."""
    parts = [
        (torch.zeros_like(p, dtype=FP) if p.grad is None else p.grad.detach().float()).reshape(-1)
        for _, p in b_params(tr)
    ]
    for _, p in H.lora_params(tr):
        p.grad = None
    return torch.cat(parts)


def b_slices(tr: torch.nn.Module) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    off = 0
    for n, p in b_params(tr):
        out.setdefault(H.block_of(n + "."), []).append((off, off + p.numel()))
        off += p.numel()
    return out


def flat_named(tr: torch.nn.Module, named: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([named[n].reshape(-1) for n, _ in b_params(tr)])


class Env:
    def __init__(self, stage, params, ctl: Optional[Ctl] = None) -> None:
        self.stage, self.params, self.ctl = stage, params, ctl
        self.tr = stage.bundle.transformer
        self.dev = stage.bundle.device


def run_pair(env: Env, pair: Dict[str, object], cot=None, cot_scale: float = 1.0):
    """One replayed (sample, step): LoRA-B gradient of ``-logp`` (``cot`` None) or of ``<cot, model output>`` -> (g, outputs, cot)."""
    for _, p_ in H.lora_params(env.tr):
        p_.grad = None
    preds: List[torch.Tensor] = []
    orig = env.stage.step.predict_noise

    def predict(*a, **k):
        v, au = orig(*a, **k)
        if v.requires_grad and cot is None:
            v.retain_grad()
            au.retain_grad()
        preds[:] = [v, au]
        return v, au

    env.stage.step.predict_noise = predict
    try:
        cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=pair["embeds"].to(env.dev)))
        res = env.stage.replay(cond, segment=pair["seg"], params=env.params, step_indices=[pair["k"]])
        v, au = preds
        if cot is None:
            (-res.log_probs.sum()).backward()
            new_cot = (v.grad.detach().float().clone(), au.grad.detach().float().clone())
        else:
            torch.autograd.backward([v, au], [(cot[0] * cot_scale).to(v.dtype), (cot[1] * cot_scale).to(au.dtype)])
            new_cot = cot
        return flat_b(env.tr), (v.detach().float(), au.detach().float()), new_cot
    finally:
        env.stage.step.predict_noise = orig


def out_rel(a, b) -> float:
    """Relative difference of the (video, audio) model outputs taken as one vector."""
    return H.rel_diff(torch.cat([a[0].reshape(-1), a[1].reshape(-1)]), torch.cat([b[0].reshape(-1), b[1].reshape(-1)]))


def metrics(g: torch.Tensor, ref: torch.Tensor, slices) -> Dict[str, object]:
    gd, rd = g.double(), ref.double()
    blocks = sorted(slices)
    sets = {}
    for name, blks in METRIC_SETS.items():
        a = torch.cat([H.take(g, slices[b]) for b in blks])
        c = torch.cat([H.take(ref, slices[b]) for b in blks])
        sets[name] = {"rel": H.rel_diff(a, c), "cos": H.cos(a, c)}
    return {
        "sets": sets,
        "rel": float((gd - rd).norm() / rd.norm()),
        "cos": float(gd @ rd / (gd.norm() * rd.norm())),
        "norm_ratio": float(gd.norm() / rd.norm()),
        "block_rel": [H.rel_diff(H.take(g, slices[b]), H.take(ref, slices[b])) for b in blocks],
        "block_cos": [H.cos(H.take(g, slices[b]), H.take(ref, slices[b])) for b in blocks],
    }


def summarize(
    per_pair: List[Dict[str, object]], diffs: List[Dict[str, float]], base: Optional[Dict[str, object]] = None
):
    """Pair-normalised summary of one variant; ``base`` (the bf16 baseline summary) adds the paired comparison."""
    rel = [m["rel"] for m in per_pair]
    nb = len(per_pair[0]["block_rel"])
    out = {
        "rel_mean": sum(rel) / len(rel),
        "rel_min": min(rel),
        "rel_max": max(rel),
        "rel_per_pair": rel,
        "cos_per_pair": [m["cos"] for m in per_pair],
        "norm_ratio_mean": sum(m["norm_ratio"] for m in per_pair) / len(per_pair),
        "block_rel_mean": [sum(m["block_rel"][b] for m in per_pair) / len(per_pair) for b in range(nb)],
        "block_cos_mean": [sum(m["block_cos"][b] for m in per_pair) / len(per_pair) for b in range(nb)],
        "by_step_rel": {},
        "diff_rel_mean": (sum(d["rel"] for d in diffs) / len(diffs)) if diffs else None,
        "diff_cos_mean": (sum(d["cos"] for d in diffs) / len(diffs)) if diffs else None,
    }
    for m in per_pair:
        out["by_step_rel"].setdefault(str(m["k"]), []).append(m["rel"])
    out["sets"] = {
        name: {
            "rel_mean": sum(m["sets"][name]["rel"] for m in per_pair) / len(per_pair),
            "cos_mean": sum(m["sets"][name]["cos"] for m in per_pair) / len(per_pair),
            "cos_min": min(m["sets"][name]["cos"] for m in per_pair),
            "rel_per_pair": [m["sets"][name]["rel"] for m in per_pair],
        }
        for name in (per_pair[0].get("sets") or {})
    }
    if base is not None:
        out["pairs_better_than_bf16"] = sum(a < b for a, b in zip(rel, base["rel_per_pair"]))
        out["rel_mean_over_bf16"] = out["rel_mean"] / base["rel_mean"]
    return out


def eval_variant(
    env: Env, pairs, cots, refs, slices, *, cot_scale: float = 1.0, base=None, per_pair_hook=None, out_refs=None
):
    """Gradients of the current precision setup on every pair against ``refs``; same-step sample differences included."""
    per_pair: List[Optional[Dict[str, object]]] = [None] * len(pairs)  # kept in ``pairs`` order for paired comparisons
    diffs = []
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    steps = sorted({p["k"] for p in pairs})
    t_core, base_alloc = 0.0, 0.0
    for k in steps:
        held = []
        for j, p in enumerate(pairs):
            if p["k"] != k:
                continue
            if per_pair_hook is not None:
                per_pair_hook(j, p)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                base_alloc = max(base_alloc, torch.cuda.memory_allocated() / 2**30)
            t1 = time.perf_counter()
            g, outs, _ = run_pair(env, p, cot=cots[j], cot_scale=cot_scale)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_core += time.perf_counter() - t1
            g = g / cot_scale
            m = metrics(g, refs[j], slices)
            m.update({"k": k, "sample": p["sample"]})
            if out_refs is not None:
                m["out_rel"] = out_rel(outs, out_refs[j])
            per_pair[j] = m
            held.append((g, refs[j]))
        for a in range(len(held)):
            for b in range(a + 1, len(held)):
                dg, dr = held[a][0] - held[b][0], held[a][1] - held[b][1]
                diffs.append({"rel": H.rel_diff(dg, dr), "cos": H.cos(dg, dr)})
        del held
    s = summarize(per_pair, diffs, base)
    s["seconds_per_pair"] = (time.perf_counter() - t0) / len(pairs)  # includes metrics and bookkeeping
    s["replay_backward_seconds_per_pair"] = t_core / len(pairs)  # forward + backward + gradient flatten only
    s["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    s["allocated_before_gb"] = base_alloc  # model + tensors the harness keeps around; peak - this = what one pair adds
    s["finite"] = all(m["rel"] == m["rel"] and abs(m["rel"]) != float("inf") for m in per_pair)
    if out_refs is not None:
        s["out_rel_mean"] = sum(m["out_rel"] for m in per_pair) / len(per_pair)
    return s


# ----------------------------------------------------------------------------------------------- local probes


def top_mask(h: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Boolean mask of each token's ``k`` largest-magnitude channels, chosen once from the reference input."""
    idx = h.float().abs().topk(k, dim=-1).indices
    return torch.zeros_like(h, dtype=torch.bool).scatter_(-1, idx, True)


def split_rel(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    a, b = a.float().to(mask.device), b.float().to(mask.device)
    return {
        "all": H.rel_diff(a, b),
        "top": H.rel_diff(a[mask], b[mask]),
        "rest": H.rel_diff(a[~mask], b[~mask]),
        "cos": H.cos(a.reshape(-1), b.reshape(-1)),
    }


def local_block(env: Env, i: int, h: torch.Tensor, c: torch.Tensor, pol, args):
    """Block ``i`` alone: output, VJP to its input and its LoRA-B gradient for a fixed input ``h`` and output cotangent ``c``."""
    ctl = env.ctl
    keep = ctl.pol[i]
    ctl.pol[i] = pol
    try:
        x = h.to(env.dev).to(pol["stream"]).detach().requires_grad_(True)
        out = fblock(ctl, i, env.tr.transformer_blocks[i], x, *args)
        bp = [p for n, p in b_params(env.tr) if H.block_of(n + ".") == i]
        gs = torch.autograd.grad(out, [x] + bp, grad_outputs=c.to(env.dev).to(out.dtype))
        return out.detach().float(), gs[0].float(), torch.cat([g.float().reshape(-1) for g in gs[1:]])
    finally:
        ctl.pol[i] = keep


def local_head(env: Env, h: torch.Tensor, cot, head_pol, stream_dt: torch.dtype, head_args, io):
    """``norm_out`` + both output heads alone: outputs and the VJP to the final stream for the fixed output cotangent."""
    ctl = env.ctl
    keep = ctl.head
    ctl.head = head_pol
    try:
        x = h.to(env.dev).to(stream_dt).detach().requires_grad_(True)
        hs = fhead(ctl, env.tr.norm_out, x, *head_args).to(env.tr.proj_out.weight.dtype)
        v = -env.tr.proj_out(hs).index_select(1, io["video_indices"])
        a = -env.tr.audio_proj_out(hs).index_select(1, io["audio_indices"])
        (g,) = torch.autograd.grad([v, a], [x], [cot[0].to(v.dtype), cot[1].to(a.dtype)])
        return torch.cat([v.detach().float().reshape(-1), a.detach().float().reshape(-1)]), g.float()
    finally:
        ctl.head = keep


def run_tail(env: Env, start: int, h_start: torch.Tensor, cot, args, head_args, io) -> torch.Tensor:
    """Blocks ``start..`` + head from a cached stream value; LoRA-B gradient of ``<cot, output>`` for those blocks only."""
    ctl = env.ctl
    h = h_start.to(env.dev)
    for i in range(start, ctl.n):
        h = fblock(ctl, i, env.tr.transformer_blocks[i], h, *args)
    hs = fhead(ctl, env.tr.norm_out, h, *head_args).to(env.tr.proj_out.weight.dtype)
    v = -env.tr.proj_out(hs).index_select(1, io["video_indices"])
    a = -env.tr.audio_proj_out(hs).index_select(1, io["audio_indices"])
    torch.autograd.backward([v, a], [cot[0].to(v.dtype), cot[1].to(a.dtype)])
    g = flat_b(env.tr)
    sl = b_slices(env.tr)
    return torch.cat([H.take(g, sl[b]) for b in sorted(sl) if b >= start])


def new_rec() -> Dict[str, object]:
    return {"h": {}, "c": {}}


# ----------------------------------------------------------------------------------------------- stage 1


def stage1(
    env_stock: Env,
    pairs: List[Dict[str, object]],
    save: Callable[[Dict[str, object]], None],
    *,
    deadline: float,
    tail_start: int,
    range_start: int,
    probe_layers: Sequence[int],
    log=print,
) -> Dict[str, object]:
    """The whole diagnosis on a fixed list of (sample, step) pairs. ``deadline`` is a ``time.perf_counter()`` value."""
    t_start = time.perf_counter()
    left = lambda: deadline - time.perf_counter()
    RESERVE = 120.0  # seconds kept for the final state checks and the last save
    tr, dev = env_stock.tr, env_stock.dev
    R: Dict[str, object] = {
        "sections_done": [],
        "skipped": [],
        "pairs": [{"sample": p["sample"], "k": p["k"]} for p in pairs],
    }

    def done(name: str) -> None:
        R["sections_done"].append(name)
        R["elapsed_s"] = time.perf_counter() - t_start
        save(R)
        log(f"section {name} done at {R['elapsed_s']:.0f}s")

    flag0 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    sdp0 = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    R["settings"] = {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": flag0,
        "sdp_flash_memeff_math_enabled": list(sdp0),
        "autocast_enabled": torch.is_autocast_enabled(),
        "attention_backend": str(tr.transformer_blocks[0].attn.processor._attention_backend),
    }
    state0 = H.state_fingerprint(tr)

    # ---- 1a. stock baselines BEFORE any patching: bf16 on every pair (records the fixed cotangent), fp32 on pair 0.
    H.cast_arm(tr, "bf16")
    stock_b, cots, outs_b = [], [], []
    t0 = time.perf_counter()
    for p in pairs:
        g, o, c = run_pair(env_stock, p)
        stock_b.append(g)
        cots.append(c)
        outs_b.append(o)
    R["bf16_seconds_per_pair"] = (time.perf_counter() - t0) / len(pairs)
    g_rep, o_rep, _ = run_pair(env_stock, pairs[0])
    rep_g, rep_o = H.rel_diff(g_rep, stock_b[0]), out_rel(o_rep, outs_b[0])
    # Stock stream adjoints at a few block outputs, to check the wrapped model's captured ones against.
    stock_c: Dict[int, torch.Tensor] = {}
    n_blocks = len(tr.transformer_blocks)
    ctrl_layers = sorted({0, n_blocks // 2, n_blocks - 1})

    def stock_hook(i: int):
        def fwd(_m, _inp, out):
            if out.requires_grad:
                out.register_hook(lambda g_, i=i: stock_c.__setitem__(i, g_.detach().float().cpu()))

        return fwd

    handles = [tr.transformer_blocks[i].register_forward_hook(stock_hook(i)) for i in ctrl_layers]
    g_cot, _, _ = run_pair(env_stock, pairs[0], cot=cots[0])
    for hd in handles:
        hd.remove()
    H.cast_arm(tr, "fp32")
    stock_f0, out_f0, _ = run_pair(env_stock, pairs[0], cot=cots[0])
    H.cast_arm(tr, "bf16")
    lora_sha = H.lora_hash(tr)

    ctl = patch(tr)
    env = Env(env_stock.stage, env_stock.params, ctl)
    slices = b_slices(tr)
    ctl.set_all(BF)
    g_wb, o_wb, _ = run_pair(env, pairs[0], cot=cots[0])
    ctl.set_all(FP)
    g_wf, o_wf, _ = run_pair(env, pairs[0], cot=cots[0])
    tol_g, tol_o = max(5 * rep_g, 1e-6), max(5 * rep_o, 1e-6)
    R["controls"] = {
        "bf16_repeat_grad_rel": rep_g,
        "bf16_repeat_out_rel": rep_o,
        "cot_backward_vs_logp_backward_rel": H.rel_diff(g_cot, stock_b[0]),
        "wrapped_bf16_vs_stock_out_rel": out_rel(o_wb, outs_b[0]),
        "wrapped_bf16_vs_stock_grad_rel": H.rel_diff(g_wb, stock_b[0]),
        "wrapped_fp32_vs_stock_out_rel": out_rel(o_wf, out_f0),
        "wrapped_fp32_vs_stock_grad_rel": H.rel_diff(g_wf, stock_f0),
        "tol_grad": tol_g,
        "tol_out": tol_o,
    }
    c = R["controls"]
    c["passed"] = (
        c["wrapped_bf16_vs_stock_out_rel"] <= tol_o
        and c["wrapped_bf16_vs_stock_grad_rel"] <= tol_g
        and c["wrapped_fp32_vs_stock_out_rel"] <= 1e-5
        and c["wrapped_fp32_vs_stock_grad_rel"] <= 1e-4
        and c["cot_backward_vs_logp_backward_rel"] <= tol_g
    )
    done("1a_controls")
    assert c["passed"], f"wrapped forward does not reproduce the stock model: {c}"
    del g_rep, g_cot, stock_f0, g_wb, g_wf

    # ---- bf16 capture run: lora z for every pair; stream values and adjoints for pair 0.
    ctl.set_all(BF)
    z_b, z_f = [], []
    rec_b = new_rec()
    for j, p in enumerate(pairs):
        ctl.z_store = {}
        ctl.rec = rec_b if j == 0 else None
        run_pair(env, p, cot=cots[j])
        z_b.append(ctl.z_store)
    ctl.z_store, ctl.rec = None, None
    c["wrapped_bf16_stream_adjoint_vs_stock_rel"] = max(
        H.rel_diff(rec_b["c"][i].float(), stock_c[i]) for i in ctrl_layers
    )
    c["passed"] = c["passed"] and c["wrapped_bf16_stream_adjoint_vs_stock_rel"] <= max(20 * rep_g, 1e-6)
    done("1a_stream_adjoint_control")
    assert c["passed"], f"wrapped stream adjoints differ from the stock model: {c}"

    # ---- 1c. fp32 reference on every pair; lora z; d_fp32 against both z; stream capture (pair 0 full, others tail start only).
    ctl.set_all(FP)
    refs, outs_f, fac_f, fac_ff = [], [], [], []
    rec_f = new_rec()
    tail_h: Dict[int, torch.Tensor] = {}
    t0 = time.perf_counter()
    for j, p in enumerate(pairs):
        ctl.z_store = {}
        ctl.rec, ctl.rec_layers = (rec_f, None) if j == 0 else (new_rec(), [tail_start])
        ctl.z_for, ctl.fac = {"zb": z_b[j], "zf": ctl.z_store}, {"zb": {}, "zf": {}}  # zf fills in during this forward
        g, o, _ = run_pair(env, p, cot=cots[j])
        refs.append(g)
        outs_f.append(o)
        z_f.append(ctl.z_store)
        fac_f.append(flat_named(tr, ctl.fac["zb"]))  # d_fp32 x z_bf16
        fac_ff.append(flat_named(tr, ctl.fac["zf"]))  # d_fp32 x z_fp32, same contraction as the other three cells
        tail_h[j] = (rec_f if j == 0 else ctl.rec)["h"][tail_start]
    ctl.z_store, ctl.rec, ctl.rec_layers, ctl.z_for, ctl.fac = None, None, None, None, None
    R["fp32_seconds_per_pair"] = (time.perf_counter() - t0) / len(pairs)
    R["bf16_baseline"] = base = summarize(
        [
            {
                **metrics(stock_b[j], refs[j], slices),
                "k": p["k"],
                "sample": p["sample"],
                "out_rel": out_rel(outs_b[j], outs_f[j]),
            }
            for j, p in enumerate(pairs)
        ],
        [],
    )
    done("1c_reference")

    # ---- 1d. factor split: B.grad = d z^T with each factor taken from either run, contracted in fp32.
    ctl.set_all(BF)
    fac = {
        "d_b.z_b_vs_native_bf16": [],
        "d_f.z_f_vs_native_fp32": [],
        "d_b.z_f": [],
        "d_f.z_b": [],
        "d_b.z_b": [],
        "term_dd": [],
        "term_dz": [],
        "term_cross": [],
    }
    for j, p in enumerate(pairs):
        ctl.z_for, ctl.fac = {"zb": z_b[j], "zf": z_f[j]}, {"zb": {}, "zf": {}}
        run_pair(env, p, cot=cots[j])
        bb, bf_ = flat_named(tr, ctl.fac["zb"]), flat_named(tr, ctl.fac["zf"])
        ff, fb = fac_ff[j], fac_f[j]
        fac["d_b.z_b_vs_native_bf16"].append(H.rel_diff(bb, stock_b[j]))
        fac["d_f.z_f_vs_native_fp32"].append(H.rel_diff(ff, refs[j]))
        fac["d_b.z_b"].append(metrics(bb, ff, slices)["rel"])
        fac["d_b.z_f"].append(metrics(bf_, ff, slices))  # only the backprop signal is bf16
        fac["d_f.z_b"].append(metrics(fb, ff, slices))  # only the forward activation is bf16
        n = ff.double().norm()
        fac["term_dd"].append(float((bf_ - ff).double().norm() / n))
        fac["term_dz"].append(float((fb - ff).double().norm() / n))
        fac["term_cross"].append(float((bb - bf_ - fb + ff).double().norm() / n))
    ctl.z_for, ctl.fac = None, None
    for key in ("d_b.z_f", "d_f.z_b"):
        fac[key] = summarize([{**m, "k": pairs[j]["k"]} for j, m in enumerate(fac[key])], [], base)
    R["factor_split"] = fac
    del z_b, z_f, fac_f, fac_ff
    done("1d_factor_split")

    # ---- 1b. bf16-speed whole-model variants.
    V: Dict[str, object] = {}
    R["variants"] = V

    def cost(fp32_fraction: float, n: int) -> float:
        """Estimated seconds for ``n`` pairs when ``fp32_fraction`` of the model runs in fp32."""
        return n * (R["bf16_seconds_per_pair"] + fp32_fraction * R["fp32_seconds_per_pair"]) * 1.3

    def variant(
        name: str,
        setup: Callable[[], None],
        teardown: Callable[[], None] = lambda: None,
        fp32_fraction: float = 0.1,
        **kw,
    ) -> None:
        if left() < RESERVE + cost(fp32_fraction, len(pairs)):
            R["skipped"].append(name)
            return
        ctl.set_all(BF)
        setup()
        try:
            V[name] = eval_variant(env, pairs, cots, refs, slices, base=base, out_refs=outs_f, **kw)
        except Exception as e:  # a variant that cannot run is reported, never silently replaced
            V[name] = {"error": repr(e)[:400]}
            for _, p_ in H.lora_params(tr):
                p_.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            teardown()
            ctl.subst, ctl.noise, ctl.rec, ctl.z_store, ctl.z_for, ctl.fac = None, None, None, None, None, None
        log(
            f"variant {name}: "
            + json.dumps(
                {
                    k: V[name].get(k)
                    for k in ("rel_mean", "rel_mean_over_bf16", "pairs_better_than_bf16", "seconds_per_pair", "error")
                },
                default=float,
            )
        )

    def set_ops(dt: torch.dtype, ops: Sequence[str], layers: Optional[Sequence[int]] = None) -> None:
        for i in range(ctl.n) if layers is None else layers:
            for op in ops:
                ctl.pol[i][op] = dt

    def head_fp32() -> None:
        ctl.head.update({k: FP for k in HEAD_OPS})

    def flag(v: bool) -> None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = v

    def sdp(flash: bool, memeff: bool, math: bool) -> None:
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_mem_efficient_sdp(memeff)
        torch.backends.cuda.enable_math_sdp(math)

    def tf32(v: bool) -> None:
        torch.backends.cuda.matmul.allow_tf32 = v

    variant("wrapped_bf16", lambda: None)
    variant("R_reduced_precision_reduction_off", lambda: flag(False), lambda: flag(flag0))
    variant("N_head_fp32", head_fp32)
    variant("S_stream_norm_modar_fp32", lambda: (set_ops(FP, ["stream", "norm", "modar"]), head_fp32()))
    variant("S+_also_modtab_fp32", lambda: (set_ops(FP, ["stream", "norm", "modar", "modtab"]), head_fp32()))
    # Attention: the SDPA kernel the dispatcher picks is not observable, so the kernel is also pinned to "math" for a like-for-like pair.
    variant("K_core_fp32_default_kernel", lambda: set_ops(FP, ["core"]), fp32_fraction=0.3)
    variant("M_bf16_math_kernel", lambda: sdp(False, False, True), lambda: sdp(*sdp0), fp32_fraction=0.3)
    variant(
        "KM_core_fp32_math_kernel",
        lambda: (set_ops(FP, ["core"]), sdp(False, False, True)),
        lambda: sdp(*sdp0),
        fp32_fraction=0.3,
    )
    variant(
        "K2_qknorm_and_core_fp32_math_kernel",
        lambda: (set_ops(FP, ["qknorm", "core"]), sdp(False, False, True)),
        lambda: sdp(*sdp0),
        fp32_fraction=0.3,
    )
    variant("X2_cot_times_2^20", lambda: None, cot_scale=float(2**20))
    done("1b_cheap_variants")

    variant("T_fp32_with_tf32", lambda: (ctl.set_all(FP), tf32(True)), lambda: tf32(False), fp32_fraction=0.5)
    done("1c_tf32")

    # ---- 1e. per-layer local comparison on pair 0 (matched input, matched incoming cotangent).
    def local_scan(rec, layers, tag: str) -> List[Dict[str, object]]:
        rows = []
        args, head_args = rec["args"], rec["head_args"]
        for i in layers:
            if left() < RESERVE:
                R["skipped"].append(f"local_scan_{tag}_from_layer_{i}")
                break
            h = rec["h"][i].to(dev)
            mask = top_mask(h)
            hq = h.to(BF).float()
            if i == ctl.n:
                cot = cots[0]
                fpo, fpg = local_head(env, h, cot, {k: FP for k in HEAD_OPS}, FP, head_args, ctl.io)
                bfo, bfg = local_head(env, h, cot, {k: BF for k in HEAD_OPS}, BF, head_args, ctl.io)
                q_o, q_g = local_head(env, hq, cot, {k: FP for k in HEAD_OPS}, FP, head_args, ctl.io)
                rows.append(
                    {
                        "layer": i,
                        "state": tag,
                        "out_bf16_vs_fp32": H.rel_diff(bfo, fpo),
                        "vjp_bf16_vs_fp32": split_rel(bfg, fpg, mask),
                        "vjp_bf16_vs_fp32_on_rounded_input": split_rel(bfg, q_g, mask),
                        "vjp_input_rounding_only": split_rel(q_g, fpg, mask),
                    }
                )
                continue
            c = rec["c"][i].to(dev)
            cq = c.to(BF).float()
            o1, v1, l1 = local_block(env, i, h, c, policy(FP), args)
            o2, v2, l2 = local_block(env, i, h, c, policy(BF), args)
            o3, v3, l3 = local_block(env, i, hq, c, policy(FP), args)
            o4, v4, l4 = local_block(env, i, hq, cq, policy(FP), args)
            rows.append(
                {
                    "layer": i,
                    "state": tag,
                    "out_bf16_vs_fp32": split_rel(o2, o1, mask),
                    "out_bf16_vs_fp32_on_rounded_io": split_rel(o2, o4, mask),
                    "vjp_bf16_vs_fp32": split_rel(v2, v1, mask),
                    "vjp_bf16_vs_fp32_on_rounded_io": split_rel(v2, v4, mask),
                    "vjp_io_rounding_only": split_rel(v4, v1, mask),
                    "vjp_input_rounding_only": split_rel(v3, v1, mask),
                    "lora_bf16_vs_fp32": {"rel": H.rel_diff(l2, l1), "cos": H.cos(l2, l1)},
                    "lora_bf16_vs_fp32_on_rounded_io": {"rel": H.rel_diff(l2, l4), "cos": H.cos(l2, l4)},
                    "lora_io_rounding_only": {"rel": H.rel_diff(l4, l1), "cos": H.cos(l4, l1)},
                    "h_absmax": float(h.float().abs().max()),
                    "h_rms": float(h.float().pow(2).mean().sqrt()),
                    "c_rms": float(c.float().pow(2).mean().sqrt()),
                    "vjp_gain": float(v1.double().norm() / c.double().norm()),
                }
            )
        return rows

    ctl.set_all(BF)
    R["local_fp32_state"] = local_scan(rec_f, list(range(ctl.n + 1)), "fp32")
    done("1e_local_fp32_state")
    scanned = [r for r in R["local_fp32_state"] if r["layer"] < ctl.n]
    worst = [r["layer"] for r in sorted(scanned, key=lambda r: -r["vjp_bf16_vs_fp32_on_rounded_io"]["all"])[:2]]
    probes = sorted(set(list(probe_layers) + worst))
    R["probe_layers"] = probes
    R["local_bf16_state"] = local_scan(rec_b, sorted(set(probes + [ctl.n - 1, ctl.n])), "bf16")
    done("1e_local_bf16_state")

    # ---- 1h. stream statistics from the pair-0 captures.
    stats = []
    for i in range(ctl.n + 1):
        hf, hb = rec_f["h"][i].float().to(dev), rec_b["h"][i].float().to(dev)
        mask = top_mask(hf)
        row = {
            "layer": i,
            "drift_bf16_vs_fp32": split_rel(hb, hf, mask),
            "token_max_over_rms_median": float((hf.abs().amax(-1) / hf.pow(2).mean(-1).sqrt()).median()),
            "token_max_over_rms_max": float((hf.abs().amax(-1) / hf.pow(2).mean(-1).sqrt()).max()),
            "top_channel_mode_share": float(hf.abs().argmax(-1).flatten().bincount().max() / hf.shape[-2]),
        }
        if i < ctl.n:
            nxt = rec_f["h"][i + 1].float().to(dev)
            u = nxt - hf  # the block's total update to the stream
            lost = nxt.to(BF).float() - nxt  # what bf16 storage of h + u throws away
            row["storage_rounding_over_update"] = {
                "all": float(lost.norm() / u.norm()),
                "top": float(lost[mask].norm() / u[mask].norm()),
                "rest": float(lost[~mask].norm() / u[~mask].norm()),
            }
            temb = rec_f["args"][0]
            lin = tr.transformer_blocks[i].adaln_proj.linear
            tb = F.linear(F.silu(temb).to(BF), lin.weight.to(BF), lin.bias.to(BF))
            tf_ = F.linear(F.silu(temb).to(FP), lin.weight.to(FP), lin.bias.to(FP))
            hid = tr.transformer_blocks[i].adaln_proj.hidden_size  # rows are (timestep, modality); columns 6 x hidden
            sb, sf = tb.view(-1, 6, hid), tf_.view(-1, 6, hid)
            ops_b = (1.0 + sb[:, [1, 4]]).float()  # the addition itself in bf16, as the block does it
            ops_f = 1.0 + sf[:, [1, 4]]
            sb = sb.float()
            row["one_plus_scale_rel_err_max"] = float(((ops_b - ops_f).abs() / ops_f.abs().clamp_min(1e-12)).max())
            row["one_plus_scale_rel_err_rms"] = H.rel_diff(ops_b, ops_f)
            row["one_plus_scale_abs_min"] = float((1 + sf[:, [1, 4]]).abs().min())
            row["gate_rel_err"] = H.rel_diff(sb[:, [2, 5]], sf[:, [2, 5]])
        stats.append(row)
    R["stream_stats"] = stats
    done("1h_stream_stats")

    # ---- 1g. per-operation ablation: (a) tail segment on three pairs, (b) single-layer local at the probe layers.
    tail_pairs = [j for j in range(len(pairs)) if pairs[j]["k"] == pairs[0]["k"]][:3]

    def pair_args(j: int):
        """Block / head arguments of pair ``j`` (timestep tables differ per step); one cheap bf16 forward under no_grad."""
        ctl.set_all(BF)
        ctl.rec, ctl.rec_layers = new_rec(), [0]
        with torch.no_grad():
            cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=pairs[j]["embeds"].to(dev)))
            env.stage.replay(cond, segment=pairs[j]["seg"], params=env.params, step_indices=[pairs[j]["k"]])
        a, ha, io = ctl.rec["args"], ctl.rec["head_args"], dict(ctl.io)
        ctl.rec, ctl.rec_layers = None, None
        return a, ha, io

    def tail_setup(default: torch.dtype, op: Optional[str], dt: Optional[torch.dtype]) -> None:
        ctl.set_all(default)
        if op == "head":
            ctl.head = {k: dt for k in HEAD_OPS}
        elif op is not None:
            set_ops(dt, [op])

    if left() > RESERVE + 240:
        tail: Dict[str, object] = {}
        ctx = {j: pair_args(j) for j in tail_pairs}
        tref = {}
        for j in tail_pairs:
            tail_setup(FP, None, None)
            tref[j] = run_tail(env, tail_start, tail_h[j], cots[j], *ctx[j])
        names = [("all_bf16", BF, None, None)]
        names += [(f"bf16_but_{op}_fp32", BF, op, FP) for op in list(BLOCK_OPS) + ["head"]]
        names += [(f"fp32_but_{op}_bf16", FP, op, BF) for op in list(BLOCK_OPS) + ["head"]]
        for name, default, op, dt in names:
            if left() < RESERVE + 30:
                R["skipped"].append(f"tail_from_{name}")
                break
            rels, coss = [], []
            for j in tail_pairs:
                tail_setup(default, op, dt)
                # the cached input is the fp32 value; an all-bf16 tail rounds it on entry, like the real bf16 stream storage
                g = run_tail(env, tail_start, tail_h[j], cots[j], *ctx[j])
                rels.append(H.rel_diff(g, tref[j]))
                coss.append(H.cos(g, tref[j]))
            tail[name] = {"rel": rels, "cos": coss, "rel_mean": sum(rels) / len(rels)}
        R["tail"] = {"start": tail_start, "pairs": tail_pairs, "variants": tail}
        del tref
        done("1g_tail")
    else:
        R["skipped"].append("1g_tail")

    if left() > RESERVE + 120:
        loc = {}
        la = rec_f["args"]
        for i in probes:
            if left() < RESERVE + 30:
                R["skipped"].append(f"local_ops_from_layer_{i}")
                break
            h, c = rec_f["h"][i].to(dev), rec_f["c"][i].to(dev)
            hq, cq = h.to(BF).float(), c.to(BF).float()
            _, v_ref, l_ref = local_block(env, i, hq, cq, policy(FP), la)
            rows = {}
            for name, pol in (
                [("all_bf16", policy(BF))]
                + [(f"bf16_but_{op}_fp32", policy(BF, **{op: FP})) for op in BLOCK_OPS]
                + [(f"fp32_but_{op}_bf16", policy(FP, **{op: BF})) for op in BLOCK_OPS]
            ):
                _, v, l = local_block(env, i, hq, cq, pol, la)
                rows[name] = {"vjp_rel": H.rel_diff(v, v_ref), "lora_rel": H.rel_diff(l, l_ref)}
            loc[str(i)] = rows
        R["local_ops"] = loc
        done("1g_local_ops")
    else:
        R["skipped"].append("1g_local_ops")

    # ---- 1i / 1f. expensive whole-model variants, most informative first.
    variant(
        f"L_blocks_{range_start}+_fp32_head_bf16",
        lambda: set_ops(FP, BLOCK_OPS, range(range_start, ctl.n)),
        fp32_fraction=(ctl.n - range_start) / ctl.n,
    )
    done("1i_L")

    def e_hook(j: int, p: Dict[str, object]) -> None:
        """Record this pair's bf16 block outputs, then make the fp32 run reuse those values exactly."""
        ctl.subst = None
        ctl.set_all(BF)
        ctl.rec, ctl.rec_layers = new_rec(), None
        with torch.no_grad():
            cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=p["embeds"].to(dev)))
            env.stage.replay(cond, segment=p["seg"], params=env.params, step_indices=[p["k"]])
        hb = ctl.rec["h"]
        ctl.rec = None
        ctl.set_all(FP)
        ctl.subst = {i: hb[i + 1].float().to(dev) for i in range(ctl.n)}

    def clear_subst() -> None:
        ctl.subst = None

    variant("E_fp32_with_bf16_block_outputs", lambda: None, clear_subst, per_pair_hook=e_hook, fp32_fraction=1.0)
    done("1i_E")

    noise_pairs = [j for j in range(len(pairs)) if pairs[j]["k"] == pairs[0]["k"]][:3]
    R["noise"] = {}
    for eps, seed in [(1e-3, 0), (4e-3, 0), (1e-4, 0), (1e-3, 1)]:
        if left() < RESERVE + cost(1.0, len(noise_pairs)):
            R["skipped"].append(f"1f_noise_eps{eps:g}_seed{seed}")
            continue
        ctl.set_all(FP)
        ctl.noise = (eps, seed)
        sub = [pairs[j] for j in noise_pairs]
        try:
            s = eval_variant(
                env,
                sub,
                [cots[j] for j in noise_pairs],
                [refs[j] for j in noise_pairs],
                slices,
                out_refs=[outs_f[j] for j in noise_pairs],
            )
        finally:
            ctl.noise = None
        R["noise"][f"eps{eps:g}_seed{seed}"] = {
            k: s[k] for k in ("rel_mean", "rel_per_pair", "cos_per_pair", "out_rel_mean", "block_rel_mean")
        }
    R["noise_ref_bf16_same_pairs_rel"] = [base["rel_per_pair"][j] for j in noise_pairs]
    done("1f_noise")

    # ---- final state checks: nothing but the LoRA dtype may have changed.
    ctl.set_all(BF)
    for mod in tr.modules():
        if hasattr(mod, "lora_A"):
            (ad,) = mod.active_adapters
            mod.lora_A[ad].to(BF)
            mod.lora_B[ad].to(BF)
    R["final_checks"] = {
        "state_moved": H.fingerprints_match(state0, H.state_fingerprint(tr)),
        "lora_sha_same": H.lora_hash(tr) == lora_sha,
        "tf32_off": not torch.backends.cuda.matmul.allow_tf32,
        "reduced_precision_flag_restored": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == flag0,
        "sdp_flags_restored": (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        == sdp0,
    }
    f = R["final_checks"]
    f["passed"] = (
        (not f["state_moved"])
        and f["lora_sha_same"]
        and f["tf32_off"]
        and f["reduced_precision_flag_restored"]
        and f["sdp_flags_restored"]
    )
    done("final_checks")
    return R


# ----------------------------------------------------------------------------------------------- stage 2


def stage2(
    env_stock: Env,
    pairs: List[Dict[str, object]],
    save: Callable[[Dict[str, object]], None],
    *,
    deadline: float,
    cut: int,
    sharp: Sequence[int],
    held: Optional[Dict[str, object]] = None,
    rollout_fn=None,
    log=print,
) -> Dict[str, object]:
    """Candidate fixes: deep-only LoRA (blocks >= ``cut``) under several precision layouts, the fp32 route's feasibility numbers,
    and a held-out check of the best deep-only layout against stored fp32 gradients."""
    t_start = time.perf_counter()
    left = lambda: deadline - time.perf_counter()
    RESERVE = 120.0
    tr, dev = env_stock.tr, env_stock.dev
    n = len(tr.transformer_blocks)
    deep = list(range(cut, n))
    METRIC_SETS.clear()
    METRIC_SETS.update(
        {"deep_without_last": deep[:-1], "deep": deep, "shallow": list(range(cut)), "all": list(range(n))}
    )
    R: Dict[str, object] = {
        "sections_done": [],
        "skipped": [],
        "cut": cut,
        "sharp_blocks": list(sharp),
        "pairs": [{"sample": p["sample"], "k": p["k"]} for p in pairs],
    }

    def done(name: str) -> None:
        R["sections_done"].append(name)
        R["elapsed_s"] = time.perf_counter() - t_start
        save(R)
        log(f"section {name} done at {R['elapsed_s']:.0f}s")

    state0 = H.state_fingerprint(tr)
    H.cast_arm(tr, "bf16")
    stock_b, cots, outs_b = [], [], []
    t0 = time.perf_counter()
    for p in pairs:
        g, o, c = run_pair(env_stock, p)
        stock_b.append(g)
        cots.append(c)
        outs_b.append(o)
    R["bf16_seconds_per_pair"] = (time.perf_counter() - t0) / len(pairs)
    g_rep, _, _ = run_pair(env_stock, pairs[0])
    rep_g = H.rel_diff(g_rep, stock_b[0])
    lora_sha = H.lora_hash(tr)
    ctl = patch(tr)
    env = Env(env_stock.stage, env_stock.params, ctl)
    slices = b_slices(tr)
    ctl.set_all(BF)
    g_wb, o_wb, _ = run_pair(env, pairs[0], cot=cots[0])
    R["controls"] = c = {
        "bf16_repeat_grad_rel": rep_g,
        "wrapped_bf16_vs_stock_out_rel": out_rel(o_wb, outs_b[0]),
        "wrapped_bf16_vs_stock_grad_rel": H.rel_diff(g_wb, stock_b[0]),
    }
    c["passed"] = c["wrapped_bf16_vs_stock_out_rel"] <= 1e-6 and c["wrapped_bf16_vs_stock_grad_rel"] <= max(
        5 * rep_g, 1e-6
    )
    done("controls")
    assert c["passed"], f"wrapped forward does not reproduce the stock model: {c}"
    del g_rep, g_wb

    ctl.set_all(FP)
    refs, outs_f = [], []
    t0 = time.perf_counter()
    for j, p in enumerate(pairs):
        g, o, _ = run_pair(env, p, cot=cots[j])
        refs.append(g)
        outs_f.append(o)
    R["fp32_seconds_per_pair"] = (time.perf_counter() - t0) / len(pairs)
    R["bf16_baseline"] = base = summarize(
        [{**metrics(stock_b[j], refs[j], slices), "k": p["k"], "sample": p["sample"]} for j, p in enumerate(pairs)], []
    )
    done("reference")

    V: Dict[str, object] = {}
    R["variants"] = V
    all_lora = list(H.lora_params(tr))  # fixed list: lora_params() filters on requires_grad, which freeze_below toggles

    def freeze_below(on: bool) -> None:
        """Deep-only training: no LoRA below the cut is differentiated and the stream is detached there."""
        ctl.cut = cut if on else None
        for name, p_ in all_lora:
            if H.block_of(name) < cut:
                p_.requires_grad_(not on)

    def set_ops(dt: torch.dtype, ops: Sequence[str], layers: Sequence[int]) -> None:
        for i in layers:
            for op in ops:
                ctl.pol[i][op] = dt

    def variant(name: str, setup: Callable[[], None], est: float, frozen: bool = False) -> None:
        if left() < RESERVE + est * len(pairs) * 1.3:
            R["skipped"].append(name)
            return
        ctl.set_all(BF)
        setup()
        freeze_below(frozen)
        try:
            V[name] = eval_variant(env, pairs, cots, refs, slices, base=base, out_refs=outs_f)
            V[name]["frozen_below_cut"] = frozen
        except Exception as e:
            V[name] = {"error": repr(e)[:400]}
            for _, p_ in H.lora_params(tr):
                p_.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            freeze_below(False)
        v = V[name]
        log(
            f"variant {name}: "
            + json.dumps(
                {
                    "finite": v.get("finite"),
                    "core_s": v.get("replay_backward_seconds_per_pair"),
                    "deep_without_last": (v.get("sets") or {}).get("deep_without_last", {}).get("rel_mean"),
                    "deep": (v.get("sets") or {}).get("deep", {}).get("rel_mean"),
                    "all": v.get("rel_mean"),
                    "s_per_pair": v.get("seconds_per_pair"),
                    "peak_gb": v.get("peak_gb"),
                    "error": v.get("error"),
                },
                default=float,
            )
        )

    fp, bf = R["fp32_seconds_per_pair"], R["bf16_seconds_per_pair"]
    shallow, attn = list(range(cut)), ["qkv", "qknorm", "core"]

    def shallow_fp32() -> None:
        ctl.set_stem(FP)
        set_ops(FP, BLOCK_OPS, shallow)

    variant("D0_all_bf16", lambda: None, bf)
    variant("D1_shallow_fp32__deep_bf16", shallow_fp32, fp)
    variant("D2_shallow_fp32__deep_bf16_attention_fp32", lambda: (shallow_fp32(), set_ops(FP, attn, deep)), fp)
    variant(
        "D3_shallow_fp32__deep_bf16_sharp_blocks_fp32",
        lambda: (shallow_fp32(), set_ops(FP, BLOCK_OPS, list(sharp))),
        fp,
    )
    variant(
        "D4_shallow_bf16__deep_fp32",
        lambda: (set_ops(FP, BLOCK_OPS, deep), ctl.head.update({k: FP for k in HEAD_OPS})),
        fp * len(deep) / n + bf,
    )
    variant("D5_deep_bf16_attention_fp32_only", lambda: set_ops(FP, attn, deep), bf * 3)
    # The peer's idea: residual stream (and the norm / modulation arithmetic on it) in fp32, every matmul path in fp16.
    variant(
        "P1_stream_fp32__rest_fp16",
        lambda: (
            set_ops(FP, ["stream", "norm", "modar"], range(n)),
            set_ops(HF, ["modtab", "qkv", "qknorm", "core", "out", "ff1", "ff2"], range(n)),
            ctl.head.update({k: FP for k in HEAD_OPS}),
        ),
        bf * 2,
    )
    variant(
        "P2_stream_fp32__rest_bf16",
        lambda: (set_ops(FP, ["stream", "norm", "modar"], range(n)), ctl.head.update({k: FP for k in HEAD_OPS})),
        bf * 2,
    )
    done("screening")

    # Real cost of deep-only training: nothing below the cut is differentiated.
    variant("T0_frozen_all_bf16", lambda: None, bf, frozen=True)
    variant("T1_frozen_shallow_fp32__deep_bf16", shallow_fp32, fp * 0.5, frozen=True)
    variant(
        "T2_frozen_shallow_fp32__deep_bf16_attention_fp32",
        lambda: (shallow_fp32(), set_ops(FP, attn, deep)),
        fp * 0.5,
        frozen=True,
    )
    variant("T3_frozen_all_fp32", lambda: ctl.set_all(FP), fp * 0.6, frozen=True)
    done("frozen_timing")

    # ---- route 1 feasibility: an fp32 rollout replayed in fp32 has no terminal-step mismatch; cost of a 10-step graph.
    p0 = pairs[0]
    cond = MiniMaxH3Conditions(text=TextEmbedCondition(embeds=p0["embeds"].to(dev)))

    def terminal_resid(seg) -> Dict[str, float]:
        k = int(seg.sde_indices.tolist()[-1])
        with torch.no_grad():
            res = env.stage.replay(cond, segment=seg, params=env.params, step_indices=[k])
        mu = res.prev_sample_means[0, 0].float().cpu()
        xp = seg.latents[0, seg.indices.tolist().index(k + 1)].float().cpu()
        return {
            "resid_rms": float((xp - mu).pow(2).mean().sqrt()),
            "fp16_quant_rms": float((mu.half().float() - mu).pow(2).mean().sqrt()),
        }

    A: Dict[str, object] = {}
    R["route1"] = A
    if rollout_fn is not None and left() > RESERVE + 20 * fp:
        ctl.set_all(FP)
        seg32 = rollout_fn()
        A["fp32_rollout__fp32_replay_terminal"] = terminal_resid(seg32)
        ctl.set_all(BF)
        A["fp32_rollout__bf16_replay_terminal"] = terminal_resid(seg32)
        A["bf16_rollout__bf16_replay_terminal"] = terminal_resid(p0["seg"])
        ctl.set_all(FP)
        A["bf16_rollout__fp32_replay_terminal"] = terminal_resid(p0["seg"])
        done("route1_rollout_consistency")
    for tag, dt in (("bf16", BF), ("fp32", FP)):
        if left() < RESERVE + (12 * fp if dt == FP else 12 * bf) * 1.5:
            R["skipped"].append(f"ten_step_graph_{tag}")
            continue
        ctl.set_all(dt)
        for _, p_ in H.lora_params(tr):
            p_.grad = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        res = None
        try:
            res = env.stage.replay(cond, segment=p0["seg"], params=env.params, step_indices=None)
            (-res.log_probs.sum()).backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            A[f"ten_step_graph_{tag}"] = {
                "seconds": time.perf_counter() - t0,
                "steps": int(res.log_probs.numel()),
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            }
        except Exception as e:
            A[f"ten_step_graph_{tag}"] = {"error": repr(e)[:300]}
        finally:
            del res
        for _, p_ in H.lora_params(tr):
            p_.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    done("route1_ten_step_graph")

    # ---- held-out check on samples that took no part in the screening, against the stored fp32 gradients.
    if held is not None:
        cands = {
            k: v["sets"]["deep_without_last"]["rel_mean"]
            for k, v in V.items()
            if k.startswith(("D1", "D2", "D3")) and "sets" in v
        }
        best = min(cands, key=cands.get) if cands else None
        layouts = {"D0_all_bf16": lambda: None}
        if best is not None:
            layouts[best] = {
                "D1": shallow_fp32,
                "D2": lambda: (shallow_fp32(), set_ops(FP, attn, deep)),
                "D3": lambda: (shallow_fp32(), set_ops(FP, BLOCK_OPS, list(sharp))),
            }[best[:2]]
        Hd: Dict[str, object] = {"best_layout": best, "samples": [], "layouts": {k: [] for k in layouts}}
        R["held_out"] = Hd
        rows: Dict[str, List[torch.Tensor]] = {k: [] for k in layouts}
        kept: List[int] = []
        for si, smp in enumerate(held["samples"]):
            ks = [k for k in smp["seg"].sde_indices.tolist() if float(smp["seg"].sigmas[k + 1]) != 0.0]
            need = len(ks) * (bf + (fp if best else 0.0)) * 1.3
            if left() < RESERVE + need:
                R["skipped"].append(f"held_out_from_sample_{smp['id']}")
                break
            acc = {k: torch.zeros_like(refs[0]) for k in layouts}
            for k in ks:
                pr = {"sample": smp["id"], "k": k, "seg": smp["seg"], "embeds": smp["embeds"]}
                ctl.set_all(BF)
                g_b, _, cot = run_pair(
                    env, pr
                )  # wrapped bf16 == stock bf16 outputs, so this is the stored run's cotangent
                acc["D0_all_bf16"] += g_b / len(ks)
                if best is not None:
                    ctl.set_all(BF)
                    layouts[best]()
                    g_c, _, _ = run_pair(env, pr, cot=cot)
                    acc[best] += g_c / len(ks)
            ref = held["ref_fp32"][si].to(dev)
            Hd["samples"].append(smp["id"])
            Hd.setdefault("recomputed_bf16_vs_stored_bf16_rel", []).append(
                H.rel_diff(acc["D0_all_bf16"], held["ref_bf16"][si].to(dev))
            )
            Hd["reference_compatible"] = max(Hd["recomputed_bf16_vs_stored_bf16_rel"]) <= max(10 * rep_g, 1e-6)
            for name in layouts:
                m = metrics(acc[name], ref, slices)
                Hd["layouts"][name].append({"sets": m["sets"], "block_cos": m["block_cos"]})
                rows[name].append(acc[name].cpu())
            kept.append(si)
            save(R)
        # Same-prompt pairwise differences on each block set.
        Hd["pair_diff"] = {}
        for name in layouts:
            Hd["pair_diff"][name] = {}
            for sname, blks in METRIC_SETS.items():
                coss = []
                for a in range(len(kept)):
                    for b in range(a + 1, len(kept)):
                        dg = torch.cat([H.take(rows[name][a] - rows[name][b], slices[x]) for x in blks])
                        dr = torch.cat(
                            [H.take(held["ref_fp32"][kept[a]] - held["ref_fp32"][kept[b]], slices[x]) for x in blks]
                        )
                        coss.append(H.cos(dg, dr))
                Hd["pair_diff"][name][sname] = (
                    {"cos_mean": sum(coss) / len(coss), "cos_min": min(coss)} if coss else None
                )
        done("held_out")

    ctl.set_all(BF)
    ctl.cut = None
    for mod in tr.modules():
        if hasattr(mod, "lora_A"):
            (ad,) = mod.active_adapters
            mod.lora_A[ad].to(BF)
            mod.lora_B[ad].to(BF)
    R["final_checks"] = f = {
        "state_moved": H.fingerprints_match(state0, H.state_fingerprint(tr)),
        "lora_sha_same": H.lora_hash(tr) == lora_sha,
        "all_lora_trainable": all(p_.requires_grad for _, p_ in all_lora),
    }
    f["passed"] = (not f["state_moved"]) and f["lora_sha_same"] and f["all_lora_trainable"]
    done("final_checks")
    return R


OP_OF_PROBE = {
    "modtab": "modtab",
    "norm1": "norm",
    "norm2": "norm",
    "modulated1": "modar",
    "modulated2": "modar",
    "to_q": "qkv",
    "to_k": "qkv",
    "to_v": "qkv",
    "q_normed": "qknorm",
    "k_normed": "qknorm",
    "attn_core": "core",
    "to_out": "out",
    "ff1": "ff1",
    "swiglu": "ff1",
    "ff2": "ff2",
    "stream_after_attn": "stream",
    "stream_after_ff": "stream",
}
FP16_OPS = ["modtab", "qkv", "qknorm", "core", "out", "ff1", "ff2"]


def stage3_fp16(
    env_stock: Env,
    pairs: List[Dict[str, object]],
    save: Callable[[Dict[str, object]], None],
    *,
    deadline: float,
    log=print,
) -> Dict[str, object]:
    """Where does 'residual stream in fp32, everything else in fp16' overflow, and what is left once those ops leave fp16?"""
    t_start = time.perf_counter()
    left = lambda: deadline - time.perf_counter()
    tr = env_stock.tr
    n = len(tr.transformer_blocks)
    METRIC_SETS.clear()
    METRIC_SETS.update({"all": list(range(n))})
    R: Dict[str, object] = {
        "sections_done": [],
        "skipped": [],
        "pairs": [{"sample": p["sample"], "k": p["k"]} for p in pairs],
    }

    def done(name: str) -> None:
        R["sections_done"].append(name)
        R["elapsed_s"] = time.perf_counter() - t_start
        save(R)
        log(f"section {name} done at {R['elapsed_s']:.0f}s")

    state0 = H.state_fingerprint(tr)
    H.cast_arm(tr, "bf16")
    stock_b, cots = [], []
    for p in pairs:
        g, _, c = run_pair(env_stock, p)
        stock_b.append(g)
        cots.append(c)
    lora_sha = H.lora_hash(tr)
    ctl = patch(tr)
    env = Env(env_stock.stage, env_stock.params, ctl)
    slices = b_slices(tr)

    def layout(fp16_ops: Dict[int, List[str]]) -> None:
        """Stream / norm / modulation arithmetic and the head in fp32; per block, the listed ops in fp16 and the rest of FP16_OPS in bf16."""
        ctl.set_all(BF)
        ctl.head.update({k: FP for k in HEAD_OPS})
        for i in range(n):
            for op in ("stream", "norm", "modar"):
                ctl.pol[i][op] = FP
            for op in fp16_ops[i]:
                ctl.pol[i][op] = HF

    def probe_once(fp16_ops) -> Dict[str, object]:
        layout(fp16_ops)
        ctl.probe = {"fwd": {}, "bwd": {}, "order": [], "bwd_order": []}
        try:
            run_pair(env, pairs[0], cot=cots[0])
        finally:
            pr, ctl.probe = ctl.probe, None
        # With checkpointing every block runs twice; later records overwrite earlier ones with identical values.
        bad_f = [k for k in dict.fromkeys(pr["order"]) if not pr["fwd"][k][0]]
        bad_b = [k for k in dict.fromkeys(pr["bwd_order"]) if not pr["bwd"][k][0]]
        tops = sorted(
            ((v[1], k, v[2]) for k, v in pr["fwd"].items() if v[2] == "torch.float16" and v[0]), reverse=True
        )[:8]
        tops_b = sorted(
            ((v[1], k, v[2]) for k, v in pr["bwd"].items() if v[2] == "torch.float16" and v[0]), reverse=True
        )[:8]
        return {
            "first_nonfinite_forward": bad_f[:6],
            "n_nonfinite_forward": len(bad_f),
            "first_nonfinite_backward": bad_b[:6],
            "n_nonfinite_backward": len(bad_b),
            "largest_finite_fp16_forward": [[k, v] for v, k, _ in tops],
            "largest_finite_fp16_backward": [[k, v] for v, k, _ in tops_b],
            "max_abs_by_op_forward": {
                op: max((v[1] for k, v in pr["fwd"].items() if k.split(".")[1] == op and v[0]), default=None)
                for op in OP_OF_PROBE
            },
            # fraction of exactly-zero gradient entries and largest |gradient| per op, fp16 ops only: underflow shows up here
            "grad_zero_fraction_by_op_fp16": {
                op: max(
                    (v[3] for k, v in pr["bwd"].items() if k.split(".")[1] == op and v[2] == "torch.float16"),
                    default=None,
                )
                for op in OP_OF_PROBE
            },
            "grad_max_abs_by_op_fp16": {
                op: max(
                    (v[1] for k, v in pr["bwd"].items() if k.split(".")[1] == op and v[2] == "torch.float16" and v[0]),
                    default=None,
                )
                for op in OP_OF_PROBE
            },
        }

    # Round 0: the layout as stated. Then move every op that went non-finite (forward or backward) out of fp16, per block, and repeat.
    fp16_ops = {i: list(FP16_OPS) for i in range(n)}
    R["rounds"] = []
    for rnd in range(6):
        if left() < 180:
            R["skipped"].append(f"probe_round_{rnd}")
            break
        pr = probe_once(fp16_ops)
        pr["fp16_ops_count"] = sum(len(v) for v in fp16_ops.values())
        R["rounds"].append(pr)
        log(
            f"round {rnd}: fwd non-finite {pr['n_nonfinite_forward']} first {pr['first_nonfinite_forward'][:3]}; "
            f"bwd non-finite {pr['n_nonfinite_backward']} first {pr['first_nonfinite_backward'][:3]}"
        )
        bad = pr["first_nonfinite_forward"][:1] or pr["first_nonfinite_backward"][:1]
        if not bad:
            break
        # Only the EARLIEST offender is a cause; everything after it is contaminated. Remove that op from fp16 in every block.
        op = OP_OF_PROBE[bad[0].split(".")[1]]
        if (
            op not in FP16_OPS
        ):  # the stream / norm / modulation arithmetic is fp32 here, so its input came out of an fp16 op
            op = {"stream": "out", "norm": "out", "modar": "modtab"}[op]
        pr["removed_from_fp16"] = op
        for i in range(n):
            if op in fp16_ops[i]:
                fp16_ops[i].remove(op)
    R["final_fp16_ops"] = sorted({op for v in fp16_ops.values() for op in v})
    done("localise")

    ctl.set_all(FP)
    refs = [run_pair(env, p, cot=cots[j])[0] for j, p in enumerate(pairs)]
    base = summarize(
        [{**metrics(stock_b[j], refs[j], slices), "k": p["k"], "sample": p["sample"]} for j, p in enumerate(pairs)], []
    )
    R["bf16_baseline_rel_mean"] = base["rel_mean"]
    V: Dict[str, object] = {}
    R["variants"] = V
    # fp16 has a narrow range at BOTH ends: the back-propagated signal here is ~1e-8 and smaller, below fp16's smallest
    # normal number (6e-5). Scaling the cotangent by a power of two (what a GradScaler does) tests that separately.
    for name, ops, scale in (
        ("stream_fp32__rest_fp16_as_stated", {i: list(FP16_OPS) for i in range(n)}, 1.0),
        ("stream_fp32__fp16_where_it_stays_finite", fp16_ops, 1.0),
        ("stream_fp32__fp16_where_finite__cotangent_x2^16", fp16_ops, float(2**16)),
        ("stream_fp32__fp16_where_finite__cotangent_x2^24", fp16_ops, float(2**24)),
        ("stream_fp32__rest_fp16_as_stated__cotangent_x2^24", {i: list(FP16_OPS) for i in range(n)}, float(2**24)),
        ("stream_fp32__rest_bf16", {i: [] for i in range(n)}, 1.0),
    ):
        if left() < 120:
            R["skipped"].append(name)
            continue
        layout(ops)
        try:
            v = eval_variant(env, pairs, cots, refs, slices, base=base, cot_scale=scale)
            V[name] = {
                k: v[k]
                for k in (
                    "finite",
                    "rel_mean",
                    "rel_mean_over_bf16",
                    "rel_per_pair",
                    "cos_per_pair",
                    "replay_backward_seconds_per_pair",
                    "peak_gb",
                )
            }
        except Exception as e:
            V[name] = {"error": repr(e)[:300]}
        log(f"variant {name}: " + json.dumps(V[name], default=float)[:300])
    done("variants")

    ctl.set_all(BF)
    for mod in tr.modules():
        if hasattr(mod, "lora_A"):
            (ad,) = mod.active_adapters
            mod.lora_A[ad].to(BF)
            mod.lora_B[ad].to(BF)
    R["final_checks"] = f = {
        "state_moved": H.fingerprints_match(state0, H.state_fingerprint(tr)),
        "lora_sha_same": H.lora_hash(tr) == lora_sha,
    }
    f["passed"] = (not f["state_moved"]) and f["lora_sha_same"]
    done("final_checks")
    return R


def stage4_scaled_fp16(
    env_stock: Env,
    pairs: List[Dict[str, object]],
    save: Callable[[Dict[str, object]], None],
    *,
    deadline: float,
    cut: int,
    log=print,
) -> Dict[str, object]:
    """The peer's fp16 layout (power-of-two input scaling on ``attn.to_out.0`` and ``ff.net.2``, loss scale 2^16) under a fixed cotangent."""
    t_start = time.perf_counter()
    left = lambda: deadline - time.perf_counter()
    tr = env_stock.tr
    n = len(tr.transformer_blocks)
    METRIC_SETS.clear()
    METRIC_SETS.update(
        {
            "all": list(range(n)),
            "shallow": list(range(cut)),
            "deep_without_last": list(range(cut, n - 1)),
            "blocks_3_18": [b for b in range(3, 19) if b < n],
            "blocks_19_plus": [b for b in range(19, n)] or [n - 1],
        }
    )
    R: Dict[str, object] = {
        "sections_done": [],
        "skipped": [],
        "pairs": [{"sample": p["sample"], "k": p["k"]} for p in pairs],
    }

    def done(name: str) -> None:
        R["sections_done"].append(name)
        R["elapsed_s"] = time.perf_counter() - t_start
        save(R)
        log(f"section {name} done at {R['elapsed_s']:.0f}s")

    state0 = H.state_fingerprint(tr)
    H.cast_arm(tr, "bf16")
    stock_b, cots = [], []
    for p in pairs:
        g, _, c = run_pair(env_stock, p)
        stock_b.append(g)
        cots.append(c)
    lora_sha = H.lora_hash(tr)
    ctl = patch(tr)
    env = Env(env_stock.stage, env_stock.params, ctl)
    slices = b_slices(tr)
    ctl.set_all(FP)
    refs = [run_pair(env, p, cot=cots[j])[0] for j, p in enumerate(pairs)]
    base = summarize(
        [{**metrics(stock_b[j], refs[j], slices), "k": p["k"], "sample": p["sample"]} for j, p in enumerate(pairs)], []
    )
    R["bf16_baseline"] = {
        "rel_mean": base["rel_mean"],
        "sets": {k: {a: v[a] for a in ("rel_mean", "cos_mean", "cos_min")} for k, v in base["sets"].items()},
        "block_cos_mean": base["block_cos_mean"],
    }
    done("reference")

    def peer_layout(
        low: torch.dtype,
        *,
        stem_fp32: bool = True,
        scaled: bool = True,
        qknorm_fp32: bool = True,
        swiglu_fp32: bool = True,
        modtab_fp32: bool = True,
    ) -> None:
        """Stem, head, stream, block RMSNorm, AdaLN arithmetic and gates in fp32; the matmul paths in ``low``."""
        ctl.set_all(BF)
        ctl.set_stem(FP if stem_fp32 else BF)
        ctl.head.update({k: FP for k in HEAD_OPS})
        for i in range(n):
            P = ctl.pol[i]
            for op in ("stream", "norm", "modar"):
                P[op] = FP
            P["modtab"] = (
                FP if modtab_fp32 else low
            )  # theirs: fp16-rounded weights, fp32 matmul; here the stored weights in fp32
            P["qkv"], P["core"], P["out"], P["ff1"], P["ff2"] = low, low, low, low, low
            P["qknorm"] = FP if qknorm_fp32 else low
            P["swiglu"] = FP if swiglu_fp32 else low
            if scaled:
                P["out_scale"], P["ff2_scale"] = 64.0, 256.0

    def probe(setup: Callable[[], None], scale: float) -> Dict[str, object]:
        setup()
        ctl.probe = {"fwd": {}, "bwd": {}, "order": [], "bwd_order": []}
        try:
            run_pair(env, pairs[0], cot=cots[0], cot_scale=scale)
        finally:
            pr, ctl.probe = ctl.probe, None
        low_f = [(v[1], k) for k, v in pr["fwd"].items() if v[2] == "torch.float16" and v[0]]
        low_b = [v for v in pr["bwd"].values() if v[2] == "torch.float16"]
        return {
            "n_nonfinite_forward": sum(not v[0] for v in pr["fwd"].values()),
            "n_nonfinite_backward": sum(not v[0] for v in pr["bwd"].values()),
            "largest_fp16_forward_value": max(low_f)[0] if low_f else None,
            "largest_fp16_forward_at": max(low_f)[1] if low_f else None,
            "fp16_grad_zero_fraction_max": max((v[3] for v in low_b), default=None),
            "fp16_grad_zero_fraction_mean": (sum(v[3] for v in low_b) / len(low_b)) if low_b else None,
        }

    V: Dict[str, object] = {}
    R["variants"] = V
    S16 = float(2**16)
    layouts = [
        ("peer_fp16_layout__loss_scale_2^16", lambda: peer_layout(HF), S16),
        ("peer_fp16_layout__loss_scale_2^7", lambda: peer_layout(HF), 128.0),
        ("peer_layout_with_bf16_instead_of_fp16__2^16", lambda: peer_layout(BF), S16),
        ("peer_fp16_layout__stem_bf16__2^16", lambda: peer_layout(HF, stem_fp32=False), S16),
        ("peer_fp16_layout__qk_norm_in_fp16__2^16", lambda: peer_layout(HF, qknorm_fp32=False), S16),
        ("peer_fp16_layout__no_input_scaling__2^16", lambda: peer_layout(HF, scaled=False), S16),
    ]
    for name, setup, scale in layouts:
        if left() < 150:
            R["skipped"].append(name)
            continue
        try:
            pr = probe(setup, scale)
            setup()
            v = eval_variant(env, pairs, cots, refs, slices, base=base, cot_scale=scale)
            V[name] = {
                "probe": pr,
                "finite": v["finite"],
                "rel_mean": v["rel_mean"],
                "rel_mean_over_bf16": v["rel_mean_over_bf16"],
                "cos_per_pair": v["cos_per_pair"],
                "norm_ratio_mean": v["norm_ratio_mean"],
                "sets": {k: {a: s_[a] for a in ("rel_mean", "cos_mean", "cos_min")} for k, s_ in v["sets"].items()},
                "block_cos_mean": v["block_cos_mean"],
                "replay_backward_seconds_per_pair": v["replay_backward_seconds_per_pair"],
                "peak_gb": v["peak_gb"],
            }
        except Exception as e:
            V[name] = {"error": repr(e)[:300]}
            for _, p_ in H.lora_params(tr):
                p_.grad = None
        log(
            f"variant {name}: "
            + json.dumps({k: V[name].get(k) for k in ("finite", "rel_mean", "probe", "error")}, default=float)[:420]
        )
        save(R)
    done("variants")

    ctl.set_all(BF)
    for mod in tr.modules():
        if hasattr(mod, "lora_A"):
            (ad,) = mod.active_adapters
            mod.lora_A[ad].to(BF)
            mod.lora_B[ad].to(BF)
    R["final_checks"] = f = {
        "state_moved": H.fingerprints_match(state0, H.state_fingerprint(tr)),
        "lora_sha_same": H.lora_hash(tr) == lora_sha,
    }
    f["passed"] = (not f["state_moved"]) and f["lora_sha_same"]
    done("final_checks")
    return R


def b_part_index(tr: torch.nn.Module) -> torch.Tensor:
    """Indices of the ``lora_B`` entries inside the full (A and B) LoRA flat vector used by the stored 32-sample gradients."""
    idx, off = [], 0
    for name, p in H.lora_params(tr):
        if ".lora_B." in name:
            idx.append(torch.arange(off, off + p.numel()))
        off += p.numel()
    return torch.cat(idx)


# ----------------------------------------------------------------------------------------------- CPU self-test


def selftest(args) -> None:
    """Tiny-model checks of every code path stage1 uses, plus identities the real run cannot afford to verify."""
    from unirl.models.minimax_h3.packing import MiniMaxH3Geometry
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.lora import inject_lora
    from unirl.types.sampling import DiffusionSamplingParams

    torch.set_num_threads(8)
    dev = torch.device("cpu")
    params = DiffusionSamplingParams(
        num_inference_steps=4, guidance_scale=1.0, height=64, width=64, num_frames=124, eta=0.7
    )
    geometry = MiniMaxH3Geometry.from_params(params)
    sigmas = get_sigma_schedule(num_steps=4, shift=12.0)
    tr = H.build_tiny(args.layers, 128, 4, 96, 0)
    for name, mod in tr.named_children():
        if not any(k in name for k in H.KEEP_FP32):
            mod.to(BF)
    tr.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr, rank=8, alpha=16, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    tr.eval()
    tr.enable_gradient_checkpointing()
    stage = H.make_stage(tr, dev)
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(1, 12, 96, generator=g).to(BF)
    pairs = []
    for s in range(3):
        seg = H.rollout(stage, emb, params, sigmas, geometry, seed=s, key=f"p0s{s}")
        pairs += [{"sample": s, "k": k, "seg": seg, "embeds": emb} for k in (1, 2)]
    env0 = Env(stage, params)
    saved: List[Dict[str, object]] = []
    n = args.layers
    R = stage1(
        env0,
        pairs,
        saved.append,
        deadline=time.perf_counter() + 1e9,
        tail_start=n - 3,
        range_start=n // 2,
        probe_layers=[1],
    )
    assert R["controls"]["passed"] and R["final_checks"]["passed"], (R["controls"], R["final_checks"])
    c = R["controls"]
    assert c["wrapped_bf16_vs_stock_out_rel"] == 0.0 and c["wrapped_bf16_vs_stock_grad_rel"] == 0.0, (
        "bf16 wrapper is not bit-identical on CPU"
    )
    assert c["wrapped_fp32_vs_stock_out_rel"] == 0.0 and c["wrapped_fp32_vs_stock_grad_rel"] == 0.0, (
        "fp32 wrapper is not bit-identical on CPU"
    )
    assert not R["skipped"], R["skipped"]
    assert not any("error" in v for v in R["variants"].values()), {
        k: v for k, v in R["variants"].items() if "error" in v
    }
    V = R["variants"]
    assert V["wrapped_bf16"]["rel_per_pair"] == R["bf16_baseline"]["rel_per_pair"], (
        "wrapped bf16 differs from the stock bf16 baseline"
    )
    fs = R["factor_split"]
    assert max(fs["d_b.z_b_vs_native_bf16"]) < 2e-2, fs[
        "d_b.z_b_vs_native_bf16"
    ]  # fp32 contraction of bf16 factors vs the bf16 GEMM
    assert max(fs["d_f.z_f_vs_native_fp32"]) < 1e-5, fs["d_f.z_f_vs_native_fp32"]
    assert c["wrapped_bf16_stream_adjoint_vs_stock_rel"] == 0.0, "wrapped stream adjoints are not bit-identical on CPU"
    assert V["X2_cot_times_2^20"]["rel_per_pair"] == V["wrapped_bf16"]["rel_per_pair"], (
        "power-of-two cotangent scaling changed the gradient"
    )

    # Identities checked on a freshly patched model (stage1 restored LoRA to bf16; patch() lifts it to fp32 again).
    tr.norm_out.__dict__.pop("forward", None)
    for blk in tr.transformer_blocks:
        blk.__dict__.pop("forward", None)
    tr._forward_pre_hooks.clear()
    ctl = patch(tr)
    env = Env(stage, params, ctl)
    sl = b_slices(tr)
    p0 = pairs[0]
    ctl.set_all(BF)
    _, _, cot = run_pair(env, p0)
    ctl.set_all(FP)
    ctl.rec, ctl.z_store = new_rec(), {}
    ctl.z_for, ctl.fac = None, None
    g_ref, o_ref, _ = run_pair(env, p0, cot=cot)
    rec, zf = ctl.rec, ctl.z_store
    ctl.rec, ctl.z_store = None, None
    # (i) d_fp32 x z_fp32 rebuilt through the hook equals the native fp32 gradient.
    ctl.z_for, ctl.fac = {"zf": zf}, {"zf": {}}
    run_pair(env, p0, cot=cot)
    i_rel = H.rel_diff(flat_named(tr, ctl.fac["zf"]), g_ref)
    ctl.z_for, ctl.fac = None, None
    # (ii) local single-layer VJP at the fp32 state reproduces the captured upstream cotangent and that block's LoRA gradient.
    ii_v, ii_l = 0.0, 0.0
    for i in range(1, n):
        _, v, l = local_block(env, i, rec["h"][i], rec["c"][i], policy(FP), rec["args"])
        ii_v = max(ii_v, H.rel_diff(v, rec["c"][i - 1]))
        ii_l = max(ii_l, H.rel_diff(l, H.take(g_ref, sl[i])))
    # (iii) the head's local VJP reproduces the cotangent captured at the last block's output.
    _, hv = local_head(env, rec["h"][n], cot, {k: FP for k in HEAD_OPS}, FP, rec["head_args"], ctl.io)
    iii = H.rel_diff(hv, rec["c"][n - 1])
    # (iv) an fp32 tail from the cached stream reproduces the full run's gradient for those blocks.
    start = n - 3
    gt = run_tail(env, start, rec["h"][start], cot, rec["args"], rec["head_args"], ctl.io)
    iv = H.rel_diff(gt, torch.cat([H.take(g_ref, sl[b]) for b in sorted(sl) if b >= start]))
    # (v) substituting the run's own block outputs changes nothing: values exact, derivative path intact.
    ctl.subst = {i: rec["h"][i + 1].float() for i in range(n)}
    g_sub, o_sub, _ = run_pair(env, p0, cot=cot)
    ctl.subst = None
    v_rel, v_out = H.rel_diff(g_sub, g_ref), out_rel(o_sub, o_ref)
    # (vi) Replace against finite differences in float64: value is the replacement, derivative is the identity.
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    val = torch.randn(5, dtype=torch.float64)
    y = Replace.apply(x * 3.0, val)
    (gx,) = torch.autograd.grad((y * torch.arange(5.0, dtype=torch.float64)).sum(), x)
    vi_ok = bool(torch.equal(y.detach(), val)) and bool(
        torch.allclose(gx, 3.0 * torch.arange(5.0, dtype=torch.float64))
    )
    # (vii) zero-amplitude noise is a no-op; a non-zero one is not and is reproducible under checkpoint recompute.
    ctl.noise = (0.0, 0)
    g0, _, _ = run_pair(env, p0, cot=cot)
    ctl.noise = (1e-2, 0)
    g1, _, _ = run_pair(env, p0, cot=cot)
    g2, _, _ = run_pair(env, p0, cot=cot)
    ctl.noise = None
    vii = (H.rel_diff(g0, g_ref), H.rel_diff(g1, g_ref), H.rel_diff(g2, g1))
    out = {
        "i_factor_ff_vs_native": i_rel,
        "ii_local_vjp_vs_captured": ii_v,
        "ii_local_lora_vs_full": ii_l,
        "iii_head_vjp_vs_captured": iii,
        "iv_tail_vs_full": iv,
        "v_self_substitution_grad": v_rel,
        "v_self_substitution_out": v_out,
        "vi_replace_ok": vi_ok,
        "vii_noise0_rel": vii[0],
        "vii_noise_changes": vii[1],
        "vii_noise_repeat": vii[2],
        "bf16_baseline_rel_mean": R["bf16_baseline"]["rel_mean"],
        "variants_rel_mean": {k: v["rel_mean"] for k, v in V.items()},
        "factor": {"d_b.z_f": fs["d_b.z_f"]["rel_mean"], "d_f.z_b": fs["d_f.z_b"]["rel_mean"]},
        "tail_all_bf16": R["tail"]["variants"]["all_bf16"]["rel_mean"],
        "sections": R["sections_done"],
    }
    print("RCTEST " + json.dumps(out, default=float))
    assert i_rel < 1e-5 and ii_v < 1e-4 and ii_l < 1e-4 and iii < 1e-4 and iv < 1e-4, (
        "an identity the diagnosis relies on does not hold"
    )
    assert v_rel < 1e-6 and v_out == 0.0 and vi_ok, "exact substitution is not exact"
    assert vii[0] == 0.0 and vii[1] > 1e-4 and vii[2] == 0.0, (
        "noise injection is not a controlled, reproducible perturbation"
    )
    json.dumps(R, default=float)

    # ---- stage 2 on a fresh tiny model; the "stored" held-out references are this model's own fp32 / bf16 gradients.
    tr2 = H.build_tiny(args.layers, 128, 4, 96, 0)
    for name, mod in tr2.named_children():
        if not any(k in name for k in H.KEEP_FP32):
            mod.to(BF)
    tr2.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr2, rank=8, alpha=16, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    tr2.eval()
    tr2.enable_gradient_checkpointing()
    stage2_ = H.make_stage(tr2, dev)
    bidx = b_part_index(tr2)
    held_samples = [
        {
            "id": 10 + s,
            "seg": H.rollout(stage2_, emb, params, sigmas, geometry, seed=50 + s, key=f"h{s}"),
            "embeds": emb,
        }
        for s in range(2)
    ]
    cot_ref = H.arm_gradients(
        stage2_,
        params,
        [(emb, h["seg"]) for h in held_samples],
        "bf16",
        record_acts=False,
        drop_terminal=True,
        record_cot=True,
    )
    fx = H.arm_gradients(
        stage2_,
        params,
        [(emb, h["seg"]) for h in held_samples],
        "fp32",
        record_acts=False,
        drop_terminal=True,
        fixed_cot=cot_ref["cot"],
    )["per_sample"]
    H.cast_arm(tr2, "bf16")
    held = {"samples": held_samples, "ref_fp32": fx[:, bidx], "ref_bf16": cot_ref["per_sample"][:, bidx]}
    pairs2 = [
        {
            "sample": s,
            "k": k,
            "seg": H.rollout(stage2_, emb, params, sigmas, geometry, seed=s, key=f"p0s{s}"),
            "embeds": emb,
        }
        for s in range(2)
        for k in (1, 2)
    ]
    R2 = stage2(
        Env(stage2_, params),
        pairs2,
        saved.append,
        deadline=time.perf_counter() + 1e9,
        cut=n - 4,
        sharp=[n - 1],
        held=held,
        rollout_fn=lambda: H.rollout(stage2_, emb, params, sigmas, geometry, seed=77, key="fp32roll"),
    )
    assert R2["controls"]["passed"] and R2["final_checks"]["passed"], (R2["controls"], R2["final_checks"])
    assert not R2["skipped"], R2["skipped"]
    errs = {
        k: v["error"] for k, v in R2["variants"].items() if "error" in v and not k.startswith("P1")
    }  # fp16 kernels may be missing on CPU
    assert not errs, errs
    V2 = R2["variants"]
    assert V2["D0_all_bf16"]["rel_per_pair"] == R2["bf16_baseline"]["rel_per_pair"]
    # frozen runs give exactly the same deep-block gradients as the unfrozen layout, and zero below the cut
    for a, b in (
        ("T0_frozen_all_bf16", "D0_all_bf16"),
        ("T1_frozen_shallow_fp32__deep_bf16", "D1_shallow_fp32__deep_bf16"),
    ):
        assert V2[a]["sets"]["deep"]["rel_per_pair"] == V2[b]["sets"]["deep"]["rel_per_pair"], (a, b)
        assert all(abs(x - 1.0) < 1e-12 for x in V2[a]["sets"]["shallow"]["rel_per_pair"]), a
    assert max(V2["T3_frozen_all_fp32"]["sets"]["deep"]["rel_per_pair"]) == 0.0, (
        "frozen all-fp32 must reproduce the reference on the deep blocks"
    )
    ho = R2["held_out"]
    assert max(ho["recomputed_bf16_vs_stored_bf16_rel"]) < 1e-6 and ho["reference_compatible"], ho[
        "recomputed_bf16_vs_stored_bf16_rel"
    ]
    r1 = R2["route1"]
    assert (
        r1["fp32_rollout__fp32_replay_terminal"]["resid_rms"]
        <= 1.5 * r1["fp32_rollout__fp32_replay_terminal"]["fp16_quant_rms"]
    )
    assert (
        r1["bf16_rollout__bf16_replay_terminal"]["resid_rms"]
        <= 1.5 * r1["bf16_rollout__bf16_replay_terminal"]["fp16_quant_rms"]
    )
    assert r1["ten_step_graph_fp32"]["steps"] == 4 and r1["ten_step_graph_bf16"]["steps"] == 4
    json.dumps(R2, default=float)
    tr3 = H.build_tiny(args.layers, 128, 4, 96, 0)
    for name, mod in tr3.named_children():
        if not any(k in name for k in H.KEEP_FP32):
            mod.to(BF)
    tr3.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr3, rank=8, alpha=16, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    tr3.eval()
    tr3.enable_gradient_checkpointing()
    stage3_ = H.make_stage(tr3, dev)
    pairs3 = [
        {
            "sample": s_,
            "k": k,
            "seg": H.rollout(stage3_, emb, params, sigmas, geometry, seed=s_, key=f"p0s{s_}"),
            "embeds": emb,
        }
        for s_ in range(2)
        for k in (1, 2)
    ]
    R3 = stage3_fp16(Env(stage3_, params), pairs3, saved.append, deadline=time.perf_counter() + 1e9)
    assert R3["final_checks"]["passed"] and not R3["skipped"], (R3["final_checks"], R3["skipped"])
    assert R3["rounds"][0]["n_nonfinite_forward"] == 0 and R3["rounds"][0]["n_nonfinite_backward"] == 0, (
        "the tiny model should not overflow fp16"
    )
    assert set(R3["rounds"][0]["max_abs_by_op_forward"]) == set(OP_OF_PROBE) and all(
        v is not None for v in R3["rounds"][0]["max_abs_by_op_forward"].values()
    )
    assert (
        R3["variants"]["stream_fp32__rest_fp16_as_stated"]["rel_per_pair"]
        == R3["variants"]["stream_fp32__fp16_where_it_stays_finite"]["rel_per_pair"]
    )
    assert not any("error" in v for v in R3["variants"].values()), R3["variants"]
    json.dumps(R3, default=float)
    print(
        "RC3TEST "
        + json.dumps(
            {
                "round0": {k: R3["rounds"][0][k] for k in ("n_nonfinite_forward", "n_nonfinite_backward")},
                "variants": {k: v.get("rel_mean") for k, v in R3["variants"].items()},
            },
            default=float,
        )
    )
    tr4 = H.build_tiny(args.layers, 128, 4, 96, 0)
    for name, mod in tr4.named_children():
        if not any(k in name for k in H.KEEP_FP32):
            mod.to(BF)
    tr4.requires_grad_(False)
    torch.manual_seed(0)
    inject_lora(tr4, rank=8, alpha=16, target_modules=H.LORA_TARGETS, module_prefix="transformer_blocks")
    tr4.eval()
    tr4.enable_gradient_checkpointing()
    stage4_ = H.make_stage(tr4, dev)
    pairs4 = [
        {
            "sample": s_,
            "k": k,
            "seg": H.rollout(stage4_, emb, params, sigmas, geometry, seed=s_, key=f"p0s{s_}"),
            "embeds": emb,
        }
        for s_ in range(2)
        for k in (1, 2)
    ]
    R4 = stage4_scaled_fp16(Env(stage4_, params), pairs4, saved.append, deadline=time.perf_counter() + 1e9, cut=n - 4)
    assert R4["final_checks"]["passed"] and not R4["skipped"], (R4["final_checks"], R4["skipped"])
    assert not any("error" in v for v in R4["variants"].values()), R4["variants"]
    v4 = R4["variants"]
    # Power-of-two input scaling is exact while nothing overflows or underflows: on the tiny model it must not change the bf16 result.
    assert (
        v4["peer_layout_with_bf16_instead_of_fp16__2^16"]["finite"]
        and v4["peer_fp16_layout__loss_scale_2^16"]["finite"]
    )
    json.dumps(R4, default=float)
    print(
        "RC4TEST "
        + json.dumps(
            {
                k: {
                    "finite": v["finite"],
                    "rel": v["rel_mean"],
                    "zero_frac": v["probe"]["fp16_grad_zero_fraction_mean"],
                }
                for k, v in v4.items()
            },
            default=float,
        )
    )
    print(
        "RC2TEST "
        + json.dumps(
            {
                "variants_deep_wo_last": {
                    k: v.get("sets", {}).get("deep_without_last", {}).get("rel_mean") for k, v in V2.items()
                },
                "held_best": ho["best_layout"],
                "route1": r1,
                "sections": R2["sections_done"],
            },
            default=float,
        )
    )
    print("RCTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    selftest(ap.parse_args())
