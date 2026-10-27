"""Efficiency metrics: params, FLOPs (MACs, the convention of the Vim/DyVM tables),
throughput (img/s) and peak GPU memory.

fvcore/thop do NOT count the selective scan, so the Vim backbone is counted analytically.
Check: vim_macs(197, 384, 24) = 5.1 G  (DyVM Table 1 reports 5.08 G for Vim-S)."""
import math
import time

import torch


def vim_macs(seq_len, dim, depth, d_state=16, expand=2, d_conv=4, patch=16, n_patches=196, num_classes=1000):
    e = expand * dim
    dt_rank = math.ceil(dim / 16)
    per_dir = (e * d_conv                       # depthwise conv1d
               + e * (dt_rank + 2 * d_state)    # x_proj
               + dt_rank * e                    # dt_proj
               + 3 * e * d_state)               # scan: dA, dB*x, C*h
    per_tok = dim * 2 * e + e * dim + 2 * per_dir   # in_proj + out_proj + 2 directions
    return seq_len * depth * per_tok + n_patches * patch * patch * 3 * dim + dim * num_classes


def vim_macs_layers(seq_lens, dim, d_state=16, expand=2, d_conv=4, patch=16, n_patches=196, num_classes=1000):
    """vim_macs with a different sequence length per block (multi-stage reduction, MTR)."""
    per_layer = vim_macs(1, dim, 1, d_state, expand, d_conv, patch, n_patches=0, num_classes=0)
    return sum(seq_lens) * per_layer + n_patches * patch * patch * 3 * dim + dim * num_classes


def reducer_macs(reducer, dim, n_tokens=196):
    """MACs of the reduction module itself, via torch's FLOP counter (counts mm/linear, FLOPs/2)."""
    from torch.utils.flop_counter import FlopCounterMode
    reducer = reducer.eval()
    x = torch.randn(1, n_tokens, dim)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        reducer(x)
    return fc.get_total_flops() / 2


def total_macs(model, dim, depth, n_tokens=196, num_classes=100):
    seq = model.layer_tokens() if hasattr(model, "layer_tokens") else [model.reducer.m + 1] * depth
    macs = vim_macs_layers(seq, dim, n_patches=n_tokens, num_classes=num_classes) + reducer_macs(model.reducer, dim, n_tokens)
    for r in getattr(model, "stages", []):
        macs += reducer_macs(r, dim, r.n)
    return macs


@torch.no_grad()
def throughput(model, batch=128, iters=30, warmup=10, device="cuda", amp=True):
    model.eval().to(device)
    x = torch.randn(batch, 3, 224, 224, device=device)
    ctx = torch.autocast(device, dtype=torch.float16, enabled=amp and device == "cuda")
    with ctx:
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t)


@torch.no_grad()
def peak_vram_mb(model, batch=128, device="cuda", amp=True):
    if device != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval().to(device)
    x = torch.randn(batch, 3, 224, 224, device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20
