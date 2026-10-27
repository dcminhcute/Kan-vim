"""Pure-PyTorch Vision Mamba (Vim, bimamba v2, middle CLS token, abs pos-embed, RMSNorm).

Why: hustvl/Vim needs causal-conv1d 1.1.1 + its mamba-1p1p1 fork (torch 2.1.1 / CUDA 11.8),
which do not build on current Colab (Python 3.13, torch 2.11, CUDA 13).

Parameter names match the official checkpoints, so `load_state_dict(strict=True)` works.
Numerics follow mamba_inner_fn_no_out_proj + selective_scan_fn of the Vim fork:
    x, z = in_proj(h).chunk(2)
    x = silu(causal_depthwise_conv1d(x))
    dt, B, C = x_proj(x).split(...);  dt = softplus(dt_proj(dt))
    h_t = exp(dt*A) h_{t-1} + dt*B_t*x_t ;  y_t = C_t h_t + D*x_t ;  y *= silu(z)
    out = out_proj((y_fwd + flip(y_bwd)) / 2)          # "div2"
If `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` is importable it is used for
the scan (fast CUDA); otherwise a chunked exact PyTorch scan is used (slower, more memory).
Throughput/VRAM measured with the PyTorch scan are NOT comparable to published Vim numbers.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
except Exception:  # noqa: BLE001
    _cuda_scan = None


def selective_scan_torch(u, delta, A, B, C, D, z, delta_bias, chunk=4, min_log_decay=-20.0):
    """u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l); D, delta_bias: (d,).

    Chunk-parallel scan. The sequence is cut into chunks of `chunk` steps; ALL chunks are
    solved at once with cumsum (h_t = e^{S_t} * sum_{s<=t} e^{-S_s} dBu_s, S = cumsum(dt*A)),
    then a short loop over chunks (l/chunk iterations on small (b,d,n) tensors) carries the state.
    Per-step log-decay is clamped at -20 (e^-20 = 2e-9: numerically 'forget everything'),
    so |S| <= 80 inside a chunk and e^{-S} never overflows fp32."""
    dtype = u.dtype
    u, delta, z, B, C = u.float(), delta.float(), z.float(), B.float(), C.float()
    delta = F.softplus(delta + delta_bias.float()[:, None])
    b, d, l = u.shape
    n = A.shape[1]
    dA = (delta.unsqueeze(-1) * A.float()[:, None, :]).clamp(min=min_log_decay)   # (b,d,l,n)
    dBu = delta.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1) * u.unsqueeze(-1)  # (b,d,l,n)
    Ct = C.transpose(1, 2)                                                       # (b,l,n)
    pad = (-l) % chunk
    if pad:
        dA, dBu, Ct = F.pad(dA, (0, 0, 0, pad)), F.pad(dBu, (0, 0, 0, pad)), F.pad(Ct, (0, 0, 0, pad))
    nc = (l + pad) // chunk
    cum = dA.view(b, d, nc, chunk, n).cumsum(3)
    e_cum = torch.exp(cum)
    local = e_cum * (torch.exp(-cum) * dBu.view(b, d, nc, chunk, n)).cumsum(3)  # chunk started from 0
    dec, end = e_cum[:, :, :, -1], local[:, :, :, -1]                             # (b,d,nc,n)
    h = u.new_zeros(b, d, n)
    h_in = []
    for c in range(nc):
        h_in.append(h)
        h = dec[:, :, c] * h + end[:, :, c]
    hs = local + e_cum * torch.stack(h_in, 2).unsqueeze(3)                       # (b,d,nc,chunk,n)
    y = (hs * Ct.view(b, 1, nc, chunk, n)).sum(-1).reshape(b, d, nc * chunk)[:, :, :l]
    y = y + u * D.float()[:, None]
    return (y * F.silu(z)).to(dtype)


def selective_scan_ref(u, delta, A, B, C, D, z, delta_bias):
    """Step-by-step reference (tests only)."""
    delta = F.softplus(delta + delta_bias[:, None])
    h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for t in range(u.shape[2]):
        h = torch.exp(delta[:, :, t, None] * A) * h + delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        ys.append((h * C[:, None, :, t]).sum(-1))
    return (torch.stack(ys, 2) + u * D[:, None]) * F.silu(z)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = None

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class BiMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di = expand * d_model
        self.d_inner, self.d_state, self.d_conv = di, d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False)
        self.out_proj = nn.Linear(di, d_model, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1).log()
        for sfx in ("", "_b"):
            setattr(self, "conv1d" + sfx, nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1))
            setattr(self, "x_proj" + sfx, nn.Linear(di, self.dt_rank + 2 * d_state, bias=False))
            setattr(self, "dt_proj" + sfx, nn.Linear(self.dt_rank, di, bias=True))
        self.A_log = nn.Parameter(A.clone())
        self.A_b_log = nn.Parameter(A.clone())
        self.D = nn.Parameter(torch.ones(di))
        self.D_b = nn.Parameter(torch.ones(di))

    def _direction(self, xz, conv, x_proj, dt_proj, A_log, D):
        L = xz.shape[-1]
        x, z = xz.chunk(2, dim=1)                                          # (b, di, l)
        x = F.silu(conv(x)[..., :L])
        x_dbl = x_proj(x.transpose(1, 2))                                  # (b, l, r+2n)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = (dt @ dt_proj.weight.t()).transpose(1, 2)                     # (b, di, l), bias added in scan
        B, C = B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous()
        A = -torch.exp(A_log.float())
        if self.record_delta:  # MTR token importance: mean over channels of Delta = softplus(dt + bias)
            self._delta_parts.append(F.softplus(dt.float() + dt_proj.bias.float()[:, None]).mean(1))
        if _cuda_scan is not None and x.is_cuda:
            return _cuda_scan(x.contiguous(), dt.contiguous(), A, B, C, D.float(), z=z.contiguous(),
                              delta_bias=dt_proj.bias.float(), delta_softplus=True)
        return selective_scan_torch(x, dt, A, B, C, D, z, dt_proj.bias)

    record_delta = False  # set True by kanvim.mtr; then last_score = mean_d(Delta_fwd + Delta_bwd), (b, l)

    def forward(self, h):                                                  # (b, l, d)
        self._delta_parts = []
        xz = self.in_proj(h).transpose(1, 2)                               # (b, 2di, l)
        y_f = self._direction(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        y_b = self._direction(xz.flip(-1), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)
        if self.record_delta:
            self.last_score = self._delta_parts[0] + self._delta_parts[1].flip(-1)
        y = (y_f + y_b.flip(-1)) / 2
        return self.out_proj(y.transpose(1, 2).to(h.dtype))


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = BiMamba(dim)
        self.norm = RMSNorm(dim)

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = hidden_states if residual is None else residual + hidden_states
        residual = residual.float()                                        # residual_in_fp32
        return self.mixer(self.norm(residual)), residual


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch=16, img=224):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.grid_size = (img // patch, img // patch)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VimPure(nn.Module):
    def __init__(self, dim=192, depth=24, num_classes=1000, img=224, patch=16):
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = PatchEmbed(dim, patch, img)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Identity()
        self.layers = nn.ModuleList(Block(dim) for _ in range(depth))
        self.norm_f = RMSNorm(dim)
        self.drop_path = nn.Identity()
        self.fused_add_norm = False
        self.head = nn.Linear(dim, num_classes)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        tp = x.shape[1] // 2
        x = torch.cat([x[:, :tp], self.cls_token.expand(x.shape[0], -1, -1), x[:, tp:]], 1) + self.pos_embed
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x = self.norm_f(residual + x)
        return self.head(x[:, tp])


VIM_CFG = {"tiny": dict(dim=192, depth=24), "small": dict(dim=384, depth=24)}
VIM_HF = {"tiny": ("hustvl/Vim-tiny-midclstok", "vim_t_midclstok_76p1acc.pth"),
          "small": ("hustvl/Vim-small-midclstok", "vim_s_midclstok_80p5acc.pth")}


def load_vim_pure(name="tiny", ckpt=None, num_classes=100):
    m = VimPure(**VIM_CFG[name])
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch\nmissing={missing}\nunexpected={unexpected}")
    if num_classes != 1000:
        m.reset_classifier(num_classes)
    return m
