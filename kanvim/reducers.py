"""Token reducers: (B, N, D) patch tokens -> (B, M, D), M = round(N * keep_ratio).

Design rules taken from the papers:
  * ORDER IS PRESERVED (MTR: Mamba is sequential, order of compressed tokens matters).
  * Kept tokens are gathered into a contiguous sequence in train AND eval, so there is no
    training/inference mismatch (DyVM: masking leaks info through the SSM hidden state).
  * M is identical for every sample -> batchable (ToMe motivation).
  * Reducers see only patch tokens; the CLS token is handled by ReducedVim.

Names: identity | avg_pool | max_pool | static_drop | random_drop |
       mlp_pool | kan_pool | mlp_prune | kan_prune | kan_prune_ms | mlp_prune_ms (multi-stage)
"""
import torch
import torch.nn as nn

from .kan_layers import make_scorer, match_mlp_hidden, count_params


def num_keep(n, ratio):
    return max(1, int(round(n * ratio)))


class Reducer(nn.Module):
    def __init__(self, n_tokens, keep_ratio):
        super().__init__()
        self.n = n_tokens
        self.m = num_keep(n_tokens, keep_ratio)
        self.aux = {}

    def extra_reg(self):
        return 0.0


class Identity(Reducer):
    def __init__(self, n_tokens, keep_ratio=1.0):
        super().__init__(n_tokens, 1.0)

    def forward(self, x):
        return x


# ----------------------------------------------------------------------------- pooling
def anchor_assignment(grid_hw, m):
    """Choose m anchors evenly along the (raster-ordered) sequence, assign every token to its
    nearest anchor in 2-D. Works for any ratio (196 -> 147/98/49) and keeps anchor order."""
    h, w = grid_hw
    n = h * w
    anchors = torch.linspace(0, n - 1, m).round().long()
    ys, xs = torch.arange(n) // w, torch.arange(n) % w
    coords = torch.stack([ys, xs], 1).float()
    assign = torch.cdist(coords, coords[anchors]).argmin(1)  # (n,) in [0, m)
    assign[anchors] = torch.arange(m)  # anchor always belongs to its own group
    return assign


def group_merge(x, logits, assign, m, mode="softmax"):
    """Merge tokens of each group. mode: softmax-weighted sum (logits) | max."""
    b, n, d = x.shape
    idx3 = assign.view(1, n, 1).expand(b, n, d)
    if mode == "max":
        out = x.new_full((b, m, d), float("-inf"))
        return out.scatter_reduce(1, idx3, x, reduce="amax", include_self=True)
    idx = assign.view(1, n).expand(b, n)
    gmax = logits.new_full((b, m), float("-inf")).scatter_reduce(1, idx, logits, "amax", include_self=True)
    e = torch.exp(logits - gmax.gather(1, idx))
    denom = e.new_zeros(b, m).scatter_add(1, idx, e)
    w = e / denom.gather(1, idx)
    return x.new_zeros(b, m, d).scatter_add(1, idx3, x * w.unsqueeze(-1))


class PoolReducer(Reducer):
    """kind: avg | max | mlp | kan.  avg/max are the fixed-weight controls; mlp/kan learn
    the merge weights (softmax over a scalar logit per token inside each group)."""

    def __init__(self, n_tokens, keep_ratio, grid_hw, kind, dim=None, hidden=None, grid_size=5):
        super().__init__(n_tokens, keep_ratio)
        self.kind = kind
        self.register_buffer("assign", anchor_assignment(grid_hw, self.m), persistent=False)
        self.scorer = make_scorer(kind, dim, hidden, grid_size) if kind in ("mlp", "kan") else None

    def forward(self, x):
        if self.kind == "max":
            return group_merge(x, None, self.assign, self.m, "max")
        logits = x.new_zeros(x.shape[:2]) if self.scorer is None else self.scorer(x).squeeze(-1)
        self.aux = {"logits": logits.detach()}
        return group_merge(x, logits, self.assign, self.m)

    def extra_reg(self):
        return 0.0 if self.scorer is None else self.scorer.l1_reg()


# ----------------------------------------------------------------------------- pruning
class StaticDrop(Reducer):
    """Spatial subsampling at fixed positions (content-agnostic control)."""

    def __init__(self, n_tokens, keep_ratio):
        super().__init__(n_tokens, keep_ratio)
        self.register_buffer("idx", torch.linspace(0, n_tokens - 1, self.m).round().long(), persistent=False)

    def forward(self, x):
        return x[:, self.idx]


class RandomDrop(Reducer):
    def forward(self, x):
        b, n, d = x.shape
        idx = torch.rand(b, n, device=x.device).topk(self.m, dim=1).indices.sort(1).values
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))


class PruneReducer(Reducer):
    """Learned importance score -> keep top-M, original order. kind: mlp | kan.

    mode "st" (default): straight-through top-M mask over ALL N tokens.
      forward:  the M-1 top-scored tokens (original order) + one fused token = mean of the dropped
                tokens (EViT), so the sequence length is still M (same GMACs as the other reducers);
      backward: the hard 0/1 mask is replaced by sigmoid((score - threshold) / temp), so a kept token
                learns "would it help to fold me into the fused token" and a dropped token learns
                "would it help to keep me" (through the fused token).
      Training adds Gumbel noise scaled by self.noise; train.py anneals it to 0, so the selection at
      the end of training is the deterministic one used in eval.
    mode "legacy": the first version (gate = 1 forward, d sigmoid backward on the kept tokens only;
      dropped tokens get no gradient). Kept only to reproduce the old results.
    norm=True (--prune_norm): the scores are z-scored per image before the noise, the top-M and the
      sigmoid, so `noise` and `temp` keep their meaning whatever the scorer's output scale. Without it the
      scale drifts freely: measured on CIFAR-100, the KAN scores collapsed to a constant (std 0.005, the
      noise then picks the tokens at random) and the MLP scores blew up (std 815, sigmoid saturated).
    frozen=True (set by train.py --freeze_reducer): the scores carry no gradient, so the selection is fixed
      and only the backbone adapts to it (the kept / fused tokens still pass gradient to x).
    cover=c (--prune_cover, st only): round(c * (M-1)) of the kept slots go to evenly spaced tokens (the Static Drop
      positions), always kept; the scorer picks the remaining slots among the other tokens. Static Drop keeps the
      spatial coverage, plain top-M loses it (the kept tokens cluster); cover keeps it and adds content selection.
    aux["raw_std"]: per-image std of the scores before the z-score, to watch the scale drift."""

    def __init__(self, n_tokens, keep_ratio, kind, dim, hidden, grid_size=5, mode="st", temp=1.0, norm=False,
                 cover=0.0):
        super().__init__(n_tokens, keep_ratio)
        assert mode in ("st", "legacy"), mode
        assert cover == 0 or mode == "st", "cover needs mode st"
        self.scorer = make_scorer(kind, dim, hidden, grid_size)
        self.mode, self.temp, self.norm = mode, temp, norm
        n_cover = int(round(cover * (self.m - 1)))
        assert 0 <= n_cover < self.m - 1, (cover, self.m)
        self.register_buffer("cover_idx", torch.linspace(0, n_tokens - 1, n_cover).round().long()
                             if n_cover else torch.zeros(0, dtype=torch.long), persistent=False)
        self.noise = 1.0  # Gumbel noise scale while training (set by train.py)
        self.frozen = False  # fixed selection, no gradient into the scores (set by train.py)

    def forward(self, x):
        b, n, d = x.shape
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            raw = self.scorer(x).squeeze(-1).float()
        logits = raw
        if self.norm:
            logits = (raw - raw.mean(1, keepdim=True)) / (raw.std(1, keepdim=True) + 1e-6)
        sel = logits
        if self.training and self.noise > 0:
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            sel = logits - self.noise * torch.log(-torch.log(u))  # + Gumbel noise
        if self.mode == "legacy":
            idx = sel.topk(self.m, dim=1).indices.sort(1).values
            kept = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
            p = torch.sigmoid(logits).gather(1, idx)
            kept = kept * (p + (1 - p).detach()).unsqueeze(-1).to(x.dtype)
            self.aux = {"logits": logits.detach(), "idx": idx.detach(), "raw_std": raw.detach().std(1).mean()}
            return kept
        k = self.m - 1  # kept tokens; the M-th slot is the fused token
        assert 1 <= k < n, (n, self.m)
        if len(self.cover_idx):  # coverage tokens: score above every other token -> always in the top k
            sel = sel.index_fill(1, self.cover_idx, 1e4)
        top = sel.topk(k + 1, dim=1).values
        thr = ((top[:, k - 1] + top[:, k]) / 2).detach().unsqueeze(1)  # between the k-th and (k+1)-th score
        idx = sel.topk(k, dim=1).indices.sort(1).values
        hard = torch.zeros_like(sel).scatter_(1, idx, 1.0)
        soft = torch.sigmoid((sel - thr) / self.temp)
        mask = (hard + soft - soft.detach()).unsqueeze(-1)  # value 0/1, gradient of the sigmoid
        xf = x.float()
        kept = (xf * mask).gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        fused = (xf * (1 - mask)).sum(1, keepdim=True) / (n - k)
        self.aux = {"logits": logits.detach(), "idx": idx.detach(), "raw_std": raw.detach().std(1).mean()}
        return torch.cat([kept, fused], 1).to(x.dtype)

    def extra_reg(self):
        return self.scorer.l1_reg()


# ----------------------------------------------------------------------------- factory
def scorer_hidden(kind, dim, kan_hidden=32, grid_size=5):
    """KAN and MLP scorers get the same parameter budget: the MLP hidden width is searched
    so that count_params(MLP) ~= count_params(KAN)."""
    if kind == "kan":
        return kan_hidden
    return match_mlp_hidden(dim, count_params(make_scorer("kan", dim, kan_hidden, grid_size)))


def build_reducer(name, dim, grid_hw, keep_ratio, kan_hidden=32, grid_size=5, prune_mode="st", prune_temp=1.0,
                  prune_norm=False, prune_cover=0.0):
    n = grid_hw[0] * grid_hw[1]
    if keep_ratio >= 1.0 or name == "identity":
        return Identity(n)
    simple = {"static_drop": StaticDrop, "random_drop": RandomDrop}
    if name in simple:
        return simple[name](n, keep_ratio)
    kind, op = name.split("_")[0], name.split("_")[1]
    if kind in ("avg", "max") and op == "pool":
        return PoolReducer(n, keep_ratio, grid_hw, kind)
    hidden = scorer_hidden(kind, dim, kan_hidden, grid_size)
    if op == "pool":
        return PoolReducer(n, keep_ratio, grid_hw, kind, dim, hidden, grid_size)
    if op == "prune":
        return PruneReducer(n, keep_ratio, kind, dim, hidden, grid_size, prune_mode, prune_temp, prune_norm,
                            prune_cover)
    raise ValueError(name)


# ----------------------------------------------------------------------------- multi-stage (DyVM-style)
MS_LAYERS = (0, 8, 16)  # prune before these blocks (0 = right after patch embedding)


def multistage_tokens(keep_ratio, depth=24, layers=MS_LAYERS, n=196):
    """Per-stage token counts. Every stage keeps the same fraction q of its input; q is chosen
    so that the mean sequence length over all blocks equals n * keep_ratio, i.e. the same GMACs
    as single-stage reduction at keep_ratio (fair comparison)."""
    def counts(q):
        out, m = [], n
        for _ in layers:
            m = num_keep(m, q)
            out.append(m)
        return out

    def mean_len(c):
        bounds = list(layers[1:]) + [depth]
        return sum(m * (e - s) for m, s, e in zip(c, layers, bounds)) / depth

    lo, hi = keep_ratio, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_len(counts(mid)) < n * keep_ratio else (lo, mid)
    return min((counts(lo), counts(hi)), key=lambda c: abs(mean_len(c) - n * keep_ratio))


def build_multistage(name, dim, keep_ratio, depth=24, kan_hidden=32, grid_size=5, layers=MS_LAYERS, n=196,
                     prune_mode="st", prune_temp=1.0, prune_norm=False):
    """name: kan_prune_ms | mlp_prune_ms -> [(layer, PruneReducer), ...]. Only pruning: after the
    first stage the tokens no longer form a 2-D grid, so grouping-based pooling is undefined."""
    kind = name.split("_")[0]
    assert name in ("kan_prune_ms", "mlp_prune_ms"), name
    hidden = scorer_hidden(kind, dim, kan_hidden, grid_size)
    out, m_in = [], n
    for layer, m in zip(layers, multistage_tokens(keep_ratio, depth, layers, n)):
        r = PruneReducer(m_in, m / m_in, kind, dim, hidden, grid_size, prune_mode, prune_temp, prune_norm)
        r.m = m
        out.append((layer, r))
        m_in = m
    return out


ALL_REDUCERS = ["avg_pool", "max_pool", "static_drop", "random_drop",
                "mlp_pool", "kan_pool", "mlp_prune", "kan_prune"]
MULTISTAGE = ["kan_prune_ms", "mlp_prune_ms"]
