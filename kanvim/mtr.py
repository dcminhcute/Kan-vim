"""MTR (Training-free Token Reduction for Vision Mamba), used as an external, training-free reference.

After a Vim block, tokens are ranked by s = mean_d(Delta_fwd + Delta_bwd) of that block (Eq. 8 of
the paper). Top-k fraction -> 'keep', bottom-k -> 'source', the rest -> 'target'; every source token
is merged (mean) into its most similar target token (cosine similarity, bipartite matching), and the
remaining tokens are put back in their original order (Algorithm 1). The CLS token is never reduced.

Schedule (our choice, the paper does not fix it for Vim-Ti): an MTR module after every block except
the last, with one constant fraction k for all of them; k is chosen so that the mean sequence length
over the blocks equals 196 * keep_ratio, i.e. the same GMACs as the single-stage reducers at that
keep ratio. Needs the pure-PyTorch backbone (kanvim.vim_pure), which exposes Delta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mtr_schedule(keep_ratio, depth=24, n=196):
    """-> number of tokens removed after each block (len depth-1) and the per-block token counts."""
    def plan(k):
        cnt, rem, m = [], [], n
        for _ in range(depth):
            cnt.append(m)
            r = min(int(round(k * m)), (m - 1) // 2)
            rem.append(r)
            m -= r
        return rem[:-1], cnt

    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(plan(mid)[1]) / depth > n * keep_ratio else (lo, mid)
    return min((plan(lo), plan(hi)), key=lambda p: abs(sum(p[1]) / depth - n * keep_ratio))


def _take(x, idx):
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def mtr_merge(x, score, r):
    """x: (B, L, D) patch tokens, score: (B, L), r: tokens to remove -> (B, L - r, D), order kept."""
    if r <= 0:
        return x
    B, L, D = x.shape
    order = score.argsort(1, descending=True)
    keep_i, tgt_i, src_i = order[:, :r], order[:, r:L - r], order[:, L - r:]
    xs, xt = _take(x, src_i), _take(x, tgt_i)
    dst = (F.normalize(xs.float(), dim=-1) @ F.normalize(xt.float(), dim=-1).transpose(1, 2)).argmax(-1)  # (B, r)
    tot = xt.float().scatter_add(1, dst.unsqueeze(-1).expand(-1, -1, D), xs.float())
    cnt = torch.ones(B, xt.shape[1], device=x.device).scatter_add(1, dst, torch.ones(B, r, device=x.device))
    xt = (tot / cnt.unsqueeze(-1)).to(x.dtype)
    idx = torch.cat([keep_i, tgt_i], 1)
    vals = torch.cat([_take(x, keep_i), xt], 1)
    return _take(vals, idx.argsort(1))


class MTRVim(nn.Module):
    """Wraps a fine-tuned ReducedVim(identity) with a pure backbone; inference only."""

    def __init__(self, reduced_vim, keep_ratio):
        super().__init__()
        self.model = reduced_vim
        self.reducer = reduced_vim.reducer  # identity, for metrics
        depth = len(reduced_vim.backbone.layers)
        n = reduced_vim.backbone.patch_embed.num_patches if hasattr(reduced_vim.backbone.patch_embed, "num_patches") else 196
        self.remove, self.tokens = mtr_schedule(keep_ratio, depth, n)
        for layer in reduced_vim.backbone.layers:
            layer.mixer.record_delta = True

    def layer_tokens(self):
        return [m + 1 for m in self.tokens]

    def extra_macs(self, dim):
        """cosine similarities source x target in every MTR module"""
        return sum(r * (m - 2 * r) * dim for r, m in zip(self.remove, self.tokens))

    @torch.no_grad()
    def forward(self, img):
        rv, b = self.model, self.model.backbone
        x = b.patch_embed(img)
        B, N, D = x.shape
        tp = N // 2
        pos = b.pos_embed
        x = x + torch.cat([pos[:, :tp], pos[:, tp + 1:]], 1)
        cls = (b.cls_token + pos[:, tp:tp + 1]).expand(B, -1, -1)
        x = torch.cat([x[:, :tp], cls, x[:, tp:]], 1)
        residual = None
        for i, layer in enumerate(b.layers):
            x, residual = layer(x, residual)
            r = self.remove[i] if i < len(self.remove) else 0
            if r:
                full = residual + x
                s = layer.mixer.last_score
                patches = torch.cat([full[:, :tp], full[:, tp + 1:]], 1)
                ps = torch.cat([s[:, :tp], s[:, tp + 1:]], 1)
                y = mtr_merge(patches, ps, r)
                ntp = y.shape[1] // 2
                x, residual, tp = torch.cat([y[:, :ntp], full[:, tp:tp + 1], y[:, ntp:]], 1), None, ntp
        x = rv._final_norm(x, residual)
        return b.head(x[:, tp])
