"""KAN layers and the scorer networks (KAN vs MLP) used by the token reducers.

The KAN layer follows FastKAN: B-splines are replaced by Gaussian RBFs, which keeps the
"learnable 1-D function on every edge" idea of the original KAN but is cheap enough to run
per token. (KAN paper, Sec. 2.2: phi(x) = w_b * silu(x) + w_s * spline(x).)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKANLayer(nn.Module):
    def __init__(self, d_in, d_out, grid_size=5, grid_range=(-2.0, 2.0), use_layernorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if use_layernorm else nn.Identity()
        self.register_buffer("grid", torch.linspace(grid_range[0], grid_range[1], grid_size))
        self.inv_h = (grid_size - 1) / (grid_range[1] - grid_range[0])
        self.base = nn.Linear(d_in, d_out)                      # residual "silu" branch
        self.spline = nn.Linear(d_in * grid_size, d_out, bias=False)  # RBF-basis branch

    def forward(self, x):
        xn = self.norm(x)
        phi = torch.exp(-(((xn.unsqueeze(-1) - self.grid) * self.inv_h) ** 2))
        return self.base(F.silu(xn)) + self.spline(phi.flatten(-2))

    def l1_reg(self):
        """L1 penalty on basis coefficients; 'KAN not Work' recommends it for generalisation."""
        return self.spline.weight.abs().mean()


class KANScorer(nn.Module):
    """[D -> hidden -> d_out] KAN. Produces an importance / merge-weight logit per token."""

    def __init__(self, dim, hidden, d_out=1, grid_size=5):
        super().__init__()
        self.l1 = RBFKANLayer(dim, hidden, grid_size)
        self.l2 = RBFKANLayer(hidden, d_out, grid_size, use_layernorm=True)

    def forward(self, x):
        return self.l2(self.l1(x))

    def l1_reg(self):
        return self.l1.l1_reg() + self.l2.l1_reg()


class MLPScorer(nn.Module):
    """[D -> hidden -> d_out] MLP, the control for KANScorer."""

    def __init__(self, dim, hidden, d_out=1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)

    def l1_reg(self):
        return 0.0


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_scorer(kind, dim, hidden, grid_size=5):
    if kind == "kan":
        return KANScorer(dim, hidden, 1, grid_size)
    if kind == "mlp":
        return MLPScorer(dim, hidden, 1)
    raise ValueError(kind)


def match_mlp_hidden(dim, target_params, max_hidden=8192):
    """Hidden width of the MLP scorer whose #params is closest to `target_params`.
    This is what makes the KAN-vs-MLP comparison (RQ2) parameter-matched."""
    best, best_diff = 1, float("inf")
    for h in range(1, max_hidden + 1):
        diff = abs(count_params(MLPScorer(dim, h)) - target_params)
        if diff < best_diff:
            best, best_diff = h, diff
        elif diff > best_diff and h > 8:
            break
    return best
