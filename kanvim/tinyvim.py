"""CPU-friendly stand-in for VisionMamba (same attributes ReducedVim touches).
Its blocks are causal + anti-causal running means, so they are order-sensitive like an SSM.
Only for tests / smoke runs; real experiments use the Vim checkpoint."""
import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = hidden_states if residual is None else residual + hidden_states
        h = self.inp(self.norm(residual))
        n = h.shape[1]
        t = torch.arange(1, n + 1, device=h.device, dtype=h.dtype).view(1, n, 1)
        fwd = h.cumsum(1) / t
        bwd = h.flip(1).cumsum(1).flip(1) / t.flip(1)
        return self.out(fwd + bwd), residual


class _PatchEmbed(nn.Module):
    def __init__(self, d, patch=16, img=224):
        super().__init__()
        self.proj = nn.Conv2d(3, d, patch, patch)
        self.grid_size = (img // patch, img // patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyVim(nn.Module):
    def __init__(self, dim=64, depth=2, num_classes=100, img=224, patch=16):
        super().__init__()
        self.patch_embed = _PatchEmbed(dim, patch, img)
        n = self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.pos_drop = nn.Identity()
        self.layers = nn.ModuleList(TinyBlock(dim) for _ in range(depth))
        self.norm_f = nn.LayerNorm(dim)
        self.drop_path = nn.Identity()
        self.fused_add_norm = False
        self.head = nn.Linear(dim, num_classes)
        self.embed_dim = dim

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.embed_dim, num_classes)
