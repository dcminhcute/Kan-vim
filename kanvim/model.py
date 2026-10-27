"""ReducedVim: Vision Mamba backbone + a token reducer right after patch embedding
(the single-stage setting of the proposal: Patch Embedding -> Token Reduction -> Vim blocks).

`forward` re-implements VisionMamba.forward_features of hustvl/Vim for the default
configuration (abs pos-embed, middle CLS token, non-RoPE, final_pool_type='none').
Check it against your checkout of models_mamba.py if you use another variant.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint


class ReducedVim(nn.Module):
    """reducer: applied right after patch embedding (single stage).
    stages: optional [(layer, reducer), ...] for multi-stage reduction; a stage at layer 0 replaces
    `reducer`, a stage at layer l>0 reduces the residual stream before block l (CLS kept aside)."""

    def __init__(self, backbone, reducer, grad_ckpt=False, stages=None):
        super().__init__()
        self.backbone = backbone
        stages = list(stages or [])
        if stages and stages[0][0] == 0:
            reducer = stages.pop(0)[1]
        self.reducer = reducer
        self.stage_layers = [l for l, _ in stages]
        self.stages = nn.ModuleList(r for _, r in stages)
        self.grad_ckpt = grad_ckpt  # recompute each block in backward: less VRAM, ~30% slower

    def layer_tokens(self):
        """Sequence length (incl. CLS) seen by every block -> GMACs."""
        b = self.backbone
        m, out = self.reducer.m, []
        stage = dict(zip(self.stage_layers, self.stages))
        for i in range(len(b.layers)):
            if i in stage:
                m = stage[i].m
            out.append(m + 1)
        return out

    @staticmethod
    def _reduce_mid(x, residual, reducer, tp):
        full = x if residual is None else residual + x       # the residual stream at this depth
        patches = torch.cat([full[:, :tp], full[:, tp + 1:]], 1)
        y = reducer(patches)
        ntp = y.shape[1] // 2
        cls = full[:, tp:tp + 1].to(y.dtype)
        return torch.cat([y[:, :ntp], cls, y[:, ntp:]], 1), ntp

    def forward(self, img):
        b = self.backbone
        x = b.patch_embed(img)                       # (B, N, D)
        B, N, D = x.shape
        tp0 = N // 2                                 # original CLS slot in pos_embed
        pos = b.pos_embed
        x = x + torch.cat([pos[:, :tp0], pos[:, tp0 + 1:]], 1)
        x = self.reducer(x)                          # (B, M, D), order preserved
        M = x.shape[1]
        tp = M // 2                                  # Vim puts CLS in the middle
        cls = (b.cls_token + pos[:, tp0:tp0 + 1]).expand(B, -1, -1)
        x = torch.cat([x[:, :tp], cls, x[:, tp:]], 1)
        x = b.pos_drop(x)

        residual = None
        stage = dict(zip(self.stage_layers, self.stages))
        for i, layer in enumerate(b.layers):
            if i in stage:
                x, tp = self._reduce_mid(x, residual, stage[i], tp)
                residual = None
            if self.grad_ckpt and self.training:
                x, residual = torch.utils.checkpoint.checkpoint(layer, x, residual, use_reentrant=False)
            else:
                x, residual = layer(x, residual)
        x = self._final_norm(x, residual)
        return b.head(x[:, tp])

    def _final_norm(self, x, residual):
        b = self.backbone
        if not getattr(b, "fused_add_norm", False):
            residual = x if residual is None else residual + b.drop_path(x)
            return b.norm_f(residual.to(dtype=b.norm_f.weight.dtype))
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
        fn = rms_norm_fn if isinstance(b.norm_f, RMSNorm) else layer_norm_fn
        return fn(x, b.norm_f.weight, b.norm_f.bias, eps=b.norm_f.eps, residual=residual,
                  prenorm=False, residual_in_fp32=b.residual_in_fp32)

    def reg_loss(self):
        return self.reducer.extra_reg() + sum(r.extra_reg() for r in self.stages)


def load_vim(name="tiny", ckpt=None, num_classes=100, impl="pure"):
    """impl='pure': kanvim.vim_pure (no CUDA build needed, default).
    impl='official': hustvl/Vim repo on PYTHONPATH + its mamba-ssm / causal-conv1d."""
    if impl == "pure":
        from .vim_pure import load_vim_pure
        return load_vim_pure(name, ckpt, num_classes)
    import os
    import sys
    import timm
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.environ.get("VIM_PATH", ""), os.path.join(here, "..", "..", "Vim", "vim"), "/workspace/Vim/vim",
              "/kaggle/working/Vim/vim"]:
        if p and os.path.isdir(p) and os.path.abspath(p) not in sys.path:
            sys.path.insert(0, os.path.abspath(p))
    import models_mamba  # noqa: F401  (registers the vim_* models with timm)
    fn = {"tiny": "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
          "small": "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"}[name]
    model = timm.create_model(fn, pretrained=False, num_classes=1000, img_size=224)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=True)
    if num_classes != 1000:  # same as load_vim_pure; VisionMamba has no reset_classifier
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model
    