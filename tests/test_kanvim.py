import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch

from kanvim.kan_layers import KANScorer, MLPScorer, count_params, match_mlp_hidden
from kanvim.metrics import vim_macs, total_macs
from kanvim.model import ReducedVim
from kanvim.reducers import ALL_REDUCERS, build_reducer, num_keep
from kanvim.tinyvim import TinyVim

D, GRID = 64, (14, 14)


@pytest.mark.parametrize("name", ALL_REDUCERS)
@pytest.mark.parametrize("ratio", [0.75, 0.5, 0.25])
def test_shapes_and_grad(name, ratio):
    r = build_reducer(name, D, GRID, ratio, kan_hidden=8)
    x = torch.randn(2, 196, D, requires_grad=True)
    y = r.train()(x)
    assert y.shape == (2, {0.75: 147, 0.5: 98, 0.25: 49}[ratio], D)
    y.sum().backward()
    assert x.grad is not None
    assert r.eval()(x).shape == y.shape


def test_prune_preserves_order_and_is_deterministic_in_eval():
    r = build_reducer("kan_prune", D, GRID, 0.5, kan_hidden=8).eval()
    x = torch.randn(3, 196, D)
    r(x)
    idx = r.aux["idx"]
    assert (idx[:, 1:] > idx[:, :-1]).all()
    assert torch.equal(r(x), r(x))


def test_scorer_gets_gradient():
    for name in ["kan_prune", "mlp_prune", "kan_pool", "mlp_pool"]:
        r = build_reducer(name, D, GRID, 0.5, kan_hidden=8).train()
        r(torch.randn(2, 196, D)).pow(2).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in r.scorer.parameters()), name


def _score_grads(r, x):
    """gradient of the loss w.r.t. every token's score, and the kept-token mask"""
    s = {}

    def keep(m, i, o):
        o.retain_grad()
        s["o"] = o

    h = r.scorer.register_forward_hook(keep)
    r(x).pow(2).sum().backward()
    h.remove()
    kept = torch.zeros(x.shape[:2], dtype=torch.bool).scatter_(1, r.aux["idx"], True)
    return s["o"].grad.squeeze(-1), kept


@pytest.mark.parametrize("name", ["kan_prune", "mlp_prune"])
def test_prune_st_dropped_tokens_get_gradient(name):
    """st: every token's score gets a gradient, the dropped ones through the fused token"""
    r = build_reducer(name, D, GRID, 0.5, kan_hidden=8).train()
    g, kept = _score_grads(r, torch.randn(2, 196, D))
    assert (g[~kept].abs() > 0).all() and (g[kept].abs() > 0).all()
    legacy = build_reducer(name, D, GRID, 0.5, kan_hidden=8, prune_mode="legacy").train()
    g, kept = _score_grads(legacy, torch.randn(2, 196, D))
    assert (g[~kept] == 0).all()  # the bug of the first version, kept for reproduction


def test_prune_st_forward_is_kept_tokens_plus_mean_of_dropped():
    r = build_reducer("kan_prune", D, GRID, 0.5, kan_hidden=8).eval()
    x = torch.randn(2, 196, D)
    y = r(x)
    idx = r.aux["idx"]
    assert idx.shape[1] == 97 and y.shape == (2, 98, D)
    assert torch.allclose(y[:, :97], x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)), atol=1e-6)
    drop = torch.ones(2, 196, dtype=torch.bool).scatter_(1, idx, False)
    for b in range(2):
        assert torch.allclose(y[b, 97], x[b][drop[b]].mean(0), atol=1e-5)


def test_prune_st_without_noise_selects_like_eval():
    r = build_reducer("mlp_prune", D, GRID, 0.5).train()
    x = torch.randn(2, 196, D)
    r.noise = 0.0
    r(x)
    i_train = r.aux["idx"]
    r.eval()(x)
    assert torch.equal(i_train, r.aux["idx"])


@pytest.mark.parametrize("name", ["kan_prune", "mlp_prune"])
def test_prune_norm_is_invariant_to_score_scale(name):
    """--prune_norm: scaling / shifting the scorer output changes neither the selection nor the output"""
    r = build_reducer(name, D, GRID, 0.5, kan_hidden=8, prune_norm=True).eval()
    x = torch.randn(2, 196, D)
    y, idx = r(x), r.aux["idx"]
    h = r.scorer.register_forward_hook(lambda m, i, o: 1000 * o - 50)
    y2 = r(x)
    h.remove()
    assert torch.equal(idx, r.aux["idx"]) and torch.allclose(y, y2, atol=1e-5)


@pytest.mark.parametrize("name", ["kan_prune", "mlp_prune"])
def test_prune_norm_keeps_gradient_when_scores_blow_up(name):
    """scores of std 1000 (what the MLP scorer reached): without --prune_norm the sigmoid saturates and
    almost no token gets a gradient; with it every token still does"""
    def share_with_grad(norm):
        r = build_reducer(name, D, GRID, 0.5, kan_hidden=8, prune_norm=norm).train()
        r.noise = 0.0
        h = r.scorer.register_forward_hook(lambda m, i, o: 1000 * o / o.detach().std())
        g, _ = _score_grads(r, torch.randn(2, 196, D))
        h.remove()
        return (g.abs() > 1e-8).float().mean().item()
    assert share_with_grad(False) < 0.5
    assert share_with_grad(True) == 1.0


def test_avg_pool_is_group_mean_and_max_pool_is_max():
    x = torch.randn(1, 196, D)
    a = build_reducer("avg_pool", D, GRID, 0.25)
    m = build_reducer("max_pool", D, GRID, 0.25)
    g = a.assign
    for j in (0, 10, 48):
        assert torch.allclose(a(x)[0, j], x[0, g == j].mean(0), atol=1e-5)
        assert torch.allclose(m(x)[0, j], x[0, g == j].max(0).values)


def test_param_matching():
    kan = count_params(KANScorer(192, 32))
    h = match_mlp_hidden(192, kan)
    mlp = count_params(MLPScorer(192, h))
    assert abs(mlp - kan) / kan < 0.01


def test_full_model_forward():
    for name in ["identity", "kan_prune", "kan_pool", "mlp_pool", "max_pool"]:
        bb = TinyVim(D, 2, 10)
        m = ReducedVim(bb, build_reducer(name, D, GRID, 0.5, kan_hidden=8))
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 10)


def test_reduced_model_equals_backbone_when_identity():
    """ReducedVim(identity) must reproduce the original TinyVim forward (CLS in the middle)."""
    bb = TinyVim(D, 2, 10).eval()
    m = ReducedVim(bb, build_reducer("identity", D, GRID, 1.0)).eval()
    x = torch.randn(2, 3, 224, 224)
    t = bb.patch_embed(x)
    tp = t.shape[1] // 2
    t = torch.cat([t[:, :tp], bb.cls_token.expand(2, -1, -1), t[:, tp:]], 1) + bb.pos_embed
    res = None
    for l in bb.layers:
        t, res = l(t, res)
    ref = bb.head(bb.norm_f(res + t)[:, tp])
    assert torch.allclose(m(x), ref, atol=1e-5)


def test_macs_match_published_vim_s():
    g = vim_macs(197, 384, 24) / 1e9
    assert 5.0 < g < 5.3, g  # DyVM Table 1: Vim-S = 5.08 G


def test_macs_decrease_with_ratio():
    f = lambda r: total_macs(ReducedVim(TinyVim(D, 2, 10), build_reducer("kan_prune", D, GRID, r, 8)), D, 2, num_classes=10)
    assert f(0.25) < f(0.5) < f(0.75)


# ----------------------------------------------------------------------------- pure Vim
from kanvim.vim_pure import VimPure, selective_scan_ref, selective_scan_torch


@pytest.mark.parametrize("l", [1, 7, 8, 9, 50])
def test_chunked_scan_matches_reference(l):
    torch.manual_seed(0)
    b, d, n = 2, 6, 4
    u, dt, z = torch.randn(b, d, l), torch.randn(b, d, l), torch.randn(b, d, l)
    A = -torch.rand(d, n) * 5
    B, C = torch.randn(b, n, l), torch.randn(b, n, l)
    D, bias = torch.randn(d), torch.randn(d)
    ref = selective_scan_ref(u, dt, A, B, C, D, z, bias)
    assert torch.allclose(selective_scan_torch(u, dt, A, B, C, D, z, bias), ref, atol=1e-5)


def test_chunked_scan_large_decay_is_finite():
    b, d, n, l = 1, 2, 3, 40
    u, z = torch.randn(b, d, l), torch.randn(b, d, l)
    dt = torch.full((b, d, l), 5.0)
    A = -torch.full((d, n), 16.0)
    y = selective_scan_torch(u, dt, A, torch.randn(b, n, l), torch.randn(b, n, l), torch.ones(d), z, torch.zeros(d))
    assert torch.isfinite(y).all()


def test_vimpure_order_sensitive_and_reduced_matches():
    torch.manual_seed(0)
    m = VimPure(dim=32, depth=2, num_classes=5).eval()
    with torch.no_grad():
        m.pos_embed.normal_(0, 0.02)
    x = torch.randn(2, 3, 224, 224)
    r = ReducedVim(m, build_reducer("identity", 32, GRID, 1.0)).eval()
    with torch.no_grad():
        assert torch.allclose(m(x), r(x), atol=1e-5)
        assert r(x).shape == (2, 5)


def test_vimpure_state_dict_keys_match_official_names():
    keys = set(VimPure(dim=192, depth=24).state_dict())
    for k in ["patch_embed.proj.weight", "cls_token", "pos_embed", "norm_f.weight", "head.weight",
              "layers.0.norm.weight", "layers.0.mixer.in_proj.weight", "layers.0.mixer.conv1d_b.bias",
              "layers.0.mixer.x_proj_b.weight", "layers.0.mixer.dt_proj_b.bias", "layers.0.mixer.A_b_log",
              "layers.0.mixer.D_b", "layers.23.mixer.out_proj.weight"]:
        assert k in keys, k
    n = sum(p.numel() for p in VimPure(dim=192, depth=24).parameters())
    assert 6.5e6 < n < 7.5e6, n  # Vim-Ti: 7M params


def test_grad_ckpt_same_grads():
    torch.manual_seed(0)
    m = VimPure(dim=32, depth=2, num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    g = []
    for ck in (False, True):
        r = ReducedVim(m, build_reducer("identity", 32, GRID, 1.0), grad_ckpt=ck).train()
        m.zero_grad()
        r(x).sum().backward()
        g.append(m.layers[0].mixer.in_proj.weight.grad.clone())
    assert torch.allclose(g[0], g[1], atol=1e-5)


def test_train_val_split_is_stratified_disjoint_and_fixed():
    from kanvim.data import split_train_val
    targets = [i % 10 for i in range(1000)]
    tr, va = split_train_val(1000, targets, 100)
    assert len(va) == 100 and len(tr) == 900 and not set(tr) & set(va)
    assert all(sum(targets[i] == c for i in va) == 10 for c in range(10))
    assert split_train_val(1000, targets, 100) == (tr, va)


def test_multistage_tokens_match_single_stage_macs_and_forward():
    from kanvim.reducers import build_multistage, multistage_tokens
    from kanvim.vim_pure import VimPure
    for r in (0.75, 0.5, 0.25):
        c = multistage_tokens(r)
        assert abs(sum(m * 8 for m in c) / 24 - 196 * r) <= 1.0
    bb = VimPure(D, 24, 10)
    m = ReducedVim(bb, None, stages=build_multistage("kan_prune_ms", D, 0.5, 24, kan_hidden=8))
    assert len(m.stages) == 2 and m.layer_tokens()[0] - 1 == m.reducer.m
    single = ReducedVim(VimPure(D, 24, 10), build_reducer("kan_prune", D, GRID, 0.5, 8))
    assert abs(sum(m.layer_tokens()) - sum(single.layer_tokens())) <= 24
    x = torch.randn(2, 3, 224, 224)
    out = m.train()(x)
    out.sum().backward()
    assert out.shape == (2, 10) and all(p.grad is not None for p in m.stages.parameters() if p.requires_grad)


def test_mtr_merge_keeps_order_and_count():
    from kanvim.mtr import mtr_merge, mtr_schedule
    x = torch.arange(10.).view(1, 10, 1).repeat(2, 1, 3)
    s = torch.tensor([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]] * 2).float()
    y = mtr_merge(x, s, 2)
    assert y.shape == (2, 8, 3)
    assert torch.equal(y[0, :2, 0], torch.tensor([0., 1.]))          # 'keep' untouched, order kept
    for r in (0.75, 0.5, 0.25):
        rem, cnt = mtr_schedule(r)
        assert abs(sum(cnt) / 24 - 196 * r) <= 2 and len(rem) == 23


def test_mtr_vim_runs_and_equals_baseline_without_reduction():
    from kanvim.mtr import MTRVim
    from kanvim.vim_pure import VimPure
    torch.manual_seed(0)
    rv = ReducedVim(VimPure(D, 4, 10), build_reducer("identity", D, GRID, 1.0)).eval()
    x = torch.randn(2, 3, 224, 224)
    ref = rv(x)
    m = MTRVim(rv, 1.0)
    assert sum(m.remove) == 0 and torch.allclose(m(x), ref, atol=1e-4)
    m = MTRVim(rv, 0.5)
    assert m(x).shape == (2, 10) and m.tokens[-1] < 196
