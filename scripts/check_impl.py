"""Can the fast official Vim kernels (hustvl/Vim + mamba-ssm) be used?  python scripts/check_impl.py vim_t.pth
Prints USE_OFFICIAL and exits 0 only if
  ReducedVim(official, identity) == official Vim  and  official Vim == kanvim.vim_pure  (same checkpoint).
Otherwise run_plan.py falls back to the pure-PyTorch backbone (slower, same accuracy)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for p in [os.environ.get("VIM_PATH", ""), os.path.join(ROOT, "..", "Vim", "vim"), "/workspace/Vim/vim",
          "/kaggle/working/Vim/vim"]:
    if p and os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))

import torch  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "vim_t.pth")
if not torch.cuda.is_available():
    sys.exit("no CUDA -> pure")
try:
    import models_mamba  # noqa: F401
    import mamba_ssm  # noqa: F401
except Exception as e:  # noqa: BLE001
    sys.exit(f"official Vim not importable ({e!r}) -> pure")

from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.reducers import build_reducer  # noqa: E402

torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224, device="cuda")
off = load_vim("tiny", ckpt, 1000, "official").cuda().eval()
pure = load_vim("tiny", ckpt, 1000, "pure").cuda().eval()
with torch.no_grad():
    a = off(x).float()
    b = ReducedVim(off, build_reducer("identity", 192, (14, 14), 1.0)).cuda().eval()(x).float()
    c = pure(x).float()
scale = a.abs().max().item()
d1, d2 = (a - b).abs().max().item() / scale, (a - c).abs().max().item() / scale
print(f"relative max diff: ReducedVim vs official {d1:.2e} | pure vs official {d2:.2e}")
print("top-1 agree pure/official:", (a.argmax(1) == c.argmax(1)).float().mean().item())
if d1 < 2e-2 and d2 < 2e-2:
    print("USE_OFFICIAL")
else:
    sys.exit("mismatch -> pure")
