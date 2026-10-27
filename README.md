# KAN-based adaptive token reduction for Vision Mamba

Đồ án: *Nghiên cứu giảm token thích nghi trong Vision Mamba bằng mạng Kolmogorov-Arnold*.

```
kanvim/kan_layers.py   RBF-KAN layer, KAN/MLP scorers, parameter matching (RQ2)
kanvim/reducers.py     avg/max pool, static/random drop, MLP/KAN pool, MLP/KAN prune
kanvim/model.py        ReducedVim = Vim + reducer after patch embedding; load_vim()
kanvim/metrics.py      GMACs (analytic Vim + reducer), throughput, peak VRAM
kanvim/data.py         CIFAR-100, Tiny-ImageNet, ImageNet-100 (all resized to 224)
kanvim/vim_pure.py     pure-PyTorch Vim (loads official ckpt; no mamba-ssm build needed)
kanvim/tinyvim.py      tiny CPU stand-in backbone for tests
kanvim/mtr.py          MTR (training-free, Delta-based) reference
scripts/run_plan.py    whole experiment plan;  scripts/report_plan.py -> BAO_CAO.md
notebooks/colab_t4.ipynb  Colab T4 pipeline
train.py               fine-tune + evaluate one (reducer, ratio, seed)
scripts/sweep.py       generate/run the whole grid;  scripts/aggregate.py -> mean±std table
tests/                 pytest (CPU): shapes, order, grads, param matching, MACs ≈ 5.08G for Vim-S
```

## Chạy toàn bộ thí nghiệm
Xem `HUONG_DAN_CHAY.md`: `python scripts/run_plan.py --profile gpu --budget_hours 30` chạy dò LR, baseline,
6 phương pháp × 3 mức giữ, 3 seed, MTR, đối chứng, multi-stage, Tiny-ImageNet, ablation KAN và viết `BAO_CAO.md`.

## Quick start
```bash
python -m pytest -q                                   # CPU, ~30s
python train.py --backbone stub --smoke --epochs 1 --reducer kan_prune --ratio 0.5
# GPU machine:
bash scripts/setup_gpu.sh
python scripts/sweep.py --dataset cifar100 --backbone tiny --ckpt vim_t.pth > cifar.sh && bash cifar.sh
python scripts/aggregate.py results
```

## Design decisions (and why)
* **Order-preserving, contiguous output** in train and eval (MTR, DyVM): Mamba is sequential and masking leaks through hidden state.
* **Fixed M per sample** (196→147/98/49) so batches stay rectangular.
* **KAN Pruning** = KAN scorer → Gumbel top-M → straight-through gate (DynamicViT/DyVM style, single stage) + distillation from the un-reduced model.
* **KANPool** = M evenly spaced anchors; each token joins its nearest anchor (2-D); KAN logit → softmax weights inside the group. Works for every ratio; Avg/Max pool use the same anchors with fixed weights.
* **Fair KAN vs MLP**: `build_reducer` searches the MLP hidden width to match the KAN scorer's parameter count.
* **KAN variant** = FastKAN (RBF) instead of B-spline for speed; L1 reg on coefficients ("KAN not Work" warns of poor generalisation).
* **MACs**, not FLOPs ×2, as in Vim/DyVM tables; selective scan counted analytically (fvcore/thop skip it).

## Backbone implementation
Default `--impl pure`: `kanvim/vim_pure.py`, because hustvl/Vim's CUDA kernels (torch 2.1.1/CUDA 11.8) do not build on Colab (Python 3.13, torch 2.11, CUDA 13). The scan is an exact chunked PyTorch scan (tested against a step-by-step reference). Accuracy and GMACs are unaffected; **throughput/VRAM are slower than the official CUDA kernels** — compare methods relative to each other, not to published Vim numbers.

## Not verified yet (needs a GPU)
* Pretrained-ckpt correctness of vim_pure: notebook cell 2 (Samoyed in top-5).
* `ReducedVim.forward` mirrors `VisionMamba.forward_features` of hustvl/Vim for the default mid-CLS config; compare with your checkout.
* `load_vim` model names / checkpoint keys; setup_gpu.sh pins (causal-conv1d 1.1.1, mamba-1p1p1) follow the Vim README at time of writing.
* No numbers have been produced on real data.

## Next steps
1. GPU: reproduce Vim-Ti baseline (CIFAR-100, 3 seeds). 2. Run `sweep.py`. 3. Ablations: multi-stage reduction (DyVM-style), MTR (Δ-based, training-free) as external reference, KAN grid size / hidden width.
