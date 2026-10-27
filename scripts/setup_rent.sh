#!/usr/bin/env bash
# Máy thuê có GPU NVIDIA (RunPod / vast.ai ...). Chạy 1 lần:  bash scripts/setup_rent.sh
# Nên chọn image PyTorch 2.1 + CUDA 11.8 "devel" để cài được kernel CUDA chính thức của Vim (nhanh hơn nhiều).
# Không cài được thì vẫn chạy bằng bản pure-PyTorch (chậm hơn, kết quả như nhau) - run_plan.py tự chọn.
set -u
cd "$(dirname "$0")/.."
nvidia-smi || { echo "KHÔNG THẤY GPU"; exit 1; }
pip install -q huggingface_hub matplotlib numpy pytest
if python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.1') and torch.version.cuda=='11.8' else 1)"; then
  echo ">> torch 2.1 + CUDA 11.8: cài kernel Vim chính thức (10-30 phút)"
  [ -d ../Vim ] || git clone -q https://github.com/hustvl/Vim.git ../Vim
  # transformers 4.36.2: mamba-1p1p1 import GreedySearchDecoderOnlyOutput (transformers 5.x đã bỏ)
  pip install -q timm==0.4.12 einops packaging ninja transformers==4.36.2
  export MAX_JOBS=${MAX_JOBS:-8}
  # causal-conv1d 1.4.0: bản mamba của Vim gọi causal_conv1d_fwd 7 tham số; 1.1.1 cài được nhưng chạy báo lỗi
  pip install --no-build-isolation causal-conv1d==1.4.0 \
    && MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation ../Vim/mamba-1p1p1 \
    && echo ">> kernel chính thức: cài xong" \
    || echo ">> cài kernel lỗi -> sẽ dùng bản pure (vẫn chạy được)"
else
  echo ">> torch $(python -c 'import torch;print(torch.__version__, torch.version.cuda)'): bỏ qua kernel chính thức, dùng bản pure"
fi
python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
