#!/usr/bin/env bash
# Run on Colab / Kaggle / WSL2 with an NVIDIA GPU. Installs Vim + its CUDA kernels and fetches a checkpoint.
set -e
git clone https://github.com/hustvl/Vim.git ../Vim
pip install -r ../Vim/vim/vim_requirements.txt || true
pip install causal-conv1d==1.1.1
pip install ../Vim/mamba-1p1p1          # Vim's modified mamba_ssm (bidirectional)
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/../Vim/vim
# Vim-Tiny middle-CLS checkpoint (76.1% ImageNet): see the Vim README for the download link, save as vim_t.pth
