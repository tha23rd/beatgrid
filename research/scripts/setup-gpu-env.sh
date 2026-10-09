#!/usr/bin/env bash
# A CUDA environment for distillation (research/beatgrid_lab/student). Separate from the
# research venv, which pins CPU-only PyTorch.
#
# torch 2.7.1+cu126 still ships kernels that run on Pascal GPUs (GTX 10xx); later CUDA builds
# dropped them.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv-gpu
VIRTUAL_ENV=.venv-gpu uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchaudio==2.7.1
VIRTUAL_ENV=.venv-gpu uv pip install "beat-this>=1.1" numpy soxr soundfile scipy
echo "run with: PYTHONPATH=. .venv-gpu/bin/python -m beatgrid_lab.student.prepare ..."
