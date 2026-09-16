#!/usr/bin/env bash
# Build and install all custom CUDA extensions so that training never
# falls back to the slow JIT path.
#
# Usage (from any directory):
#   bash llmfoundry/models/ops/float8/cuda_kernels/install_cuda_extensions.sh
#
# In a Dockerfile (H100 example):
#   ENV TORCH_CUDA_ARCH_LIST="9.0"
#   RUN bash llmfoundry/models/ops/float8/cuda_kernels/install_cuda_extensions.sh
#
# TORCH_CUDA_ARCH_LIST controls which SM architectures are compiled.
# Leave unset to auto-detect from the build machine's GPU (requires a GPU
# at image build time, e.g. --gpus all with BuildKit).
# Common values:
#   "8.0"       – A100 only
#   "9.0"       – H100 only
#   "8.0 9.0"   – A100 + H100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building all CUDA extensions ==="
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<native>}"

MAX_JOBS=4

build_ext() {
    local name="$1"
    local dir="$SCRIPT_DIR/$name"
    echo ""
    echo "--- $name ---"
    pip install --no-build-isolation -v "$dir"
}

build_ext row2col_dq_quantization
build_ext row2col_te_quantization
build_ext scaling_aware_transpose

echo ""
echo "=== All CUDA extensions installed successfully ==="
