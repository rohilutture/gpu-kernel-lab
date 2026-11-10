#!/usr/bin/env bash
# Build the CUDA kernels inside WSL (nvcc 12.8, sm_120). Build dir lives on the Linux fs for speed.
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/rohil/Downloads/gap-projects/02-gpu-kernel-lab/scripts/build_cuda.sh
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:/usr/local/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${KL_BUILD_DIR:-$HOME/kl-build}"
cmake -S "$HERE/cuda" -B "$BUILD" -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc >/dev/null
cmake --build "$BUILD" 2>&1 | tail -40
ls -la "$BUILD/bench"
echo "built: $BUILD/bench"
