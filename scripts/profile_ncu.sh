#!/usr/bin/env bash
# Nsight Compute profile of one CUDA kernel from the bench binary (run inside WSL).
#   bash scripts/profile_ncu.sh flash_attn_fwd          # kernel-name regex
# Output: results/ncu_<kernel>.ncu-rep (open in the Nsight Compute GUI on Windows) + a text summary.
#
# CAVEAT: hardware counters on WSL2 require "Allow access to the GPU performance counters to all
# users" to be enabled on the Windows side (NVIDIA Control Panel > Developer > Manage GPU Performance
# Counters), which needs admin. If ncu reports ERR_NVGPUCTRPERM, either have an admin enable it or
# fall back to torch.profiler / CUPTI timing (no counters) and static analysis (ptxas -v).
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
K="${1:?kernel name regex}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${KL_BUILD_DIR:-$HOME/kl-build}"
mkdir -p "$HERE/results"
ncu --set full -k "regex:$K" --launch-skip 5 --launch-count 1 \
    -o "$HERE/results/ncu_$K" -f "$BUILD/bench" --check
ncu -i "$HERE/results/ncu_$K.ncu-rep" --page details | head -120
