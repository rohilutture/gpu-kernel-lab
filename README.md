# GPU kernel lab: CUDA C++ and Triton (RTX 5090, sm_120)

Status: **all kernels pass correctness on the RTX 5090 and a first baseline benchmark has run (single runs,
desktop session active on the same GPU, no profiler data yet). Not yet optimized or profiled.**

## Baseline results (one RTX 5090, sm_120, single run; full table in `results/summary.md`)

Correctness: every CUDA kernel matches a CPU reference (max abs err: softmax 2.8e-8, RMSNorm 4.8e-7, attention
3.1e-5 / causal 2.4e-4, W4A16 GEMV 3.5e-3); the Triton kernels pass the GPU tests (12 tests total).

- **Memory-bound kernels (softmax, RMSNorm, rows=8192):** fused CUDA and Triton reach about 1.35 to 1.5 TB/s
  (75 to 84% of the 1792 GB/s spec peak, ideal bytes) and are roughly on par with PyTorch eager (within ~10%);
  Triton RMSNorm is 1.4x faster than eager at cols=8192 (0.356 vs 0.499 ms). The naive one-thread-per-row CUDA
  softmax reaches only ~0.26 to 0.28 TB/s; the fused kernel is 4.9x to 5.7x faster (0.237 vs 0.049 ms at cols=1024; 2.02 vs 0.40 ms at 8192).
- **Attention (head_dim 64, causal, bh=32):** the CUDA-core flash-style kernel is **much slower than the library**:
  6.9 TFLOP/s vs 51 (PyTorch SDPA) at n=1024, 13.6 vs 87 at n=4096 (6x to 7.6x slower across n=512 to 4096). The Triton tensor-core
  kernel is faster than SDPA in this setup (92 vs 51 TFLOP/s at n=1024; 163 vs 87 at n=4096). SDPA backend
  selection was not inspected, so do not read this as "Triton beats FlashAttention".
- **W4A16 GEMV (batch 1):** the CUDA kernel is 2.5x (N=K=4096) and 2.1x (8192) faster than the fp16 GEMV, short of
  the 4x byte reduction; the Triton version is 2.0x and 1.6x faster. Neither is bandwidth-saturating yet.
- Caveats: one run each; the desktop was using the GPU; CUDA numbers come from WSL2, Triton/PyTorch from Windows.

Four kernels, each written twice (CUDA C++ and Triton) and compared against PyTorch:

| Kernel | CUDA (`cuda/src`) | Triton (`triton_kernels/kernels.py`) | PyTorch baseline | Bound by |
|---|---|---|---|---|
| softmax | naive (1 thread/row, 3 passes) and fused (block/row, online softmax, warp shuffles) | row-per-program | `torch.softmax` | memory |
| RMSNorm | block/row, shuffle + smem reduction | row-per-program | `F.rms_norm` | memory |
| attention (head_dim 64) | FlashAttention-style, CUDA cores, K/V tiles in smem, online softmax | flash fwd with `tl.dot` (tensor cores) | `F.scaled_dot_product_attention` | compute |
| W4A16 GEMV (batch-1 decode) | warp/row, fused int4 dequant | row-block, unpack 8 nibbles | fp16 GEMV (4x the weight bytes) | memory |

The CUDA attention kernel is intentionally the readable CUDA-core baseline (no `mma`); the
interesting question for the writeup is how far it is from the tensor-core Triton and PyTorch
kernels and why (it uses 255 registers per thread with 64-thread blocks, so occupancy is low;
this is the first thing to profile).

## Layout

- `cuda/` kernels, `bench/bench.cu` (CPU-reference correctness check, then L2-flushed median timing, CSV)
- `triton_kernels/` Triton kernels, `quant.py` (int4 pack/unpack), `bench.py`
- `tests/` `test_quant.py` (CPU), `test_triton_gpu.py` (marked `gpu`)
- `report/roofline.py` merges `results/*.csv` into `summary.md` and `bandwidth.png`
- `scripts/build_cuda.sh`, `scripts/profile_ncu.sh`

## Run (when the GPU is free)

```powershell
# CUDA, inside WSL (builds to ~/kl-build)
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/rohil/Downloads/gap-projects/02-gpu-kernel-lab/scripts/build_cuda.sh
wsl -d Ubuntu-24.04 -- /root/kl-build/bench --check          # correctness first
wsl -d Ubuntu-24.04 -- /root/kl-build/bench --csv /mnt/c/Users/rohil/Downloads/gap-projects/02-gpu-kernel-lab/results/cuda.csv

# Triton + PyTorch, Windows venv
.venv\Scripts\Activate.ps1
python -m pytest -q                      # CPU tests plus Triton correctness on GPU
python -m triton_kernels.bench --csv results/triton.csv
python report/roofline.py                # --peak-gbps defaults to 1792; verify against your card
```

Make `results/` first. The benchmark should be run with nothing else on the GPU (another process
sharing it will distort every number).

## Plan beyond the baseline

1. Get all kernels correct, record the baseline table.
2. Profile (see `profile_ncu.sh` caveat: hardware counters on WSL may need an admin-enabled setting).
3. Optimize with measured reasons: vectorized `float4`/`uint4` loads in softmax, RMSNorm and GEMV;
   fewer registers and more warps in attention, then a `mma.sync`/WMMA tensor-core version; FP8 and
   NVFP4 variants of the GEMV on Blackwell, compared against the existing 4-bit GPTQ results.
4. Report each change as before/after with the profiler metric that explains it.

## Phrasing rules (once there are results)

- One RTX 5090, one driver and toolkit version; say so. CUDA runs under WSL2, Triton/PyTorch under
  Windows, same card; kernel times use CUDA events or `do_bench`, but the two stacks differ in launch
  overhead, so compare within a stack first.
- Percent-of-peak bandwidth uses ideal bytes and the spec-sheet peak, not a measured copy bandwidth.
- Do not call the CUDA attention kernel "FlashAttention" without "-style" and do not claim parity with
  the library kernels unless measured.
- The W4A16 correctness tolerance is loose (fp16 accumulate/scale rounding); state the error you measured.
