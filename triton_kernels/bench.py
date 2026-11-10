"""Benchmark Triton kernels vs PyTorch eager. Same CSV schema as cuda/bench/bench.cu.

    python -m triton_kernels.bench --csv results/triton.csv

Timing: triton.testing.do_bench (median, L2 flushed between calls).
"""
from __future__ import annotations

import argparse
import csv

import torch
import torch.nn.functional as F
import triton

from . import kernels as K
from .quant import dequantize, quantize_pack


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/triton.csv")
    args = ap.parse_args()
    dev = "cuda"
    rows = []

    def add(impl, kernel, shape, ms, bytes_, flops=0.0):
        rows.append([impl, kernel, shape, f"{ms:.5f}", f"{bytes_ / (ms * 1e-3) / 1e9:.2f}",
                     f"{flops / (ms * 1e-3) / 1e12:.3f}"])

    for cols in (1024, 4096, 8192):
        r = 8192
        sh = f"rows={r},cols={cols}"
        x = torch.randn(r, cols, device=dev)
        w = torch.rand(cols, device=dev) + 0.5
        b = 2.0 * x.numel() * 4
        add("torch_eager", "softmax", sh, bench(lambda: torch.softmax(x, dim=-1)), b)
        add("triton", "softmax", sh, bench(lambda: K.softmax(x)), b)
        add("torch_eager", "rmsnorm", sh, bench(lambda: F.rms_norm(x, (cols,), w, 1e-6)), b)
        add("triton", "rmsnorm", sh, bench(lambda: K.rmsnorm(x, w)), b)

    for n in (512, 1024, 2048, 4096):
        bh = 32
        q, k, v = (torch.randn(bh, n, 64, device=dev, dtype=torch.float16) for _ in range(3))
        sh = f"bh={bh},n={n},d=64,causal"
        flops, b = 4.0 * bh * n * n * 64 / 2, 4.0 * q.numel() * 2
        q4, k4, v4 = (t.view(1, bh, n, 64) for t in (q, k, v))
        add("torch_sdpa", "attention", sh, bench(lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)), b, flops)
        add("triton", "attention", sh, bench(lambda: K.attention(q, k, v, causal=True)), b, flops)

    for nk in (4096, 8192):
        wf = torch.randn(nk, nk) * 0.02
        wq, sc = quantize_pack(wf, 128)
        wq, sc = wq.to(dev), sc.to(dev)
        x = torch.randn(nk, device=dev, dtype=torch.float16)
        sh = f"N={nk},K={nk}"
        b = nk * nk / 2 + nk * (nk // 128) * 2 + nk * 2 * 2
        # baseline: fp16 weights (4x the bytes of int4) -- the speedup int4 can at best deliver is bandwidth-bound
        wdev = dequantize(wq.cpu(), sc.cpu()).to(dev).to(torch.float16)
        add("torch_fp16_gemv", "gemv_int4", sh, bench(lambda: wdev @ x), nk * nk * 2 + nk * 2 * 2)  # fp16 weights: 4x the bytes
        add("triton", "gemv_int4", sh, bench(lambda: K.gemv_w4a16(wq, sc, x)), b)

    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["impl", "kernel", "shape", "ms", "gbps", "tflops"])
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
