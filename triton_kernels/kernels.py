"""Triton versions of the four kernels (same semantics as cuda/)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- softmax
@triton.jit
def _softmax_k(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(y_ptr + row * n_cols + offs, num / tl.sum(num, axis=0), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    _softmax_k[(rows,)](x, y, cols, BLOCK=triton.next_power_of_2(cols))
    return y


# ---------------------------------------------------------------- rmsnorm
@triton.jit
def _rmsnorm_k(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(y_ptr + row * n_cols + offs, x * inv * w, mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_like(x)
    _rmsnorm_k[(rows,)](x, w, y, cols, eps, BLOCK=triton.next_power_of_2(cols))
    return y


# ---------------------------------------------------------------- flash attention fwd
@triton.jit
def _attn_fwd(Q, K, V, O, n, sm_scale,
              CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh.to(tl.int64) * n * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < n, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    if CAUSAL:
        hi = tl.minimum(n, (pid_m + 1) * BLOCK_M)
    else:
        hi = n
    for start in range(0, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[None, :] * D + offs_d[:, None], mask=offs_n[None, :] < n, other=0.0)
        qk = tl.dot(q, k) * sm_scale
        keep = offs_n[None, :] < n
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(keep, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(V + base + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < n, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * D + offs_d[None, :], acc.to(tl.float16), mask=offs_m[:, None] < n)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
              block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """q,k,v: [bh, n, 64] fp16 contiguous."""
    bh, n, d = q.shape
    assert d == 64 and q.dtype == torch.float16 and q.is_contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(n, block_m), bh)
    _attn_fwd[grid](q, k, v, o, n, 1.0 / (d ** 0.5), CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n, D=d,
                    num_warps=4, num_stages=2)
    return o


# ---------------------------------------------------------------- W4A16 GEMV
@triton.jit
def _gemv_w4a16(wq, sc, x, y, N, K, G: tl.constexpr, BN: tl.constexpr, BKP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    rmask = rows < N
    KP = K // 8
    acc = tl.zeros([BN], tl.float32)
    for kp0 in range(0, KP, BKP):
        cols = kp0 + tl.arange(0, BKP)
        cmask = cols < KP
        m2 = rmask[:, None] & cmask[None, :]
        w = tl.load(wq + rows[:, None].to(tl.int64) * KP + cols[None, :], mask=m2, other=0)
        s = tl.load(sc + rows[:, None].to(tl.int64) * (K // G) + (cols[None, :] * 8) // G, mask=m2, other=0.0)
        s = s.to(tl.float32)
        for i in tl.static_range(8):
            nib = ((w >> (4 * i)) & 0xF) - 8
            xi = tl.load(x + cols * 8 + i, mask=cmask, other=0.0).to(tl.float32)
            acc += tl.sum(nib.to(tl.float32) * s * xi[None, :], axis=1)
    tl.store(y + rows, acc.to(tl.float16), mask=rmask)


def gemv_w4a16(wq: torch.Tensor, scales: torch.Tensor, x: torch.Tensor, group: int = 128,
               bn: int = 16, bkp: int = 128) -> torch.Tensor:
    """wq: [N, K/8] int32 (packed nibbles), scales: [N, K/group] fp16, x: [K] fp16."""
    N, KP = wq.shape
    K = KP * 8
    y = torch.empty(N, device=x.device, dtype=torch.float16)
    _gemv_w4a16[(triton.cdiv(N, bn),)](wq, scales, x, y, N, K, G=group, BN=bn, BKP=bkp)
    return y
