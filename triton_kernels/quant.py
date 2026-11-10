"""int4 packing helpers shared by tests and the Triton bench. Pure torch, CPU-safe."""
from __future__ import annotations

import torch


def quantize_pack(w: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [N, K] float. Symmetric per-group int4: q = clamp(round(w/scale)+8, 0, 15), scale = max|w|/7.

    Returns (wq [N, K/8] int32, scales [N, K/group] fp16). Nibble i of a word holds element 8*p+i
    (little-end nibble first), matching cuda/src/gemv_int4.cu and triton_kernels/kernels.py.
    """
    N, K = w.shape
    assert K % group == 0 and group % 8 == 0
    g = w.float().view(N, K // group, group)
    scale = (g.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = (torch.round(g / scale) + 8).clamp(0, 15).to(torch.int64).view(N, K)
    q8 = q.view(N, K // 8, 8)
    word = torch.zeros(N, K // 8, dtype=torch.int64)
    for i in range(8):
        word |= q8[..., i] << (4 * i)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)  # reinterpret as signed int32
    return word, scale.squeeze(-1).to(torch.float16)


def dequantize(wq: torch.Tensor, scales: torch.Tensor, group: int = 128) -> torch.Tensor:
    """Inverse of quantize_pack -> [N, K] fp32."""
    N, KP = wq.shape
    u = wq.to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(u >> (4 * i)) & 0xF for i in range(8)], dim=-1).view(N, KP * 8)
    s = scales.float().repeat_interleave(group, dim=1)
    return (nib.float() - 8.0) * s
