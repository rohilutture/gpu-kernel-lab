#include "common.cuh"
#include "kernels.h"

// W4A16 GEMV for batch-1 decode: y = W x with W stored as packed 4-bit ints.
// Memory bound: the whole kernel is "stream N*K/2 weight bytes once". One warp per output row;
// lanes stride over the row's packed words so weight loads are coalesced (32 x 4 B per step).
// Dequant is fused: (nibble - 8) * group_scale, accumulate in fp32.
__global__ void gemv_w4a16_k(const uint32_t* __restrict__ wq, const half* __restrict__ sc,
                             const half* __restrict__ x, half* __restrict__ y, int N, int K, int group) {
  const int warps = blockDim.x >> 5;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const int KP = K >> 3;  // packed words per row
  const uint32_t* wr = wq + (size_t)row * KP;
  const half* sr = sc + (size_t)row * (K / group);
  float acc = 0.f;
  for (int p = lane; p < KP; p += 32) {
    uint32_t word = wr[p];
    const int k0 = p << 3;
    const float s = __half2float(sr[k0 / group]);
    float part = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int nib = (int)((word >> (4 * i)) & 0xFu) - 8;
      part += (float)nib * __half2float(x[k0 + i]);
    }
    acc += part * s;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

void gemv_w4a16(const uint32_t* wq, const half* scales, const half* x, half* y, int N, int K, int group,
                cudaStream_t s) {
  const int threads = 128;  // 4 rows per block
  gemv_w4a16_k<<<(N + 3) / 4, threads, 0, s>>>(wq, scales, x, y, N, K, group);
}
