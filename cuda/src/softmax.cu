#include "common.cuh"
#include "kernels.h"

// Strawman: one thread per row, three passes over global memory (max, sum, write).
__global__ void softmax_naive_k(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + (size_t)r * cols;
  float* yr = y + (size_t)r * cols;
  float m = NEG_BIG;
  for (int i = 0; i < cols; ++i) m = fmaxf(m, xr[i]);
  float d = 0.f;
  for (int i = 0; i < cols; ++i) d += __expf(xr[i] - m);
  float inv = 1.f / d;
  for (int i = 0; i < cols; ++i) yr[i] = __expf(xr[i] - m) * inv;
}

// One block per row. Pass 1: single-sweep online (max, sum) per thread, merged by warp
// shuffles then across warps through shared memory. Pass 2: write normalized values.
// Global traffic: 2 reads + 1 write per element (second read mostly hits L1/L2).
__global__ void softmax_fused_k(const float* __restrict__ x, float* __restrict__ y, int cols) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  float m = NEG_BIG, d = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = xr[i];
    float mn = fmaxf(m, v);
    d = d * __expf(m - mn) + __expf(v - mn);
    m = mn;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float m2 = __shfl_xor_sync(FULL_MASK, m, o);
    float d2 = __shfl_xor_sync(FULL_MASK, d, o);
    merge_md(m, d, m2, d2);
  }
  __shared__ float sm[32], sd[32];
  int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
  if (l == 0) { sm[w] = m; sd[w] = d; }
  __syncthreads();
  if (w == 0) {
    m = l < nw ? sm[l] : NEG_BIG;
    d = l < nw ? sd[l] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float m2 = __shfl_xor_sync(FULL_MASK, m, o);
      float d2 = __shfl_xor_sync(FULL_MASK, d, o);
      merge_md(m, d, m2, d2);
    }
    if (l == 0) { sm[0] = m; sd[0] = d; }
  }
  __syncthreads();
  float M = sm[0], inv = 1.f / sd[0];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = __expf(xr[i] - M) * inv;
}

void softmax_naive(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  softmax_naive_k<<<(rows + 127) / 128, 128, 0, s>>>(x, y, rows, cols);
}
void softmax_fused(const float* x, float* y, int rows, int cols, cudaStream_t s) {
  softmax_fused_k<<<rows, 256, 0, s>>>(x, y, cols);
}
