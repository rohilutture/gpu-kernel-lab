#include "common.cuh"
#include "kernels.h"

// FlashAttention-style forward, head_dim = 64, one thread per query row.
//
// Each block owns BQ query rows. K/V are streamed through shared memory in tiles of BK keys
// (all threads read the same smem address per key, so reads broadcast without bank conflicts).
// Per tile: scores for BK keys, ONE rescale of the running output by exp(m_old - m_new), then
// accumulate P@V. The n x n score matrix is never written to global memory.
//
// This is deliberately a CUDA-core kernel (no mma/tensor cores): it is the readable baseline
// that the Triton version (which lowers tl.dot to tensor cores) is compared against.
constexpr int D = 64;
constexpr int BQ = 64;
constexpr int BK = 32;

__global__ void flash_attn_fwd(const half* __restrict__ Q, const half* __restrict__ K,
                               const half* __restrict__ V, half* __restrict__ O, int n, bool causal) {
  __shared__ half Ks[BK][D];
  __shared__ half Vs[BK][D];
  const int bh = blockIdx.y;
  const int qi = blockIdx.x * BQ + threadIdx.x;
  const bool valid = qi < n;
  const size_t base = (size_t)bh * n * D;

  float q[D], o[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    q[d] = valid ? __half2float(Q[base + (size_t)qi * D + d]) * 0.125f : 0.f;  // 1/sqrt(64)
    o[d] = 0.f;
  }
  float m = NEG_BIG, l = 0.f;
  const int kmax = causal ? min(n, (int)(blockIdx.x + 1) * BQ) : n;

  for (int k0 = 0; k0 < kmax; k0 += BK) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < BK * D; idx += BQ) {
      int r = idx / D, c = idx % D, kj = k0 + r;
      Ks[r][c] = kj < n ? K[base + (size_t)kj * D + c] : __float2half(0.f);
      Vs[r][c] = kj < n ? V[base + (size_t)kj * D + c] : __float2half(0.f);
    }
    __syncthreads();

    float s[BK];
    float tmax = NEG_BIG;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) acc += q[d] * __half2float(Ks[j][d]);
      int kj = k0 + j;
      bool ok = kj < n && (!causal || kj <= qi);
      s[j] = ok ? acc : NEG_BIG;
      tmax = fmaxf(tmax, s[j]);
    }
    float mn = fmaxf(m, tmax);
    float alpha = __expf(m - mn);
    l *= alpha;
#pragma unroll
    for (int d = 0; d < D; ++d) o[d] *= alpha;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
      float p = s[j] > -1e38f ? __expf(s[j] - mn) : 0.f;  // masked keys contribute exactly 0
      l += p;
#pragma unroll
      for (int d = 0; d < D; ++d) o[d] += p * __half2float(Vs[j][d]);
    }
    m = mn;
  }

  if (valid) {
    float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < D; ++d) O[base + (size_t)qi * D + d] = __float2half(o[d] * inv);
  }
}

void attention_flash(const half* q, const half* k, const half* v, half* o, int bh, int n, bool causal,
                     cudaStream_t s) {
  dim3 grid((n + BQ - 1) / BQ, bh);
  flash_attn_fwd<<<grid, BQ, 0, s>>>(q, k, v, o, n, causal);
}
