#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                            \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                               \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr float NEG_BIG = -3.0e38f;  // finite stand-in for -inf (avoids inf-inf = NaN)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
  return v;
}

// Merge two online-softmax partial states (max, sum of exp(x - max)).
__device__ __forceinline__ void merge_md(float& m, float& d, float m2, float d2) {
  float M = fmaxf(m, m2);
  d = d * __expf(m - M) + d2 * __expf(m2 - M);
  m = M;
}
