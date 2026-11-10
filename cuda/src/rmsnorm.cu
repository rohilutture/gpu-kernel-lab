#include "common.cuh"
#include "kernels.h"

// One block per row: sum of squares via warp shuffles + shared memory, then scale.
__global__ void rmsnorm_k(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                          int cols, float eps) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  float ss = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = xr[i];
    ss += v * v;
  }
  ss = warp_sum(ss);
  __shared__ float part[32];
  __shared__ float inv_rms;
  int wid = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
  if (l == 0) part[wid] = ss;
  __syncthreads();
  if (wid == 0) {
    float t = l < nw ? part[l] : 0.f;
    t = warp_sum(t);
    if (l == 0) inv_rms = rsqrtf(t / cols + eps);
  }
  __syncthreads();
  float r = inv_rms;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

void rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s) {
  rmsnorm_k<<<rows, 256, 0, s>>>(x, w, y, cols, eps);
}
