#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Row-wise softmax over [rows, cols] fp32.
void softmax_naive(const float* x, float* y, int rows, int cols, cudaStream_t s);  // 1 thread/row, 3 passes (strawman)
void softmax_fused(const float* x, float* y, int rows, int cols, cudaStream_t s);  // 1 block/row, online softmax, 2 reads

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w, [rows, cols] fp32.
void rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s);

// FlashAttention-style forward (no tensor cores): fp16 in/out, fp32 accumulate, head_dim = 64.
// Q,K,V,O: [bh, n, 64]. Online softmax, K/V tiles staged in shared memory, never materializes n x n.
void attention_flash(const half* q, const half* k, const half* v, half* o, int bh, int n, bool causal,
                     cudaStream_t s);

// W4A16 GEMV: y[N] = W[N,K] x[K]. W packed 8 x uint4 per uint32 (little-end nibble first),
// symmetric quant: w = (nibble - 8) * scale[row, k / group]. K % 8 == 0, group % 8 == 0.
void gemv_w4a16(const uint32_t* wq, const half* scales, const half* x, half* y, int N, int K, int group,
                cudaStream_t s);
