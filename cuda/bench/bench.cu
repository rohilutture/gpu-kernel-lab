// Correctness check (vs CPU reference) then benchmark for every CUDA kernel.
//   ./bench --check            correctness only
//   ./bench --csv out.csv      check, then benchmark, write CSV
// CSV columns: impl,kernel,shape,ms,gbps,tflops   (same schema as triton/bench.py)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"
#include "kernels.h"

// ---------------- CPU references ----------------
static void softmax_ref(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[(size_t)r * cols];
    float m = -INFINITY;
    for (int i = 0; i < cols; ++i) m = std::max(m, xr[i]);
    double d = 0;
    for (int i = 0; i < cols; ++i) d += std::exp((double)xr[i] - m);
    for (int i = 0; i < cols; ++i) y[(size_t)r * cols + i] = (float)(std::exp((double)xr[i] - m) / d);
  }
}
static void rmsnorm_ref(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y,
                        int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < cols; ++i) ss += (double)x[(size_t)r * cols + i] * x[(size_t)r * cols + i];
    float inv = (float)(1.0 / std::sqrt(ss / cols + eps));
    for (int i = 0; i < cols; ++i) y[(size_t)r * cols + i] = x[(size_t)r * cols + i] * inv * w[i];
  }
}
static void attn_ref(const std::vector<half>& q, const std::vector<half>& k, const std::vector<half>& v,
                     std::vector<float>& o, int bh, int n, bool causal) {
  const int D = 64;
  std::vector<double> s(n);
  for (int b = 0; b < bh; ++b)
    for (int i = 0; i < n; ++i) {
      double m = -1e300;
      for (int j = 0; j < n; ++j) {
        if (causal && j > i) { s[j] = -1e300; continue; }
        double a = 0;
        for (int d = 0; d < D; ++d)
          a += (double)__half2float(q[((size_t)b * n + i) * D + d]) * __half2float(k[((size_t)b * n + j) * D + d]);
        s[j] = a / 8.0;
        m = std::max(m, s[j]);
      }
      double l = 0;
      for (int j = 0; j < n; ++j) { s[j] = (s[j] <= -1e299) ? 0 : std::exp(s[j] - m); l += s[j]; }
      for (int d = 0; d < D; ++d) {
        double a = 0;
        for (int j = 0; j < n; ++j) a += s[j] * __half2float(v[((size_t)b * n + j) * D + d]);
        o[((size_t)b * n + i) * D + d] = (float)(a / l);
      }
    }
}
static void gemv_ref(const std::vector<uint32_t>& wq, const std::vector<half>& sc, const std::vector<half>& x,
                     std::vector<float>& y, int N, int K, int group) {
  for (int r = 0; r < N; ++r) {
    double acc = 0;
    for (int k = 0; k < K; ++k) {
      uint32_t word = wq[(size_t)r * (K / 8) + k / 8];
      int nib = (int)((word >> (4 * (k % 8))) & 0xF) - 8;
      acc += (double)nib * __half2float(sc[(size_t)r * (K / group) + k / group]) * __half2float(x[k]);
    }
    y[r] = (float)acc;
  }
}

// ---------------- helpers ----------------
static std::mt19937 rng(0);
static std::vector<float> randf(size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (auto& e : v) e = u(rng);
  return v;
}
static std::vector<half> randh(size_t n, float lo, float hi) {
  auto f = randf(n, lo, hi);
  std::vector<half> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __float2half(f[i]);
  return h;
}
template <class T> static T* to_dev(const std::vector<T>& h) {
  T* d; CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> static std::vector<T> to_host(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, (double)std::fabs(a[i] - b[i]));
  return e;
}
static std::vector<float> h2f(const std::vector<half>& h) {
  std::vector<float> f(h.size());
  for (size_t i = 0; i < h.size(); ++i) f[i] = __half2float(h[i]);
  return f;
}

// Median of `reps` single-call timings; L2 is flushed before every timed call.
static void* g_flush = nullptr;
static constexpr size_t FLUSH_BYTES = 256u << 20;
static double time_ms(const std::function<void()>& f, int warm = 10, int reps = 50) {
  if (!g_flush) CUDA_CHECK(cudaMalloc(&g_flush, FLUSH_BYTES));
  for (int i = 0; i < warm; ++i) f();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaMemset(g_flush, i & 0xff, FLUSH_BYTES));
    cudaEventRecord(a); f(); cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  CUDA_CHECK(cudaGetLastError());
  return t[t.size() / 2];
}

static int g_fail = 0;
static void expect(const char* name, double err, double tol) {
  bool ok = err <= tol;
  printf("[%s] %-26s max_abs_err=%.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", name, err, tol);
  if (!ok) ++g_fail;
}

// ---------------- correctness ----------------
static void run_checks() {
  {  // softmax, odd cols to exercise the strided tail
    int rows = 64, cols = 1000;
    auto x = randf((size_t)rows * cols, -8, 8); std::vector<float> ref(x.size());
    softmax_ref(x, ref, rows, cols);
    float *dx = to_dev(x), *dy; CUDA_CHECK(cudaMalloc(&dy, x.size() * 4));
    softmax_naive(dx, dy, rows, cols, 0); CUDA_CHECK(cudaDeviceSynchronize());
    expect("softmax_naive", max_abs_err(to_host(dy, x.size()), ref), 1e-5);
    softmax_fused(dx, dy, rows, cols, 0); CUDA_CHECK(cudaDeviceSynchronize());
    expect("softmax_fused", max_abs_err(to_host(dy, x.size()), ref), 1e-5);
    cudaFree(dx); cudaFree(dy);
  }
  {
    int rows = 64, cols = 1000; float eps = 1e-6f;
    auto x = randf((size_t)rows * cols, -3, 3), w = randf(cols, 0.5, 1.5); std::vector<float> ref(x.size());
    rmsnorm_ref(x, w, ref, rows, cols, eps);
    float *dx = to_dev(x), *dw = to_dev(w), *dy; CUDA_CHECK(cudaMalloc(&dy, x.size() * 4));
    rmsnorm(dx, dw, dy, rows, cols, eps, 0); CUDA_CHECK(cudaDeviceSynchronize());
    expect("rmsnorm", max_abs_err(to_host(dy, x.size()), ref), 1e-4);
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
  }
  for (bool causal : {false, true}) {  // n=200 is not a multiple of BQ/BK: exercises tails and masking
    int bh = 2, n = 200;
    size_t sz = (size_t)bh * n * 64;
    auto q = randh(sz, -1, 1), k = randh(sz, -1, 1), v = randh(sz, -1, 1);
    std::vector<float> ref(sz); attn_ref(q, k, v, ref, bh, n, causal);
    half *dq = to_dev(q), *dk = to_dev(k), *dv = to_dev(v), *dout; CUDA_CHECK(cudaMalloc(&dout, sz * 2));
    attention_flash(dq, dk, dv, dout, bh, n, causal, 0); CUDA_CHECK(cudaDeviceSynchronize());
    expect(causal ? "attention_flash(causal)" : "attention_flash", max_abs_err(h2f(to_host(dout, sz)), ref), 5e-3);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
  }
  {
    int N = 256, K = 1024, G = 128;
    std::uniform_int_distribution<uint32_t> u;
    std::vector<uint32_t> wq((size_t)N * K / 8); for (auto& w : wq) w = u(rng);
    auto sc = randh((size_t)N * (K / G), 0.01f, 0.05f), x = randh(K, -1, 1);
    std::vector<float> ref(N); gemv_ref(wq, sc, x, ref, N, K, G);
    uint32_t* dw = to_dev(wq); half *ds = to_dev(sc), *dx = to_dev(x), *dy; CUDA_CHECK(cudaMalloc(&dy, N * 2));
    gemv_w4a16(dw, ds, dx, dy, N, K, G, 0); CUDA_CHECK(cudaDeviceSynchronize());
    expect("gemv_w4a16", max_abs_err(h2f(to_host(dy, N)), ref), 5e-2);
    cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
  }
}

// ---------------- benchmark ----------------
static void row(FILE* f, const char* impl, const char* kernel, const std::string& shape, double ms, double bytes,
                double flops) {
  fprintf(f, "%s,%s,\"%s\",%.5f,%.2f,%.3f\n", impl, kernel, shape.c_str(), ms, bytes / (ms * 1e-3) / 1e9,
          flops / (ms * 1e-3) / 1e12);
}

static void run_bench(FILE* out) {
  fprintf(out, "impl,kernel,shape,ms,gbps,tflops\n");
  for (int cols : {1024, 4096, 8192}) {
    int rows = 8192;
    std::string sh = "rows=" + std::to_string(rows) + ",cols=" + std::to_string(cols);
    size_t n = (size_t)rows * cols;
    float *dx, *dy, *dw; CUDA_CHECK(cudaMalloc(&dx, n * 4)); CUDA_CHECK(cudaMalloc(&dy, n * 4));
    CUDA_CHECK(cudaMalloc(&dw, cols * 4));
    CUDA_CHECK(cudaMemset(dx, 0, n * 4)); CUDA_CHECK(cudaMemset(dw, 0, cols * 4));
    double bytes = 2.0 * n * 4;  // ideal: read once + write once
    row(out, "cuda_naive", "softmax", sh, time_ms([&] { softmax_naive(dx, dy, rows, cols, 0); }), bytes, 0);
    row(out, "cuda_fused", "softmax", sh, time_ms([&] { softmax_fused(dx, dy, rows, cols, 0); }), bytes, 0);
    row(out, "cuda_fused", "rmsnorm", sh, time_ms([&] { rmsnorm(dx, dw, dy, rows, cols, 1e-6f, 0); }), bytes, 0);
    cudaFree(dx); cudaFree(dy); cudaFree(dw);
  }
  for (int n : {512, 1024, 2048, 4096}) {
    int bh = 32;  // e.g. 32 heads, batch 1
    size_t sz = (size_t)bh * n * 64;
    half *q, *k, *v, *o; for (half** p : {&q, &k, &v, &o}) { CUDA_CHECK(cudaMalloc(p, sz * 2)); CUDA_CHECK(cudaMemset(*p, 0, sz * 2)); }
    std::string sh = "bh=" + std::to_string(bh) + ",n=" + std::to_string(n) + ",d=64,causal";
    double flops = 4.0 * bh * (double)n * n * 64 / 2;  // causal: half the pairs
    double bytes = 4.0 * sz * 2;                       // Q,K,V read + O write, fp16
    row(out, "cuda_flash", "attention", sh, time_ms([&] { attention_flash(q, k, v, o, bh, n, true, 0); }), bytes, flops);
    for (half* p : {q, k, v, o}) cudaFree(p);
  }
  for (int NK : {4096, 8192}) {
    int N = NK, K = NK, G = 128;
    uint32_t* w; half *s, *x, *y;
    CUDA_CHECK(cudaMalloc(&w, (size_t)N * K / 2)); CUDA_CHECK(cudaMemset(w, 0x77, (size_t)N * K / 2));
    CUDA_CHECK(cudaMalloc(&s, (size_t)N * (K / G) * 2)); CUDA_CHECK(cudaMemset(s, 0, (size_t)N * (K / G) * 2));
    CUDA_CHECK(cudaMalloc(&x, K * 2)); CUDA_CHECK(cudaMemset(x, 0, K * 2));
    CUDA_CHECK(cudaMalloc(&y, N * 2));
    double bytes = (double)N * K / 2 + (double)N * (K / G) * 2 + K * 2 + N * 2;
    row(out, "cuda_w4a16", "gemv_int4", "N=" + std::to_string(N) + ",K=" + std::to_string(K),
        time_ms([&] { gemv_w4a16(w, s, x, y, N, K, G, 0); }), bytes, 2.0 * N * K);
    cudaFree(w); cudaFree(s); cudaFree(x); cudaFree(y);
  }
}

int main(int argc, char** argv) {
  const char* csv = nullptr; bool check_only = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--check")) check_only = true;
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
  }
  cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  fprintf(stderr, "device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  run_checks();
  if (g_fail) { fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
  if (check_only) return 0;
  FILE* f = csv ? fopen(csv, "w") : stdout;
  if (!f) { perror("csv"); return 1; }
  run_bench(f);
  if (csv) fclose(f);
  return 0;
}
