"""Correctness of the Triton kernels vs PyTorch. Needs a CUDA device (briefly uses the GPU)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")]

from triton_kernels import kernels as K  # noqa: E402
from triton_kernels.quant import dequantize, quantize_pack  # noqa: E402


@pytest.mark.parametrize("cols", [1000, 4096])
def test_softmax(cols):
    x = torch.randn(64, cols, device="cuda") * 4
    torch.testing.assert_close(K.softmax(x), torch.softmax(x, -1), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("cols", [1000, 4096])
def test_rmsnorm(cols):
    x, w = torch.randn(64, cols, device="cuda"), torch.rand(cols, device="cuda") + 0.5
    torch.testing.assert_close(K.rmsnorm(x, w), F.rms_norm(x, (cols,), w, 1e-6), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [200, 512])  # 200 is not a multiple of the block size
def test_attention(n, causal):
    bh = 4
    q, k, v = (torch.randn(bh, n, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    ref = F.scaled_dot_product_attention(*(t.view(1, bh, n, 64) for t in (q, k, v)), is_causal=causal)
    torch.testing.assert_close(K.attention(q, k, v, causal=causal).view(1, bh, n, 64), ref, atol=2e-2, rtol=2e-2)


def test_gemv_w4a16_matches_dequant_reference():
    torch.manual_seed(0)
    N, Kd = 256, 1024
    wq, sc = quantize_pack(torch.randn(N, Kd) * 0.02, 128)
    x = torch.randn(Kd, device="cuda", dtype=torch.float16)
    ref = (dequantize(wq, sc, 128).cuda() @ x.float())
    out = K.gemv_w4a16(wq.cuda(), sc.cuda(), x, 128).float()
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=2e-2)
