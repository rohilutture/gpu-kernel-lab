import torch

from triton_kernels.quant import dequantize, quantize_pack


def test_roundtrip_error_bounded_by_half_step():
    torch.manual_seed(0)
    w = torch.randn(16, 256)
    wq, sc = quantize_pack(w, group=128)
    assert wq.dtype == torch.int32 and wq.shape == (16, 32) and sc.shape == (16, 2)
    wd = dequantize(wq, sc, group=128)
    step = sc.float().repeat_interleave(128, dim=1)
    # fp16 scale rounding adds a little on top of the half-step quantization bound
    assert ((w - wd).abs() <= 0.5 * step * 1.01 + 1e-6).all()


def test_nibble_layout_little_end_first():
    # element 0 -> lowest nibble of word 0; value 15 (=+7 after -8) at index 0, 8 (=0) elsewhere
    w = torch.zeros(1, 128)
    w[0, 0] = 7.0  # scale becomes 1.0, q = 15
    wq, sc = quantize_pack(w, group=128)
    word0 = int(wq[0, 0].item()) & 0xFFFFFFFF
    assert word0 & 0xF == 15
    assert (word0 >> 4) & 0xF == 8  # zero maps to 8


def test_high_bit_words_survive_int32_reinterpretation():
    w = torch.full((1, 128), 7.0)  # every nibble = 15 -> word 0xFFFFFFFF -> int32 -1
    wq, sc = quantize_pack(w, group=128)
    assert (wq == -1).all()
    assert torch.allclose(dequantize(wq, sc), w, atol=1e-2)
