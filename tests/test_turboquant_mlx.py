import numpy as np
import mlx.core as mx

from turboquant_mlx.core import (
    TurboQuantMSE,
    TurboQuantProd,
    _pack_bits,
    _unpack_bits,
    dequantize_kv_cache,
    quantize_kv_cache,
)


def _mse(a: mx.array, b: mx.array) -> float:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    return float(np.mean((arr_a - arr_b) ** 2))


def test_bit_packing_roundtrip():
    rng = np.random.default_rng(123)

    for bits in (1, 2, 3, 4, 5, 6, 7, 8):
        size = 257
        values = mx.array(rng.integers(0, 1 << bits, size=size, dtype=np.int32))
        packed = _pack_bits(values, bits)
        unpacked = _unpack_bits(packed, bits, size)
        np.testing.assert_array_equal(np.asarray(values), np.asarray(unpacked))


def test_mse_quality_improves_with_more_bits():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((256, 64), dtype=np.float32))

    quantizer_2bit = TurboQuantMSE(dimension=64, bit_width=2, seed=0)
    quantizer_4bit = TurboQuantMSE(dimension=64, bit_width=4, seed=0)

    recon_2bit = quantizer_2bit.dequantize(quantizer_2bit.quantize(x))
    recon_4bit = quantizer_4bit.dequantize(quantizer_4bit.quantize(x))

    err_2bit = _mse(x, recon_2bit)
    err_4bit = _mse(x, recon_4bit)

    assert recon_2bit.shape == x.shape
    assert recon_4bit.shape == x.shape
    assert err_4bit < err_2bit


def test_prod_quantization_tracks_inner_products():
    rng = np.random.default_rng(1)

    x = mx.array(rng.standard_normal((3, 17, 64), dtype=np.float32))
    queries = mx.array(rng.standard_normal((5, 64), dtype=np.float32))

    quantizer = TurboQuantProd(dimension=64, bit_width=3, seed=4)
    codes = quantizer.quantize(x)

    dequantized = quantizer.dequantize(codes)
    estimated_scores = quantizer.estimate_inner_products(queries, codes)

    true_scores = np.asarray(mx.matmul(queries, x.reshape((-1, 64)).T)).reshape(5, 3, 17)
    approx_scores = np.asarray(estimated_scores)

    corr = float(np.corrcoef(true_scores.reshape(-1), approx_scores.reshape(-1))[0, 1])

    assert dequantized.shape == x.shape
    assert estimated_scores.shape == (5, 3, 17)
    assert np.isfinite(approx_scores).all()
    assert corr > 0.7


def test_kv_cache_helpers_roundtrip_shapes():
    rng = np.random.default_rng(2)

    keys = mx.array(rng.standard_normal((2, 11, 32), dtype=np.float32))
    values = mx.array(rng.standard_normal((2, 11, 48), dtype=np.float32))

    cache = quantize_kv_cache(keys, values, key_bit_width=3, value_bit_width=4, seed=11)
    keys_hat, values_hat = dequantize_kv_cache(cache)

    assert keys_hat.shape == keys.shape
    assert values_hat.shape == values.shape

    # Ensure quantized representations are genuinely compact.
    assert cache.key_codes.storage_bytes() < int(np.asarray(keys).nbytes)
    assert cache.value_codes.storage_bytes() < int(np.asarray(values).nbytes)
