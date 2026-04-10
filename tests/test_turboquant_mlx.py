import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from turboquant_mlx.core import (
    QuantizedModelWeights,
    TurboQuantMSE,
    TurboQuantProd,
    _pack_bits,
    _unpack_bits,
    dequantize_model_weights,
    dequantize_kv_cache,
    load_turboquant_weights,
    quantize_model_weights,
    quantize_kv_cache,
    save_turboquant_weights,
    turboquantize_model_weights,
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


def test_model_weight_helpers_roundtrip_state_and_compression():
    rng = np.random.default_rng(5)
    weights = {
        "layer": {
            "weight": mx.array(rng.standard_normal((32, 64), dtype=np.float32)),
            "bias": mx.array(rng.standard_normal((64,), dtype=np.float32)),
        },
        # Simulate already-quantized/non-float metadata that should be passed through.
        "meta": mx.array(rng.integers(0, 16, size=(32,), dtype=np.uint32)),
    }

    quantized = quantize_model_weights(weights, bit_width=3, seed=9)
    restored = dequantize_model_weights(quantized)

    assert isinstance(quantized, QuantizedModelWeights)
    assert quantized.stats.total_tensors == 3
    assert quantized.stats.quantized_tensors == 2
    assert quantized.stats.skipped_tensors == 1
    assert quantized.stats.compression_ratio > 1.0

    original_leaves = dict(tree_flatten(weights))
    restored_leaves = dict(tree_flatten(restored))
    assert set(original_leaves) == set(restored_leaves)

    for path, original in original_leaves.items():
        recon = restored_leaves[path]
        assert recon.shape == original.shape
        assert recon.dtype == original.dtype

    np.testing.assert_array_equal(
        np.asarray(restored["meta"]),
        np.asarray(weights["meta"]),
    )
    assert _mse(weights["layer"]["weight"], restored["layer"]["weight"]) > 0.0


def test_model_weight_helpers_support_prod_algorithm():
    rng = np.random.default_rng(6)
    weights = {"w": mx.array(rng.standard_normal((12, 16), dtype=np.float32))}

    quantized = quantize_model_weights(weights, bit_width=3, seed=3, algorithm="prod")
    restored = dequantize_model_weights(quantized)

    assert quantized.stats.quantized_tensors == 1
    assert restored["w"].shape == weights["w"].shape
    assert restored["w"].dtype == weights["w"].dtype


def test_model_weight_helpers_support_bfloat16_tensors():
    rng = np.random.default_rng(14)
    weights = {"w": mx.array(rng.standard_normal((8, 16), dtype=np.float32)).astype(mx.bfloat16)}

    quantized = quantize_model_weights(weights, bit_width=3, seed=1)
    restored = dequantize_model_weights(quantized)

    assert quantized.stats.quantized_tensors == 1
    assert restored["w"].dtype == mx.bfloat16
    assert restored["w"].shape == weights["w"].shape


def test_turboquantize_model_weights_updates_model_in_place():
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

    before = {
        path: np.asarray(value).copy()
        for path, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array)
    }

    quantized = turboquantize_model_weights(model, bit_width=2, seed=4)

    after = {
        path: np.asarray(value).copy()
        for path, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array)
    }

    assert quantized.stats.quantized_tensors > 0
    assert set(before) == set(after)

    changed_paths = [
        path
        for path, before_value in before.items()
        if np.issubdtype(before_value.dtype, np.floating)
        and not np.allclose(before_value, after[path])
    ]
    assert changed_paths


def test_model_weight_helpers_validate_inputs():
    weights = {"w": mx.array(np.ones((4, 8), dtype=np.float32))}

    with pytest.raises(ValueError):
        quantize_model_weights(weights, bit_width=0)

    with pytest.raises(ValueError):
        quantize_model_weights(weights, algorithm="unknown")


def test_save_and_load_turboquant_weights_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    weights = {
        "layer": {
            "weight": mx.array(rng.standard_normal((24, 32), dtype=np.float32)),
            "bias": mx.array(rng.standard_normal((32,), dtype=np.float32)),
        },
        "meta": mx.array(rng.integers(0, 10, size=(32,), dtype=np.uint32)),
    }

    output_dir = tmp_path / "tq-artifact"
    saved = save_turboquant_weights(weights, output_dir, bit_width=3, seed=5)
    restored, loaded_quantized = load_turboquant_weights(output_dir, return_quantized=True)

    assert saved.stats.quantized_tensors == 2
    assert saved.stats.skipped_tensors == 1
    assert (output_dir / "config.json").exists()
    assert (output_dir / "turboquant-metadata.json").exists()
    assert (output_dir / "turboquant-weights.index.json").exists()
    assert (output_dir / "turboquant-passthrough.index.json").exists()

    assert loaded_quantized.stats.quantized_tensors == saved.stats.quantized_tensors
    assert loaded_quantized.stats.skipped_tensors == saved.stats.skipped_tensors

    np.testing.assert_array_equal(
        np.asarray(restored["meta"]),
        np.asarray(weights["meta"]),
    )

    original_w = weights["layer"]["weight"]
    restored_w = restored["layer"]["weight"]
    assert restored_w.dtype == original_w.dtype
    assert restored_w.shape == original_w.shape
    assert _mse(original_w, restored_w) > 0.0


def test_load_turboquant_weights_validates_missing_shard(tmp_path):
    rng = np.random.default_rng(19)
    weights = {"w": mx.array(rng.standard_normal((16, 16), dtype=np.float32))}

    output_dir = tmp_path / "tq-artifact"
    save_turboquant_weights(weights, output_dir, bit_width=2, seed=3)

    index_path = output_dir / "turboquant-weights.index.json"
    with open(index_path, "r") as f:
        index_payload = json.load(f)
    first_shard = next(iter(index_payload["weight_map"].values()))
    (output_dir / first_shard).unlink()

    with pytest.raises(FileNotFoundError):
        load_turboquant_weights(output_dir)
