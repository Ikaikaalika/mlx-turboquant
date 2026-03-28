import numpy as np
import mlx.core as mx

from mlx_lm.models import cache as mlx_cache

from turboquant_mlx.mlx_lm_integration import (
    MLXLMTurboQuantPatcher,
    TurboQuantBatchKVCache,
    TurboQuantChunkedKVCache,
    TurboQuantKVCache,
    make_turbo_prompt_cache,
    turboquantize_prompt_cache,
)


class _DummyModel:
    def __init__(self, num_layers: int):
        self.layers = [object() for _ in range(num_layers)]


def test_make_turbo_prompt_cache_wraps_default_kv_caches():
    model = _DummyModel(num_layers=3)
    prompt_cache = make_turbo_prompt_cache(model, key_bit_width=3, value_bit_width=3)

    assert len(prompt_cache) == 3
    assert all(isinstance(c, TurboQuantKVCache) for c in prompt_cache)


def test_turboquantized_kv_cache_updates_and_reports_stats():
    rng = np.random.default_rng(0)

    base_cache = mlx_cache.KVCache()
    wrapped = turboquantize_prompt_cache([base_cache], key_bit_width=3, value_bit_width=3)[0]

    keys = mx.array(rng.standard_normal((1, 4, 64, 64), dtype=np.float32))
    values = mx.array(rng.standard_normal((1, 4, 64, 64), dtype=np.float32))

    k_hat, v_hat = wrapped.update_and_fetch(keys, values)

    assert isinstance(wrapped, TurboQuantKVCache)
    assert k_hat.shape == keys.shape
    assert v_hat.shape == values.shape

    stats = wrapped.last_turboquant_stats
    assert stats is not None
    assert stats.quantized_bytes < stats.original_bytes
    assert stats.compression_ratio > 1.0
    # Canonical cache state is compressed; dense tensors are released.
    assert wrapped.keys is None
    assert wrapped.values is None


def test_turboquantize_prompt_cache_wraps_cachelist_entries():
    cache_list = mlx_cache.CacheList(mlx_cache.KVCache(), mlx_cache.KVCache())
    wrapped = turboquantize_prompt_cache([cache_list], key_bit_width=3, value_bit_width=3)

    assert isinstance(wrapped[0], mlx_cache.CacheList)
    assert all(isinstance(c, TurboQuantKVCache) for c in wrapped[0].caches)


def test_mlx_lm_patcher_swaps_make_prompt_cache_and_restores():
    original = mlx_cache.make_prompt_cache

    patcher = MLXLMTurboQuantPatcher(key_bit_width=3, value_bit_width=3).apply()
    try:
        patched_cache = mlx_cache.make_prompt_cache(_DummyModel(num_layers=2))
        assert all(isinstance(c, TurboQuantKVCache) for c in patched_cache)
    finally:
        patcher.restore()

    assert mlx_cache.make_prompt_cache is original


def test_turboquant_kvcache_from_state_remains_usable():
    rng = np.random.default_rng(7)

    cache = TurboQuantKVCache()
    keys = mx.array(rng.standard_normal((1, 2, 16, 32), dtype=np.float32))
    values = mx.array(rng.standard_normal((1, 2, 16, 32), dtype=np.float32))
    cache.update_and_fetch(keys, values)

    restored = TurboQuantKVCache.from_state(cache.state, cache.meta_state)
    assert restored.offset == 16
    assert restored.keys is None
    assert restored.values is None

    new_keys = mx.array(rng.standard_normal((1, 2, 1, 32), dtype=np.float32))
    new_values = mx.array(rng.standard_normal((1, 2, 1, 32), dtype=np.float32))
    k_hat, v_hat = restored.update_and_fetch(new_keys, new_values)

    assert k_hat.shape[-2] == 17
    assert v_hat.shape[-2] == 17


def test_turboquant_batch_kvcache_releases_dense_storage():
    rng = np.random.default_rng(11)
    cache = TurboQuantBatchKVCache(left_padding=[0, 1], key_bit_width=3, value_bit_width=3)

    keys = mx.array(rng.standard_normal((2, 2, 8, 16), dtype=np.float32))
    values = mx.array(rng.standard_normal((2, 2, 8, 16), dtype=np.float32))
    k_hat, v_hat = cache.update_and_fetch(keys, values)

    assert k_hat.shape == keys.shape
    assert v_hat.shape == values.shape
    assert cache.keys is None
    assert cache.values is None


def test_turboquantize_prompt_cache_wraps_chunked_cache():
    chunked = mlx_cache.ChunkedKVCache(chunk_size=1024)
    wrapped = turboquantize_prompt_cache([chunked], key_bit_width=3, value_bit_width=3)[0]
    assert isinstance(wrapped, TurboQuantChunkedKVCache)


def test_turboquant_chunked_kvcache_roundtrip_from_state():
    rng = np.random.default_rng(13)
    cache = TurboQuantChunkedKVCache(chunk_size=1024, key_bit_width=3, value_bit_width=3)

    keys = mx.array(rng.standard_normal((1, 2, 12, 32), dtype=np.float32))
    values = mx.array(rng.standard_normal((1, 2, 12, 32), dtype=np.float32))
    cache.update_and_fetch(keys, values)

    restored = TurboQuantChunkedKVCache.from_state(cache.state, cache.meta_state)
    assert restored.chunk_size == 1024
    assert restored.offset == 12
    assert restored.keys is None
    assert restored.values is None

    new_keys = mx.array(rng.standard_normal((1, 2, 2, 32), dtype=np.float32))
    new_values = mx.array(rng.standard_normal((1, 2, 2, 32), dtype=np.float32))
    k_hat, v_hat = restored.update_and_fetch(new_keys, new_values)

    assert k_hat.shape[-2] == 14
    assert v_hat.shape[-2] == 14
