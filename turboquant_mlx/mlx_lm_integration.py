from __future__ import annotations

import copy
import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .core import dequantize_kv_cache, quantize_kv_cache

try:
    from mlx_lm.models import cache as mlx_cache
except Exception:  # pragma: no cover - handled by helper checks in runtime paths
    mlx_cache = None


@dataclass(frozen=True)
class TurboQuantCacheStats:
    original_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes <= 0:
            return float("inf")
        return float(self.original_bytes / self.quantized_bytes)


class _TurboQuantMixin:
    def _tq_setup(
        self,
        *,
        key_bit_width: int,
        value_bit_width: int,
        seed: int,
        pack: bool,
        cache_id: int = 0,
    ) -> None:
        self._tq_key_bit_width = int(key_bit_width)
        self._tq_value_bit_width = int(value_bit_width)
        self._tq_seed = int(seed)
        self._tq_cache_id = int(cache_id)
        self._tq_pack = bool(pack)
        self._tq_stats: Optional[TurboQuantCacheStats] = None

    @property
    def last_turboquant_stats(self) -> Optional[TurboQuantCacheStats]:
        return self._tq_stats

    def _tq_quantize_pair(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        cache = quantize_kv_cache(
            keys=keys,
            values=values,
            key_bit_width=self._tq_key_bit_width,
            value_bit_width=self._tq_value_bit_width,
            seed=self._tq_seed + self._tq_cache_id,
            pack=self._tq_pack,
        )
        keys_hat, values_hat = dequantize_kv_cache(cache)

        original_bytes = int(np.asarray(keys).nbytes + np.asarray(values).nbytes)
        quantized_bytes = int(cache.key_codes.storage_bytes() + cache.value_codes.storage_bytes())
        self._tq_stats = TurboQuantCacheStats(
            original_bytes=original_bytes,
            quantized_bytes=quantized_bytes,
        )

        return keys_hat.astype(keys.dtype), values_hat.astype(values.dtype)

    def _tq_write_back(self, keys: mx.array, values: mx.array) -> None:
        if getattr(self, "keys", None) is None or getattr(self, "values", None) is None:
            return

        seq_len = int(keys.shape[2])
        self.keys[..., :seq_len, :] = keys
        self.values[..., :seq_len, :] = values

    def _tq_current_kv(self) -> tuple[mx.array, mx.array]:
        state = self.state
        if isinstance(state, tuple) and len(state) >= 2:
            return state[0], state[1]
        raise RuntimeError("Unexpected cache state format")


def _ensure_mlx_lm_available() -> ModuleType:
    if mlx_cache is None:
        raise RuntimeError(
            "mlx_lm is not available. Install mlx-lm to use turboquant_mlx.mlx_lm_integration"
        )
    return mlx_cache


def _copy_cache_state(dst: Any, src: Any) -> None:
    # Some mlx_lm cache classes (e.g. KVCache) raise when state is accessed
    # before first update because `keys`/`values` are still None.
    if hasattr(src, "keys") and getattr(src, "keys") is None:
        if hasattr(dst, "keys"):
            dst.keys = None
        if hasattr(dst, "values"):
            dst.values = None
    else:
        dst.state = src.state
    try:
        dst.meta_state = src.meta_state
    except Exception:
        pass

    for attr in (
        "offset",
        "_idx",
        "_offset",
        "rotated",
        "keep",
        "max_size",
        "left_padding",
        "start_position",
        "chunk_size",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, copy.deepcopy(getattr(src, attr)))


if mlx_cache is not None:

    class TurboQuantConcatenateKVCache(_TurboQuantMixin, mlx_cache.ConcatenateKVCache):
        def __init__(
            self,
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__()
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantConcatenateKVCache":
            obj = cls(**kwargs)
            _copy_cache_state(obj, cache)
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            obj = cls()
            obj.state = state
            try:
                obj.meta_state = meta_state
            except Exception:
                pass
            return obj

        def update_and_fetch(self, keys, values):
            keys, values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_quantize_pair(keys, values)
            self._tq_write_back(keys_hat, values_hat)
            return self._tq_current_kv()


    class TurboQuantKVCache(_TurboQuantMixin, mlx_cache.KVCache):
        def __init__(
            self,
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__()
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantKVCache":
            obj = cls(**kwargs)
            _copy_cache_state(obj, cache)
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            obj = cls()
            obj.state = state
            try:
                obj.meta_state = meta_state
            except Exception:
                pass
            return obj

        def update_and_fetch(self, keys, values):
            keys, values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_quantize_pair(keys, values)
            self._tq_write_back(keys_hat, values_hat)
            return self._tq_current_kv()

        def to_quantized(self, group_size: int = 64, bits: int = 4):
            # Keep TurboQuant active even when mlx_lm tries to switch to built-in quantization.
            return self


    class TurboQuantRotatingKVCache(_TurboQuantMixin, mlx_cache.RotatingKVCache):
        def __init__(
            self,
            max_size: int,
            keep: int = 0,
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__(max_size=max_size, keep=keep)
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantRotatingKVCache":
            obj = cls(max_size=cache.max_size, keep=cache.keep, **kwargs)
            _copy_cache_state(obj, cache)
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            keep, max_size, _, _ = map(int, meta_state)
            obj = cls(max_size=max_size, keep=keep)
            obj.state = state
            obj.meta_state = meta_state
            return obj

        def update_and_fetch(self, keys, values):
            keys, values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_quantize_pair(keys, values)
            self._tq_write_back(keys_hat, values_hat)
            return self._tq_current_kv()

        def to_quantized(self, group_size: int = 64, bits: int = 4):
            return self


    class TurboQuantBatchKVCache(_TurboQuantMixin, mlx_cache.BatchKVCache):
        def __init__(
            self,
            left_padding: List[int],
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__(left_padding=left_padding)
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantBatchKVCache":
            left_padding = np.asarray(cache.left_padding).astype(np.int32).tolist()
            obj = cls(left_padding=left_padding, **kwargs)
            _copy_cache_state(obj, cache)
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            _, _, _, left_padding = state
            obj = cls(left_padding=np.asarray(left_padding).astype(np.int32).tolist())
            obj.state = state
            try:
                obj.meta_state = meta_state
            except Exception:
                pass
            return obj

        def update_and_fetch(self, keys, values):
            keys, values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_quantize_pair(keys, values)
            self._tq_write_back(keys_hat, values_hat)
            return self._tq_current_kv()


    class TurboQuantBatchRotatingKVCache(_TurboQuantMixin, mlx_cache.BatchRotatingKVCache):
        def __init__(
            self,
            max_size: int,
            left_padding: List[int],
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__(max_size=max_size, left_padding=left_padding)
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantBatchRotatingKVCache":
            left_padding = np.asarray(cache.left_padding).astype(np.int32).tolist()
            obj = cls(max_size=cache.max_size, left_padding=left_padding, **kwargs)
            _copy_cache_state(obj, cache)
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            _, _, _, left_padding = state
            max_size = int(meta_state[0])
            obj = cls(
                max_size=max_size,
                left_padding=np.asarray(left_padding).astype(np.int32).tolist(),
            )
            obj.state = state
            obj.meta_state = meta_state
            return obj

        def update_and_fetch(self, keys, values):
            keys, values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_quantize_pair(keys, values)
            self._tq_write_back(keys_hat, values_hat)
            return self._tq_current_kv()

else:  # pragma: no cover - mlx_lm unavailable fallback types

    class TurboQuantConcatenateKVCache:  # type: ignore[override]
        pass

    class TurboQuantKVCache:  # type: ignore[override]
        pass

    class TurboQuantRotatingKVCache:  # type: ignore[override]
        pass

    class TurboQuantBatchKVCache:  # type: ignore[override]
        pass

    class TurboQuantBatchRotatingKVCache:  # type: ignore[override]
        pass


if mlx_cache is not None:
    # Register classes for mlx_lm cache (de)serialization resolution.
    mlx_cache.TurboQuantKVCache = TurboQuantKVCache
    mlx_cache.TurboQuantRotatingKVCache = TurboQuantRotatingKVCache
    mlx_cache.TurboQuantBatchKVCache = TurboQuantBatchKVCache
    mlx_cache.TurboQuantBatchRotatingKVCache = TurboQuantBatchRotatingKVCache
    mlx_cache.TurboQuantConcatenateKVCache = TurboQuantConcatenateKVCache


def _wrap_single_cache(cache_obj: Any, *, key_bit_width: int, value_bit_width: int, seed: int, pack: bool, cache_id: int) -> Any:
    cache_mod = _ensure_mlx_lm_available()

    if isinstance(
        cache_obj,
        (
            TurboQuantConcatenateKVCache,
            TurboQuantKVCache,
            TurboQuantRotatingKVCache,
            TurboQuantBatchKVCache,
            TurboQuantBatchRotatingKVCache,
        ),
    ):
        return cache_obj

    kwargs = {
        "key_bit_width": key_bit_width,
        "value_bit_width": value_bit_width,
        "seed": seed,
        "pack": pack,
        "cache_id": cache_id,
    }

    if isinstance(cache_obj, cache_mod.CacheList):
        wrapped = [
            _wrap_single_cache(
                c,
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id + i,
            )
            for i, c in enumerate(cache_obj.caches)
        ]
        return cache_mod.CacheList(*wrapped)

    if isinstance(cache_obj, cache_mod.BatchRotatingKVCache):
        return TurboQuantBatchRotatingKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.BatchKVCache):
        return TurboQuantBatchKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.RotatingKVCache):
        return TurboQuantRotatingKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.KVCache):
        return TurboQuantKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.ConcatenateKVCache):
        return TurboQuantConcatenateKVCache.from_cache(cache_obj, **kwargs)

    return cache_obj


def turboquantize_prompt_cache(
    prompt_cache: List[Any],
    *,
    key_bit_width: int = 3,
    value_bit_width: int = 3,
    seed: int = 0,
    pack: bool = True,
) -> List[Any]:
    """
    Wrap a prompt cache so KV updates use TurboQuant across supported mlx_lm cache types.

    Unsupported cache entries are left untouched.
    """
    _ensure_mlx_lm_available()

    wrapped: List[Any] = []
    for i, c in enumerate(prompt_cache):
        wrapped.append(
            _wrap_single_cache(
                c,
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=i,
            )
        )
    return wrapped


def make_turbo_prompt_cache(
    model: Any,
    max_kv_size: Optional[int] = None,
    *,
    key_bit_width: int = 3,
    value_bit_width: int = 3,
    seed: int = 0,
    pack: bool = True,
) -> List[Any]:
    """Create an mlx_lm prompt cache and wrap it with TurboQuant adapters."""
    cache_mod = _ensure_mlx_lm_available()
    base_cache = cache_mod.make_prompt_cache(model, max_kv_size=max_kv_size)
    return turboquantize_prompt_cache(
        base_cache,
        key_bit_width=key_bit_width,
        value_bit_width=value_bit_width,
        seed=seed,
        pack=pack,
    )


class MLXLMTurboQuantPatcher:
    """
    Monkey patch mlx_lm cache construction so TurboQuant is used across common mlx_lm features.

    Patched targets include:
    - ``mlx_lm.models.cache.make_prompt_cache``
    - ``mlx_lm.generate.cache.make_prompt_cache`` (indirectly via models.cache)
    - ``mlx_lm.chat.make_prompt_cache``
    - ``mlx_lm.evaluate.make_prompt_cache``
    - ``mlx_lm.cache_prompt.make_prompt_cache``
    - ``mlx_lm.generate.BatchKVCache`` / ``BatchRotatingKVCache``
    """

    def __init__(
        self,
        *,
        key_bit_width: int = 3,
        value_bit_width: int = 3,
        seed: int = 0,
        pack: bool = True,
    ):
        self.key_bit_width = int(key_bit_width)
        self.value_bit_width = int(value_bit_width)
        self.seed = int(seed)
        self.pack = bool(pack)
        self._originals: Dict[Tuple[str, str], Any] = {}
        self._applied = False

    def _patch_attr(self, module: ModuleType, attr: str, value: Any) -> None:
        key = (module.__name__, attr)
        if key not in self._originals:
            self._originals[key] = getattr(module, attr)
        setattr(module, attr, value)

    def apply(self) -> "MLXLMTurboQuantPatcher":
        if self._applied:
            return self

        cache_mod = importlib.import_module("mlx_lm.models.cache")

        # Register wrappers on mlx_lm cache module so serialization helpers can resolve class names.
        cache_mod.TurboQuantKVCache = TurboQuantKVCache
        cache_mod.TurboQuantRotatingKVCache = TurboQuantRotatingKVCache
        cache_mod.TurboQuantBatchKVCache = TurboQuantBatchKVCache
        cache_mod.TurboQuantBatchRotatingKVCache = TurboQuantBatchRotatingKVCache
        cache_mod.TurboQuantConcatenateKVCache = TurboQuantConcatenateKVCache

        original_make_prompt_cache = cache_mod.make_prompt_cache

        def patched_make_prompt_cache(model: Any, max_kv_size: Optional[int] = None):
            base_cache = original_make_prompt_cache(model, max_kv_size=max_kv_size)
            return turboquantize_prompt_cache(
                base_cache,
                key_bit_width=self.key_bit_width,
                value_bit_width=self.value_bit_width,
                seed=self.seed,
                pack=self.pack,
            )

        self._patch_attr(cache_mod, "make_prompt_cache", patched_make_prompt_cache)

        for module_name in ("mlx_lm.chat", "mlx_lm.evaluate", "mlx_lm.cache_prompt"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            if hasattr(module, "make_prompt_cache"):
                self._patch_attr(module, "make_prompt_cache", patched_make_prompt_cache)

        try:
            generate_mod = importlib.import_module("mlx_lm.generate")

            key_bits = self.key_bit_width
            value_bits = self.value_bit_width
            seed = self.seed
            pack = self.pack

            class PatchedBatchKVCache(TurboQuantBatchKVCache):
                def __init__(self, left_padding):
                    super().__init__(
                        left_padding=left_padding,
                        key_bit_width=key_bits,
                        value_bit_width=value_bits,
                        seed=seed,
                        pack=pack,
                    )

            class PatchedBatchRotatingKVCache(TurboQuantBatchRotatingKVCache):
                def __init__(self, max_size, left_padding):
                    super().__init__(
                        max_size=max_size,
                        left_padding=left_padding,
                        key_bit_width=key_bits,
                        value_bit_width=value_bits,
                        seed=seed,
                        pack=pack,
                    )

            self._patch_attr(generate_mod, "BatchKVCache", PatchedBatchKVCache)
            self._patch_attr(generate_mod, "BatchRotatingKVCache", PatchedBatchRotatingKVCache)
        except Exception:
            pass

        self._applied = True
        return self

    def restore(self) -> None:
        if not self._applied:
            return

        for (module_name, attr), original in self._originals.items():
            module = importlib.import_module(module_name)
            setattr(module, attr, original)

        self._originals.clear()
        self._applied = False


@contextmanager
def turboquantize_mlx_lm(
    *,
    key_bit_width: int = 3,
    value_bit_width: int = 3,
    seed: int = 0,
    pack: bool = True,
) -> Iterator[MLXLMTurboQuantPatcher]:
    """Context manager that temporarily patches mlx_lm to use TurboQuant caches."""
    patcher = MLXLMTurboQuantPatcher(
        key_bit_width=key_bit_width,
        value_bit_width=value_bit_width,
        seed=seed,
        pack=pack,
    ).apply()
    try:
        yield patcher
    finally:
        patcher.restore()


def patch_mlx_lm(
    *,
    key_bit_width: int = 3,
    value_bit_width: int = 3,
    seed: int = 0,
    pack: bool = True,
) -> MLXLMTurboQuantPatcher:
    """Apply persistent mlx_lm TurboQuant patching and return the patcher."""
    return MLXLMTurboQuantPatcher(
        key_bit_width=key_bit_width,
        value_bit_width=value_bit_width,
        seed=seed,
        pack=pack,
    ).apply()


__all__ = [
    "TurboQuantCacheStats",
    "TurboQuantKVCache",
    "TurboQuantRotatingKVCache",
    "TurboQuantBatchKVCache",
    "TurboQuantBatchRotatingKVCache",
    "TurboQuantConcatenateKVCache",
    "turboquantize_prompt_cache",
    "make_turbo_prompt_cache",
    "MLXLMTurboQuantPatcher",
    "turboquantize_mlx_lm",
    "patch_mlx_lm",
]
