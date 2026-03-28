from __future__ import annotations

import importlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .core import (
    PackedMSECodes,
    PackedProdCodes,
    QuantizedKVCache,
    TurboQuantMSE,
    TurboQuantProd,
    dequantize_kv_cache,
    quantize_kv_cache,
)

try:
    from mlx_lm.models import cache as mlx_cache
except Exception:  # pragma: no cover - handled by helper checks in runtime paths
    mlx_cache = None


_DTYPE_TO_NAME: Dict[Any, str] = {
    mx.float16: "float16",
    mx.float32: "float32",
    mx.bfloat16: "bfloat16",
}
_NAME_TO_DTYPE: Dict[str, Any] = {v: k for k, v in _DTYPE_TO_NAME.items()}


@dataclass(frozen=True)
class TurboQuantCacheStats:
    original_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes <= 0:
            return float("inf")
        return float(self.original_bytes / self.quantized_bytes)


@dataclass(frozen=True)
class _TurboQuantConfig:
    key_bit_width: int
    value_bit_width: int
    seed: int
    pack: bool
    cache_id: int


def _dtype_name(dtype: Any) -> str:
    for known, name in _DTYPE_TO_NAME.items():
        if dtype == known:
            return name
    return "float32"


def _dtype_from_name(name: str) -> Any:
    return _NAME_TO_DTYPE.get(name, mx.float32)


def _ensure_mlx_lm_available() -> ModuleType:
    if mlx_cache is None:
        raise RuntimeError(
            "mlx_lm is not available. Install mlx-lm to use turboquant_mlx.mlx_lm_integration"
        )
    return mlx_cache


class _TurboQuantCompressedMixin:
    """Mixin that stores compressed KV codes as canonical cache state."""

    def _tq_setup(
        self,
        *,
        key_bit_width: int,
        value_bit_width: int,
        seed: int,
        pack: bool,
        cache_id: int = 0,
    ) -> None:
        self._tq_cfg = _TurboQuantConfig(
            key_bit_width=int(key_bit_width),
            value_bit_width=int(value_bit_width),
            seed=int(seed),
            pack=bool(pack),
            cache_id=int(cache_id),
        )
        self._tq_cache: Optional[QuantizedKVCache] = None
        self._tq_stats: Optional[TurboQuantCacheStats] = None
        self._tq_key_dim: int = 0
        self._tq_value_dim: int = 0
        self._tq_key_dtype: Any = mx.float32
        self._tq_value_dtype: Any = mx.float32

    @property
    def last_turboquant_stats(self) -> Optional[TurboQuantCacheStats]:
        return self._tq_stats

    def _tq_update_from_dense(
        self,
        keys: mx.array,
        values: mx.array,
        *,
        key_dtype: Any,
        value_dtype: Any,
    ) -> tuple[mx.array, mx.array]:
        qcache = quantize_kv_cache(
            keys=keys,
            values=values,
            key_bit_width=self._tq_cfg.key_bit_width,
            value_bit_width=self._tq_cfg.value_bit_width,
            seed=self._tq_cfg.seed + self._tq_cfg.cache_id,
            pack=self._tq_cfg.pack,
        )
        keys_hat, values_hat = dequantize_kv_cache(qcache)

        self._tq_cache = qcache
        self._tq_key_dim = int(keys.shape[-1])
        self._tq_value_dim = int(values.shape[-1])
        self._tq_key_dtype = key_dtype
        self._tq_value_dtype = value_dtype

        original_bytes = int(np.asarray(keys).nbytes + np.asarray(values).nbytes)
        quantized_bytes = int(
            qcache.key_codes.storage_bytes() + qcache.value_codes.storage_bytes()
        )
        self._tq_stats = TurboQuantCacheStats(
            original_bytes=original_bytes,
            quantized_bytes=quantized_bytes,
        )

        return keys_hat.astype(key_dtype), values_hat.astype(value_dtype)

    def _tq_decode_to_dense(self) -> tuple[mx.array, mx.array]:
        if self._tq_cache is None:
            raise RuntimeError("No TurboQuant cache is available to decode.")
        keys, values = dequantize_kv_cache(self._tq_cache)
        return keys.astype(self._tq_key_dtype), values.astype(self._tq_value_dtype)

    def _tq_clear_dense_storage(self) -> None:
        if hasattr(self, "keys"):
            self.keys = None
        if hasattr(self, "values"):
            self.values = None

    def _tq_serialize_quantized_state(self) -> Any:
        if self._tq_cache is None:
            return []

        key_codes = self._tq_cache.key_codes
        value_codes = self._tq_cache.value_codes

        key_indices = key_codes.packed_indices
        if key_indices is None:
            key_indices = mx.array(np.zeros((0,), dtype=np.uint8))

        leading_shape = mx.array(np.asarray(key_codes.leading_shape, dtype=np.int32))

        return (
            key_indices,
            key_codes.packed_qjl,
            key_codes.residual_norms,
            key_codes.norms,
            value_codes.packed_indices,
            value_codes.norms,
            leading_shape,
        )

    def _tq_deserialize_quantized_state(self, state: Any) -> None:
        if not state:
            self._tq_cache = None
            return

        if self._tq_key_dim <= 0 or self._tq_value_dim <= 0:
            raise ValueError(
                "Cannot deserialize TurboQuant state before key/value dimensions are known."
            )

        (
            key_indices,
            key_signs,
            key_residual_norms,
            key_norms,
            value_indices,
            value_norms,
            leading_shape_array,
        ) = state

        leading_shape = tuple(int(v) for v in np.asarray(leading_shape_array).tolist())

        key_codes = PackedProdCodes(
            packed_indices=(
                None
                if self._tq_cfg.key_bit_width == 1
                else key_indices.astype(mx.uint8)
            ),
            packed_qjl=key_signs.astype(mx.uint8),
            residual_norms=key_residual_norms.astype(mx.float32),
            norms=key_norms.astype(mx.float32),
            leading_shape=leading_shape,
            dimension=self._tq_key_dim,
            bit_width=self._tq_cfg.key_bit_width,
            packed=True,
        )
        value_codes = PackedMSECodes(
            packed_indices=value_indices.astype(mx.uint8),
            norms=value_norms.astype(mx.float32),
            leading_shape=leading_shape,
            dimension=self._tq_value_dim,
            bit_width=self._tq_cfg.value_bit_width,
            packed=True,
        )

        key_quantizer = TurboQuantProd(
            dimension=self._tq_key_dim,
            bit_width=self._tq_cfg.key_bit_width,
            seed=self._tq_cfg.seed + self._tq_cfg.cache_id,
        )
        value_quantizer = TurboQuantMSE(
            dimension=self._tq_value_dim,
            bit_width=self._tq_cfg.value_bit_width,
            seed=self._tq_cfg.seed + 13 + self._tq_cfg.cache_id,
        )

        self._tq_cache = QuantizedKVCache(
            key_quantizer=key_quantizer,
            value_quantizer=value_quantizer,
            key_codes=key_codes,
            value_codes=value_codes,
        )


class _LegacyTurboQuantMixin:
    """Fallback mixin for cache types not yet using compressed canonical state."""

    _warned_legacy: bool = False

    def _legacy_tq_setup(
        self,
        *,
        key_bit_width: int,
        value_bit_width: int,
        seed: int,
        pack: bool,
        cache_id: int = 0,
    ) -> None:
        self._legacy_tq_cfg = _TurboQuantConfig(
            key_bit_width=int(key_bit_width),
            value_bit_width=int(value_bit_width),
            seed=int(seed),
            pack=bool(pack),
            cache_id=int(cache_id),
        )
        self._legacy_stats: Optional[TurboQuantCacheStats] = None

    @property
    def last_turboquant_stats(self) -> Optional[TurboQuantCacheStats]:
        return self._legacy_stats

    def _legacy_quantize_and_writeback(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if not self.__class__._warned_legacy:
            warnings.warn(
                f"{self.__class__.__name__} currently uses compatibility mode (no compressed canonical state).",
                RuntimeWarning,
                stacklevel=2,
            )
            self.__class__._warned_legacy = True

        qcache = quantize_kv_cache(
            keys=keys,
            values=values,
            key_bit_width=self._legacy_tq_cfg.key_bit_width,
            value_bit_width=self._legacy_tq_cfg.value_bit_width,
            seed=self._legacy_tq_cfg.seed + self._legacy_tq_cfg.cache_id,
            pack=self._legacy_tq_cfg.pack,
        )
        keys_hat, values_hat = dequantize_kv_cache(qcache)
        self._legacy_stats = TurboQuantCacheStats(
            original_bytes=int(np.asarray(keys).nbytes + np.asarray(values).nbytes),
            quantized_bytes=int(
                qcache.key_codes.storage_bytes() + qcache.value_codes.storage_bytes()
            ),
        )

        if getattr(self, "keys", None) is not None and getattr(self, "values", None) is not None:
            seq_len = int(keys_hat.shape[2])
            self.keys[..., :seq_len, :] = keys_hat.astype(self.keys.dtype)
            self.values[..., :seq_len, :] = values_hat.astype(self.values.dtype)

        return keys_hat.astype(keys.dtype), values_hat.astype(values.dtype)


if mlx_cache is not None:

    class TurboQuantKVCache(_TurboQuantCompressedMixin, mlx_cache.KVCache):
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

        def _materialize_dense_base(self) -> None:
            if self._tq_cache is None:
                self.keys = None
                self.values = None
                return
            keys, values = self._tq_decode_to_dense()
            self.keys = keys
            self.values = values
            self.offset = int(keys.shape[2])

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantKVCache":
            obj = cls(**kwargs)
            if getattr(cache, "keys", None) is not None:
                keys, values = cache.state
                obj._tq_update_from_dense(
                    keys,
                    values,
                    key_dtype=keys.dtype,
                    value_dtype=values.dtype,
                )
                obj.offset = int(keys.shape[2])
            else:
                obj.offset = int(getattr(cache, "offset", 0))
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            if not meta_state or len(meta_state) < 10:
                obj = cls()
                obj.state = state
                return obj

            (
                offset,
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
            ) = meta_state
            obj = cls(
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            obj.offset = int(offset)
            obj._tq_key_dim = int(key_dim)
            obj._tq_value_dim = int(value_dim)
            obj._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            obj._tq_value_dtype = _dtype_from_name(str(value_dtype_name))
            obj.state = state
            return obj

        @property
        def meta_state(self):
            return tuple(
                map(
                    str,
                    (
                        int(getattr(self, "offset", 0)),
                        self._tq_cfg.key_bit_width,
                        self._tq_cfg.value_bit_width,
                        self._tq_cfg.seed,
                        int(self._tq_cfg.pack),
                        self._tq_key_dim,
                        self._tq_value_dim,
                        _dtype_name(self._tq_key_dtype),
                        _dtype_name(self._tq_value_dtype),
                        self._tq_cfg.cache_id,
                    ),
                )
            )

        @meta_state.setter
        def meta_state(self, v):
            (
                offset,
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
            ) = v
            self.offset = int(offset)
            self._tq_cfg = _TurboQuantConfig(
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            self._tq_key_dim = int(key_dim)
            self._tq_value_dim = int(value_dim)
            self._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            self._tq_value_dtype = _dtype_from_name(str(value_dtype_name))

        @property
        def state(self):
            return self._tq_serialize_quantized_state()

        @state.setter
        def state(self, v):
            self._tq_deserialize_quantized_state(v)
            if self._tq_cache is not None:
                self.offset = int(self._tq_cache.key_codes.leading_shape[-1])

        def update_and_fetch(self, keys, values):
            self._materialize_dense_base()
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_update_from_dense(
                dense_keys,
                dense_values,
                key_dtype=keys.dtype,
                value_dtype=values.dtype,
            )
            self._tq_clear_dense_storage()
            self.offset = int(keys_hat.shape[2])
            return keys_hat, values_hat

        def trim(self, n):
            if self._tq_cache is None:
                return 0

            self._materialize_dense_base()
            trimmed = super().trim(n)
            if self.offset > 0:
                dense_keys, dense_values = super().state
                self._tq_update_from_dense(
                    dense_keys,
                    dense_values,
                    key_dtype=self._tq_key_dtype,
                    value_dtype=self._tq_value_dtype,
                )
            else:
                self._tq_cache = None

            self._tq_clear_dense_storage()
            return trimmed

        def to_quantized(self, group_size: int = 64, bits: int = 4):
            return self


    class TurboQuantChunkedKVCache(_TurboQuantCompressedMixin, mlx_cache.ChunkedKVCache):
        def __init__(
            self,
            chunk_size: int,
            *,
            key_bit_width: int = 3,
            value_bit_width: int = 3,
            seed: int = 0,
            pack: bool = True,
            cache_id: int = 0,
        ):
            super().__init__(chunk_size=chunk_size)
            self._tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )
            self._tq_start_position = int(self.start_position)

        def _materialize_dense_base(self) -> None:
            self.start_position = int(self._tq_start_position)
            if self._tq_cache is None:
                self.keys = None
                self.values = None
                self.offset = int(self.start_position)
                return

            keys, values = self._tq_decode_to_dense()
            self.keys = keys
            self.values = values
            self.offset = int(self.start_position + keys.shape[2])

        def _update_from_current_dense(self) -> None:
            if self.keys is None or self.values is None:
                self._tq_cache = None
                return
            active_len = max(0, int(self.offset - self.start_position))
            self._tq_update_from_dense(
                self.keys[..., :active_len, :],
                self.values[..., :active_len, :],
                key_dtype=self._tq_key_dtype,
                value_dtype=self._tq_value_dtype,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantChunkedKVCache":
            chunk_size = int(getattr(cache, "chunk_size"))
            obj = cls(chunk_size=chunk_size, **kwargs)
            obj.start_position = int(getattr(cache, "start_position", 0))
            obj._tq_start_position = int(obj.start_position)
            obj.offset = int(getattr(cache, "offset", obj.start_position))
            if getattr(cache, "keys", None) is not None:
                dense_keys, dense_values = cache.state
                obj._tq_update_from_dense(
                    dense_keys,
                    dense_values,
                    key_dtype=dense_keys.dtype,
                    value_dtype=dense_values.dtype,
                )
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            if not meta_state or len(meta_state) < 12:
                obj = cls(chunk_size=256)
                obj.state = state
                return obj

            (
                offset,
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
                chunk_size,
                start_position,
            ) = meta_state
            obj = cls(
                chunk_size=int(chunk_size),
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            obj.offset = int(offset)
            obj.start_position = int(start_position)
            obj._tq_start_position = int(start_position)
            obj._tq_key_dim = int(key_dim)
            obj._tq_value_dim = int(value_dim)
            obj._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            obj._tq_value_dtype = _dtype_from_name(str(value_dtype_name))
            obj.state = state
            return obj

        @property
        def meta_state(self):
            return tuple(
                map(
                    str,
                    (
                        int(getattr(self, "offset", 0)),
                        self._tq_cfg.key_bit_width,
                        self._tq_cfg.value_bit_width,
                        self._tq_cfg.seed,
                        int(self._tq_cfg.pack),
                        self._tq_key_dim,
                        self._tq_value_dim,
                        _dtype_name(self._tq_key_dtype),
                        _dtype_name(self._tq_value_dtype),
                        self._tq_cfg.cache_id,
                        int(self.chunk_size),
                        int(self._tq_start_position),
                    ),
                )
            )

        @meta_state.setter
        def meta_state(self, v):
            (
                offset,
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
                chunk_size,
                start_position,
            ) = v
            self.offset = int(offset)
            self._tq_cfg = _TurboQuantConfig(
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            self._tq_key_dim = int(key_dim)
            self._tq_value_dim = int(value_dim)
            self._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            self._tq_value_dtype = _dtype_from_name(str(value_dtype_name))
            self.chunk_size = int(chunk_size)
            self.start_position = int(start_position)
            self._tq_start_position = int(start_position)

        @property
        def state(self):
            return self._tq_serialize_quantized_state()

        @state.setter
        def state(self, v):
            self._tq_deserialize_quantized_state(v)
            if self._tq_cache is not None:
                active_len = int(self._tq_cache.key_codes.leading_shape[-1])
                self.offset = int(self._tq_start_position + active_len)

        def maybe_trim_front(self):
            self._materialize_dense_base()
            super().maybe_trim_front()
            self._tq_start_position = int(self.start_position)
            self._update_from_current_dense()
            self._tq_clear_dense_storage()

        def update_and_fetch(self, keys, values):
            self._materialize_dense_base()
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            keys_hat, values_hat = self._tq_update_from_dense(
                dense_keys,
                dense_values,
                key_dtype=keys.dtype,
                value_dtype=values.dtype,
            )
            self._tq_start_position = int(self.start_position)
            self._tq_clear_dense_storage()
            self.offset = int(self.start_position + keys_hat.shape[2])
            return keys_hat, values_hat

        def trim(self, n):
            if self._tq_cache is None:
                return 0

            self._materialize_dense_base()
            trimmed = super().trim(n)
            if self.offset > self.start_position:
                self._update_from_current_dense()
            else:
                self._tq_cache = None
            self._tq_start_position = int(self.start_position)
            self._tq_clear_dense_storage()
            return trimmed

        def to_quantized(self, group_size: int = 64, bits: int = 4):
            return self


    class TurboQuantBatchKVCache(_TurboQuantCompressedMixin, mlx_cache.BatchKVCache):
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
            self._tq_offset = self.offset
            self._tq_left_padding = self.left_padding
            self._tq_idx = int(self._idx)

        def _materialize_dense_base(self) -> None:
            self.offset = self._tq_offset
            self.left_padding = self._tq_left_padding
            self._idx = int(self._tq_idx)

            if self._tq_cache is None:
                self.keys = None
                self.values = None
                return

            keys, values = self._tq_decode_to_dense()
            self.keys = keys
            self.values = values

        def _sync_snapshot_from_base(self) -> None:
            self._tq_offset = self.offset
            self._tq_left_padding = self.left_padding
            self._tq_idx = int(self._idx)

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantBatchKVCache":
            left_padding = (
                np.asarray(cache.left_padding).astype(np.int32).tolist()
                if getattr(cache, "left_padding", None) is not None
                else []
            )
            obj = cls(left_padding=left_padding, **kwargs)
            obj._tq_offset = copy_array(getattr(cache, "offset", obj.offset))
            obj._tq_left_padding = copy_array(getattr(cache, "left_padding", obj.left_padding))
            obj._tq_idx = int(getattr(cache, "_idx", obj._idx))

            if getattr(cache, "keys", None) is not None:
                dense_keys, dense_values, _, _ = cache.state
                obj._tq_update_from_dense(
                    dense_keys,
                    dense_values,
                    key_dtype=dense_keys.dtype,
                    value_dtype=dense_values.dtype,
                )
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            if not meta_state or len(meta_state) < 9:
                obj = cls(left_padding=[])
                obj.state = state
                return obj

            (
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
            ) = meta_state

            left_padding = []
            if state and len(state) >= 10:
                left_padding = np.asarray(state[8]).astype(np.int32).tolist()

            obj = cls(
                left_padding=left_padding,
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            obj._tq_key_dim = int(key_dim)
            obj._tq_value_dim = int(value_dim)
            obj._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            obj._tq_value_dtype = _dtype_from_name(str(value_dtype_name))
            obj.state = state
            return obj

        @property
        def meta_state(self):
            return tuple(
                map(
                    str,
                    (
                        self._tq_cfg.key_bit_width,
                        self._tq_cfg.value_bit_width,
                        self._tq_cfg.seed,
                        int(self._tq_cfg.pack),
                        self._tq_key_dim,
                        self._tq_value_dim,
                        _dtype_name(self._tq_key_dtype),
                        _dtype_name(self._tq_value_dtype),
                        self._tq_cfg.cache_id,
                    ),
                )
            )

        @meta_state.setter
        def meta_state(self, v):
            (
                key_bits,
                value_bits,
                seed,
                pack,
                key_dim,
                value_dim,
                key_dtype_name,
                value_dtype_name,
                cache_id,
            ) = v
            self._tq_cfg = _TurboQuantConfig(
                key_bit_width=int(key_bits),
                value_bit_width=int(value_bits),
                seed=int(seed),
                pack=bool(int(pack)),
                cache_id=int(cache_id),
            )
            self._tq_key_dim = int(key_dim)
            self._tq_value_dim = int(value_dim)
            self._tq_key_dtype = _dtype_from_name(str(key_dtype_name))
            self._tq_value_dtype = _dtype_from_name(str(value_dtype_name))

        @property
        def state(self):
            core = self._tq_serialize_quantized_state()
            if not core:
                return []
            return (
                core[0],
                core[1],
                core[2],
                core[3],
                core[4],
                core[5],
                core[6],
                self._tq_offset,
                self._tq_left_padding,
                mx.array(np.array([self._tq_idx], dtype=np.int32)),
            )

        @state.setter
        def state(self, v):
            if not v:
                self._tq_cache = None
                return
            core = v[:7]
            self._tq_deserialize_quantized_state(core)
            self._tq_offset = v[7]
            self._tq_left_padding = v[8]
            self._tq_idx = int(np.asarray(v[9])[0])
            self.offset = self._tq_offset
            self.left_padding = self._tq_left_padding
            self._idx = self._tq_idx

        def update_and_fetch(self, keys, values):
            self._materialize_dense_base()
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            self._sync_snapshot_from_base()
            keys_hat, values_hat = self._tq_update_from_dense(
                dense_keys,
                dense_values,
                key_dtype=keys.dtype,
                value_dtype=values.dtype,
            )
            self._tq_clear_dense_storage()
            return keys_hat, values_hat

        def filter(self, batch_indices):
            self._materialize_dense_base()
            super().filter(batch_indices)
            self._sync_snapshot_from_base()

            if self.keys is not None:
                self._tq_update_from_dense(
                    self.keys[..., : self._idx, :],
                    self.values[..., : self._idx, :],
                    key_dtype=self._tq_key_dtype,
                    value_dtype=self._tq_value_dtype,
                )
            else:
                self._tq_cache = None
            self._tq_clear_dense_storage()

        def extend(self, other):
            self._materialize_dense_base()
            if hasattr(other, "_materialize_dense_base"):
                other._materialize_dense_base()
            super().extend(other)
            self._sync_snapshot_from_base()

            self._tq_update_from_dense(
                self.keys[..., : self._idx, :],
                self.values[..., : self._idx, :],
                key_dtype=self._tq_key_dtype,
                value_dtype=self._tq_value_dtype,
            )
            self._tq_clear_dense_storage()


    class TurboQuantConcatenateKVCache(_LegacyTurboQuantMixin, mlx_cache.ConcatenateKVCache):
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
            self._legacy_tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantConcatenateKVCache":
            obj = cls(**kwargs)
            obj.keys = getattr(cache, "keys", None)
            obj.values = getattr(cache, "values", None)
            obj.offset = int(getattr(cache, "offset", 0))
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
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            return self._legacy_quantize_and_writeback(dense_keys, dense_values)


    class TurboQuantRotatingKVCache(_LegacyTurboQuantMixin, mlx_cache.RotatingKVCache):
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
            self._legacy_tq_setup(
                key_bit_width=key_bit_width,
                value_bit_width=value_bit_width,
                seed=seed,
                pack=pack,
                cache_id=cache_id,
            )

        @classmethod
        def from_cache(cls, cache: Any, **kwargs) -> "TurboQuantRotatingKVCache":
            obj = cls(max_size=cache.max_size, keep=cache.keep, **kwargs)
            obj.state = cache.state
            obj.meta_state = cache.meta_state
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            keep, max_size, _, _ = map(int, meta_state)
            obj = cls(max_size=max_size, keep=keep)
            obj.state = state
            obj.meta_state = meta_state
            return obj

        def update_and_fetch(self, keys, values):
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            return self._legacy_quantize_and_writeback(dense_keys, dense_values)

        def to_quantized(self, group_size: int = 64, bits: int = 4):
            return self


    class TurboQuantBatchRotatingKVCache(_LegacyTurboQuantMixin, mlx_cache.BatchRotatingKVCache):
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
            self._legacy_tq_setup(
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
            obj.state = cache.state
            obj.meta_state = cache.meta_state
            return obj

        @classmethod
        def from_state(cls, state, meta_state):
            _, _, _, left_padding = state
            max_size = int(meta_state[0])
            obj = cls(max_size=max_size, left_padding=np.asarray(left_padding).tolist())
            obj.state = state
            obj.meta_state = meta_state
            return obj

        def update_and_fetch(self, keys, values):
            dense_keys, dense_values = super().update_and_fetch(keys, values)
            return self._legacy_quantize_and_writeback(dense_keys, dense_values)


else:  # pragma: no cover - mlx_lm unavailable fallback types

    class TurboQuantKVCache:  # type: ignore[override]
        pass

    class TurboQuantChunkedKVCache:  # type: ignore[override]
        pass

    class TurboQuantBatchKVCache:  # type: ignore[override]
        pass

    class TurboQuantConcatenateKVCache:  # type: ignore[override]
        pass

    class TurboQuantRotatingKVCache:  # type: ignore[override]
        pass

    class TurboQuantBatchRotatingKVCache:  # type: ignore[override]
        pass


def copy_array(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, mx.array):
        return mx.array(np.asarray(x))
    return x


if mlx_cache is not None:
    # Register classes for mlx_lm cache (de)serialization resolution.
    mlx_cache.TurboQuantKVCache = TurboQuantKVCache
    mlx_cache.TurboQuantChunkedKVCache = TurboQuantChunkedKVCache
    mlx_cache.TurboQuantBatchKVCache = TurboQuantBatchKVCache
    mlx_cache.TurboQuantConcatenateKVCache = TurboQuantConcatenateKVCache
    mlx_cache.TurboQuantRotatingKVCache = TurboQuantRotatingKVCache
    mlx_cache.TurboQuantBatchRotatingKVCache = TurboQuantBatchRotatingKVCache


def _wrap_single_cache(
    cache_obj: Any,
    *,
    key_bit_width: int,
    value_bit_width: int,
    seed: int,
    pack: bool,
    cache_id: int,
) -> Any:
    cache_mod = _ensure_mlx_lm_available()

    if isinstance(
        cache_obj,
        (
            TurboQuantKVCache,
            TurboQuantChunkedKVCache,
            TurboQuantBatchKVCache,
            TurboQuantConcatenateKVCache,
            TurboQuantRotatingKVCache,
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

    if isinstance(cache_obj, cache_mod.BatchKVCache):
        return TurboQuantBatchKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.ChunkedKVCache):
        return TurboQuantChunkedKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.KVCache):
        return TurboQuantKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.BatchRotatingKVCache):
        return TurboQuantBatchRotatingKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.RotatingKVCache):
        return TurboQuantRotatingKVCache.from_cache(cache_obj, **kwargs)
    if isinstance(cache_obj, cache_mod.ConcatenateKVCache):
        return TurboQuantConcatenateKVCache.from_cache(cache_obj, **kwargs)

    warnings.warn(
        f"TurboQuant integration is skipping unsupported cache type: {type(cache_obj)!r}",
        RuntimeWarning,
        stacklevel=2,
    )
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

    Unsupported cache entries are left untouched with a runtime warning.
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

        cache_mod.TurboQuantKVCache = TurboQuantKVCache
        cache_mod.TurboQuantChunkedKVCache = TurboQuantChunkedKVCache
        cache_mod.TurboQuantBatchKVCache = TurboQuantBatchKVCache
        cache_mod.TurboQuantConcatenateKVCache = TurboQuantConcatenateKVCache
        cache_mod.TurboQuantRotatingKVCache = TurboQuantRotatingKVCache
        cache_mod.TurboQuantBatchRotatingKVCache = TurboQuantBatchRotatingKVCache

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
    "TurboQuantChunkedKVCache",
    "TurboQuantBatchKVCache",
    "TurboQuantConcatenateKVCache",
    "TurboQuantRotatingKVCache",
    "TurboQuantBatchRotatingKVCache",
    "turboquantize_prompt_cache",
    "make_turbo_prompt_cache",
    "MLXLMTurboQuantPatcher",
    "turboquantize_mlx_lm",
    "patch_mlx_lm",
]
